package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mrsibe/goat/agent"
	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/ai/faux"
)

// pi 一致性基线：把 pi-agent-core/test/agent-loop.test.ts 与 agent.test.ts
// 中当前 API 可表达的行为锁定下来。阶段 2/3 引入 role-specific 消息、
// convertToLlm、runToolCall 等能力后，只需替换构造部分，断言语义保持不变。

func piParityTextTurn(text string) faux.Turn {
	return faux.Turn{Events: []ai.Event{
		{Type: ai.EventTextDelta, TextDelta: text},
		{Type: ai.EventDone, StopReason: ai.StopReasonStop},
	}}
}

func piParityToolCall(index int, id, name, arguments string) ai.Event {
	return ai.Event{
		Type:         ai.EventToolCallDelta,
		ContentIndex: index,
		ToolCallDelta: &ai.ToolCallDelta{
			ID:             id,
			Name:           name,
			ArgumentsDelta: arguments,
		},
	}
}

func piParityToolTurn(calls ...ai.Event) faux.Turn {
	events := append([]ai.Event(nil), calls...)
	events = append(events, ai.Event{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls})
	return faux.Turn{Events: events}
}

func piParityToolMessageIDs(messages []ai.Message) []string {
	var ids []string
	for _, message := range messages {
		if message.Role() == ai.RoleTool {
			ids = append(ids, ai.AsToolResult(message).ToolCallID)
		}
	}
	return ids
}

// pi: "should emit tool_execution_end in completion order but persist tool results in source order"
func TestPiParityToolEndCompletionOrder(t *testing.T) {
	model := faux.New(
		piParityToolTurn(
			piParityToolCall(0, "call_slow", "slow", `{}`),
			piParityToolCall(1, "call_fast", "fast", `{}`),
		),
		piParityTextTurn("done"),
	)
	slowStarted := make(chan struct{})
	fastDone := make(chan struct{})
	slow := agent.NewTextTool(agent.ToolSpec{Name: "slow"}, func(ctx context.Context, _ string, _ json.RawMessage) (string, error) {
		close(slowStarted)
		select {
		case <-fastDone:
			return "slow result", nil
		case <-ctx.Done():
			return "", ctx.Err()
		}
	})
	fast := agent.NewTextTool(agent.ToolSpec{Name: "fast"}, func(ctx context.Context, _ string, _ json.RawMessage) (string, error) {
		select {
		case <-slowStarted:
			close(fastDone)
			return "fast result", nil
		case <-ctx.Done():
			return "", ctx.Err()
		}
	})
	a := newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{slow, fast}})
	log := subscribe(a)

	result, err := a.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}

	var endIDs []string
	for _, event := range log.snapshot() {
		if event.Type == agent.EventToolEnd {
			endIDs = append(endIDs, event.ToolCall.ID)
		}
	}
	if want := []string{"call_fast", "call_slow"}; !equalStrings(endIDs, want) {
		t.Fatalf("tool_end order = %v, want %v", endIDs, want)
	}
	if got, want := piParityToolMessageIDs(result.Messages), []string{"call_slow", "call_fast"}; !equalStrings(got, want) {
		t.Fatalf("tool result message order = %v, want %v", got, want)
	}

	var turnResults []string
	for _, event := range log.snapshot() {
		if event.Type == agent.EventTurnEnd {
			for _, toolResult := range event.ToolResults {
				turnResults = append(turnResults, toolResult.Text())
			}
		}
	}
	if want := []string{"slow result", "fast result"}; !equalStrings(turnResults, want) {
		t.Fatalf("turn_end tool result order = %v, want %v", turnResults, want)
	}
}

// pi: "should inject queued messages after all tool calls complete"
func TestPiParitySteeringInjectedAfterToolBatch(t *testing.T) {
	var a *agent.Agent
	model := faux.New(
		piParityToolTurn(
			piParityToolCall(0, "call_1", "echo", `{"value":"first"}`),
			piParityToolCall(1, "call_2", "echo", `{"value":"second"}`),
		),
		piParityTextTurn("done"),
	)
	var mu sync.Mutex
	var executed []string
	echo := agent.NewTool(agent.ToolSpec{Name: "echo"}, func(_ context.Context, _ string, raw json.RawMessage) (agent.ToolResult, error) {
		var parsed struct {
			Value string `json:"value"`
		}
		if err := json.Unmarshal(raw, &parsed); err != nil {
			return agent.ToolResult{}, err
		}
		mu.Lock()
		executed = append(executed, parsed.Value)
		first := len(executed) == 1
		mu.Unlock()
		if first {
			a.Steer(ai.TextMessage(ai.RoleUser, "interrupt"))
		}
		return agent.TextResult("ok:" + parsed.Value), nil
	})
	a = newAgent(t, agent.Config{
		Model:         model,
		Tools:         []agent.Tool{echo},
		ToolExecution: agent.ToolExecutionSequential,
	})
	log := subscribe(a)

	if _, err := a.Prompt(context.Background(), "start"); err != nil {
		t.Fatal(err)
	}

	mu.Lock()
	gotExecuted := append([]string(nil), executed...)
	mu.Unlock()
	if want := []string{"first", "second"}; !equalStrings(gotExecuted, want) {
		t.Fatalf("executed = %v, want %v", gotExecuted, want)
	}

	var sequence []string
	for _, event := range log.snapshot() {
		if event.Type != agent.EventMessageStart || event.Message == nil {
			continue
		}
		switch event.Message.Role() {
		case ai.RoleToolResult:
			sequence = append(sequence, "tool:"+ai.AsToolResult(event.Message).ToolCallID)
		case ai.RoleUser:
			sequence = append(sequence, event.Message.Text())
		}
	}
	toolOne := indexOf(sequence, "tool:call_1")
	toolTwo := indexOf(sequence, "tool:call_2")
	interrupt := indexOf(sequence, "interrupt")
	if toolOne < 0 || toolTwo < 0 || interrupt < 0 || toolOne > interrupt || toolTwo > interrupt {
		t.Fatalf("steering must follow both tool results: %v", sequence)
	}

	if got := len(model.Requests); got != 2 {
		t.Fatalf("requests = %d, want 2", got)
	}
	second := model.Requests[1].Messages
	if !containsText(second, "interrupt") || !containsToolID(second, "call_1") || !containsToolID(second, "call_2") {
		t.Fatalf("second request missing steering or tool results: %+v", second)
	}
}

// pi: "answers every tool call of a sequential batch aborted mid-batch"
func TestPiParityAbortMidBatchSequential(t *testing.T) {
	var a *agent.Agent
	model := faux.New(piParityToolTurn(
		piParityToolCall(0, "call_1", "first", `{}`),
		piParityToolCall(1, "call_2", "second", `{}`),
	))
	first := agent.NewTextTool(agent.ToolSpec{Name: "first"}, func(context.Context, string, json.RawMessage) (string, error) {
		return "first", nil
	})
	second := agent.NewTextTool(agent.ToolSpec{Name: "second"}, func(context.Context, string, json.RawMessage) (string, error) {
		return "second", nil
	})
	a = newAgent(t, agent.Config{
		Model: model,
		Tools: []agent.Tool{first, second},
		BeforeToolCall: func(_ context.Context, call agent.BeforeToolCallContext) (*agent.BeforeToolCallResult, error) {
			if call.ToolCall.Name == "second" {
				a.Abort()
			}
			return nil, nil
		},
	})

	result, err := a.Prompt(context.Background(), "go")
	if err == nil {
		t.Fatal("expected the run to end with an abort error")
	}
	for _, id := range []string{"call_1", "call_2"} {
		message := findToolMessage(result.Messages, id)
		if message == nil {
			t.Fatalf("tool call %s has no result: %+v", id, result.Messages)
		}
		if !message.IsError {
			t.Fatalf("tool call %s should be an error result: %+v", id, message)
		}
	}
}

// pi: "answers every tool call of a parallel batch aborted while calls are prepared"
func TestPiParityAbortWhilePreparingParallelBatch(t *testing.T) {
	var a *agent.Agent
	model := faux.New(piParityToolTurn(
		piParityToolCall(0, "call_1", "first", `{}`),
		piParityToolCall(1, "call_2", "second", `{}`),
	))
	first := agent.NewTextTool(agent.ToolSpec{Name: "first"}, func(context.Context, string, json.RawMessage) (string, error) {
		return "first", nil
	})
	second := agent.NewTextTool(agent.ToolSpec{Name: "second"}, func(context.Context, string, json.RawMessage) (string, error) {
		return "second", nil
	})
	a = newAgent(t, agent.Config{
		Model:         model,
		Tools:         []agent.Tool{first, second},
		ToolExecution: agent.ToolExecutionParallel,
		BeforeToolCall: func(_ context.Context, call agent.BeforeToolCallContext) (*agent.BeforeToolCallResult, error) {
			if call.ToolCall.Name == "second" {
				a.Abort()
			}
			return nil, nil
		},
	})

	result, err := a.Prompt(context.Background(), "go")
	if err == nil {
		t.Fatal("expected the run to end with an abort error")
	}
	for _, id := range []string{"call_1", "call_2"} {
		message := findToolMessage(result.Messages, id)
		if message == nil || !message.IsError {
			t.Fatalf("tool call %s should have an error result: %+v", id, result.Messages)
		}
	}
}

// pi: "should prepare tool arguments for validation"
func TestPiParityPrepareArgumentsBeforeValidation(t *testing.T) {
	model := faux.New(
		piParityToolTurn(piParityToolCall(0, "call_1", "edit", `{"oldText":"before","newText":"after"}`)),
		piParityTextTurn("done"),
	)
	var seen []string
	edit := agent.NewTextTool(agent.ToolSpec{
		Name: "edit",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"edits": map[string]any{
					"type": "array",
					"items": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"oldText": map[string]any{"type": "string"},
							"newText": map[string]any{"type": "string"},
						},
						"required": []any{"oldText", "newText"},
					},
				},
			},
			"required": []any{"edits"},
		},
		PrepareArguments: func(json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"edits":[{"oldText":"before","newText":"after"}]}`), nil
		},
	}, func(_ context.Context, _ string, args json.RawMessage) (string, error) {
		seen = append(seen, string(args))
		return "edited", nil
	})
	a := newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{edit}})

	result, err := a.Prompt(context.Background(), "edit")
	if err != nil {
		t.Fatal(err)
	}
	if len(seen) != 1 || !strings.Contains(seen[0], `"edits"`) {
		t.Fatalf("prepared arguments = %v", seen)
	}
	if toolMessage := findToolMessage(result.Messages, "call_1"); toolMessage == nil || toolMessage.IsError {
		t.Fatalf("prepared call must validate: %+v", result.Messages)
	}
}

// pi: "does not poll steering after prepareRequest"
func TestPiParityPrepareRequestDoesNotPollSteering(t *testing.T) {
	var a *agent.Agent
	model := faux.New(piParityTextTurn("first"), piParityTextTurn("second"))
	var requests int
	a = newAgent(t, agent.Config{
		Model: model,
		PrepareRequest: func(context.Context, agent.PrepareRequestContext) (*agent.RequestUpdate, error) {
			requests++
			if requests == 1 {
				a.Steer(ai.TextMessage(ai.RoleUser, "late"))
			}
			return nil, nil
		},
	})

	if _, err := a.Prompt(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	if got := len(model.Requests); got != 2 {
		t.Fatalf("requests = %d, want 2", got)
	}
	if containsText(model.Requests[0].Messages, "late") {
		t.Fatalf("steering queued during prepareRequest leaked into the same request: %+v", model.Requests[0].Messages)
	}
	if !containsText(model.Requests[1].Messages, "late") {
		t.Fatalf("steering was not injected on the next request: %+v", model.Requests[1].Messages)
	}
}

// pi: "picks up steering queued during prepareNextTurn before the next request"
func TestPiParityPrepareNextTurnPicksUpSteering(t *testing.T) {
	var a *agent.Agent
	model := faux.New(
		piParityToolTurn(piParityToolCall(0, "call_1", "echo", `{}`)),
		piParityTextTurn("done"),
	)
	echo := agent.NewTextTool(agent.ToolSpec{Name: "echo"}, func(context.Context, string, json.RawMessage) (string, error) {
		return "ok", nil
	})
	a = newAgent(t, agent.Config{
		Model: model,
		Tools: []agent.Tool{echo},
		PrepareNextTurn: func(context.Context, agent.TurnContext) (*agent.TurnUpdate, error) {
			a.Steer(ai.TextMessage(ai.RoleUser, "during-next"))
			return nil, nil
		},
	})

	result, err := a.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	if got := len(model.Requests); got != 2 {
		t.Fatalf("requests = %d, want 2", got)
	}
	if !containsText(model.Requests[1].Messages, "during-next") {
		t.Fatalf("steering queued during prepareNextTurn is missing: %+v", model.Requests[1].Messages)
	}
	if countText(result.Messages, "during-next") != 1 {
		t.Fatalf("steering must be injected once: %+v", result.Messages)
	}
}

// pi: "action:end skips queue polling and next-turn preparation"
func TestPiParityFinishTurnEndSkipsQueues(t *testing.T) {
	model := faux.New(piParityTextTurn("done"))
	var preparedNext bool
	var a *agent.Agent
	a = newAgent(t, agent.Config{
		Model: model,
		FinishTurn: func(context.Context, agent.TurnContext) (agent.TurnDecision, error) {
			// 在 turn 结束前入队，验证 action:end 之后不再轮询队列。
			a.Steer(ai.TextMessage(ai.RoleUser, "steer"))
			a.FollowUp(ai.TextMessage(ai.RoleUser, "follow"))
			return agent.TurnEnd, nil
		},
		PrepareNextTurn: func(context.Context, agent.TurnContext) (*agent.TurnUpdate, error) {
			preparedNext = true
			return nil, nil
		},
	})

	if _, err := a.Prompt(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	if preparedNext {
		t.Fatal("prepareNextTurn must not run after action: end")
	}
	if got := len(model.Requests); got != 1 {
		t.Fatalf("requests = %d, want 1", got)
	}
	if !a.HasQueuedMessages() {
		t.Fatal("queues must be left untouched when finishTurn ends the run")
	}
	if got := a.PeekQueuedMessages(); len(got) != 1 || got[0].Text() != "steer" {
		t.Fatalf("peek = %+v", got)
	}
}

// pi: "makes exactly one context-only request when no natural request satisfies continuation"
func TestPiParityContinuationContextOnlyRequest(t *testing.T) {
	model := faux.New(piParityTextTurn("first"), piParityTextTurn("second"))
	finishCalls := 0
	a := newAgent(t, agent.Config{
		Model: model,
		FinishTurn: func(context.Context, agent.TurnContext) (agent.TurnDecision, error) {
			finishCalls++
			if finishCalls == 1 {
				return agent.TurnContinue, nil
			}
			return agent.TurnDefault, nil
		},
	})
	if _, err := a.Prompt(context.Background(), "run"); err != nil {
		t.Fatal(err)
	}
	if got := len(model.Requests); got != 2 {
		t.Fatalf("requests = %d, want 2", got)
	}
	if finishCalls != 2 {
		t.Fatalf("finishTurn calls = %d, want 2", finishCalls)
	}
}

// pi: "lets a natural tool-result request satisfy continuation"
func TestPiParityNaturalToolResultSatisfiesContinuation(t *testing.T) {
	model := faux.New(
		piParityToolTurn(piParityToolCall(0, "call_1", "noop", `{}`)),
		piParityTextTurn("done"),
	)
	noop := agent.NewTextTool(agent.ToolSpec{Name: "noop"}, func(context.Context, string, json.RawMessage) (string, error) {
		return "done", nil
	})
	finishCalls := 0
	a := newAgent(t, agent.Config{
		Model: model,
		Tools: []agent.Tool{noop},
		FinishTurn: func(context.Context, agent.TurnContext) (agent.TurnDecision, error) {
			finishCalls++
			if finishCalls == 1 {
				return agent.TurnContinue, nil
			}
			return agent.TurnDefault, nil
		},
	})
	if _, err := a.Prompt(context.Background(), "run"); err != nil {
		t.Fatal(err)
	}
	if got := len(model.Requests); got != 2 {
		t.Fatalf("requests = %d, want 2 (the tool-result turn satisfies continuation)", got)
	}
}

// pi: "keeps steering ahead of follow-up from a non-assistant continuation tail"
func TestPiParitySteeringAheadOfFollowUp(t *testing.T) {
	model := faux.New(
		piParityTextTurn("first"),
		piParityTextTurn("second"),
	)
	a := newAgent(t, agent.Config{Model: model})
	a.Steer(ai.TextMessage(ai.RoleUser, "steer"))
	a.FollowUp(ai.TextMessage(ai.RoleUser, "follow"))

	result, err := a.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	var users []string
	for _, message := range result.Messages {
		if message.Role() == ai.RoleUser {
			users = append(users, message.Text())
		}
	}
	if want := []string{"go", "steer", "follow"}; !equalStrings(users, want) {
		t.Fatalf("user message order = %v, want %v", users, want)
	}
}

// pi: "should reject reset while processing without corrupting the transcript"
// pi: "should throw when prompt() called while streaming"
func TestPiParityResetAndPromptWhileBusy(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	model := faux.New(
		piParityToolTurn(piParityToolCall(0, "call_1", "block", `{}`)),
		piParityTextTurn("done"),
	)
	var a *agent.Agent
	block := agent.NewTextTool(agent.ToolSpec{Name: "block"}, func(context.Context, string, json.RawMessage) (string, error) {
		a.Steer(ai.TextMessage(ai.RoleUser, "queued"))
		once.Do(func() { close(started) })
		<-release
		return "released", nil
	})
	a = newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{block}})

	done := make(chan error, 1)
	go func() {
		_, err := a.Prompt(context.Background(), "go")
		done <- err
	}()
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("tool did not start")
	}

	if _, err := a.Prompt(context.Background(), "again"); !errors.Is(err, agent.ErrBusy) {
		t.Fatalf("prompt while busy = %v, want ErrBusy", err)
	}
	if err := a.Reset(); !errors.Is(err, agent.ErrBusy) {
		t.Fatalf("reset while busy = %v, want ErrBusy", err)
	}
	if got := a.State().Messages; len(got) == 0 {
		t.Fatal("transcript must stay intact after a rejected reset")
	}
	if !a.HasQueuedMessages() {
		t.Fatal("busy prompt must not consume queues")
	}

	close(release)
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("run did not finish")
	}
}

// pi: "records how long execute() took on the tool result, excluding hooks"
// 只锁定"同一个耗时同时出现在事件与消息上"，不锁定具体数值，避免 CI 抖动。
func TestPiParityToolDurationOnEventAndMessage(t *testing.T) {
	model := faux.New(
		piParityToolTurn(piParityToolCall(0, "call_1", "echo", `{}`)),
		piParityTextTurn("done"),
	)
	echo := agent.NewTextTool(agent.ToolSpec{Name: "echo"}, func(context.Context, string, json.RawMessage) (string, error) {
		time.Sleep(time.Millisecond)
		return "ok", nil
	})
	a := newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{echo}})
	log := subscribe(a)

	result, err := a.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	var eventDuration int64 = -1
	for _, event := range log.snapshot() {
		if event.Type == agent.EventToolEnd {
			eventDuration = event.DurationMs
		}
	}
	toolMessage := findToolMessage(result.Messages, "call_1")
	if toolMessage == nil {
		t.Fatalf("missing tool result: %+v", result.Messages)
	}
	if eventDuration < 0 || eventDuration != ai.AsToolResult(toolMessage).DurationMs {
		t.Fatalf("duration mismatch: event=%d message=%d", eventDuration, ai.AsToolResult(toolMessage).DurationMs)
	}
}

func equalStrings(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func indexOf(values []string, target string) int {
	for i, value := range values {
		if value == target {
			return i
		}
	}
	return -1
}

func containsText(messages []ai.Message, text string) bool {
	for _, message := range messages {
		if message.Text() == text {
			return true
		}
	}
	return false
}

func countText(messages []ai.Message, text string) int {
	count := 0
	for _, message := range messages {
		if message.Text() == text {
			count++
		}
	}
	return count
}

func containsToolID(messages []ai.Message, id string) bool {
	for _, message := range messages {
		if message.Role() == ai.RoleTool && ai.AsToolResult(message).ToolCallID == id {
			return true
		}
	}
	return false
}

func findToolMessage(messages []ai.Message, id string) *ai.ToolResultMessage {
	for _, message := range messages {
		result := ai.AsToolResult(message)
		if result != nil && result.ToolCallID == id {
			return result
		}
	}
	return nil
}

// pi: "declares tool loadout changes to the model before the next request"
func TestPiParityDeclaresToolLoadoutChanges(t *testing.T) {
	model := faux.New(piParityTextTurn("first"), piParityTextTurn("second"))
	echo := agent.NewTextTool(agent.ToolSpec{Name: "echo", Description: "echo"}, func(context.Context, string, json.RawMessage) (string, error) {
		return "ok", nil
	})
	read := agent.NewTextTool(agent.ToolSpec{Name: "read", Description: "read"}, func(context.Context, string, json.RawMessage) (string, error) {
		return "ok", nil
	})
	a := newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{echo}})

	if _, err := a.Prompt(context.Background(), "one"); err != nil {
		t.Fatal(err)
	}
	if err := a.SetTools([]agent.Tool{read}); err != nil {
		t.Fatal(err)
	}
	if _, err := a.Prompt(context.Background(), "two"); err != nil {
		t.Fatal(err)
	}

	second := model.Requests[1].Messages
	var system *ai.SystemMessage
	systemIndex := -1
	for i, message := range second {
		if candidate := ai.AsSystem(message); candidate != nil && (candidate.ToolsAdded != nil || candidate.ToolsRemoved != nil) {
			system = candidate
			systemIndex = i
		}
	}
	if system == nil {
		t.Fatalf("tool change must be declared in a system message: %+v", second)
	}
	if systemIndex != len(second)-2 {
		t.Fatalf("declaration must precede the pending user message: index=%d messages=%+v", systemIndex, second)
	}
	if len(system.ToolsAdded) != 1 || system.ToolsAdded[0].Name != "read" {
		t.Fatalf("toolsAdded = %+v", system.ToolsAdded)
	}
	if len(system.ToolsRemoved) != 1 || system.ToolsRemoved[0].Name != "echo" {
		t.Fatalf("toolsRemoved = %+v", system.ToolsRemoved)
	}
	if tools := ai.GetCurrentTools(second); len(tools) != 1 || tools[0].Name != "read" {
		t.Fatalf("replayed tools = %+v", tools)
	}
	for _, message := range model.Requests[0].Messages {
		if ai.AsSystem(message) != nil {
			t.Fatalf("unchanged tools must not be redeclared: %+v", model.Requests[0].Messages)
		}
	}
}

// pi: assistant messages record the requested thinking level.
func TestPiParityAssistantRecordsThinkingLevel(t *testing.T) {
	model := faux.New(piParityTextTurn("done"))
	a := newAgent(t, agent.Config{Model: model, ThinkingLevel: agent.ThinkingHigh})
	result, err := a.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	assistant := ai.AsAssistant(result.Messages[len(result.Messages)-1])
	if assistant == nil || assistant.ThinkingLevel != ai.ThinkingHigh {
		t.Fatalf("assistant = %+v", assistant)
	}
}

// pi: "should apply transformContext before convertToLlm"
func TestPiParityTransformContextBeforeConvertToLLM(t *testing.T) {
	model := faux.New(piParityTextTurn("done"))
	var order []string
	a := newAgent(t, agent.Config{
		Model: model,
		TransformContext: func(_ context.Context, messages []ai.Message) ([]ai.Message, error) {
			order = append(order, "transform")
			return append(messages, ai.TextMessage(ai.RoleUser, "kept")), nil
		},
		ConvertToLLM: func(messages []ai.Message) []ai.Message {
			order = append(order, "convert")
			converted := make([]ai.Message, 0, len(messages))
			for _, message := range messages {
				if message.Text() == "drop" {
					continue
				}
				converted = append(converted, message)
			}
			return converted
		},
	})
	if _, err := a.Prompt(context.Background(), "drop"); err != nil {
		t.Fatal(err)
	}
	if !equalStrings(order, []string{"transform", "convert"}) {
		t.Fatalf("hook order = %v", order)
	}
	request := model.Requests[0].Messages
	if containsText(request, "drop") || !containsText(request, "kept") {
		t.Fatalf("request = %+v", request)
	}
}

// pi: "runToolCall validates, runs the hooks, and reports failures as error outcomes"
func TestPiParityRunToolCallHooks(t *testing.T) {
	var beforeArgs any
	var executedArgs string
	echo := agent.NewTool(agent.ToolSpec{Name: "echo", Parameters: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"value": map[string]any{"type": "string"},
		},
		"required": []any{"value"},
	}}, func(_ context.Context, callID string, args json.RawMessage) (agent.ToolResult, error) {
		executedArgs = callID + ":" + string(args)
		return agent.TextResult("ok"), nil
	})

	outcome := agent.RunToolCall(context.Background(), ai.ToolCall{
		ID:        "call_1",
		Name:      "echo",
		Arguments: json.RawMessage(`{"value":"x"}`),
	}, agent.RunToolCallOptions{
		Tools: []agent.Tool{echo},
		BeforeToolCall: func(_ context.Context, call agent.BeforeToolCallContext) (*agent.BeforeToolCallResult, error) {
			beforeArgs = call.Args
			return nil, nil
		},
		AfterToolCall: func(_ context.Context, call agent.AfterToolCallContext) (*agent.AfterToolCallResult, error) {
			return &agent.AfterToolCallResult{
				Content: []ai.ContentBlock{{Type: ai.BlockText, Text: "overridden"}},
			}, nil
		},
	})
	if outcome.IsError || !strings.Contains(outcome.Result.Text(), "overridden") {
		t.Fatalf("outcome = %+v", outcome)
	}
	if executedArgs != `call_1:{"value":"x"}` {
		t.Fatalf("executed args = %q", executedArgs)
	}
	if args, ok := beforeArgs.(map[string]any); !ok || args["value"] != "x" {
		t.Fatalf("before args = %+v", beforeArgs)
	}

	invalid := agent.RunToolCall(context.Background(), ai.ToolCall{
		ID:        "call_2",
		Name:      "echo",
		Arguments: json.RawMessage(`{"value":1}`),
	}, agent.RunToolCallOptions{Tools: []agent.Tool{echo}})
	if !invalid.IsError || !strings.Contains(invalid.Result.Text(), "expected string") {
		t.Fatalf("invalid outcome = %+v", invalid)
	}

	unknown := agent.RunToolCall(context.Background(), ai.ToolCall{ID: "call_3", Name: "missing"}, agent.RunToolCallOptions{Tools: []agent.Tool{echo}})
	if !unknown.IsError || !strings.Contains(unknown.Result.Text(), "unknown tool") {
		t.Fatalf("unknown outcome = %+v", unknown)
	}
}

// pi: "should pass the active abort signal to subscribers" / Agent.signal
func TestPiParityDoneSignalAndQueueModes(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	model := faux.New(
		piParityToolTurn(piParityToolCall(0, "call_1", "block", `{}`)),
		piParityTextTurn("done"),
	)
	block := agent.NewTextTool(agent.ToolSpec{Name: "block"}, func(context.Context, string, json.RawMessage) (string, error) {
		once.Do(func() { close(started) })
		<-release
		return "released", nil
	})
	a := newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{block}})
	if a.Done() != nil {
		t.Fatal("Done must be nil without an active run")
	}
	done := make(chan error, 1)
	go func() {
		_, err := a.Prompt(context.Background(), "go")
		done <- err
	}()
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("tool did not start")
	}
	signal := a.Done()
	if signal == nil {
		t.Fatal("Done must expose the active run")
	}
	if err := a.SetSteeringMode(agent.QueueAll); err != nil {
		t.Fatal(err)
	}
	if err := a.SetFollowUpMode(agent.QueueOneAtATime); err != nil {
		t.Fatal(err)
	}
	if err := a.SetSteeringMode(agent.QueueMode("bad")); err == nil {
		t.Fatal("invalid queue mode must be rejected")
	}
	close(release)
	select {
	case <-signal:
	case <-time.After(2 * time.Second):
		t.Fatal("Done was not closed after the run")
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

// pi event names: tool_execution_start/update/end.
func TestPiParityToolExecutionEventNames(t *testing.T) {
	if string(agent.EventToolExecutionStart) != "tool_execution_start" ||
		string(agent.EventToolExecutionUpdate) != "tool_execution_update" ||
		string(agent.EventToolExecutionEnd) != "tool_execution_end" {
		t.Fatalf("unexpected event names: %q %q %q",
			agent.EventToolExecutionStart,
			agent.EventToolExecutionUpdate,
			agent.EventToolExecutionEnd,
		)
	}
}

// pi: nested tool calls are recorded on the tool result.
func TestPiParityNestedToolCalls(t *testing.T) {
	model := faux.New(
		piParityToolTurn(piParityToolCall(0, "call_outer", "outer", `{}`)),
		piParityTextTurn("done"),
	)
	inner := agent.NewTextTool(agent.ToolSpec{Name: "inner"}, func(context.Context, string, json.RawMessage) (string, error) {
		return "inner result", nil
	})
	outer := agent.NewTool(agent.ToolSpec{Name: "outer"}, func(ctx context.Context, _ string, _ json.RawMessage) (agent.ToolResult, error) {
		nested := agent.RunToolCall(ctx, ai.ToolCall{
			ID:        "inner_1",
			Name:      "inner",
			Arguments: json.RawMessage(`{"value":"x"}`),
		}, agent.RunToolCallOptions{Tools: []agent.Tool{inner}})
		return agent.TextResult(nested.Result.Text()), nil
	})
	a := newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{outer}})

	result, err := a.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	toolMessage := findToolMessage(result.Messages, "call_outer")
	if toolMessage == nil || toolMessage.NestedCalls == nil {
		t.Fatalf("missing nested calls: %+v", toolMessage)
	}
	calls := toolMessage.NestedCalls.Calls
	if len(calls) != 1 || calls[0].Name != "inner" || calls[0].Status != "ok" {
		t.Fatalf("nested calls = %+v", calls)
	}
	if !toolMessage.NestedCalls.Complete {
		t.Fatal("nested calls must be complete")
	}
}

// pi: "Resolves an API key dynamically for each LLM call."
func TestPiParityGetAPIKeyPerRequest(t *testing.T) {
	model := faux.New(
		piParityToolTurn(piParityToolCall(0, "call_1", "echo", `{}`)),
		piParityTextTurn("done"),
	)
	echo := agent.NewTextTool(agent.ToolSpec{Name: "echo"}, func(context.Context, string, json.RawMessage) (string, error) {
		return "ok", nil
	})
	keys := []string{"key-one", "key-two"}
	index := 0
	a := newAgent(t, agent.Config{
		Model: model,
		Tools: []agent.Tool{echo},
		GetAPIKey: func(provider string) (string, error) {
			if provider == "" {
				t.Fatalf("provider must not be empty")
			}
			key := keys[index]
			index++
			return key, nil
		},
	})
	if _, err := a.Prompt(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	if got := model.Requests[0].Options.APIKey; got != "key-one" {
		t.Fatalf("first key = %q", got)
	}
	if got := model.Requests[1].Options.APIKey; got != "key-two" {
		t.Fatalf("second key = %q", got)
	}

	failing := newAgent(t, agent.Config{
		Model: faux.New(piParityTextTurn("never")),
		GetAPIKey: func(string) (string, error) {
			return "", errors.New("no credential")
		},
	})
	if _, err := failing.Prompt(context.Background(), "go"); err == nil || !strings.Contains(err.Error(), "no credential") {
		t.Fatalf("error = %v", err)
	}
}

// pi: subscribers receive the active run signal.
func TestPiParitySubscriberReceivesRunContext(t *testing.T) {
	started := make(chan struct{})
	var once sync.Once
	model := faux.New(
		piParityToolTurn(piParityToolCall(0, "call_1", "block", `{}`)),
		piParityTextTurn("done"),
	)
	block := agent.NewTextTool(agent.ToolSpec{Name: "block"}, func(ctx context.Context, _ string, _ json.RawMessage) (string, error) {
		once.Do(func() { close(started) })
		<-ctx.Done()
		return "", ctx.Err()
	})
	a := newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{block}})

	var runCtx context.Context
	a.Subscribe(func(_ agent.Event, ctx context.Context) {
		runCtx = ctx
	})

	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _ = a.Prompt(context.Background(), "go")
	}()
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("tool did not start")
	}
	if runCtx == nil {
		t.Fatal("subscriber did not receive a context")
	}
	a.Abort()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("run did not finish")
	}
	if runCtx.Err() == nil {
		t.Fatal("run context must be canceled after abort")
	}
}
