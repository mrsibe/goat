package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/mrsibe/goat/agent"
	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/ai/faux"
)

func TestRunDirectAnswer(t *testing.T) {
	model := faux.New(faux.Turn{Events: []ai.Event{
		{Type: ai.EventStart},
		{Type: ai.EventTextDelta, TextDelta: "你"},
		{Type: ai.EventTextDelta, TextDelta: "好"},
		{Type: ai.EventUsage, Usage: &ai.Usage{PromptTokens: 10, CompletionTokens: 2, TotalTokens: 12}},
		{Type: ai.EventDone, StopReason: ai.StopReasonStop},
	}})
	assistant, err := agent.New(agent.Config{
		Model:        model,
		SystemPrompt: "system",
	})
	if err != nil {
		t.Fatal(err)
	}
	var events []agent.Event
	result, err := assistant.Run(context.Background(), "你好", agent.WithOnEvent(func(event agent.Event) {
		events = append(events, event)
	}))
	if err != nil {
		t.Fatal(err)
	}
	if result.Answer != "你好" || result.Steps != 1 || result.Usage.TotalTokens != 12 {
		t.Fatalf("unexpected result: %+v", result)
	}
	if result.StopReason != ai.StopReasonStop {
		t.Fatalf("stop reason = %q", result.StopReason)
	}
	wantTypes := []agent.EventType{
		agent.EventAgentStart,
		agent.EventMessageStart,
		agent.EventMessageEnd,
		agent.EventTurnStart,
		agent.EventMessageStart,
		agent.EventMessageDelta,
		agent.EventMessageDelta,
		agent.EventMessageEnd,
		agent.EventTurnEnd,
		agent.EventAgentEnd,
	}
	if got := eventTypes(events); !reflect.DeepEqual(got, wantTypes) {
		t.Fatalf("events = %v, want %v", got, wantTypes)
	}
	wantRoles := []ai.Role{ai.RoleSystem, ai.RoleUser, ai.RoleAssistant}
	if got := messageRoles(result.Messages); !reflect.DeepEqual(got, wantRoles) {
		t.Fatalf("roles = %v, want %v", got, wantRoles)
	}
	if len(model.Requests) != 1 || len(model.Requests[0].Messages) != 2 {
		t.Fatalf("model requests = %+v", model.Requests)
	}
}

func TestRunExecutesToolAndContinues(t *testing.T) {
	model := faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventStart},
			{Type: ai.EventTextDelta, TextDelta: "先查天气"},
			{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{
				ID: "call_1", Name: "weather", ArgumentsDelta: `{"city":`,
			}},
			{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{
				ArgumentsDelta: `"北京"}`,
			}},
			{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventStart},
			{Type: ai.EventTextDelta, TextDelta: "晴，25°C"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	var received ai.ToolCall
	weather := agent.NewTextTool(ai.ToolSpec{Name: "weather"}, func(_ context.Context, call ai.ToolCall) (string, error) {
		received = call
		return "sunny", nil
	})
	assistant, err := agent.New(agent.Config{
		Model: model,
		Tools: []agent.Tool{weather},
	})
	if err != nil {
		t.Fatal(err)
	}
	var events []agent.Event
	result, err := assistant.Run(context.Background(), "天气", agent.WithOnEvent(func(event agent.Event) {
		events = append(events, event)
	}))
	if err != nil {
		t.Fatal(err)
	}
	if result.Answer != "晴，25°C" || result.Steps != 2 {
		t.Fatalf("result = %+v", result)
	}
	if received.ID != "call_1" || received.Name != "weather" || string(received.Arguments) != `{"city":"北京"}` {
		t.Fatalf("received call = %+v", received)
	}
	if len(model.Requests) != 2 {
		t.Fatalf("requests = %d", len(model.Requests))
	}
	wantRoles := []ai.Role{ai.RoleUser, ai.RoleAssistant, ai.RoleTool}
	if got := messageRoles(model.Requests[1].Messages); !reflect.DeepEqual(got, wantRoles) {
		t.Fatalf("second request roles = %v", got)
	}
	if len(result.Messages[1].ToolCalls()) != 1 {
		t.Fatalf("assistant message = %+v", result.Messages[1])
	}
	toolResult := result.Messages[2]
	if toolResult.Role != ai.RoleTool || toolResult.ToolCallID != "call_1" || toolResult.Text() != "sunny" {
		t.Fatalf("tool message = %+v", toolResult)
	}
	var sawToolStart, sawToolEnd bool
	for _, event := range events {
		sawToolStart = sawToolStart || event.Type == agent.EventToolStart
		sawToolEnd = sawToolEnd || event.Type == agent.EventToolEnd
	}
	if !sawToolStart || !sawToolEnd {
		t.Fatalf("missing tool events: %v", eventTypes(events))
	}
}

func TestToolCallLifecycleWithoutDoneMessage(t *testing.T) {
	model := faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventStart},
			{Type: ai.EventToolCallStart, ContentIndex: 0, ToolCall: &ai.ToolCall{
				ID: "call_1", Name: "weather", Arguments: json.RawMessage(`{}`),
			}},
			{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{
				ID: "call_1", Name: "weather", ArgumentsDelta: `{"city":`,
			}},
			{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{
				ID: "call_1", Name: "weather", ArgumentsDelta: `"北京"}`,
			}},
			{Type: ai.EventToolCallEnd, ContentIndex: 0, ToolCall: &ai.ToolCall{
				ID: "call_1", Name: "weather", Arguments: json.RawMessage(`{"city":"北京"}`),
			}},
			{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "done"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	var received ai.ToolCall
	weather := agent.NewTextTool(ai.ToolSpec{Name: "weather"}, func(_ context.Context, call ai.ToolCall) (string, error) {
		received = call
		return "sunny", nil
	})
	assistant, err := agent.New(agent.Config{Model: model, Tools: []agent.Tool{weather}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := assistant.Run(context.Background(), "天气"); err != nil {
		t.Fatal(err)
	}
	if received.Name != "weather" || string(received.Arguments) != `{"city":"北京"}` {
		t.Fatalf("received call = %+v", received)
	}
}

func TestToolErrorsAreObservations(t *testing.T) {
	for _, testCase := range []struct {
		name string
		tool agent.Tool
	}{
		{
			name: "tool error",
			tool: agent.NewTextTool(ai.ToolSpec{Name: "weather"}, func(context.Context, ai.ToolCall) (string, error) {
				return "", errors.New("upstream failed")
			}),
		},
		{
			name: "panic",
			tool: agent.NewTextTool(ai.ToolSpec{Name: "weather"}, func(context.Context, ai.ToolCall) (string, error) {
				panic("boom")
			}),
		},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			model := faux.New(
				faux.Turn{Events: []ai.Event{
					{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{ID: "call_1", Name: "weather", ArgumentsDelta: "{}"}},
					{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
				}},
				faux.Turn{Events: []ai.Event{
					{Type: ai.EventTextDelta, TextDelta: "recovered"},
					{Type: ai.EventDone, StopReason: ai.StopReasonStop},
				}},
			)
			assistant, err := agent.New(agent.Config{Model: model, Tools: []agent.Tool{testCase.tool}})
			if err != nil {
				t.Fatal(err)
			}
			result, err := assistant.Run(context.Background(), "go")
			if err != nil {
				t.Fatal(err)
			}
			if result.Answer != "recovered" {
				t.Fatalf("answer = %q", result.Answer)
			}
			toolMessage := result.Messages[len(result.Messages)-2]
			if !strings.Contains(toolMessage.Text(), "failed") && !strings.Contains(toolMessage.Text(), "panicked") {
				t.Fatalf("tool observation = %q", toolMessage.Text())
			}
			if !toolMessage.IsError {
				t.Fatal("tool error was not preserved in message history")
			}
		})
	}
}

func TestBeforeToolCallCanBlock(t *testing.T) {
	model := faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{ID: "call_1", Name: "echo", ArgumentsDelta: "{}"}},
			{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "done"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	echo := agent.NewTextTool(ai.ToolSpec{Name: "echo"}, func(context.Context, ai.ToolCall) (string, error) {
		return "ok", nil
	})
	var beforeCalled bool
	assistant, err := agent.New(agent.Config{
		Model: model,
		Tools: []agent.Tool{echo},
		BeforeToolCall: func(context.Context, ai.ToolCall) error {
			beforeCalled = true
			return errors.New("blocked by policy")
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := assistant.Run(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	if !beforeCalled {
		t.Fatal("before hook was not called")
	}
	if got := result.Messages[len(result.Messages)-2].Text(); !strings.Contains(got, "blocked by policy") {
		t.Fatalf("tool observation = %q", got)
	}
}

func TestUnknownToolIsObservation(t *testing.T) {
	model := faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{ID: "call_1", Name: "missing", ArgumentsDelta: "{}"}},
			{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "done"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	assistant, err := agent.New(agent.Config{Model: model})
	if err != nil {
		t.Fatal(err)
	}
	result, err := assistant.Run(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	toolMessage := result.Messages[2]
	if !toolMessage.IsError || !strings.Contains(toolMessage.Text(), "unknown tool") {
		t.Fatalf("tool message = %+v", toolMessage)
	}
}

func TestMaxStepsAndHistoryRestore(t *testing.T) {
	turn := faux.Turn{Events: []ai.Event{
		{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{ID: "call_1", Name: "echo", ArgumentsDelta: "{}"}},
		{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
	}}
	model := faux.New(turn, turn)
	echo := agent.NewTextTool(ai.ToolSpec{Name: "echo"}, func(context.Context, ai.ToolCall) (string, error) {
		return "ok", nil
	})
	assistant, err := agent.New(agent.Config{Model: model, Tools: []agent.Tool{echo}, MaxSteps: 2})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := assistant.Run(context.Background(), "go"); !errors.Is(err, agent.ErrMaxSteps) {
		t.Fatalf("error = %v, want ErrMaxSteps", err)
	}

	session := assistant.NewSessionWithHistory([]ai.Message{
		ai.TextMessage(ai.RoleSystem, "stale"),
		ai.TextMessage(ai.RoleUser, "old"),
		ai.TextMessage(ai.RoleAssistant, "answer"),
	})
	if got := messageRoles(session.Messages()); !reflect.DeepEqual(got, []ai.Role{ai.RoleUser, ai.RoleAssistant}) {
		t.Fatalf("restored roles = %v", got)
	}
}

func TestOutputLengthIsFatal(t *testing.T) {
	model := faux.New(faux.Turn{Events: []ai.Event{
		{Type: ai.EventTextDelta, TextDelta: "partial"},
		{Type: ai.EventDone, StopReason: ai.StopReasonLength},
	}})
	assistant, err := agent.New(agent.Config{Model: model})
	if err != nil {
		t.Fatal(err)
	}
	result, err := assistant.Run(context.Background(), "go")
	if !errors.Is(err, agent.ErrOutputLimit) {
		t.Fatalf("error = %v", err)
	}
	if result.Answer != "partial" {
		t.Fatalf("answer = %q", result.Answer)
	}
}

func TestOutputLengthDoesNotExecuteToolCalls(t *testing.T) {
	call := ai.ToolCall{
		ID: "call_1", Name: "weather", Arguments: json.RawMessage(`{"city":"北京"}`),
	}
	truncated := ai.Message{
		Role: ai.RoleAssistant,
		Content: []ai.ContentBlock{{
			Type: ai.BlockToolCall, ToolCall: &call,
		}},
	}
	model := faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventDone, Message: &truncated, StopReason: ai.StopReasonLength},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "recovered"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	executed := false
	weather := agent.NewTextTool(ai.ToolSpec{
		Name: "weather",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city": map[string]any{"type": "string"},
			},
			"required": []string{"city"},
		},
	}, func(context.Context, ai.ToolCall) (string, error) {
		executed = true
		return "sunny", nil
	})
	assistant, err := agent.New(agent.Config{Model: model, Tools: []agent.Tool{weather}})
	if err != nil {
		t.Fatal(err)
	}
	result, err := assistant.Run(context.Background(), "天气")
	if err != nil {
		t.Fatal(err)
	}
	if executed {
		t.Fatal("truncated tool call was executed")
	}
	if result.Answer != "recovered" || result.Steps != 2 {
		t.Fatalf("result = %+v", result)
	}
	toolMessage := result.Messages[2]
	if toolMessage.Role != ai.RoleTool || !toolMessage.IsError ||
		!strings.Contains(toolMessage.Text(), "output token limit") {
		t.Fatalf("tool message = %+v", toolMessage)
	}
}

func TestInvalidToolArgumentsAreRejected(t *testing.T) {
	call := ai.ToolCall{
		ID: "call_1", Name: "weather", Arguments: json.RawMessage(`{"city":123}`),
	}
	assistantCall := ai.Message{
		Role: ai.RoleAssistant,
		Content: []ai.ContentBlock{{
			Type: ai.BlockToolCall, ToolCall: &call,
		}},
	}
	model := faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventDone, Message: &assistantCall, StopReason: ai.StopReasonToolCalls},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "recovered"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	executed := false
	weather := agent.NewTextTool(ai.ToolSpec{
		Name: "weather",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city": map[string]any{"type": "string"},
			},
			"required": []string{"city"},
		},
	}, func(context.Context, ai.ToolCall) (string, error) {
		executed = true
		return "sunny", nil
	})
	assistant, err := agent.New(agent.Config{Model: model, Tools: []agent.Tool{weather}})
	if err != nil {
		t.Fatal(err)
	}
	result, err := assistant.Run(context.Background(), "天气")
	if err != nil {
		t.Fatal(err)
	}
	if executed {
		t.Fatal("invalid tool arguments were executed")
	}
	toolMessage := result.Messages[2]
	if !toolMessage.IsError || !strings.Contains(toolMessage.Text(), "expected string") {
		t.Fatalf("tool message = %+v", toolMessage)
	}
}

func TestModelErrorStopIsFatal(t *testing.T) {
	model := faux.New(faux.Turn{Events: []ai.Event{
		{Type: ai.EventTextDelta, TextDelta: "partial"},
		{Type: ai.EventDone, StopReason: ai.StopReasonError},
	}})
	assistant, err := agent.New(agent.Config{Model: model})
	if err != nil {
		t.Fatal(err)
	}
	result, err := assistant.Run(context.Background(), "go")
	if !errors.Is(err, agent.ErrModelStop) {
		t.Fatalf("error = %v", err)
	}
	if result.Answer != "partial" {
		t.Fatalf("answer = %q", result.Answer)
	}
}

func TestDoneMessageAndUsageAreAuthoritative(t *testing.T) {
	final := ai.TextMessage(ai.RoleAssistant, "final")
	model := faux.New(faux.Turn{Events: []ai.Event{
		{Type: ai.EventTextDelta, TextDelta: "ignored"},
		{Type: ai.EventDone, Message: &final, StopReason: ai.StopReasonStop, Usage: &ai.Usage{
			PromptTokens: 10, CompletionTokens: 2, TotalTokens: 12,
		}},
	}})
	assistant, err := agent.New(agent.Config{Model: model})
	if err != nil {
		t.Fatal(err)
	}
	result, err := assistant.Run(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	if result.Answer != "final" {
		t.Fatalf("answer = %q", result.Answer)
	}
	if result.Usage.TotalTokens != 12 {
		t.Fatalf("usage = %+v", result.Usage)
	}
}

func TestStreamErrorEventIsFatal(t *testing.T) {
	streamErr := errors.New("provider failed")
	partial := ai.TextMessage(ai.RoleAssistant, "partial")
	model := faux.New(faux.Turn{Events: []ai.Event{
		{Type: ai.EventTextDelta, TextDelta: "partial"},
		{
			Type:       ai.EventError,
			Err:        streamErr,
			Message:    &partial,
			Usage:      &ai.Usage{PromptTokens: 10, CompletionTokens: 2, TotalTokens: 12},
			StopReason: ai.StopReasonError,
		},
	}})
	assistant, err := agent.New(agent.Config{Model: model})
	if err != nil {
		t.Fatal(err)
	}
	var events []agent.Event
	result, err := assistant.Run(context.Background(), "go", agent.WithOnEvent(func(event agent.Event) {
		events = append(events, event)
	}))
	if !errors.Is(err, streamErr) {
		t.Fatalf("error = %v", err)
	}
	if result.Answer != "partial" || result.Usage.TotalTokens != 12 {
		t.Fatalf("result = %+v", result)
	}
	if len(result.Messages) != 2 || result.Messages[1].Text() != "partial" {
		t.Fatalf("messages = %+v", result.Messages)
	}
	if !strings.Contains(strings.Join(eventTypeStrings(events), ","), "turn_end,agent_error") {
		t.Fatalf("events = %v", eventTypeStrings(events))
	}
}

func TestThinkingEventsAndSignaturesArePreserved(t *testing.T) {
	final := ai.Message{
		Role: ai.RoleAssistant,
		Content: []ai.ContentBlock{
			{Type: ai.BlockThinking, Thinking: "plan", Signature: "sig"},
			{Type: ai.BlockText, Text: "answer"},
		},
	}
	model := faux.New(faux.Turn{Events: []ai.Event{
		{Type: ai.EventThinkingStart},
		{Type: ai.EventThinkingDelta, ThinkingDelta: "plan"},
		{Type: ai.EventThinkingEnd, Thinking: "plan"},
		{Type: ai.EventTextDelta, TextDelta: "answer"},
		{Type: ai.EventDone, Message: &final, StopReason: ai.StopReasonStop},
	}})
	assistant, err := agent.New(agent.Config{Model: model})
	if err != nil {
		t.Fatal(err)
	}
	var events []agent.Event
	result, err := assistant.Run(context.Background(), "go", agent.WithOnEvent(func(event agent.Event) {
		events = append(events, event)
	}))
	if err != nil {
		t.Fatal(err)
	}
	if result.Answer != "answer" || result.Messages[1].Thinking() != "plan" {
		t.Fatalf("result = %+v", result)
	}
	if got := result.Messages[1].Content[0].Signature; got != "sig" {
		t.Fatalf("thinking signature = %q", got)
	}
	for _, eventType := range []agent.EventType{
		agent.EventThinkingStart,
		agent.EventThinkingDelta,
		agent.EventThinkingEnd,
	} {
		found := false
		for _, event := range events {
			if event.Type == eventType {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("missing %s in %v", eventType, eventTypeStrings(events))
		}
	}
}

func TestParallelToolExecutionPreservesModelOrder(t *testing.T) {
	model := faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{
				ID: "call_slow", Name: "slow", ArgumentsDelta: "{}",
			}},
			{Type: ai.EventToolCallDelta, ContentIndex: 1, ToolCallDelta: &ai.ToolCallDelta{
				ID: "call_fast", Name: "fast", ArgumentsDelta: "{}",
			}},
			{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "done"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	slowStarted := make(chan struct{})
	fastDone := make(chan struct{})
	slow := agent.NewTextTool(ai.ToolSpec{Name: "slow"}, func(ctx context.Context, _ ai.ToolCall) (string, error) {
		close(slowStarted)
		select {
		case <-fastDone:
			return "slow result", nil
		case <-ctx.Done():
			return "", ctx.Err()
		}
	})
	fast := agent.NewTextTool(ai.ToolSpec{Name: "fast"}, func(ctx context.Context, _ ai.ToolCall) (string, error) {
		select {
		case <-slowStarted:
			close(fastDone)
			return "fast result", nil
		case <-ctx.Done():
			return "", ctx.Err()
		}
	})
	assistant, err := agent.New(agent.Config{
		Model:         model,
		Tools:         []agent.Tool{slow, fast},
		ToolExecution: agent.ToolExecutionParallel,
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	result, err := assistant.Run(ctx, "go")
	if err != nil {
		t.Fatal(err)
	}
	if result.Answer != "done" {
		t.Fatalf("answer = %q", result.Answer)
	}
	slowMessage := result.Messages[2]
	fastMessage := result.Messages[3]
	if slowMessage.ToolCallID != "call_slow" || slowMessage.ToolName != "slow" ||
		slowMessage.Text() != "slow result" {
		t.Fatalf("slow message = %+v", slowMessage)
	}
	if fastMessage.ToolCallID != "call_fast" || fastMessage.ToolName != "fast" ||
		fastMessage.Text() != "fast result" {
		t.Fatalf("fast message = %+v", fastMessage)
	}
}

func TestTransformMessagesAndRequestOptions(t *testing.T) {
	temperature := 0.2
	maxTokens := 128
	overrideTemperature := 0.8
	topP := 0.9
	model := faux.New(faux.Turn{Events: []ai.Event{
		{Type: ai.EventTextDelta, TextDelta: "ok"},
		{Type: ai.EventDone, StopReason: ai.StopReasonStop},
	}})
	transformed := false
	assistant, err := agent.New(agent.Config{
		Model: model,
		RequestOptions: ai.RequestOptions{
			Temperature: &temperature,
			MaxTokens:   maxTokens,
		},
		TransformMessages: func(_ context.Context, messages []ai.Message) ([]ai.Message, error) {
			transformed = true
			return append(messages, ai.TextMessage(ai.RoleUser, "injected")), nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := assistant.Run(
		context.Background(),
		"hello",
		agent.WithRequestOptions(ai.RequestOptions{
			Temperature: &overrideTemperature,
			TopP:        &topP,
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if !transformed {
		t.Fatal("message transformer was not called")
	}
	if len(model.Requests) != 1 {
		t.Fatalf("requests = %d", len(model.Requests))
	}
	request := model.Requests[0]
	if len(request.Messages) != 2 || request.Messages[1].Text() != "injected" {
		t.Fatalf("transformed messages = %+v", request.Messages)
	}
	options := request.Options
	if options.Temperature == nil || *options.Temperature != overrideTemperature {
		t.Fatalf("temperature = %v", options.Temperature)
	}
	if options.TopP == nil || *options.TopP != topP {
		t.Fatalf("top_p = %v", options.TopP)
	}
	if options.MaxTokens != maxTokens {
		t.Fatalf("max tokens = %d", options.MaxTokens)
	}
	if len(result.Messages) != 2 {
		t.Fatalf("session was polluted by transformed messages: %+v", result.Messages)
	}
}

func TestToolResultImagesArePreserved(t *testing.T) {
	model := faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{
				ID: "call_1", Name: "screenshot", ArgumentsDelta: "{}",
			}},
			{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "done"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	screenshot := agent.NewTool(ai.ToolSpec{Name: "screenshot"}, func(context.Context, ai.ToolCall) (agent.ToolResult, error) {
		return agent.ToolResult{Content: []ai.ContentBlock{
			{Type: ai.BlockText, Text: "preview"},
			{Type: ai.BlockImage, Image: &ai.ImageContent{Data: "aGVsbG8=", MediaType: "image/png"}},
		}}, nil
	})
	assistant, err := agent.New(agent.Config{Model: model, Tools: []agent.Tool{screenshot}})
	if err != nil {
		t.Fatal(err)
	}
	result, err := assistant.Run(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	toolMessage := result.Messages[2]
	if toolMessage.IsError || len(toolMessage.Content) != 2 ||
		toolMessage.Content[1].Type != ai.BlockImage ||
		toolMessage.Content[1].Image == nil ||
		toolMessage.Content[1].Image.MediaType != "image/png" {
		t.Fatalf("tool message = %+v", toolMessage)
	}
	requestToolMessage := model.Requests[1].Messages[2]
	if len(requestToolMessage.Content) != 2 || requestToolMessage.Content[1].Image == nil {
		t.Fatalf("request tool message = %+v", requestToolMessage)
	}
}

func TestInvalidToolResultBecomesErrorObservation(t *testing.T) {
	model := faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{
				ID: "call_1", Name: "screenshot", ArgumentsDelta: "{}",
			}},
			{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "done"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	screenshot := agent.NewTool(ai.ToolSpec{Name: "screenshot"}, func(context.Context, ai.ToolCall) (agent.ToolResult, error) {
		return agent.ToolResult{Content: []ai.ContentBlock{{
			Type: ai.BlockImage,
			Image: &ai.ImageContent{
				Data:      "aGVsbG8=",
				URL:       "https://example.test/image.png",
				MediaType: "image/png",
			},
		}}}, nil
	})
	assistant, err := agent.New(agent.Config{Model: model, Tools: []agent.Tool{screenshot}})
	if err != nil {
		t.Fatal(err)
	}
	result, err := assistant.Run(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	toolMessage := result.Messages[2]
	if !toolMessage.IsError || !strings.Contains(toolMessage.Text(), "exactly one") {
		t.Fatalf("tool message = %+v", toolMessage)
	}
}

func TestStreamedArgumentErrorIsObservation(t *testing.T) {
	call := ai.ToolCall{
		ID:             "call_1",
		Name:           "weather",
		Arguments:      json.RawMessage(`{}`),
		ArgumentsError: "tool call arguments are not valid JSON",
	}
	assistantCall := ai.Message{
		Role: ai.RoleAssistant,
		Content: []ai.ContentBlock{{
			Type: ai.BlockToolCall, ToolCall: &call,
		}},
	}
	model := faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventDone, Message: &assistantCall, StopReason: ai.StopReasonToolCalls},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "recovered"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	executed := false
	weather := agent.NewTextTool(ai.ToolSpec{Name: "weather"}, func(context.Context, ai.ToolCall) (string, error) {
		executed = true
		return "sunny", nil
	})
	assistant, err := agent.New(agent.Config{Model: model, Tools: []agent.Tool{weather}})
	if err != nil {
		t.Fatal(err)
	}
	result, err := assistant.Run(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	if executed {
		t.Fatal("tool with invalid streamed arguments was executed")
	}
	toolMessage := result.Messages[2]
	if !toolMessage.IsError || !strings.Contains(toolMessage.Text(), "not valid JSON") {
		t.Fatalf("tool message = %+v", toolMessage)
	}
}

func eventTypes(events []agent.Event) []agent.EventType {
	types := make([]agent.EventType, len(events))
	for i, event := range events {
		types[i] = event.Type
	}
	return types
}

func eventTypeStrings(events []agent.Event) []string {
	types := make([]string, len(events))
	for i, event := range events {
		types[i] = string(event.Type)
	}
	return types
}

func messageRoles(messages []ai.Message) []ai.Role {
	roles := make([]ai.Role, len(messages))
	for i, message := range messages {
		roles[i] = message.Role
	}
	return roles
}
