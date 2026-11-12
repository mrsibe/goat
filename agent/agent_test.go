package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrsibe/goat/agent"
	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/ai/faux"
)

func TestPromptDirectAnswer(t *testing.T) {
	model := faux.New(faux.Turn{Events: []ai.Event{
		{Type: ai.EventStart},
		{Type: ai.EventTextDelta, TextDelta: "你"},
		{Type: ai.EventTextDelta, TextDelta: "好"},
		{Type: ai.EventUsage, Usage: &ai.Usage{PromptTokens: 10, CompletionTokens: 2, TotalTokens: 12}},
		{Type: ai.EventDone, StopReason: ai.StopReasonStop},
	}})
	a := newAgent(t, agent.Config{Model: model, SystemPrompt: "system"})
	log := subscribe(a)

	result, err := a.Prompt(context.Background(), "你好")
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
		agent.EventTurnStart,
		agent.EventMessageStart,
		agent.EventMessageEnd,
		agent.EventMessageStart,
		agent.EventMessageUpdate,
		agent.EventMessageDelta,
		agent.EventMessageUpdate,
		agent.EventMessageDelta,
		agent.EventMessageEnd,
		agent.EventTurnEnd,
		agent.EventAgentEnd,
	}
	if got := eventTypes(log.snapshot()); !reflect.DeepEqual(got, wantTypes) {
		t.Fatalf("events = %v, want %v", got, wantTypes)
	}
	if got := messageRoles(result.Messages); !reflect.DeepEqual(got, []ai.Role{ai.RoleSystem, ai.RoleUser, ai.RoleAssistant}) {
		t.Fatalf("roles = %v", got)
	}
	if got := messageRoles(result.NewMessages); !reflect.DeepEqual(got, []ai.Role{ai.RoleUser, ai.RoleAssistant}) {
		t.Fatalf("new message roles = %v", got)
	}
	if len(model.Requests) != 1 || len(model.Requests[0].Messages) != 2 {
		t.Fatalf("model requests = %+v", model.Requests)
	}
	state := a.State()
	if state.IsStreaming || len(state.Messages) != 3 || state.SystemPrompt != "system" {
		t.Fatalf("state = %+v", state)
	}
}

func TestMessageUpdateExposesStreamingAssistantEvents(t *testing.T) {
	model := faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventStart},
			{Type: ai.EventTextStart, ContentIndex: 0},
			{Type: ai.EventTextDelta, ContentIndex: 0, TextDelta: "hello"},
			{Type: ai.EventTextEnd, ContentIndex: 0, Text: "hello"},
			{Type: ai.EventToolCallStart, ContentIndex: 1, ToolCall: &ai.ToolCall{
				ID:     "call_1",
				ItemID: "item_1",
				Name:   "echo",
			}},
			{Type: ai.EventToolCallDelta, ContentIndex: 1, ToolCallDelta: &ai.ToolCallDelta{
				ArgumentsDelta: `{"value":"x"}`,
			}},
			{Type: ai.EventToolCallEnd, ContentIndex: 1, ToolCall: &ai.ToolCall{
				ID:        "call_1",
				ItemID:    "item_1",
				Name:      "echo",
				Arguments: json.RawMessage(`{"value":"x"}`),
			}},
			{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "done"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	echo := agent.NewTextTool(agent.ToolSpec{Name: "echo"}, func(context.Context, ai.ToolCall) (string, error) {
		return "ok", nil
	})
	a := newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{echo}})
	log := subscribe(a)

	result, err := a.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	var sawTextDelta, sawToolCallDelta bool
	for _, event := range log.snapshot() {
		if event.Type != agent.EventMessageUpdate {
			continue
		}
		if event.Message == nil || event.AssistantMessageEvent == nil {
			t.Fatalf("message_update is missing payloads: %+v", event)
		}
		switch event.AssistantMessageEvent.Type {
		case ai.EventTextDelta:
			if event.Step != 1 {
				continue
			}
			sawTextDelta = true
			if event.Text != "hello" || event.Message.Text() != "hello" {
				t.Fatalf("text update = %+v", event)
			}
		case ai.EventToolCallDelta:
			sawToolCallDelta = true
			if event.AssistantMessageEvent.ToolCallDelta == nil ||
				event.AssistantMessageEvent.ToolCallDelta.ArgumentsDelta != `{"value":"x"}` {
				t.Fatalf("tool call delta = %+v", event.AssistantMessageEvent)
			}
			calls := event.Message.ToolCalls()
			if len(calls) != 1 || calls[0].ItemID != "item_1" ||
				!strings.Contains(string(calls[0].Arguments), `"value"`) {
				t.Fatalf("partial tool call = %+v", calls)
			}
		}
	}
	if !sawTextDelta || !sawToolCallDelta {
		t.Fatalf("message updates = %v", eventTypes(log.snapshot()))
	}
	calls := result.Messages[1].ToolCalls()
	if len(calls) != 1 || calls[0].ItemID != "item_1" {
		t.Fatalf("tool call transcript = %+v", calls)
	}
}

func TestAbortDuringToolPreflightPreventsExecution(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	model := faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{
				ID:             "call_1",
				Name:           "side_effect",
				ArgumentsDelta: "{}",
			}},
			{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "should not run"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	var executed atomic.Bool
	tool := agent.NewTextTool(agent.ToolSpec{Name: "side_effect"}, func(context.Context, ai.ToolCall) (string, error) {
		executed.Store(true)
		return "ran", nil
	})
	a := newAgent(t, agent.Config{
		Model: model,
		Tools: []agent.Tool{tool},
		BeforeToolCall: func(context.Context, agent.BeforeToolCallContext) (*agent.BeforeToolCallResult, error) {
			cancel()
			return nil, nil
		},
	})

	result, err := a.Prompt(ctx, "go")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("prompt error = %v", err)
	}
	if executed.Load() {
		t.Fatal("tool executed after context cancellation")
	}
	if len(model.Requests) != 1 {
		t.Fatalf("model requests = %d, want 1", len(model.Requests))
	}
	var observation string
	for _, message := range result.Messages {
		if message.Role == ai.RoleTool {
			observation = message.Text()
		}
	}
	if !strings.Contains(observation, "Operation aborted") {
		t.Fatalf("tool observation = %q", observation)
	}
}

func TestToolExecutionAndContinue(t *testing.T) {
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
	weather := agent.NewTextTool(agent.ToolSpec{Name: "weather"}, func(_ context.Context, call ai.ToolCall) (string, error) {
		received = call
		return "sunny", nil
	})
	a := newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{weather}})
	log := subscribe(a)

	result, err := a.Prompt(context.Background(), "天气")
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
	if got := messageRoles(model.Requests[1].Messages); !reflect.DeepEqual(got, []ai.Role{ai.RoleUser, ai.RoleAssistant, ai.RoleTool}) {
		t.Fatalf("second request roles = %v", got)
	}
	toolResult := result.Messages[2]
	if toolResult.Role != ai.RoleTool || toolResult.ToolCallID != "call_1" || toolResult.Text() != "sunny" {
		t.Fatalf("tool message = %+v", toolResult)
	}
	if !hasEvent(log.snapshot(), agent.EventToolStart) || !hasEvent(log.snapshot(), agent.EventToolEnd) {
		t.Fatalf("missing tool events: %v", eventTypes(log.snapshot()))
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
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "done"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	var received ai.ToolCall
	weather := agent.NewTextTool(agent.ToolSpec{Name: "weather"}, func(_ context.Context, call ai.ToolCall) (string, error) {
		received = call
		return "sunny", nil
	})
	a := newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{weather}})
	if _, err := a.Prompt(context.Background(), "天气"); err != nil {
		t.Fatal(err)
	}
	if received.Name != "weather" || string(received.Arguments) != `{"city":"北京"}` {
		t.Fatalf("received call = %+v", received)
	}
}

func TestStreamingPreservesContentBlockOrder(t *testing.T) {
	model := faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventToolCallStart, ContentIndex: 0, ToolCall: &ai.ToolCall{
				ID: "call_1", Name: "echo",
			}},
			{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{
				ArgumentsDelta: `{}`,
			}},
			{Type: ai.EventTextDelta, ContentIndex: 1, TextDelta: "after tool"},
			{Type: ai.EventThinkingDelta, ContentIndex: 2, ThinkingDelta: "after text"},
			{Type: ai.EventTextDelta, ContentIndex: 1, TextDelta: "!"},
			{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "done"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	echo := agent.NewTextTool(agent.ToolSpec{Name: "echo"}, func(context.Context, ai.ToolCall) (string, error) {
		return "ok", nil
	})
	a := newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{echo}})

	result, err := a.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	assistant := result.Messages[1]
	if assistant.Role != ai.RoleAssistant || len(assistant.Content) != 3 {
		t.Fatalf("assistant = %+v", assistant)
	}
	want := []ai.BlockType{ai.BlockToolCall, ai.BlockText, ai.BlockThinking}
	for i, block := range assistant.Content {
		if block.Type != want[i] {
			t.Fatalf("content[%d] = %q, want %q", i, block.Type, want[i])
		}
	}
	if assistant.Content[1].Text != "after tool!" || assistant.Content[2].Thinking != "after text" {
		t.Fatalf("content = %+v", assistant.Content)
	}
}

func TestToolErrorsAndPanicsAreObservations(t *testing.T) {
	for _, testCase := range []struct {
		name string
		tool agent.Tool
		want string
	}{
		{
			name: "tool error",
			tool: agent.NewTextTool(agent.ToolSpec{Name: "weather"}, func(context.Context, ai.ToolCall) (string, error) {
				return "", errors.New("upstream failed")
			}),
			want: "failed",
		},
		{
			name: "panic",
			tool: agent.NewTextTool(agent.ToolSpec{Name: "weather"}, func(context.Context, ai.ToolCall) (string, error) {
				panic("boom")
			}),
			want: "panicked",
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
			a := newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{testCase.tool}})
			result, err := a.Prompt(context.Background(), "go")
			if err != nil {
				t.Fatal(err)
			}
			toolMessage := result.Messages[len(result.Messages)-2]
			if !toolMessage.IsError || !strings.Contains(toolMessage.Text(), testCase.want) {
				t.Fatalf("tool observation = %+v", toolMessage)
			}
		})
	}
}

func TestBeforeAndAfterToolHooks(t *testing.T) {
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
	echo := agent.NewTextTool(agent.ToolSpec{Name: "echo"}, func(context.Context, ai.ToolCall) (string, error) {
		return "ok", nil
	})
	var beforeCalled, afterCalled bool
	a := newAgent(t, agent.Config{
		Model: model,
		Tools: []agent.Tool{echo},
		BeforeToolCall: func(context.Context, agent.BeforeToolCallContext) (*agent.BeforeToolCallResult, error) {
			beforeCalled = true
			return nil, nil
		},
		AfterToolCall: func(_ context.Context, call agent.AfterToolCallContext) (*agent.AfterToolCallResult, error) {
			afterCalled = true
			isError := false
			return &agent.AfterToolCallResult{
				Content:   []ai.ContentBlock{{Type: ai.BlockText, Text: "audited " + call.Result.Text()}},
				IsError:   &isError,
				Terminate: boolPtr(true),
			}, nil
		},
	})
	result, err := a.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	if !beforeCalled || !afterCalled {
		t.Fatalf("hooks called = before:%v after:%v", beforeCalled, afterCalled)
	}
	if result.Steps != 1 {
		t.Fatalf("terminate should stop after one turn, steps = %d", result.Steps)
	}
	toolMessage := result.Messages[2]
	if toolMessage.Text() != "audited ok" {
		t.Fatalf("tool message = %q", toolMessage.Text())
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
	echo := agent.NewTextTool(agent.ToolSpec{Name: "echo"}, func(context.Context, ai.ToolCall) (string, error) {
		return "ok", nil
	})
	a := newAgent(t, agent.Config{
		Model: model,
		Tools: []agent.Tool{echo},
		BeforeToolCall: func(context.Context, agent.BeforeToolCallContext) (*agent.BeforeToolCallResult, error) {
			return &agent.BeforeToolCallResult{Block: true, Reason: "blocked by policy"}, nil
		},
	})
	result, err := a.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	if got := result.Messages[2].Text(); !strings.Contains(got, "blocked by policy") {
		t.Fatalf("tool observation = %q", got)
	}
}

func TestUnknownAndInvalidToolCallsAreObservations(t *testing.T) {
	call := ai.ToolCall{ID: "call_1", Name: "weather", Arguments: json.RawMessage(`{"city":123}`)}
	model := faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventDone, Message: &ai.Message{
				Role: ai.RoleAssistant,
				Content: []ai.ContentBlock{{
					Type: ai.BlockToolCall, ToolCall: &call,
				}},
			}, StopReason: ai.StopReasonToolCalls},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "recovered"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	executed := false
	weather := agent.NewTextTool(agent.ToolSpec{
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
	a := newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{weather}})
	result, err := a.Prompt(context.Background(), "天气")
	if err != nil {
		t.Fatal(err)
	}
	if executed {
		t.Fatal("invalid tool arguments were executed")
	}
	if toolMessage := result.Messages[2]; !toolMessage.IsError || !strings.Contains(toolMessage.Text(), "expected string") {
		t.Fatalf("tool message = %+v", toolMessage)
	}

	unknownModel := faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{ID: "call_2", Name: "missing", ArgumentsDelta: "{}"}},
			{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "done"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	unknown := newAgent(t, agent.Config{Model: unknownModel})
	result, err = unknown.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	if toolMessage := result.Messages[2]; !toolMessage.IsError || !strings.Contains(toolMessage.Text(), "unknown tool") {
		t.Fatalf("tool message = %+v", toolMessage)
	}
}

func TestMaxStepsAndHistoryMutation(t *testing.T) {
	turn := faux.Turn{Events: []ai.Event{
		{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{ID: "call_1", Name: "echo", ArgumentsDelta: "{}"}},
		{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
	}}
	model := faux.New(turn, turn)
	echo := agent.NewTextTool(agent.ToolSpec{Name: "echo"}, func(context.Context, ai.ToolCall) (string, error) {
		return "ok", nil
	})
	a := newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{echo}, MaxSteps: 2})
	if _, err := a.Prompt(context.Background(), "go"); !errors.Is(err, agent.ErrMaxSteps) {
		t.Fatalf("error = %v, want ErrMaxSteps", err)
	}

	a.SetMessages([]ai.Message{ai.TextMessage(ai.RoleUser, "old")})
	if got := messageRoles(a.Messages()); !reflect.DeepEqual(got, []ai.Role{ai.RoleUser}) {
		t.Fatalf("messages = %v", got)
	}
	if err := a.Reset(); err != nil {
		t.Fatal(err)
	}
	if got := messageRoles(a.Messages()); len(got) != 0 {
		t.Fatalf("reset without baseline messages = %v", got)
	}
}

func TestOutputLengthSemantics(t *testing.T) {
	model := faux.New(faux.Turn{Events: []ai.Event{
		{Type: ai.EventTextDelta, TextDelta: "partial"},
		{Type: ai.EventDone, StopReason: ai.StopReasonLength},
	}})
	a := newAgent(t, agent.Config{Model: model})
	result, err := a.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatalf("length without tools should be a normal stop: %v", err)
	}
	if result.Answer != "partial" || result.StopReason != ai.StopReasonLength {
		t.Fatalf("result = %+v", result)
	}

	call := ai.ToolCall{ID: "call_1", Name: "weather", Arguments: json.RawMessage(`{"city":"北京"}`)}
	truncated := ai.Message{
		Role: ai.RoleAssistant,
		Content: []ai.ContentBlock{{
			Type: ai.BlockToolCall, ToolCall: &call,
		}},
	}
	model = faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventDone, Message: &truncated, StopReason: ai.StopReasonLength},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "recovered"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	executed := false
	weather := agent.NewTextTool(agent.ToolSpec{Name: "weather"}, func(context.Context, ai.ToolCall) (string, error) {
		executed = true
		return "sunny", nil
	})
	a = newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{weather}})
	result, err = a.Prompt(context.Background(), "天气")
	if err != nil {
		t.Fatal(err)
	}
	if executed || result.Answer != "recovered" || result.Steps != 2 {
		t.Fatalf("result = %+v executed=%v", result, executed)
	}
	if toolMessage := result.Messages[2]; !toolMessage.IsError || !strings.Contains(toolMessage.Text(), "output token limit") {
		t.Fatalf("tool message = %+v", toolMessage)
	}
}

func TestModelAndStreamErrorsPreservePartialMessages(t *testing.T) {
	model := faux.New(faux.Turn{Events: []ai.Event{
		{Type: ai.EventTextDelta, TextDelta: "partial"},
		{Type: ai.EventDone, StopReason: ai.StopReasonError},
	}})
	a := newAgent(t, agent.Config{Model: model})
	result, err := a.Prompt(context.Background(), "go")
	var modelErr *agent.ModelError
	if !errors.As(err, &modelErr) {
		t.Fatalf("error = %v, want ModelError", err)
	}
	if result.Answer != "partial" || result.StopReason != ai.StopReasonError {
		t.Fatalf("result = %+v", result)
	}

	streamErr := errors.New("provider failed")
	partial := ai.TextMessage(ai.RoleAssistant, "partial")
	model = faux.New(faux.Turn{Events: []ai.Event{
		{Type: ai.EventTextDelta, TextDelta: "partial"},
		{
			Type:       ai.EventError,
			Err:        streamErr,
			Message:    &partial,
			Usage:      &ai.Usage{PromptTokens: 10, CompletionTokens: 2, TotalTokens: 12},
			StopReason: ai.StopReasonError,
		},
	}})
	a = newAgent(t, agent.Config{Model: model})
	log := subscribe(a)
	result, err = a.Prompt(context.Background(), "go")
	if !errors.Is(err, streamErr) {
		t.Fatalf("error = %v", err)
	}
	if result.Answer != "partial" || result.Usage.TotalTokens != 12 {
		t.Fatalf("result = %+v", result)
	}
	if !hasEvent(log.snapshot(), agent.EventAgentError) || !hasEvent(log.snapshot(), agent.EventAgentEnd) {
		t.Fatalf("events = %v", eventTypes(log.snapshot()))
	}
}

func TestDoneMessageUsageAndThinkingSignature(t *testing.T) {
	final := ai.Message{
		Role: ai.RoleAssistant,
		Content: []ai.ContentBlock{
			{Type: ai.BlockThinking, Thinking: "plan", ThinkingSignature: "sig"},
			{Type: ai.BlockText, Text: "final"},
		},
	}
	model := faux.New(faux.Turn{Events: []ai.Event{
		{Type: ai.EventTextDelta, TextDelta: "ignored"},
		{Type: ai.EventThinkingStart},
		{Type: ai.EventThinkingDelta, ThinkingDelta: "plan"},
		{Type: ai.EventThinkingEnd, Thinking: "plan"},
		{Type: ai.EventDone, Message: &final, StopReason: ai.StopReasonStop, Usage: &ai.Usage{
			PromptTokens: 10, CompletionTokens: 2, TotalTokens: 12,
		}},
	}})
	a := newAgent(t, agent.Config{Model: model})
	result, err := a.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	if result.Answer != "final" || result.Usage.TotalTokens != 12 {
		t.Fatalf("result = %+v", result)
	}
	if result.Messages[1].Thinking() != "plan" || result.Messages[1].Content[0].ThinkingSignature != "sig" {
		t.Fatalf("thinking message = %+v", result.Messages[1])
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
	slow := agent.NewTextTool(agent.ToolSpec{Name: "slow"}, func(ctx context.Context, _ ai.ToolCall) (string, error) {
		close(slowStarted)
		select {
		case <-fastDone:
			return "slow result", nil
		case <-ctx.Done():
			return "", ctx.Err()
		}
	})
	fast := agent.NewTextTool(agent.ToolSpec{Name: "fast"}, func(ctx context.Context, _ ai.ToolCall) (string, error) {
		select {
		case <-slowStarted:
			close(fastDone)
			return "fast result", nil
		case <-ctx.Done():
			return "", ctx.Err()
		}
	})
	a := newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{slow, fast}})
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	result, err := a.Prompt(ctx, "go")
	if err != nil {
		t.Fatal(err)
	}
	if result.Messages[2].ToolName != "slow" || result.Messages[3].ToolName != "fast" {
		t.Fatalf("tool result order = %+v", result.Messages)
	}
}

func TestPerToolSequentialForcesBatchOrder(t *testing.T) {
	model := faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{ID: "call_1", Name: "slow", ArgumentsDelta: "{}"}},
			{Type: ai.EventToolCallDelta, ContentIndex: 1, ToolCallDelta: &ai.ToolCallDelta{ID: "call_2", Name: "fast", ArgumentsDelta: "{}"}},
			{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "done"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	slow := agent.NewTextTool(agent.ToolSpec{Name: "slow", ExecutionMode: agent.ToolExecutionSequential}, func(context.Context, ai.ToolCall) (string, error) {
		time.Sleep(20 * time.Millisecond)
		return "slow", nil
	})
	fast := agent.NewTextTool(agent.ToolSpec{Name: "fast"}, func(context.Context, ai.ToolCall) (string, error) {
		return "fast", nil
	})
	a := newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{slow, fast}})
	log := subscribe(a)
	if _, err := a.Prompt(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	var endOrder []string
	for _, event := range log.snapshot() {
		if event.Type == agent.EventToolEnd && event.ToolCall != nil {
			endOrder = append(endOrder, event.ToolCall.Name)
		}
	}
	if !reflect.DeepEqual(endOrder, []string{"slow", "fast"}) {
		t.Fatalf("tool_end order = %v", endOrder)
	}
}

func TestStreamingToolUpdatesAndTerminate(t *testing.T) {
	model := faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{ID: "call_1", Name: "stream", ArgumentsDelta: "{}"}},
			{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "done"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	streamTool := agent.NewStreamingTool(agent.ToolSpec{Name: "stream"}, func(_ context.Context, _ ai.ToolCall, update func(agent.ToolResult)) (agent.ToolResult, error) {
		update(agent.TextResult("first"))
		update(agent.TextResult("second"))
		result := agent.TextResult("done")
		result.Terminate = true
		return result, nil
	})
	a := newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{streamTool}})
	log := subscribe(a)
	result, err := a.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	if result.Steps != 1 {
		t.Fatalf("terminate should stop after one turn, steps = %d", result.Steps)
	}
	updates := 0
	for _, event := range log.snapshot() {
		if event.Type == agent.EventToolUpdate {
			updates++
		}
	}
	if updates != 2 {
		t.Fatalf("tool updates = %d, events = %v", updates, eventTypes(log.snapshot()))
	}
}

func TestLateToolUpdatesAreIgnored(t *testing.T) {
	model := faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{ID: "call_1", Name: "stream", ArgumentsDelta: "{}"}},
			{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "done"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	release := make(chan struct{})
	streamTool := agent.NewStreamingTool(agent.ToolSpec{Name: "stream"}, func(_ context.Context, _ ai.ToolCall, update func(agent.ToolResult)) (agent.ToolResult, error) {
		go func() {
			<-release
			update(agent.TextResult("late"))
		}()
		return agent.TextResult("done"), nil
	})
	a := newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{streamTool}})
	log := subscribe(a)
	if _, err := a.Prompt(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	close(release)
	time.Sleep(30 * time.Millisecond)
	for _, event := range log.snapshot() {
		if event.Type == agent.EventToolUpdate {
			t.Fatalf("late tool update was emitted: %+v", event)
		}
	}
}

func TestToolUpdatesFinishBeforeToolEnd(t *testing.T) {
	model := faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{ID: "call_1", Name: "stream", ArgumentsDelta: "{}"}},
			{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "done"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	inListener := make(chan struct{})
	releaseListener := make(chan struct{})
	streamTool := agent.NewStreamingTool(agent.ToolSpec{Name: "stream"}, func(_ context.Context, _ ai.ToolCall, update func(agent.ToolResult)) (agent.ToolResult, error) {
		go update(agent.TextResult("progress"))
		<-inListener
		return agent.TextResult("done"), nil
	})
	a := newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{streamTool}})
	log := subscribe(a)
	var once sync.Once
	a.Subscribe(func(event agent.Event) {
		if event.Type == agent.EventToolUpdate {
			once.Do(func() { close(inListener) })
			<-releaseListener
		}
	})
	done := make(chan error, 1)
	go func() {
		_, err := a.Prompt(context.Background(), "go")
		done <- err
	}()
	select {
	case err := <-done:
		t.Fatalf("run returned while a tool update was still in flight: %v", err)
	case <-time.After(30 * time.Millisecond):
	}
	close(releaseListener)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	types := eventTypes(log.snapshot())
	updateIndex := -1
	toolEndIndex := -1
	agentEndIndex := -1
	for i, eventType := range types {
		switch eventType {
		case agent.EventToolUpdate:
			if updateIndex < 0 {
				updateIndex = i
			}
		case agent.EventToolEnd:
			toolEndIndex = i
		case agent.EventAgentEnd:
			agentEndIndex = i
		}
	}
	if updateIndex < 0 || toolEndIndex < 0 || agentEndIndex < 0 ||
		!(updateIndex < toolEndIndex && toolEndIndex < agentEndIndex) {
		t.Fatalf("events = %v", types)
	}
}

func TestParallelToolsPreflightBeforeExecution(t *testing.T) {
	model := faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{ID: "call_1", Name: "first", ArgumentsDelta: "{}"}},
			{Type: ai.EventToolCallDelta, ContentIndex: 1, ToolCallDelta: &ai.ToolCallDelta{ID: "call_2", Name: "second", ArgumentsDelta: "{}"}},
			{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "done"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	var firstStarted sync.Once
	firstStartedCh := make(chan struct{})
	var mu sync.Mutex
	preflightSawExecution := false
	first := agent.NewTextTool(agent.ToolSpec{Name: "first"}, func(context.Context, ai.ToolCall) (string, error) {
		firstStarted.Do(func() { close(firstStartedCh) })
		return "first", nil
	})
	second := agent.NewTextTool(agent.ToolSpec{Name: "second"}, func(context.Context, ai.ToolCall) (string, error) {
		return "second", nil
	})
	a := newAgent(t, agent.Config{
		Model: model,
		Tools: []agent.Tool{first, second},
		BeforeToolCall: func(_ context.Context, call agent.BeforeToolCallContext) (*agent.BeforeToolCallResult, error) {
			if call.ToolCall.Name == "second" {
				time.Sleep(10 * time.Millisecond)
				select {
				case <-firstStartedCh:
					mu.Lock()
					preflightSawExecution = true
					mu.Unlock()
				default:
				}
			}
			return nil, nil
		},
	})
	if _, err := a.Prompt(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if preflightSawExecution {
		t.Fatal("a tool started executing before all tool calls were preflighted")
	}
}

func TestAfterToolCallContentReplacementClearsStructuredContent(t *testing.T) {
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
	echo := agent.NewTool(agent.ToolSpec{Name: "echo"}, func(context.Context, ai.ToolCall) (agent.ToolResult, error) {
		return agent.ToolResult{
			Content:           []ai.ContentBlock{{Type: ai.BlockText, Text: "original"}},
			StructuredContent: json.RawMessage(`{"v":1}`),
		}, nil
	})
	a := newAgent(t, agent.Config{
		Model: model,
		Tools: []agent.Tool{echo},
		AfterToolCall: func(context.Context, agent.AfterToolCallContext) (*agent.AfterToolCallResult, error) {
			return &agent.AfterToolCallResult{
				Content: []ai.ContentBlock{{Type: ai.BlockText, Text: "changed"}},
			}, nil
		},
	})
	result, err := a.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	toolMessage := result.Messages[2]
	if toolMessage.Text() != "changed" || toolMessage.StructuredContent != nil {
		t.Fatalf("tool message = %+v", toolMessage)
	}
}

func TestContinueBusyKeepsQueuedMessages(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	blocking := agent.NewTextTool(agent.ToolSpec{Name: "block"}, func(context.Context, ai.ToolCall) (string, error) {
		close(started)
		<-release
		return "ok", nil
	})
	model := faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{ID: "call_1", Name: "block", ArgumentsDelta: "{}"}},
			{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "done"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	a := newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{blocking}})
	done := make(chan error, 1)
	go func() {
		_, err := a.Prompt(context.Background(), "go")
		done <- err
	}()
	<-started
	a.Steer(ai.TextMessage(ai.RoleUser, "steer"))
	a.FollowUp(ai.TextMessage(ai.RoleUser, "follow"))
	if _, err := a.Continue(context.Background()); !errors.Is(err, agent.ErrBusy) {
		t.Fatalf("error = %v", err)
	}
	if !a.HasQueuedMessages() {
		t.Fatal("busy Continue consumed queued messages")
	}
	a.ClearAllQueues()
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func TestFailureBeforeAssistantSynthesizesMessage(t *testing.T) {
	model := faux.New()
	a := newAgent(t, agent.Config{
		Model: model,
		PrepareRequest: func(context.Context, agent.PrepareRequestContext) (*agent.RequestUpdate, error) {
			return nil, errors.New("prepare failed")
		},
	})
	log := subscribe(a)
	result, err := a.Prompt(context.Background(), "go")
	if err == nil || !strings.Contains(err.Error(), "prepare failed") {
		t.Fatalf("error = %v", err)
	}
	if len(result.Messages) != 2 || result.Messages[1].Role != ai.RoleAssistant ||
		result.Messages[1].StopReason != ai.StopReasonError ||
		!strings.Contains(result.Messages[1].ErrorMessage, "prepare failed") {
		t.Fatalf("result messages = %+v", result.Messages)
	}
	types := eventTypes(log.snapshot())
	wantPrefix := []agent.EventType{
		agent.EventAgentStart,
		agent.EventTurnStart,
		agent.EventMessageStart,
		agent.EventMessageEnd,
		agent.EventMessageStart,
		agent.EventMessageEnd,
		agent.EventTurnEnd,
	}
	if len(types) < len(wantPrefix) || !reflect.DeepEqual(types[:len(wantPrefix)], wantPrefix) {
		t.Fatalf("events = %v", types)
	}
}

func TestTransformMessagesAndRequestOptions(t *testing.T) {
	temperature := 0.2
	model := faux.New(faux.Turn{Events: []ai.Event{
		{Type: ai.EventTextDelta, TextDelta: "ok"},
		{Type: ai.EventDone, StopReason: ai.StopReasonStop},
	}})
	transformed := false
	a := newAgent(t, agent.Config{
		Model: model,
		TransformMessages: func(_ context.Context, messages []ai.Message) ([]ai.Message, error) {
			transformed = true
			return append(messages, ai.TextMessage(ai.RoleUser, "injected")), nil
		},
	})
	a.SetRequestOptions(ai.RequestOptions{Temperature: &temperature, MaxTokens: 128})
	result, err := a.Prompt(context.Background(), "hello")
	if err != nil {
		t.Fatal(err)
	}
	if !transformed {
		t.Fatal("message transformer was not called")
	}
	request := model.Requests[0]
	if len(request.Messages) != 2 || request.Messages[1].Text() != "injected" {
		t.Fatalf("transformed messages = %+v", request.Messages)
	}
	if request.Options.Temperature == nil || *request.Options.Temperature != temperature || request.Options.MaxTokens != 128 {
		t.Fatalf("request options = %+v", request.Options)
	}
	if len(result.Messages) != 2 {
		t.Fatalf("session was polluted by transformed messages: %+v", result.Messages)
	}
}

func TestPrepareRequestFinishTurnAndPrepareNextTurn(t *testing.T) {
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
	echo := agent.NewTextTool(agent.ToolSpec{Name: "echo"}, func(context.Context, ai.ToolCall) (string, error) {
		return "ok", nil
	})
	var preparedRequest bool
	var preparedNext bool
	a := newAgent(t, agent.Config{
		Model: model,
		Tools: []agent.Tool{echo},
		PrepareRequest: func(_ context.Context, request agent.PrepareRequestContext) (*agent.RequestUpdate, error) {
			preparedRequest = true
			context := request.Context
			context.Messages = append(context.Messages, ai.TextMessage(ai.RoleUser, "request-context"))
			return &agent.RequestUpdate{Context: &context}, nil
		},
		PrepareNextTurn: func(_ context.Context, turn agent.TurnContext) (*agent.TurnUpdate, error) {
			preparedNext = true
			return &agent.TurnUpdate{Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "prepared-next")}}, nil
		},
	})
	if _, err := a.Prompt(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	if !preparedRequest || !preparedNext {
		t.Fatalf("hooks called = request:%v next:%v", preparedRequest, preparedNext)
	}
	if got := model.Requests[0].Messages[len(model.Requests[0].Messages)-1].Text(); got != "request-context" {
		t.Fatalf("prepare request message = %q", got)
	}
	foundPreparedNext := false
	foundRequestContext := false
	for _, message := range model.Requests[1].Messages {
		if message.Text() == "prepared-next" {
			foundPreparedNext = true
		}
		if message.Text() == "request-context" {
			foundRequestContext = true
		}
	}
	if !foundPreparedNext || !foundRequestContext {
		t.Fatalf("prepared context missing from second request: %+v", model.Requests[1].Messages)
	}
}

func TestFinishTurnCanEndRun(t *testing.T) {
	model := faux.New(faux.Turn{Events: []ai.Event{
		{Type: ai.EventTextDelta, TextDelta: "done"},
		{Type: ai.EventDone, StopReason: ai.StopReasonStop},
	}})
	a := newAgent(t, agent.Config{
		Model: model,
		FinishTurn: func(context.Context, agent.TurnContext) (agent.TurnDecision, error) {
			return agent.TurnEnd, nil
		},
	})
	result, err := a.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	if result.Steps != 1 || len(model.Requests) != 1 {
		t.Fatalf("result = %+v requests = %d", result, len(model.Requests))
	}
}

func TestSteeringAndFollowUpQueues(t *testing.T) {
	model := faux.New(
		faux.Turn{Events: []ai.Event{{Type: ai.EventTextDelta, TextDelta: "first"}, {Type: ai.EventDone, StopReason: ai.StopReasonStop}}},
		faux.Turn{Events: []ai.Event{{Type: ai.EventTextDelta, TextDelta: "second"}, {Type: ai.EventDone, StopReason: ai.StopReasonStop}}},
		faux.Turn{Events: []ai.Event{{Type: ai.EventTextDelta, TextDelta: "third"}, {Type: ai.EventDone, StopReason: ai.StopReasonStop}}},
	)
	a := newAgent(t, agent.Config{Model: model})
	a.Steer(ai.TextMessage(ai.RoleUser, "steer"))
	a.FollowUp(ai.TextMessage(ai.RoleUser, "follow"))
	if !a.HasQueuedMessages() {
		t.Fatal("queues should be non-empty")
	}
	if got := a.PeekQueuedMessages(); len(got) != 1 || got[0].Text() != "steer" {
		t.Fatalf("peek = %+v", got)
	}
	result, err := a.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	if result.Answer != "second" || result.Steps != 2 {
		t.Fatalf("result = %+v", result)
	}
	roles := messageRoles(result.Messages)
	if !reflect.DeepEqual(roles, []ai.Role{
		ai.RoleUser, ai.RoleUser, ai.RoleAssistant,
		ai.RoleUser, ai.RoleAssistant,
	}) {
		t.Fatalf("roles = %v", roles)
	}
	firstRequest := model.Requests[0].Messages
	if len(firstRequest) != 2 || firstRequest[0].Text() != "go" || firstRequest[1].Text() != "steer" {
		t.Fatalf("steering should be present in the first request: %+v", firstRequest)
	}
}

func TestContinueValidationAndAssistantQueueFallback(t *testing.T) {
	model := faux.New(faux.Turn{Events: []ai.Event{
		{Type: ai.EventTextDelta, TextDelta: "continued"},
		{Type: ai.EventDone, StopReason: ai.StopReasonStop},
	}})
	a := newAgent(t, agent.Config{Model: model, Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "old")}})
	result, err := a.Continue(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if result.Answer != "continued" {
		t.Fatalf("result = %+v", result)
	}

	model = faux.New(faux.Turn{Events: []ai.Event{
		{Type: ai.EventTextDelta, TextDelta: "queued"},
		{Type: ai.EventDone, StopReason: ai.StopReasonStop},
	}})
	a = newAgent(t, agent.Config{Model: model, Messages: []ai.Message{
		ai.TextMessage(ai.RoleUser, "old"),
		ai.TextMessage(ai.RoleAssistant, "answer"),
	}})
	if _, err := a.Continue(context.Background()); !errors.Is(err, agent.ErrCannotContinue) {
		t.Fatalf("error = %v", err)
	}
	a.Steer(ai.TextMessage(ai.RoleUser, "queued"))
	result, err = a.Continue(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if result.Answer != "queued" {
		t.Fatalf("result = %+v", result)
	}
}

func TestAbortAndWaitForIdle(t *testing.T) {
	started := make(chan struct{})
	blocking := agent.NewTool(agent.ToolSpec{Name: "block"}, func(ctx context.Context, _ ai.ToolCall) (agent.ToolResult, error) {
		close(started)
		<-ctx.Done()
		return agent.ToolResult{}, ctx.Err()
	})
	model := faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{ID: "call_1", Name: "block", ArgumentsDelta: "{}"}},
			{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "should not run"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	a := newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{blocking}})
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
		t.Fatalf("concurrent prompt error = %v", err)
	}
	a.Abort()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("abort error = %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("prompt did not abort")
	}
	waitCtx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := a.WaitForIdle(waitCtx); err != nil {
		t.Fatal(err)
	}
}

func TestSubscribeOrderAndUnsubscribe(t *testing.T) {
	model := faux.New(
		faux.Turn{Events: []ai.Event{{Type: ai.EventDone, StopReason: ai.StopReasonStop}}},
		faux.Turn{Events: []ai.Event{{Type: ai.EventDone, StopReason: ai.StopReasonStop}}},
	)
	a := newAgent(t, agent.Config{Model: model})
	var order []string
	var mu sync.Mutex
	unsubscribeFirst := a.Subscribe(func(event agent.Event) {
		if event.Type == agent.EventAgentStart {
			mu.Lock()
			order = append(order, "first")
			mu.Unlock()
		}
	})
	a.Subscribe(func(event agent.Event) {
		if event.Type == agent.EventAgentStart {
			mu.Lock()
			order = append(order, "second")
			mu.Unlock()
		}
	})
	if _, err := a.Prompt(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	unsubscribeFirst()
	if _, err := a.Prompt(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if !reflect.DeepEqual(order, []string{"first", "second", "second"}) {
		t.Fatalf("listener order = %v", order)
	}
}

func TestToolResultDetailsStructuredContentAndUsage(t *testing.T) {
	model := faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{ID: "call_1", Name: "detail", ArgumentsDelta: "{}"}},
			{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "done"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	detailTool := agent.NewTool(agent.ToolSpec{Name: "detail"}, func(context.Context, ai.ToolCall) (agent.ToolResult, error) {
		return agent.ToolResult{
			Content:           []ai.ContentBlock{{Type: ai.BlockText, Text: "preview"}},
			Details:           json.RawMessage(`{"audited":true}`),
			StructuredContent: json.RawMessage(`{"value":42}`),
			Usage:             &ai.Usage{TotalTokens: 3},
		}, nil
	})
	a := newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{detailTool}})
	result, err := a.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	toolMessage := result.Messages[2]
	if string(toolMessage.Details) != `{"audited":true}` ||
		string(toolMessage.StructuredContent) != `{"value":42}` ||
		toolMessage.Usage == nil || toolMessage.Usage.TotalTokens != 3 ||
		toolMessage.DurationMs < 0 {
		t.Fatalf("tool message = %+v", toolMessage)
	}
}

func TestToolResultImagesAndInvalidContent(t *testing.T) {
	model := faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{ID: "call_1", Name: "screenshot", ArgumentsDelta: "{}"}},
			{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "done"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	screenshot := agent.NewTool(agent.ToolSpec{Name: "screenshot"}, func(context.Context, ai.ToolCall) (agent.ToolResult, error) {
		return agent.ToolResult{Content: []ai.ContentBlock{
			{Type: ai.BlockText, Text: "preview"},
			{Type: ai.BlockImage, Image: &ai.ImageContent{Data: "aGVsbG8=", MediaType: "image/png"}},
		}}, nil
	})
	a := newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{screenshot}})
	result, err := a.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	toolMessage := result.Messages[2]
	if toolMessage.IsError || len(toolMessage.Content) != 2 || toolMessage.Content[1].Image == nil {
		t.Fatalf("tool message = %+v", toolMessage)
	}

	invalid := agent.NewTool(agent.ToolSpec{Name: "screenshot"}, func(context.Context, ai.ToolCall) (agent.ToolResult, error) {
		return agent.ToolResult{Content: []ai.ContentBlock{{
			Type: ai.BlockImage,
			Image: &ai.ImageContent{
				Data:      "aGVsbG8=",
				URL:       "https://example.test/image.png",
				MediaType: "image/png",
			},
		}}}, nil
	})
	model = faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{ID: "call_1", Name: "screenshot", ArgumentsDelta: "{}"}},
			{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "done"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	a = newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{invalid}})
	result, err = a.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	if toolMessage := result.Messages[2]; !toolMessage.IsError || !strings.Contains(toolMessage.Text(), "exactly one") {
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
	weather := agent.NewTextTool(agent.ToolSpec{Name: "weather"}, func(context.Context, ai.ToolCall) (string, error) {
		executed = true
		return "sunny", nil
	})
	a := newAgent(t, agent.Config{Model: model, Tools: []agent.Tool{weather}})
	result, err := a.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	if executed {
		t.Fatal("tool with invalid streamed arguments was executed")
	}
	if toolMessage := result.Messages[2]; !toolMessage.IsError || !strings.Contains(toolMessage.Text(), "not valid JSON") {
		t.Fatalf("tool message = %+v", toolMessage)
	}
}

func newAgent(t *testing.T, config agent.Config) *agent.Agent {
	t.Helper()
	a, err := agent.New(config)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

type eventLog struct {
	mu     sync.Mutex
	events []agent.Event
}

func subscribe(a *agent.Agent) *eventLog {
	log := &eventLog{}
	a.Subscribe(func(event agent.Event) {
		log.mu.Lock()
		log.events = append(log.events, event)
		log.mu.Unlock()
	})
	return log
}

func (l *eventLog) snapshot() []agent.Event {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]agent.Event(nil), l.events...)
}

func eventTypes(events []agent.Event) []agent.EventType {
	types := make([]agent.EventType, len(events))
	for i, event := range events {
		types[i] = event.Type
	}
	return types
}

func hasEvent(events []agent.Event, eventType agent.EventType) bool {
	for _, event := range events {
		if event.Type == eventType {
			return true
		}
	}
	return false
}

func messageRoles(messages []ai.Message) []ai.Role {
	roles := make([]ai.Role, len(messages))
	for i, message := range messages {
		roles[i] = message.Role
	}
	return roles
}

func boolPtr(value bool) *bool {
	return &value
}
