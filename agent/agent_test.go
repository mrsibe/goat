package agent_test

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/mrsibe/goat/agent"
	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/internal/faux"
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
			{Type: ai.EventToolCallDelta, ToolCall: &ai.ToolCallDelta{
				Index: 0, ID: "call_1", Name: "weather", ArgumentsDelta: `{"city":`,
			}},
			{Type: ai.EventToolCallDelta, ToolCall: &ai.ToolCallDelta{
				Index: 0, ArgumentsDelta: `"北京"}`,
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
	weather := agent.NewTool(ai.ToolSpec{Name: "weather"}, func(_ context.Context, call ai.ToolCall) (string, error) {
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

func TestToolErrorsAreObservations(t *testing.T) {
	for _, testCase := range []struct {
		name string
		tool agent.Tool
	}{
		{
			name: "tool error",
			tool: agent.NewTool(ai.ToolSpec{Name: "weather"}, func(context.Context, ai.ToolCall) (string, error) {
				return "", errors.New("upstream failed")
			}),
		},
		{
			name: "panic",
			tool: agent.NewTool(ai.ToolSpec{Name: "weather"}, func(context.Context, ai.ToolCall) (string, error) {
				panic("boom")
			}),
		},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			model := faux.New(
				faux.Turn{Events: []ai.Event{
					{Type: ai.EventToolCallDelta, ToolCall: &ai.ToolCallDelta{Index: 0, ID: "call_1", Name: "weather", ArgumentsDelta: "{}"}},
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
		})
	}
}

func TestUnknownToolAndHooks(t *testing.T) {
	model := faux.New(
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventToolCallDelta, ToolCall: &ai.ToolCallDelta{Index: 0, ID: "call_1", Name: "missing", ArgumentsDelta: "{}"}},
			{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
		}},
		faux.Turn{Events: []ai.Event{
			{Type: ai.EventTextDelta, TextDelta: "done"},
			{Type: ai.EventDone, StopReason: ai.StopReasonStop},
		}},
	)
	var beforeCalled bool
	assistant, err := agent.New(agent.Config{
		Model: model,
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

func TestMaxStepsAndHistoryRestore(t *testing.T) {
	turn := faux.Turn{Events: []ai.Event{
		{Type: ai.EventToolCallDelta, ToolCall: &ai.ToolCallDelta{Index: 0, ID: "call_1", Name: "echo", ArgumentsDelta: "{}"}},
		{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
	}}
	model := faux.New(turn, turn)
	echo := agent.NewTool(ai.ToolSpec{Name: "echo"}, func(context.Context, ai.ToolCall) (string, error) {
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

func eventTypes(events []agent.Event) []agent.EventType {
	types := make([]agent.EventType, len(events))
	for i, event := range events {
		types[i] = event.Type
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
