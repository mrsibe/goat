package agent_test

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/mrsibe/goat/agent"
	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/ai/faux"
)

func TestAgentConformanceMatrix(t *testing.T) {
	echo := agent.NewTextTool(agent.ToolSpec{Name: "echo"}, func(context.Context, ai.ToolCall) (string, error) {
		return "ok", nil
	})
	terminate := agent.NewTool(agent.ToolSpec{Name: "stop"}, func(context.Context, ai.ToolCall) (agent.ToolResult, error) {
		result := agent.TextResult("stopped")
		result.Terminate = true
		return result, nil
	})

	cases := []struct {
		name         string
		model        func() *faux.Model
		tools        []agent.Tool
		maxSteps     int
		prepare      agent.PrepareRequest
		prompt       string
		steer        string
		wantErr      string
		wantRoles    []ai.Role
		wantNewRoles []ai.Role
		wantStop     ai.StopReason
		wantSteps    int
	}{
		{
			name: "direct answer",
			model: func() *faux.Model {
				return faux.New(faux.Turn{Events: []ai.Event{
					{Type: ai.EventTextDelta, TextDelta: "hello"},
					{Type: ai.EventDone, StopReason: ai.StopReasonStop},
				}})
			},
			prompt:       "hi",
			wantRoles:    []ai.Role{ai.RoleUser, ai.RoleAssistant},
			wantNewRoles: []ai.Role{ai.RoleUser, ai.RoleAssistant},
			wantStop:     ai.StopReasonStop,
			wantSteps:    1,
		},
		{
			name: "tool round trip",
			model: func() *faux.Model {
				return faux.New(
					faux.Turn{Events: []ai.Event{
						{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{
							ID: "call_1", Name: "echo", ArgumentsDelta: "{}",
						}},
						{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
					}},
					faux.Turn{Events: []ai.Event{
						{Type: ai.EventTextDelta, TextDelta: "done"},
						{Type: ai.EventDone, StopReason: ai.StopReasonStop},
					}},
				)
			},
			tools:        []agent.Tool{echo},
			prompt:       "go",
			wantRoles:    []ai.Role{ai.RoleUser, ai.RoleAssistant, ai.RoleTool, ai.RoleAssistant},
			wantNewRoles: []ai.Role{ai.RoleUser, ai.RoleAssistant, ai.RoleTool, ai.RoleAssistant},
			wantStop:     ai.StopReasonStop,
			wantSteps:    2,
		},
		{
			name: "provider error",
			model: func() *faux.Model {
				partial := ai.TextMessage(ai.RoleAssistant, "partial")
				return faux.New(faux.Turn{Events: []ai.Event{{
					Type:       ai.EventError,
					Err:        errors.New("provider failed"),
					Message:    &partial,
					StopReason: ai.StopReasonError,
				}}})
			},
			prompt:       "go",
			wantErr:      "provider failed",
			wantRoles:    []ai.Role{ai.RoleUser, ai.RoleAssistant},
			wantNewRoles: []ai.Role{ai.RoleUser, ai.RoleAssistant},
			wantStop:     ai.StopReasonError,
			wantSteps:    1,
		},
		{
			name:  "pre-assistant failure",
			model: func() *faux.Model { return faux.New() },
			prepare: func(context.Context, agent.PrepareRequestContext) (*agent.RequestUpdate, error) {
				return nil, errors.New("prepare failed")
			},
			prompt:       "go",
			wantErr:      "prepare failed",
			wantRoles:    []ai.Role{ai.RoleUser, ai.RoleAssistant},
			wantNewRoles: []ai.Role{ai.RoleUser, ai.RoleAssistant},
			wantStop:     ai.StopReasonError,
			wantSteps:    0,
		},
		{
			name: "terminate",
			model: func() *faux.Model {
				return faux.New(faux.Turn{Events: []ai.Event{
					{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{
						ID: "call_1", Name: "stop", ArgumentsDelta: "{}",
					}},
					{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
				}})
			},
			tools:        []agent.Tool{terminate},
			prompt:       "go",
			wantRoles:    []ai.Role{ai.RoleUser, ai.RoleAssistant, ai.RoleTool},
			wantNewRoles: []ai.Role{ai.RoleUser, ai.RoleAssistant, ai.RoleTool},
			wantStop:     ai.StopReasonToolCalls,
			wantSteps:    1,
		},
		{
			name: "steering before first request",
			model: func() *faux.Model {
				return faux.New(faux.Turn{Events: []ai.Event{
					{Type: ai.EventTextDelta, TextDelta: "steered"},
					{Type: ai.EventDone, StopReason: ai.StopReasonStop},
				}})
			},
			prompt:       "go",
			steer:        "steer",
			wantRoles:    []ai.Role{ai.RoleUser, ai.RoleUser, ai.RoleAssistant},
			wantNewRoles: []ai.Role{ai.RoleUser, ai.RoleUser, ai.RoleAssistant},
			wantStop:     ai.StopReasonStop,
			wantSteps:    1,
		},
		{
			name: "max steps",
			model: func() *faux.Model {
				turn := faux.Turn{Events: []ai.Event{
					{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{
						ID: "call_1", Name: "echo", ArgumentsDelta: "{}",
					}},
					{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
				}}
				return faux.New(turn, turn)
			},
			tools:        []agent.Tool{echo},
			maxSteps:     2,
			prompt:       "go",
			wantErr:      agent.ErrMaxSteps.Error(),
			wantRoles:    []ai.Role{ai.RoleUser, ai.RoleAssistant, ai.RoleTool, ai.RoleAssistant, ai.RoleTool, ai.RoleAssistant},
			wantNewRoles: []ai.Role{ai.RoleUser, ai.RoleAssistant, ai.RoleTool, ai.RoleAssistant, ai.RoleTool, ai.RoleAssistant},
			wantStop:     ai.StopReasonError,
			wantSteps:    2,
		},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			a := newAgent(t, agent.Config{
				Model:          testCase.model(),
				Tools:          testCase.tools,
				MaxSteps:       testCase.maxSteps,
				PrepareRequest: testCase.prepare,
			})
			if testCase.steer != "" {
				a.Steer(ai.TextMessage(ai.RoleUser, testCase.steer))
			}
			log := subscribe(a)
			result, err := a.Prompt(context.Background(), testCase.prompt)
			if testCase.wantErr == "" {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil || !strings.Contains(err.Error(), testCase.wantErr) {
				t.Fatalf("error = %v, want %q", err, testCase.wantErr)
			}
			if got := messageRoles(result.Messages); !reflect.DeepEqual(got, testCase.wantRoles) {
				t.Fatalf("roles = %v, want %v", got, testCase.wantRoles)
			}
			if got := messageRoles(result.NewMessages); !reflect.DeepEqual(got, testCase.wantNewRoles) {
				t.Fatalf("new roles = %v, want %v", got, testCase.wantNewRoles)
			}
			if result.StopReason != testCase.wantStop || result.Steps != testCase.wantSteps {
				t.Fatalf("result = %+v", result)
			}
			assertConformanceEvents(t, log.snapshot(), testCase.wantErr != "")
		})
	}
}

func assertConformanceEvents(t *testing.T, events []agent.Event, wantError bool) {
	t.Helper()
	if len(events) == 0 {
		t.Fatal("no events")
	}
	if events[0].Type != agent.EventAgentStart || events[len(events)-1].Type != agent.EventAgentEnd {
		t.Fatalf("events must start with agent_start and end with agent_end: %v", eventTypes(events))
	}
	openMessages := 0
	turnDepth := 0
	openTools := make(map[string]bool)
	sawError := false
	for _, event := range events {
		switch event.Type {
		case agent.EventAgentStart:
			if turnDepth != 0 || openMessages != 0 || len(openTools) != 0 {
				t.Fatalf("agent_start with open state: %v", eventTypes(events))
			}
		case agent.EventTurnStart:
			turnDepth++
		case agent.EventTurnEnd:
			turnDepth--
			if turnDepth < 0 {
				t.Fatalf("turn_end without turn_start: %v", eventTypes(events))
			}
		case agent.EventMessageStart:
			openMessages++
		case agent.EventMessageEnd:
			openMessages--
			if openMessages < 0 {
				t.Fatalf("message_end without message_start: %v", eventTypes(events))
			}
		case agent.EventToolStart:
			if event.ToolCall == nil {
				t.Fatal("tool_start without tool call")
			}
			openTools[event.ToolCall.ID] = true
		case agent.EventToolUpdate:
			if event.ToolCall == nil || !openTools[event.ToolCall.ID] {
				t.Fatalf("tool_update without open tool: %+v", event)
			}
		case agent.EventToolEnd:
			if event.ToolCall == nil || !openTools[event.ToolCall.ID] {
				t.Fatalf("tool_end without open tool: %+v", event)
			}
			delete(openTools, event.ToolCall.ID)
		case agent.EventAgentError:
			sawError = true
		case agent.EventAgentEnd:
			if turnDepth != 0 || openMessages != 0 || len(openTools) != 0 {
				t.Fatalf("agent_end with open state: turns=%d messages=%d tools=%v", turnDepth, openMessages, openTools)
			}
		}
	}
	if sawError != wantError {
		t.Fatalf("agent_error present = %v, want %v: %v", sawError, wantError, eventTypes(events))
	}
}
