package streaming

import (
	"encoding/json"
	"testing"

	"github.com/mrsibe/goat/ai"
)

func TestBuilderKeepsPartialToolArgumentsSerializable(t *testing.T) {
	builder := New(ai.ModelInfo{API: ai.APIOpenAICompletions})
	builder.Start()
	builder.ToolStart("tool:0", "call_1", "get_weather")

	incomplete := builder.ToolDelta("tool:0", `{"city":`)
	if _, err := json.Marshal(incomplete); err != nil {
		t.Fatalf("marshal incomplete tool delta: %v", err)
	}
	calls := incomplete.Partial.ToolCalls()
	if len(calls) != 1 || string(calls[0].Arguments) != `{}` {
		t.Fatalf("incomplete partial tool calls = %+v", calls)
	}

	builder.ToolDelta("tool:0", `"北京"}`)
	done, err := builder.ToolEnd("tool:0", nil)
	if err != nil {
		t.Fatalf("finish tool call: %v", err)
	}
	if done.ToolCall == nil || string(done.ToolCall.Arguments) != `{"city":"北京"}` {
		t.Fatalf("finished tool call = %+v", done.ToolCall)
	}
	if _, err := json.Marshal(done); err != nil {
		t.Fatalf("marshal finished tool call: %v", err)
	}
}

func TestBuilderPreservesInvalidFinalToolArgumentsAsObservation(t *testing.T) {
	builder := New(ai.ModelInfo{API: ai.APIOpenAICompletions})
	builder.Start()
	builder.ToolStart("tool:0", "call_1", "get_weather")
	builder.ToolDelta("tool:0", `{"city":`)

	done, err := builder.ToolEnd("tool:0", nil)
	if err != nil {
		t.Fatalf("finish tool call: %v", err)
	}
	if done.ToolCall == nil {
		t.Fatal("finished tool call is nil")
	}
	if string(done.ToolCall.Arguments) != `{}` {
		t.Fatalf("arguments = %s, want {}", done.ToolCall.Arguments)
	}
	if done.ToolCall.ArgumentsError == "" {
		t.Fatal("invalid JSON was not preserved as ArgumentsError")
	}
}
