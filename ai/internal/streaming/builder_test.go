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

func TestBuilderPreservesProviderReplayMetadata(t *testing.T) {
	builder := New(ai.ModelInfo{API: ai.APIOpenAIResponses})
	builder.Start()
	builder.TextDelta("text:msg_1:0", "hello")
	builder.SetTextMeta("text:msg_1:0", `{"id":"msg_1","phase":"final_answer"}`)
	builder.SetToolItemID("tool:fc_1", "fc_1")
	builder.ToolStart("tool:fc_1", "call_1", "echo")

	message := builder.Message()
	if len(message.Content) != 2 {
		t.Fatalf("content = %+v", message.Content)
	}
	if got := message.Content[0].TextSignature; got != `{"id":"msg_1","phase":"final_answer"}` {
		t.Fatalf("text signature = %q", got)
	}
	if call := message.Content[1].ToolCall; call == nil || call.ID != "call_1" || call.ItemID != "fc_1" {
		t.Fatalf("tool call = %+v", call)
	}
}
