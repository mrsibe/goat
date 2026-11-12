package ai_test

import (
	"encoding/json"
	"testing"

	"github.com/mrsibe/goat/ai"
)

func TestCloneMessageCopiesRuntimeMetadata(t *testing.T) {
	usage := &ai.Usage{TotalTokens: 7}
	message := ai.Message{
		Role:              ai.RoleTool,
		Content:           []ai.ContentBlock{{Type: ai.BlockText, Text: "ok"}},
		Details:           json.RawMessage(`{"a":1}`),
		StructuredContent: json.RawMessage(`{"b":2}`),
		Usage:             usage,
		StopReason:        ai.StopReasonStop,
		Provider:          "test",
		Model:             "model",
		API:               ai.APIOpenAIResponses,
		ErrorMessage:      "none",
		DurationMs:        12,
	}
	cloned := ai.CloneMessage(message)
	cloned.Content[0].Text = "changed"
	cloned.Details[0] = '['
	cloned.StructuredContent[0] = '['
	cloned.Usage.TotalTokens = 9

	if message.Content[0].Text != "ok" || string(message.Details) != `{"a":1}` ||
		string(message.StructuredContent) != `{"b":2}` || usage.TotalTokens != 7 {
		t.Fatalf("clone mutated original: %+v usage=%+v", message, usage)
	}
	if cloned.StopReason != ai.StopReasonStop || cloned.Provider != "test" ||
		cloned.Model != "model" || cloned.API != ai.APIOpenAIResponses ||
		cloned.ErrorMessage != "none" || cloned.DurationMs != 12 {
		t.Fatalf("metadata missing from clone: %+v", cloned)
	}
}

func TestCloneToolCallCopiesReplayMetadata(t *testing.T) {
	call := ai.ToolCall{
		ID:        "call_1",
		ItemID:    "fc_1",
		Name:      "echo",
		Arguments: json.RawMessage(`{"x":1}`),
	}
	cloned := ai.CloneToolCall(call)
	cloned.Arguments[0] = '['
	if call.ItemID != "fc_1" || string(call.Arguments) != `{"x":1}` {
		t.Fatalf("clone mutated original: %+v", call)
	}
}
