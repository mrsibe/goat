package ai_test

import (
	"encoding/json"
	"testing"

	"github.com/mrsibe/goat/ai"
)

func TestCloneMessageCopiesRuntimeMetadata(t *testing.T) {
	usage := &ai.Usage{TotalTokens: 7}
	message := &ai.AssistantMessage{
		Content:      []ai.ContentBlock{{Type: ai.BlockText, Text: "ok"}},
		Usage:        ai.Usage{TotalTokens: 7, Cost: ai.Cost{Total: 0.5}},
		StopReason:   ai.StopReasonStop,
		Provider:     "test",
		Model:        "model",
		API:          ai.APIOpenAIResponses,
		ErrorMessage: "none",
		DurationMs:   12,
	}
	cloned := ai.CloneMessage(message)
	cloned.(*ai.AssistantMessage).Content[0].Text = "changed"

	result := &ai.ToolResultMessage{
		ToolCallID:        "call_1",
		ToolName:          "echo",
		Content:           []ai.ContentBlock{{Type: ai.BlockText, Text: "ok"}},
		Details:           json.RawMessage(`{"a":1}`),
		StructuredContent: json.RawMessage(`{"b":2}`),
		Usage:             usage,
		IsError:           true,
		DurationMs:        12,
	}
	clonedResult := ai.CloneMessage(result).(*ai.ToolResultMessage)
	clonedResult.Content[0].Text = "changed"
	clonedResult.Details[0] = '['
	clonedResult.StructuredContent[0] = '['
	clonedResult.Usage.TotalTokens = 9

	if message.Content[0].Text != "ok" || message.Usage.TotalTokens != 7 {
		t.Fatalf("clone mutated assistant: %+v", message)
	}
	if cloned.(*ai.AssistantMessage).StopReason != ai.StopReasonStop ||
		cloned.(*ai.AssistantMessage).Provider != "test" ||
		cloned.(*ai.AssistantMessage).Model != "model" ||
		cloned.(*ai.AssistantMessage).API != ai.APIOpenAIResponses ||
		cloned.(*ai.AssistantMessage).ErrorMessage != "none" ||
		cloned.(*ai.AssistantMessage).DurationMs != 12 {
		t.Fatalf("metadata missing from clone: %+v", cloned)
	}
	if result.Content[0].Text != "ok" || string(result.Details) != `{"a":1}` ||
		string(result.StructuredContent) != `{"b":2}` || usage.TotalTokens != 7 {
		t.Fatalf("clone mutated original: %+v usage=%+v", result, usage)
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
