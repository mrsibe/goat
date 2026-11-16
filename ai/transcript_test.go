package ai_test

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/mrsibe/goat/ai"
)

func TestNormalizeContextFoldsPromptAndTools(t *testing.T) {
	context := ai.NormalizeContext(ai.Context{
		SystemPrompt: "base",
		Messages:     []ai.Message{ai.NewUserMessage(ai.ContentBlock{Type: ai.BlockText, Text: "hi"})},
		Tools:        []ai.ToolSpec{{Name: "echo"}},
	})
	if len(context.Messages) != 2 {
		t.Fatalf("messages = %+v", context.Messages)
	}
	system := ai.AsSystem(context.Messages[0])
	if system == nil || system.Content != "base" || len(system.ToolsAdded) != 1 {
		t.Fatalf("leading system = %+v", context.Messages[0])
	}
	if got := ai.GetCurrentTools(context.Messages); len(got) != 1 || got[0].Name != "echo" {
		t.Fatalf("tools = %+v", got)
	}
}

func TestSystemMessageReplaySectionsAndTools(t *testing.T) {
	remove := (*string)(nil)
	replacement := "new section"
	messages := []ai.Message{
		&ai.SystemMessage{
			Content:    "base",
			Sections:   map[string]*string{"a": strPtr("one"), "b": strPtr("two")},
			ToolsAdded: []ai.ToolSpec{{Name: "echo"}, {Name: "read"}},
		},
		ai.NewUserMessage(ai.ContentBlock{Type: ai.BlockText, Text: "hi"}),
		&ai.SystemMessage{
			Content:      "extra",
			Sections:     map[string]*string{"a": &replacement, "b": remove},
			ToolsAdded:   []ai.ToolSpec{{Name: "write"}},
			ToolsRemoved: []ai.ToolReference{{Name: "read"}},
		},
	}

	prompt := ai.GetCurrentSystemPrompt(messages)
	if !contains(prompt, "base") || !contains(prompt, "extra") || !contains(prompt, "new section") || contains(prompt, "two") {
		t.Fatalf("prompt = %q", prompt)
	}
	tools := ai.GetCurrentTools(messages)
	var names []string
	for _, spec := range tools {
		names = append(names, spec.Name)
	}
	if !reflect.DeepEqual(names, []string{"echo", "write"}) {
		t.Fatalf("tools = %v", names)
	}
}

func TestGetToolStateChanges(t *testing.T) {
	changes := ai.GetToolStateChanges(
		[]ai.ToolSpec{{Name: "echo", Description: "old"}, {Name: "read"}},
		[]ai.ToolSpec{{Name: "echo", Description: "new"}, {Name: "write"}},
	)
	if len(changes.ToolsAdded) != 2 || changes.ToolsAdded[0].Name != "echo" {
		t.Fatalf("added = %+v", changes.ToolsAdded)
	}
	if len(changes.ToolsRemoved) != 1 || changes.ToolsRemoved[0].Name != "read" {
		t.Fatalf("removed = %+v", changes.ToolsRemoved)
	}
	if !changes.HasToolChanges() {
		t.Fatal("expected changes")
	}
	if ai.GetToolStateChanges([]ai.ToolSpec{{Name: "echo"}}, []ai.ToolSpec{{Name: "echo"}}).HasToolChanges() {
		t.Fatal("identical tools must not produce changes")
	}
}

func TestMessageJSONRoundTrip(t *testing.T) {
	call := ai.ToolCall{ID: "call_1", Name: "echo", Arguments: json.RawMessage(`{"x":1}`)}
	messages := []ai.Message{
		&ai.SystemMessage{Content: "sys", ToolsAdded: []ai.ToolSpec{{Name: "echo"}}},
		ai.NewUserMessage(ai.ContentBlock{Type: ai.BlockText, Text: "hi"}),
		&ai.AssistantMessage{
			Content:       []ai.ContentBlock{{Type: ai.BlockToolCall, ToolCall: &call}},
			Provider:      "test",
			Model:         "model",
			API:           ai.APIOpenAICompletions,
			Usage:         ai.Usage{Input: 1, Output: 2, TotalTokens: 3},
			StopReason:    ai.StopReasonToolUse,
			ThinkingLevel: ai.ThinkingLow,
		},
		&ai.ToolResultMessage{ToolCallID: "call_1", ToolName: "echo", IsError: true, Content: []ai.ContentBlock{{Type: ai.BlockText, Text: "ok"}}},
	}

	data, err := json.Marshal(messages)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded) != len(messages) {
		t.Fatalf("decoded = %+v", decoded)
	}
	if got := decoded[2].(*ai.AssistantMessage); got.StopReason != ai.StopReasonToolUse || got.ThinkingLevel != ai.ThinkingLow || got.Usage.TotalTokens != 3 {
		t.Fatalf("assistant = %+v", got)
	}
	if got := decoded[3].(*ai.ToolResultMessage); !got.IsError || got.ToolCallID != "call_1" {
		t.Fatalf("tool result = %+v", got)
	}
	if got := decoded[0].(*ai.SystemMessage); got.Content != "sys" || len(got.ToolsAdded) != 1 {
		t.Fatalf("system = %+v", got)
	}
}

func strPtr(value string) *string { return &value }

func contains(haystack, needle string) bool {
	return len(haystack) >= len(needle) && (haystack == needle || len(needle) == 0 || indexOf(haystack, needle) >= 0)
}

func indexOf(haystack, needle string) int {
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return i
		}
	}
	return -1
}

func TestRenderSystemMessagePreservesInsertionOrder(t *testing.T) {
	system := &ai.SystemMessage{Content: "base"}
	system.SetSection("b", strPtr("B"))
	system.SetSection("a", strPtr("A"))
	system.SetSection("b", nil)
	system.SetSection("c", strPtr("C"))
	if got, want := ai.RenderSystemMessage(*system), "base\n\nA\n\nC"; got != want {
		t.Fatalf("render = %q, want %q", got, want)
	}
}
