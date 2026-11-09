// Package ai defines the provider-neutral message, tool, and streaming types used
// by the agent runtime.
package ai

import (
	"encoding/json"
	"strings"
)

// Role identifies the author of a message.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// BlockType identifies a content block within a message.
type BlockType string

const (
	BlockText     BlockType = "text"
	BlockToolCall BlockType = "tool_call"
)

// ToolCall is a completed tool invocation requested by the model.
type ToolCall struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// ContentBlock is one piece of provider-neutral message content.
type ContentBlock struct {
	Type     BlockType `json:"type"`
	Text     string    `json:"text,omitempty"`
	ToolCall *ToolCall `json:"tool_call,omitempty"`
}

// Message is a provider-neutral chat message. Assistant messages use tool-call
// blocks for model-requested tools; tool messages use ToolCallID to correlate
// the result with its request.
type Message struct {
	Role       Role           `json:"role"`
	Content    []ContentBlock `json:"content"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
}

// TextMessage creates a message containing one text block.
func TextMessage(role Role, text string) Message {
	return Message{
		Role:    role,
		Content: []ContentBlock{{Type: BlockText, Text: text}},
	}
}

// ToolMessage creates a tool-result message correlated with a tool call.
func ToolMessage(callID, text string) Message {
	return Message{
		Role:       RoleTool,
		Content:    []ContentBlock{{Type: BlockText, Text: text}},
		ToolCallID: callID,
	}
}

// Text returns all text blocks in the message joined without separators.
func (m Message) Text() string {
	var text strings.Builder
	for _, block := range m.Content {
		if block.Type == BlockText {
			text.WriteString(block.Text)
		}
	}
	return text.String()
}

// ToolCalls returns all tool-call blocks in source order.
func (m Message) ToolCalls() []ToolCall {
	var calls []ToolCall
	for _, block := range m.Content {
		if block.Type == BlockToolCall && block.ToolCall != nil {
			calls = append(calls, CloneToolCall(*block.ToolCall))
		}
	}
	return calls
}

// ToolSpec describes a tool to the model.
type ToolSpec struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

// Usage contains token counts reported by a model provider.
type Usage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

// Add accumulates another usage report.
func (u *Usage) Add(other Usage) {
	u.PromptTokens += other.PromptTokens
	u.CompletionTokens += other.CompletionTokens
	u.TotalTokens += other.TotalTokens
}

// StopReason describes why a model turn ended.
type StopReason string

const (
	StopReasonStop      StopReason = "stop"
	StopReasonLength    StopReason = "length"
	StopReasonToolCalls StopReason = "tool_calls"
	StopReasonError     StopReason = "error"
	StopReasonAborted   StopReason = "aborted"
)

// CloneToolCall deep-copies a tool call.
func CloneToolCall(call ToolCall) ToolCall {
	call.Arguments = append(json.RawMessage(nil), call.Arguments...)
	return call
}

// CloneMessage deep-copies a message.
func CloneMessage(message Message) Message {
	cloned := message
	if message.Content == nil {
		return cloned
	}
	cloned.Content = make([]ContentBlock, len(message.Content))
	for i, block := range message.Content {
		cloned.Content[i] = block
		if block.ToolCall != nil {
			call := CloneToolCall(*block.ToolCall)
			cloned.Content[i].ToolCall = &call
		}
	}
	return cloned
}

// CloneMessages deep-copies a message slice.
func CloneMessages(messages []Message) []Message {
	if messages == nil {
		return nil
	}
	cloned := make([]Message, len(messages))
	for i, message := range messages {
		cloned[i] = CloneMessage(message)
	}
	return cloned
}
