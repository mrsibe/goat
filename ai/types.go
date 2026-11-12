package ai

import (
	"encoding/json"
	"strings"
	"time"
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
	BlockThinking BlockType = "thinking"
	BlockImage    BlockType = "image"
	BlockToolCall BlockType = "tool_call"
)

// ImageContent is an inline image or image URL.
type ImageContent struct {
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
	MediaType string `json:"media_type,omitempty"`
}

// ToolCall is a completed tool invocation requested by the model.
type ToolCall struct {
	ID               string          `json:"id"`
	ItemID           string          `json:"item_id,omitempty"`
	Name             string          `json:"name"`
	Arguments        json.RawMessage `json:"arguments"`
	ArgumentsError   string          `json:"arguments_error,omitempty"`
	ThoughtSignature string          `json:"thought_signature,omitempty"`
	Namespace        string          `json:"namespace,omitempty"`
}

// ContentBlock is one piece of provider-neutral message content.
type ContentBlock struct {
	Type              BlockType     `json:"type"`
	Text              string        `json:"text,omitempty"`
	TextSignature     string        `json:"text_signature,omitempty"`
	Thinking          string        `json:"thinking,omitempty"`
	ThinkingSignature string        `json:"thinking_signature,omitempty"`
	ID                string        `json:"id,omitempty"`
	Redacted          bool          `json:"redacted,omitempty"`
	Image             *ImageContent `json:"image,omitempty"`
	ToolCall          *ToolCall     `json:"tool_call,omitempty"`
}

// Message is a provider-neutral chat message. Assistant messages use tool-call
// blocks for model-requested tools; tool messages use ToolCallID to correlate
// the result with its request.
type Message struct {
	Role              Role            `json:"role"`
	Content           []ContentBlock  `json:"content"`
	ToolCallID        string          `json:"tool_call_id,omitempty"`
	ToolName          string          `json:"tool_name,omitempty"`
	IsError           bool            `json:"is_error,omitempty"`
	Timestamp         int64           `json:"timestamp,omitempty"`
	Details           json.RawMessage `json:"details,omitempty"`
	StructuredContent json.RawMessage `json:"structured_content,omitempty"`
	Usage             *Usage          `json:"usage,omitempty"`
	StopReason        StopReason      `json:"stop_reason,omitempty"`
	Provider          string          `json:"provider,omitempty"`
	Model             string          `json:"model,omitempty"`
	API               API             `json:"api,omitempty"`
	ResponseModel     string          `json:"response_model,omitempty"`
	ResponseID        string          `json:"response_id,omitempty"`
	ErrorMessage      string          `json:"error_message,omitempty"`
	DurationMs        int64           `json:"duration_ms,omitempty"`
}

// TextMessage creates a message containing one text block.
func TextMessage(role Role, text string) Message {
	return Message{
		Role:      role,
		Content:   []ContentBlock{{Type: BlockText, Text: text}},
		Timestamp: time.Now().UnixMilli(),
	}
}

// ToolMessage creates a successful text tool-result message.
func ToolMessage(callID, toolName, text string) Message {
	return ToolTextMessage(callID, toolName, text, false)
}

// ToolTextMessage creates a tool-result message containing one text block.
func ToolTextMessage(callID, toolName, text string, isError bool) Message {
	return ToolResultMessage(
		callID,
		toolName,
		[]ContentBlock{{Type: BlockText, Text: text}},
		isError,
	)
}

// ToolResultMessage creates a tool-result message with arbitrary content blocks.
func ToolResultMessage(callID, toolName string, content []ContentBlock, isError bool) Message {
	return Message{
		Role:       RoleTool,
		Content:    CloneContentBlocks(content),
		ToolCallID: callID,
		ToolName:   toolName,
		IsError:    isError,
		Timestamp:  time.Now().UnixMilli(),
	}
}

// Text returns all text blocks in the message joined without separators.
func (m Message) Text() string {
	return joinBlocks(m.Content, BlockText, func(block ContentBlock) string {
		return block.Text
	})
}

// Thinking returns all thinking blocks joined without separators.
func (m Message) Thinking() string {
	return joinBlocks(m.Content, BlockThinking, func(block ContentBlock) string {
		return block.Thinking
	})
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

func joinBlocks(blocks []ContentBlock, blockType BlockType, value func(ContentBlock) string) string {
	var text strings.Builder
	for _, block := range blocks {
		if block.Type == blockType {
			text.WriteString(value(block))
		}
	}
	return text.String()
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
	CacheReadTokens  int64 `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens int64 `json:"cache_write_tokens,omitempty"`
	ReasoningTokens  int64 `json:"reasoning_tokens,omitempty"`
	TotalTokens      int64 `json:"total_tokens"`
}

// Add accumulates another usage report.
func (u *Usage) Add(other Usage) {
	u.PromptTokens += other.PromptTokens
	u.CompletionTokens += other.CompletionTokens
	u.CacheReadTokens += other.CacheReadTokens
	u.CacheWriteTokens += other.CacheWriteTokens
	u.ReasoningTokens += other.ReasoningTokens
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

// CloneRawMessage deep-copies a JSON payload.
func CloneRawMessage(raw json.RawMessage) json.RawMessage {
	if raw == nil {
		return nil
	}
	return append(json.RawMessage(nil), raw...)
}

// CloneContentBlock deep-copies a content block.
func CloneContentBlock(block ContentBlock) ContentBlock {
	if block.ToolCall != nil {
		call := CloneToolCall(*block.ToolCall)
		block.ToolCall = &call
	}
	if block.Image != nil {
		image := *block.Image
		block.Image = &image
	}
	return block
}

// CloneContentBlocks deep-copies content blocks.
func CloneContentBlocks(blocks []ContentBlock) []ContentBlock {
	if blocks == nil {
		return nil
	}
	cloned := make([]ContentBlock, len(blocks))
	for i, block := range blocks {
		cloned[i] = CloneContentBlock(block)
	}
	return cloned
}

// CloneMessage deep-copies a message.
func CloneMessage(message Message) Message {
	cloned := message
	cloned.Content = CloneContentBlocks(message.Content)
	cloned.Details = CloneRawMessage(message.Details)
	cloned.StructuredContent = CloneRawMessage(message.StructuredContent)
	if message.Usage != nil {
		usage := *message.Usage
		cloned.Usage = &usage
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
