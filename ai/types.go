package ai

import (
	"encoding/json"
	"errors"
	"strings"
	"time"
)

// Role identifies the author of a message.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	// RoleToolResult is pi's tool-result role name.
	RoleToolResult Role = "toolResult"
	// Deprecated: use RoleToolResult.
	RoleTool = RoleToolResult
)

// ThinkingLevel is the requested reasoning level for a model turn.
type ThinkingLevel string

const (
	ThinkingOff     ThinkingLevel = "off"
	ThinkingMinimal ThinkingLevel = "minimal"
	ThinkingLow     ThinkingLevel = "low"
	ThinkingMedium  ThinkingLevel = "medium"
	ThinkingHigh    ThinkingLevel = "high"
	ThinkingXHigh   ThinkingLevel = "xhigh"
	ThinkingMax     ThinkingLevel = "max"
)

// BlockType identifies a content block within a message.
type BlockType string

const (
	BlockText     BlockType = "text"
	BlockThinking BlockType = "thinking"
	BlockImage    BlockType = "image"
	BlockToolCall BlockType = "toolCall"
)

// ImageContent is an inline image or image URL.
type ImageContent struct {
	Data     string `json:"data,omitempty"`
	URL      string `json:"url,omitempty"`
	MimeType string `json:"mimeType,omitempty"`
}

// ToolCall is a completed tool invocation requested by the model.
type ToolCall struct {
	ID               string          `json:"id"`
	ItemID           string          `json:"itemId,omitempty"`
	Name             string          `json:"name"`
	Arguments        json.RawMessage `json:"arguments"`
	ArgumentsError   string          `json:"argumentsError,omitempty"`
	ThoughtSignature string          `json:"thoughtSignature,omitempty"`
	Namespace        string          `json:"namespace,omitempty"`
}

// ContentBlock is one piece of provider-neutral message content.
type ContentBlock struct {
	Type              BlockType     `json:"type"`
	Text              string        `json:"text,omitempty"`
	TextSignature     string        `json:"textSignature,omitempty"`
	Thinking          string        `json:"thinking,omitempty"`
	ThinkingSignature string        `json:"thinkingSignature,omitempty"`
	ID                string        `json:"id,omitempty"`
	Redacted          bool          `json:"redacted,omitempty"`
	Image             *ImageContent `json:"image,omitempty"`
	ToolCall          *ToolCall     `json:"toolCall,omitempty"`
}

// Message is one provider-neutral transcript message. It is a closed union:
// only the four message types in this package implement it.
type Message interface {
	Role() Role
	Text() string
	Thinking() string
	ToolCalls() []ToolCall
	Timestamp() int64
	message()
}

// SystemMessage carries instructions and, from the first message on, the tool
// declarations. Later system messages add instructions, replace or remove named
// sections, and add or remove tools.
type SystemMessage struct {
	Content  string             `json:"content,omitempty"`
	Sections map[string]*string `json:"sections,omitempty"`
	// SectionOrder preserves insertion order, which Go maps cannot express.
	SectionOrder []string        `json:"sectionOrder,omitempty"`
	ToolsAdded   []ToolSpec      `json:"toolsAdded,omitempty"`
	ToolsRemoved []ToolReference `json:"toolsRemoved,omitempty"`
	At           int64           `json:"timestamp,omitempty"`
}

// SetSection sets or removes a named section, preserving insertion order.
func (m *SystemMessage) SetSection(name string, value *string) {
	if m.Sections == nil {
		m.Sections = make(map[string]*string)
	}
	if _, exists := m.Sections[name]; !exists {
		m.SectionOrder = append(m.SectionOrder, name)
	}
	if value == nil {
		delete(m.Sections, name)
		m.SectionOrder = removeSectionName(m.SectionOrder, name)
		return
	}
	m.Sections[name] = value
}

// UserMessage is a user turn.
type UserMessage struct {
	Content []ContentBlock `json:"content"`
	At      int64          `json:"timestamp,omitempty"`
}

// AssistantMessage is a model turn.
type AssistantMessage struct {
	Content  []ContentBlock `json:"content"`
	API      API            `json:"api,omitempty"`
	Provider string         `json:"provider,omitempty"`
	Model    string         `json:"model,omitempty"`
	// ResponseModel is the concrete model reported by the provider, when reported.
	ResponseModel string `json:"responseModel,omitempty"`
	// ResponseID is a provider-specific response or message identifier.
	ResponseID string `json:"responseId,omitempty"`
	// ProviderThinkingLevel is the exact provider-native effort level used.
	ProviderThinkingLevel string `json:"providerThinkingLevel,omitempty"`
	// ThinkingLevel is the pi thinking level requested for this response.
	ThinkingLevel ThinkingLevel   `json:"thinkingLevel,omitempty"`
	Diagnostics   []Diagnostic    `json:"diagnostics,omitempty"`
	Usage         Usage           `json:"usage"`
	StopReason    StopReason      `json:"stopReason,omitempty"`
	Deferred      *DeferredHandle `json:"deferred,omitempty"`
	ErrorMessage  string          `json:"errorMessage,omitempty"`
	RawStopReason string          `json:"rawStopReason,omitempty"`
	// EndTurn reports whether the model explicitly ended its turn.
	EndTurn    bool  `json:"endTurn,omitempty"`
	At         int64 `json:"timestamp,omitempty"`
	DurationMs int64 `json:"durationMs,omitempty"`
}

// ToolReference names a tool that stops being available.
type ToolReference struct {
	Name string `json:"name"`
}

// NestedToolCallRecord is a tool call another tool made while it ran.
type NestedToolCallRecord struct {
	ID             string          `json:"id"`
	Name           string          `json:"name"`
	Arguments      json.RawMessage `json:"arguments,omitempty"`
	ArgumentsBytes int             `json:"argumentsBytes,omitempty"`
	Status         string          `json:"status"`
	DurationMs     int64           `json:"durationMs,omitempty"`
	Error          string          `json:"error,omitempty"`
}

// NestedToolCalls is a bounded record of the calls a tool made.
type NestedToolCalls struct {
	Calls    []NestedToolCallRecord `json:"calls"`
	Complete bool                   `json:"complete"`
}

// ToolResultMessage is the observation returned to the model after a tool call.
type ToolResultMessage struct {
	ToolCallID        string           `json:"toolCallId"`
	ToolName          string           `json:"toolName"`
	Content           []ContentBlock   `json:"content"`
	Details           json.RawMessage  `json:"details,omitempty"`
	StructuredContent json.RawMessage  `json:"structuredContent,omitempty"`
	Usage             *Usage           `json:"usage,omitempty"`
	NestedCalls       *NestedToolCalls `json:"nestedCalls,omitempty"`
	IsError           bool             `json:"isError,omitempty"`
	At                int64            `json:"timestamp,omitempty"`
	DurationMs        int64            `json:"durationMs,omitempty"`
}

// Diagnostic is a redacted provider or runtime diagnostic attached to an
// assistant message.
type Diagnostic struct {
	Type      string           `json:"type"`
	Timestamp int64            `json:"timestamp"`
	Error     *DiagnosticError `json:"error,omitempty"`
	Details   json.RawMessage  `json:"details,omitempty"`
}

// DiagnosticError describes the captured failure of a diagnostic.
type DiagnosticError struct {
	Name    string `json:"name,omitempty"`
	Message string `json:"message"`
	Stack   string `json:"stack,omitempty"`
	Code    any    `json:"code,omitempty"`
}

// DeferredHandle identifies a provider-side deferred response.
type DeferredHandle struct {
	Provider    string          `json:"provider"`
	ModelID     string          `json:"modelId"`
	API         string          `json:"api"`
	ID          string          `json:"id"`
	ExpiresAt   int64           `json:"expiresAt,omitempty"`
	PollAfterMs int64           `json:"pollAfterMs,omitempty"`
	Data        json.RawMessage `json:"data,omitempty"`
}

// Role implements Message.
func (m *SystemMessage) Role() Role { return RoleSystem }

// Role implements Message.
func (m *UserMessage) Role() Role { return RoleUser }

// Role implements Message.
func (m *AssistantMessage) Role() Role { return RoleAssistant }

// Role implements Message.
func (m *ToolResultMessage) Role() Role { return RoleToolResult }

func (m *SystemMessage) message()     {}
func (m *UserMessage) message()       {}
func (m *AssistantMessage) message()  {}
func (m *ToolResultMessage) message() {}

// Timestamp returns the message creation time in Unix milliseconds.
func (m *SystemMessage) Timestamp() int64     { return m.At }
func (m *UserMessage) Timestamp() int64       { return m.At }
func (m *AssistantMessage) Timestamp() int64  { return m.At }
func (m *ToolResultMessage) Timestamp() int64 { return m.At }

// Text returns the message's text blocks joined without separators.
func (m *SystemMessage) Text() string { return m.Content }
func (m *UserMessage) Text() string {
	return joinBlocks(m.Content, BlockText, func(block ContentBlock) string { return block.Text })
}
func (m *AssistantMessage) Text() string {
	return joinBlocks(m.Content, BlockText, func(block ContentBlock) string { return block.Text })
}
func (m *ToolResultMessage) Text() string {
	return joinBlocks(m.Content, BlockText, func(block ContentBlock) string { return block.Text })
}

// Thinking returns the message's thinking blocks joined without separators.
func (m *SystemMessage) Thinking() string { return "" }
func (m *UserMessage) Thinking() string   { return "" }
func (m *AssistantMessage) Thinking() string {
	return joinBlocks(m.Content, BlockThinking, func(block ContentBlock) string { return block.Thinking })
}
func (m *ToolResultMessage) Thinking() string { return "" }

// ToolCalls returns the message's tool calls in source order.
func (m *SystemMessage) ToolCalls() []ToolCall     { return nil }
func (m *UserMessage) ToolCalls() []ToolCall       { return nil }
func (m *ToolResultMessage) ToolCalls() []ToolCall { return nil }
func (m *AssistantMessage) ToolCalls() []ToolCall {
	return collectToolCalls(m.Content)
}

func collectToolCalls(blocks []ContentBlock) []ToolCall {
	var calls []ToolCall
	for _, block := range blocks {
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

// Usage contains token counts and derived cost reported by a model provider.
type Usage struct {
	Input        int64 `json:"input"`
	Output       int64 `json:"output"`
	CacheRead    int64 `json:"cacheRead"`
	CacheWrite   int64 `json:"cacheWrite"`
	CacheWrite1H int64 `json:"cacheWrite1h,omitempty"`
	Reasoning    int64 `json:"reasoning,omitempty"`
	TotalTokens  int64 `json:"totalTokens"`
	Cost         Cost  `json:"cost"`
}

// Cost is the monetary cost of one request.
type Cost struct {
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cacheRead"`
	CacheWrite float64 `json:"cacheWrite"`
	Total      float64 `json:"total"`
	// TotalFromProvider is true when Total is billed cost reported by the
	// provider rather than a catalog estimate.
	TotalFromProvider bool `json:"totalFromProvider,omitempty"`
}

// Add accumulates another usage report.
func (u *Usage) Add(other Usage) {
	u.Input += other.Input
	u.Output += other.Output
	u.CacheRead += other.CacheRead
	u.CacheWrite += other.CacheWrite
	u.CacheWrite1H += other.CacheWrite1H
	u.Reasoning += other.Reasoning
	u.TotalTokens += other.TotalTokens
	u.Cost.Input += other.Cost.Input
	u.Cost.Output += other.Cost.Output
	u.Cost.CacheRead += other.Cost.CacheRead
	u.Cost.CacheWrite += other.Cost.CacheWrite
	u.Cost.Total += other.Cost.Total
	if other.Cost.TotalFromProvider {
		u.Cost.TotalFromProvider = true
	}
}

// StopReason describes why a model turn ended.
type StopReason string

const (
	StopReasonPending  StopReason = "pending"
	StopReasonStop     StopReason = "stop"
	StopReasonLength   StopReason = "length"
	StopReasonToolUse  StopReason = "toolUse"
	StopReasonError    StopReason = "error"
	StopReasonAborted  StopReason = "aborted"
	StopReasonDeferred StopReason = "deferred"
	// Deprecated: use StopReasonToolUse.
	StopReasonToolCalls = StopReasonToolUse
)

// NewSystemMessage creates a system message.
func NewSystemMessage(text string) *SystemMessage {
	return &SystemMessage{Content: text, At: time.Now().UnixMilli()}
}

// NewUserMessage creates a user message.
func NewUserMessage(content ...ContentBlock) *UserMessage {
	return &UserMessage{Content: CloneContentBlocks(content), At: time.Now().UnixMilli()}
}

// NewAssistantMessage creates an assistant message.
func NewAssistantMessage(content ...ContentBlock) *AssistantMessage {
	return &AssistantMessage{Content: CloneContentBlocks(content), At: time.Now().UnixMilli()}
}

// NewToolResultMessage creates a tool-result message with arbitrary blocks.
func NewToolResultMessage(callID, toolName string, content []ContentBlock, isError bool) *ToolResultMessage {
	return &ToolResultMessage{
		ToolCallID: callID,
		ToolName:   toolName,
		Content:    CloneContentBlocks(content),
		IsError:    isError,
		At:         time.Now().UnixMilli(),
	}
}

// TextMessage creates a message of the given role containing one text block.
func TextMessage(role Role, text string) Message {
	switch role {
	case RoleSystem:
		return NewSystemMessage(text)
	case RoleUser:
		return NewUserMessage(ContentBlock{Type: BlockText, Text: text})
	case RoleAssistant:
		return NewAssistantMessage(ContentBlock{Type: BlockText, Text: text})
	case RoleToolResult:
		return NewToolResultMessage("", "", []ContentBlock{{Type: BlockText, Text: text}}, false)
	default:
		return NewUserMessage(ContentBlock{Type: BlockText, Text: text})
	}
}

// ToolMessage creates a successful text tool-result message.
func ToolMessage(callID, toolName, text string) Message {
	return ToolTextMessage(callID, toolName, text, false)
}

// ToolTextMessage creates a tool-result message containing one text block.
func ToolTextMessage(callID, toolName, text string, isError bool) Message {
	return NewToolResultMessage(callID, toolName, []ContentBlock{{Type: BlockText, Text: text}}, isError)
}

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
	switch typed := message.(type) {
	case *SystemMessage:
		cloned := *typed
		cloned.ToolsAdded = cloneToolSpecs(typed.ToolsAdded)
		cloned.ToolsRemoved = append([]ToolReference(nil), typed.ToolsRemoved...)
		cloned.SectionOrder = append([]string(nil), typed.SectionOrder...)
		if typed.Sections != nil {
			cloned.Sections = make(map[string]*string, len(typed.Sections))
			for name, value := range typed.Sections {
				if value == nil {
					cloned.Sections[name] = nil
					continue
				}
				copied := *value
				cloned.Sections[name] = &copied
			}
		}
		return &cloned
	case *UserMessage:
		cloned := *typed
		cloned.Content = CloneContentBlocks(typed.Content)
		return &cloned
	case *AssistantMessage:
		cloned := *typed
		cloned.Content = CloneContentBlocks(typed.Content)
		cloned.Diagnostics = append([]Diagnostic(nil), typed.Diagnostics...)
		cloned.Usage = typed.Usage
		if typed.Deferred != nil {
			deferred := *typed.Deferred
			deferred.Data = CloneRawMessage(typed.Deferred.Data)
			cloned.Deferred = &deferred
		}
		return &cloned
	case *ToolResultMessage:
		cloned := *typed
		cloned.Content = CloneContentBlocks(typed.Content)
		cloned.Details = CloneRawMessage(typed.Details)
		cloned.StructuredContent = CloneRawMessage(typed.StructuredContent)
		if typed.Usage != nil {
			usage := *typed.Usage
			cloned.Usage = &usage
		}
		if typed.NestedCalls != nil {
			nested := *typed.NestedCalls
			nested.Calls = append([]NestedToolCallRecord(nil), typed.NestedCalls.Calls...)
			cloned.NestedCalls = &nested
		}
		return &cloned
	default:
		return message
	}
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

func cloneToolSpecs(specs []ToolSpec) []ToolSpec {
	if specs == nil {
		return nil
	}
	cloned := make([]ToolSpec, len(specs))
	for i, spec := range specs {
		cloned[i] = ToolSpec{
			Name:        spec.Name,
			Description: spec.Description,
			Parameters:  cloneMap(spec.Parameters),
		}
	}
	return cloned
}

func cloneMap(values map[string]any) map[string]any {
	if values == nil {
		return nil
	}
	cloned := make(map[string]any, len(values))
	for key, value := range values {
		cloned[key] = value
	}
	return cloned
}

// MarshalJSON adds the role discriminator to a system message.
func (m SystemMessage) MarshalJSON() ([]byte, error) {
	type plain SystemMessage
	return json.Marshal(struct {
		Role Role `json:"role"`
		plain
	}{RoleSystem, plain(m)})
}

// MarshalJSON adds the role discriminator to a user message.
func (m UserMessage) MarshalJSON() ([]byte, error) {
	type plain UserMessage
	return json.Marshal(struct {
		Role Role `json:"role"`
		plain
	}{RoleUser, plain(m)})
}

// MarshalJSON adds the role discriminator to an assistant message.
func (m AssistantMessage) MarshalJSON() ([]byte, error) {
	type plain AssistantMessage
	return json.Marshal(struct {
		Role Role `json:"role"`
		plain
	}{RoleAssistant, plain(m)})
}

// MarshalJSON adds the role discriminator to a tool-result message.
func (m ToolResultMessage) MarshalJSON() ([]byte, error) {
	type plain ToolResultMessage
	return json.Marshal(struct {
		Role Role `json:"role"`
		plain
	}{RoleToolResult, plain(m)})
}

// UnmarshalMessage decodes one message using its role discriminator.
func UnmarshalMessage(data []byte) (Message, error) {
	var envelope struct {
		Role Role `json:"role"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		return nil, err
	}
	switch envelope.Role {
	case RoleSystem:
		message := &SystemMessage{}
		return message, json.Unmarshal(data, message)
	case RoleUser:
		message := &UserMessage{}
		return message, json.Unmarshal(data, message)
	case RoleAssistant:
		message := &AssistantMessage{}
		return message, json.Unmarshal(data, message)
	case RoleToolResult:
		message := &ToolResultMessage{}
		return message, json.Unmarshal(data, message)
	case "":
		return nil, errors.New("ai: message role must not be empty")
	default:
		return nil, errors.New("ai: unknown message role " + string(envelope.Role))
	}
}

// UnmarshalMessages decodes a JSON array of messages.
func UnmarshalMessages(data []byte) ([]Message, error) {
	var raws []json.RawMessage
	if err := json.Unmarshal(data, &raws); err != nil {
		return nil, err
	}
	messages := make([]Message, len(raws))
	for i, raw := range raws {
		message, err := UnmarshalMessage(raw)
		if err != nil {
			return nil, err
		}
		messages[i] = message
	}
	return messages, nil
}

// AsSystem returns the message as a system message, or nil.
func AsSystem(message Message) *SystemMessage {
	typed, _ := message.(*SystemMessage)
	return typed
}

// AsUser returns the message as a user message, or nil.
func AsUser(message Message) *UserMessage {
	typed, _ := message.(*UserMessage)
	return typed
}

// AsAssistant returns the message as an assistant message, or nil.
func AsAssistant(message Message) *AssistantMessage {
	typed, _ := message.(*AssistantMessage)
	return typed
}

// AsToolResult returns the message as a tool-result message, or nil.
func AsToolResult(message Message) *ToolResultMessage {
	typed, _ := message.(*ToolResultMessage)
	return typed
}
