package ai

import "context"

// API identifies a provider protocol.
type API string

const (
	APIOpenAICompletions    API = "openai-completions"
	APIOpenAIResponses      API = "openai-responses"
	APIAzureOpenAIResponses API = "azure-openai-responses"
	APIGoogleGenerativeAI   API = "google-generative-ai"
	APIAnthropicMessages    API = "anthropic-messages"
)

// ModelInfo describes a concrete model and the protocol capabilities exposed
// by its adapter. Model-specific limitations still apply.
type ModelInfo struct {
	Provider                 string `json:"provider"`
	Model                    string `json:"model"`
	API                      API    `json:"api"`
	ContextWindow            int    `json:"context_window,omitempty"`
	MaxOutputTokens          int    `json:"max_output_tokens,omitempty"`
	SupportsTools            bool   `json:"supports_tools"`
	SupportsImages           bool   `json:"supports_images,omitempty"`
	SupportsToolResultImages bool   `json:"supports_tool_result_images,omitempty"`
	SupportsThinking         bool   `json:"supports_thinking,omitempty"`
}

// Model is the boundary between the agent loop and a model provider.
type Model interface {
	Info() ModelInfo
	Stream(ctx context.Context, request Request) (Stream, error)
}

// Stream yields normalized model events. Next returns io.EOF after the final
// event. The caller always closes the stream.
type Stream interface {
	Next() (Event, error)
	Close() error
}

// ToolChoiceMode controls how the model may use tools.
type ToolChoiceMode string

const (
	ToolChoiceAuto     ToolChoiceMode = "auto"
	ToolChoiceNone     ToolChoiceMode = "none"
	ToolChoiceRequired ToolChoiceMode = "required"
	ToolChoiceNamed    ToolChoiceMode = "named"
)

// ToolChoice selects a tool-use policy. Name is used only with ToolChoiceNamed.
type ToolChoice struct {
	Mode ToolChoiceMode `json:"mode,omitempty"`
	Name string         `json:"name,omitempty"`
}

// RequestOptions contains per-request sampling and routing controls. Empty
// values leave the provider default in place.
type RequestOptions struct {
	Temperature          *float64   `json:"temperature,omitempty"`
	TopP                 *float64   `json:"top_p,omitempty"`
	MaxTokens            int        `json:"max_tokens,omitempty"`
	ToolChoice           ToolChoice `json:"tool_choice,omitempty"`
	SessionID            string     `json:"session_id,omitempty"`
	ReasoningEffort      string     `json:"reasoning_effort,omitempty"`
	ThinkingBudgetTokens int        `json:"thinking_budget_tokens,omitempty"`
}

// Request is one provider-neutral model turn.
type Request struct {
	Messages []Message      `json:"messages"`
	Tools    []ToolSpec     `json:"tools,omitempty"`
	Options  RequestOptions `json:"options,omitempty"`
}

// EventType identifies one normalized model stream event.
type EventType string

const (
	EventStart         EventType = "start"
	EventTextStart     EventType = "text_start"
	EventTextDelta     EventType = "text_delta"
	EventTextEnd       EventType = "text_end"
	EventThinkingStart EventType = "thinking_start"
	EventThinkingDelta EventType = "thinking_delta"
	EventThinkingEnd   EventType = "thinking_end"
	EventToolCallStart EventType = "tool_call_start"
	EventToolCallDelta EventType = "tool_call_delta"
	EventToolCallEnd   EventType = "tool_call_end"
	EventUsage         EventType = "usage"
	EventDone          EventType = "done"
	EventError         EventType = "error"
)

// ToolCallDelta is one streamed fragment of a tool call. ContentIndex on the
// enclosing Event identifies the content block. ID and Name are complete values
// when present; ArgumentsDelta is incremental.
type ToolCallDelta struct {
	ID             string `json:"id,omitempty"`
	Name           string `json:"name,omitempty"`
	ArgumentsDelta string `json:"arguments_delta,omitempty"`
}

// Event is one normalized model stream event. Partial is a deep copy of the
// assistant message accumulated so far. Done carries Message and Usage; Error
// carries Err and the partial message.
type Event struct {
	Type          EventType      `json:"type"`
	ContentIndex  int            `json:"content_index,omitempty"`
	Text          string         `json:"text,omitempty"`
	TextDelta     string         `json:"text_delta,omitempty"`
	Thinking      string         `json:"thinking,omitempty"`
	ThinkingDelta string         `json:"thinking_delta,omitempty"`
	ToolCall      *ToolCall      `json:"tool_call,omitempty"`
	ToolCallDelta *ToolCallDelta `json:"tool_call_delta,omitempty"`
	Partial       *Message       `json:"partial,omitempty"`
	Message       *Message       `json:"message,omitempty"`
	Usage         *Usage         `json:"usage,omitempty"`
	StopReason    StopReason     `json:"stop_reason,omitempty"`
	Err           error          `json:"-"`
	Error         string         `json:"error,omitempty"`
}
