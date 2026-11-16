package ai

import (
	"context"
	"net/http"
)

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
	// Reasoning reports whether the model exposes a reasoning/thinking mode.
	Reasoning bool `json:"reasoning,omitempty"`
	// ThinkingLevels lists explicitly supported levels. Empty means off..high
	// when Reasoning is true; xhigh and max always require an explicit entry.
	ThinkingLevels []ThinkingLevel `json:"thinking_levels,omitempty"`
	// Pricing is the per-million-token price used by Usage.ApplyCost.
	Pricing ModelPricing `json:"pricing,omitempty"`
}

// ModelPricing is a per-million-token price table.
type ModelPricing struct {
	Input      float64 `json:"input,omitempty"`
	Output     float64 `json:"output,omitempty"`
	CacheRead  float64 `json:"cache_read,omitempty"`
	CacheWrite float64 `json:"cache_write,omitempty"`
}

// Calculate returns the cost of one usage report at this price table. Anthropic
// 1h cache writes are billed at twice the base input rate.
func (p ModelPricing) Calculate(usage Usage) Cost {
	longWrite := usage.CacheWrite1H
	shortWrite := usage.CacheWrite - longWrite
	cost := Cost{
		Input:      (p.Input / 1_000_000) * float64(usage.Input),
		Output:     (p.Output / 1_000_000) * float64(usage.Output),
		CacheRead:  (p.CacheRead / 1_000_000) * float64(usage.CacheRead),
		CacheWrite: (p.CacheWrite*float64(shortWrite) + p.Input*2*float64(longWrite)) / 1_000_000,
	}
	cost.Total = cost.Input + cost.Output + cost.CacheRead + cost.CacheWrite
	return cost
}

// ApplyCost fills usage.Cost from the pricing table.
func (p ModelPricing) ApplyCost(usage *Usage) {
	if usage == nil {
		return
	}
	usage.Cost = p.Calculate(*usage)
}

// SupportsThinkingLevel reports whether the model accepts the level.
func (info ModelInfo) SupportsThinkingLevel(level ThinkingLevel) bool {
	if !info.Reasoning && !info.SupportsThinking {
		return level == ThinkingOff
	}
	if len(info.ThinkingLevels) > 0 {
		for _, supported := range info.ThinkingLevels {
			if supported == level {
				return true
			}
		}
		return false
	}
	return level != ThinkingXHigh && level != ThinkingMax
}

// ClampThinkingLevel returns the requested level when supported, otherwise the
// closest supported lower level.
func (info ModelInfo) ClampThinkingLevel(level ThinkingLevel) ThinkingLevel {
	levels := []ThinkingLevel{
		ThinkingOff, ThinkingMinimal, ThinkingLow, ThinkingMedium, ThinkingHigh, ThinkingXHigh, ThinkingMax,
	}
	index := 0
	for i, candidate := range levels {
		if candidate == level {
			index = i
			break
		}
	}
	for i := index; i >= 0; i-- {
		if info.SupportsThinkingLevel(levels[i]) {
			return levels[i]
		}
	}
	return ThinkingOff
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

	// APIKey overrides the provider credential for this request.
	APIKey string `json:"-"`
	// OnPayload inspects or replaces the provider request body. Returning nil
	// keeps the payload unchanged.
	OnPayload func(map[string]any) (map[string]any, error) `json:"-"`
	// OnResponse observes the raw HTTP response before its body is consumed.
	OnResponse func(status int, headers http.Header) error `json:"-"`
	// OnProviderStreamEvent observes each raw provider stream event before
	// normalization.
	OnProviderStreamEvent func(event any) error `json:"-"`
	// Identity overrides User-Agent and provider attribution headers.
	Identity *Identity `json:"-"`
	// Transport selects a preferred transport; GOAT rejects "websocket".
	Transport Transport `json:"transport,omitempty"`
	// TimeoutMs caps the provider request duration when the adapter supports it.
	TimeoutMs int `json:"timeout_ms,omitempty"`
	// MaxRetries caps client-side retries when the adapter supports it.
	MaxRetries int `json:"max_retries,omitempty"`
	// MaxRetryDelayMs caps provider-requested retry delays. Adapters that rely
	// on SDK retries ignore it.
	MaxRetryDelayMs int `json:"max_retry_delay_ms,omitempty"`
	// SamplingParams is merged into OpenAI-compatible request bodies.
	SamplingParams map[string]any `json:"sampling_params,omitempty"`
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
	Partial       Message        `json:"partial,omitempty"`
	Message       Message        `json:"message,omitempty"`
	Usage         *Usage         `json:"usage,omitempty"`
	StopReason    StopReason     `json:"stop_reason,omitempty"`
	Err           error          `json:"-"`
	Error         string         `json:"error,omitempty"`
}
