package ai

import "context"

// Request is one provider-neutral model turn.
type Request struct {
	Messages []Message
	Tools    []ToolSpec
}

// Model is the boundary between the agent loop and a model provider.
type Model interface {
	Stream(ctx context.Context, request Request) (Stream, error)
}

// Stream yields normalized model events. Next returns io.EOF after the final
// event. The caller always closes the stream.
type Stream interface {
	Next() (Event, error)
	Close() error
}

// EventType identifies one normalized model stream event.
type EventType string

const (
	EventStart         EventType = "start"
	EventTextDelta     EventType = "text_delta"
	EventToolCallDelta EventType = "tool_call_delta"
	EventUsage         EventType = "usage"
	EventDone          EventType = "done"
)

// ToolCallDelta is one fragment of a streamed tool call.
type ToolCallDelta struct {
	Index          int    `json:"index"`
	ID             string `json:"id,omitempty"`
	Name           string `json:"name,omitempty"`
	ArgumentsDelta string `json:"arguments_delta,omitempty"`
}

// Event is one normalized model stream event.
type Event struct {
	Type       EventType      `json:"type"`
	TextDelta  string         `json:"text_delta,omitempty"`
	ToolCall   *ToolCallDelta `json:"tool_call,omitempty"`
	Usage      *Usage         `json:"usage,omitempty"`
	StopReason StopReason     `json:"stop_reason,omitempty"`
}
