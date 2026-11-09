package agent

import "github.com/mrsibe/goat/ai"

// EventType identifies a lifecycle or streaming event.
type EventType string

const (
	EventAgentStart    EventType = "agent_start"
	EventTurnStart     EventType = "turn_start"
	EventMessageStart  EventType = "message_start"
	EventMessageDelta  EventType = "message_delta"
	EventMessageEnd    EventType = "message_end"
	EventThinkingStart EventType = "thinking_start"
	EventThinkingDelta EventType = "thinking_delta"
	EventThinkingEnd   EventType = "thinking_end"
	EventToolStart     EventType = "tool_start"
	EventToolEnd       EventType = "tool_end"
	EventTurnEnd       EventType = "turn_end"
	EventAgentEnd      EventType = "agent_end"
	EventAgentError    EventType = "agent_error"
)

// Event is delivered synchronously while an agent run is in progress.
type Event struct {
	Type        EventType    `json:"type"`
	Step        int          `json:"step,omitempty"`
	Text        string       `json:"text,omitempty"`
	Thinking    string       `json:"thinking,omitempty"`
	Message     *ai.Message  `json:"message,omitempty"`
	ToolCall    *ai.ToolCall `json:"tool_call,omitempty"`
	ToolResult  *ToolResult  `json:"tool_result,omitempty"`
	ToolResults []ToolResult `json:"tool_results,omitempty"`
	Result      *Result      `json:"result,omitempty"`
	Err         error        `json:"-"`
	Error       string       `json:"error,omitempty"`
}

// Result is the complete outcome of one agent run.
type Result struct {
	Answer     string        `json:"answer"`
	Steps      int           `json:"steps"`
	StopReason ai.StopReason `json:"stop_reason"`
	Messages   []ai.Message  `json:"messages"`
	Usage      ai.Usage      `json:"usage"`
}

type runOptions struct {
	onEvent        func(Event)
	requestOptions *ai.RequestOptions
}

// RunOption customizes one agent run.
type RunOption func(*runOptions)

// WithOnEvent registers a synchronous event listener for a run.
func WithOnEvent(handler func(Event)) RunOption {
	return func(options *runOptions) {
		options.onEvent = handler
	}
}

// WithRequestOptions overrides non-zero fields of the agent's default request
// options for one run.
func WithRequestOptions(options ai.RequestOptions) RunOption {
	return func(run *runOptions) {
		run.requestOptions = &options
	}
}
