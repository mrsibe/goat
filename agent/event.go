package agent

import (
	"context"
	"encoding/json"

	"github.com/mrsibe/goat/ai"
)

// ToolExecutionMode controls how tool calls in one assistant message execute.
type ToolExecutionMode string

const (
	ToolExecutionSequential ToolExecutionMode = "sequential"
	ToolExecutionParallel   ToolExecutionMode = "parallel"
)

// QueueMode controls how many queued messages are drained at one point.
type QueueMode string

const (
	QueueAll        QueueMode = "all"
	QueueOneAtATime QueueMode = "one-at-a-time"
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

// EventType identifies a lifecycle or streaming event.
type EventType string

const (
	EventAgentStart    EventType = "agent_start"
	EventTurnStart     EventType = "turn_start"
	EventMessageStart  EventType = "message_start"
	EventMessageUpdate EventType = "message_update"
	EventMessageDelta  EventType = "message_delta"
	EventMessageEnd    EventType = "message_end"
	EventThinkingStart EventType = "thinking_start"
	EventThinkingDelta EventType = "thinking_delta"
	EventThinkingEnd   EventType = "thinking_end"
	EventToolStart     EventType = "tool_start"
	EventToolUpdate    EventType = "tool_update"
	EventToolEnd       EventType = "tool_end"
	EventTurnEnd       EventType = "turn_end"
	EventAgentEnd      EventType = "agent_end"
	EventAgentError    EventType = "agent_error"
)

// Event is delivered synchronously while an agent run is in progress.
type Event struct {
	Type                  EventType    `json:"type"`
	Step                  int          `json:"step,omitempty"`
	Text                  string       `json:"text,omitempty"`
	Thinking              string       `json:"thinking,omitempty"`
	Message               *ai.Message  `json:"message,omitempty"`
	AssistantMessageEvent *ai.Event    `json:"assistant_message_event,omitempty"`
	ToolCall              *ai.ToolCall `json:"tool_call,omitempty"`
	ToolResult            *ToolResult  `json:"tool_result,omitempty"`
	PartialResult         *ToolResult  `json:"partial_result,omitempty"`
	ToolResults           []ToolResult `json:"tool_results,omitempty"`
	DurationMs            int64        `json:"duration_ms,omitempty"`
	Result                *Result      `json:"result,omitempty"`
	Err                   error        `json:"-"`
	Error                 string       `json:"error,omitempty"`
}

// Result is the complete outcome of one agent run.
type Result struct {
	Answer       string        `json:"answer"`
	Steps        int           `json:"steps"`
	StopReason   ai.StopReason `json:"stop_reason"`
	Messages     []ai.Message  `json:"messages"`
	NewMessages  []ai.Message  `json:"new_messages"`
	Usage        ai.Usage      `json:"usage"`
	ErrorMessage string        `json:"error_message,omitempty"`
}

// State is a read-only snapshot of the agent's current conversation state.
type State struct {
	SystemPrompt     string
	Model            ai.Model
	ThinkingLevel    ThinkingLevel
	Tools            []Tool
	Messages         []ai.Message
	IsStreaming      bool
	StreamingMessage *ai.Message
	PendingToolCalls []string
	ErrorMessage     string
}

// AgentContext is the transcript and executable tools visible to one loop.
type AgentContext struct {
	Messages []ai.Message
	Tools    []Tool
}

// MessageTransformer rewrites messages immediately before a model request.
// The returned messages are request-only and do not mutate the transcript.
type MessageTransformer func(context.Context, []ai.Message) ([]ai.Message, error)

// BeforeToolCallContext describes a validated tool call before execution.
type BeforeToolCallContext struct {
	AssistantMessage ai.Message
	ToolCall         ai.ToolCall
	Args             any
	Context          AgentContext
}

// BeforeToolCallResult can block a tool call and optionally terminate the batch.
type BeforeToolCallResult struct {
	Block     bool
	Reason    string
	Terminate bool
}

// BeforeToolCall runs after argument validation and before execution.
type BeforeToolCall func(context.Context, BeforeToolCallContext) (*BeforeToolCallResult, error)

// AfterToolCallContext describes an executed tool call before finalization.
type AfterToolCallContext struct {
	AssistantMessage ai.Message
	ToolCall         ai.ToolCall
	Args             any
	Result           ToolResult
	IsError          bool
	Context          AgentContext
}

// AfterToolCallResult replaces selected fields of an executed tool result.
type AfterToolCallResult struct {
	Content           []ai.ContentBlock
	Details           json.RawMessage
	StructuredContent json.RawMessage
	Usage             *ai.Usage
	IsError           *bool
	Terminate         *bool
}

// AfterToolCall runs after tool execution and before tool_end is emitted.
type AfterToolCall func(context.Context, AfterToolCallContext) (*AfterToolCallResult, error)

// PrepareRequestContext describes the state immediately before a provider call.
type PrepareRequestContext struct {
	Context       AgentContext
	Model         ai.Model
	ThinkingLevel ThinkingLevel
}

// RequestUpdate replaces selected request runtime state.
type RequestUpdate struct {
	Context       *AgentContext
	Model         ai.Model
	ThinkingLevel *ThinkingLevel
}

// PrepareRequest runs before every provider request, including the first.
type PrepareRequest func(context.Context, PrepareRequestContext) (*RequestUpdate, error)

// TurnContext describes a completed assistant turn and its tool results.
type TurnContext struct {
	Message     ai.Message
	ToolResults []ai.Message
	Context     AgentContext
	NewMessages []ai.Message
}

// TurnDecision tells the loop whether to continue or stop after a turn.
type TurnDecision string

const (
	TurnDefault  TurnDecision = ""
	TurnContinue TurnDecision = "continue"
	TurnEnd      TurnDecision = "end"
)

// TurnUpdate replaces selected state before the next turn.
type TurnUpdate struct {
	Context       *AgentContext
	Messages      []ai.Message
	Model         ai.Model
	ThinkingLevel *ThinkingLevel
}

// PrepareNextTurn runs after turn_end and before the next turn starts.
type PrepareNextTurn func(context.Context, TurnContext) (*TurnUpdate, error)

// FinishTurn runs after the assistant message and tool results are appended,
// immediately before turn_end.
type FinishTurn func(context.Context, TurnContext) (TurnDecision, error)
