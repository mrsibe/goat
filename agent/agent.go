package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/mrsibe/goat/ai"
)

const defaultMaxSteps = 10

var (
	// ErrMaxSteps is returned when a run exhausts its configured model turns.
	ErrMaxSteps = errors.New("agent: maximum steps reached")
	// ErrOutputLimit is returned when the model reaches its output length limit.
	ErrOutputLimit = errors.New("agent: model output length limit reached")
)

// BeforeToolCall can block a tool call. Returning an error prevents execution
// and sends the error text to the model as an error result.
type BeforeToolCall func(context.Context, ai.ToolCall) error

// AfterToolCall can replace the result of an executed tool call.
type AfterToolCall func(context.Context, ai.ToolCall, ToolResult) ToolResult

// Config controls the reusable agent runtime.
type Config struct {
	Model          ai.Model
	Tools          []Tool
	SystemPrompt   string
	MaxSteps       int
	BeforeToolCall BeforeToolCall
	AfterToolCall  AfterToolCall
}

// Agent coordinates model turns and tool execution.
type Agent struct {
	model          ai.Model
	tools          toolRegistry
	systemPrompt   string
	maxSteps       int
	beforeToolCall BeforeToolCall
	afterToolCall  AfterToolCall
}

// New creates an agent. Agent values are immutable and may be shared across
// concurrent sessions.
func New(config Config) (*Agent, error) {
	if config.Model == nil {
		return nil, errors.New("agent: model must not be nil")
	}
	tools, err := newToolRegistry(config.Tools)
	if err != nil {
		return nil, err
	}
	maxSteps := config.MaxSteps
	if maxSteps <= 0 {
		maxSteps = defaultMaxSteps
	}
	return &Agent{
		model:          config.Model,
		tools:          tools,
		systemPrompt:   config.SystemPrompt,
		maxSteps:       maxSteps,
		beforeToolCall: config.BeforeToolCall,
		afterToolCall:  config.AfterToolCall,
	}, nil
}

// Run executes a one-shot conversation in a new session.
func (a *Agent) Run(ctx context.Context, input string, options ...RunOption) (Result, error) {
	return a.NewSession().Run(ctx, input, options...)
}

// NewSession creates an in-memory conversation.
func (a *Agent) NewSession() *Session {
	messages := make([]ai.Message, 0, 1)
	if a.systemPrompt != "" {
		messages = append(messages, ai.TextMessage(ai.RoleSystem, a.systemPrompt))
	}
	return &Session{agent: a, messages: messages}
}

// NewSessionWithHistory restores a conversation. System messages in history are
// ignored and the agent's current system prompt is prepended instead.
func (a *Agent) NewSessionWithHistory(history []ai.Message) *Session {
	session := a.NewSession()
	for _, message := range history {
		if message.Role == ai.RoleSystem {
			continue
		}
		session.messages = append(session.messages, ai.CloneMessage(message))
	}
	return session
}

// Session stores the message history for a multi-turn conversation. A Session
// must not be run concurrently.
type Session struct {
	agent    *Agent
	messages []ai.Message
}

// Messages returns a copy of the session history.
func (s *Session) Messages() []ai.Message {
	if s == nil {
		return nil
	}
	return ai.CloneMessages(s.messages)
}

// Run adds one user turn and continues until the model produces a final answer,
// a tool-free turn fails, or the configured step limit is reached.
func (s *Session) Run(ctx context.Context, input string, options ...RunOption) (Result, error) {
	if s == nil || s.agent == nil {
		return Result{}, errors.New("agent: session is not initialized")
	}

	runConfig := runOptions{}
	for _, option := range options {
		if option != nil {
			option(&runConfig)
		}
	}
	emit := func(event Event) {
		if runConfig.onEvent != nil {
			runConfig.onEvent(event)
		}
	}

	result := Result{}
	emit(Event{Type: EventAgentStart})
	s.appendMessage(ai.TextMessage(ai.RoleUser, input), 0, emit)

	fail := func(err error) (Result, error) {
		result.Messages = ai.CloneMessages(s.messages)
		eventResult := cloneResult(result)
		emit(Event{
			Type:   EventAgentError,
			Step:   result.Steps,
			Result: &eventResult,
			Err:    err,
			Error:  err.Error(),
		})
		return result, err
	}

	for step := 1; step <= s.agent.maxSteps; step++ {
		if err := ctx.Err(); err != nil {
			return fail(err)
		}

		result.Steps = step
		emit(Event{Type: EventTurnStart, Step: step})
		assistant, stopReason, usage, err := s.runModelTurn(ctx, step, emit)
		result.Usage.Add(usage)
		if err != nil {
			return fail(fmt.Errorf("agent: model call %d: %w", step, err))
		}

		result.StopReason = stopReason
		result.Answer = assistant.Text()
		s.messages = append(s.messages, ai.CloneMessage(assistant))

		toolCalls := assistant.ToolCalls()
		if len(toolCalls) == 0 {
			if stopReason == ai.StopReasonLength {
				return fail(ErrOutputLimit)
			}
			if stopReason == ai.StopReasonAborted {
				return fail(context.Canceled)
			}
			emit(Event{Type: EventTurnEnd, Step: step})
			result.Messages = ai.CloneMessages(s.messages)
			eventResult := cloneResult(result)
			emit(Event{Type: EventAgentEnd, Step: step, Result: &eventResult})
			return result, nil
		}

		for _, call := range toolCalls {
			toolResult := s.executeTool(ctx, call, step, emit)
			s.appendMessage(ai.ToolMessage(call.ID, toolResult.Content), step, emit)
		}
		emit(Event{Type: EventTurnEnd, Step: step})
	}

	return fail(fmt.Errorf("%w after %d steps", ErrMaxSteps, s.agent.maxSteps))
}

func (s *Session) appendMessage(message ai.Message, step int, emit func(Event)) {
	message = ai.CloneMessage(message)
	s.messages = append(s.messages, message)
	messageCopy := ai.CloneMessage(message)
	emit(Event{Type: EventMessageStart, Step: step, Message: &messageCopy})
	messageCopy = ai.CloneMessage(message)
	emit(Event{Type: EventMessageEnd, Step: step, Message: &messageCopy})
}

func (s *Session) runModelTurn(
	ctx context.Context,
	step int,
	emit func(Event),
) (ai.Message, ai.StopReason, ai.Usage, error) {
	stream, err := s.agent.model.Stream(ctx, ai.Request{
		Messages: ai.CloneMessages(s.messages),
		Tools:    s.agent.tools.list(),
	})
	if err != nil {
		return ai.Message{}, "", ai.Usage{}, err
	}
	if stream == nil {
		return ai.Message{}, "", ai.Usage{}, errors.New("model returned a nil stream")
	}
	defer stream.Close()

	emit(Event{
		Type:    EventMessageStart,
		Step:    step,
		Message: &ai.Message{Role: ai.RoleAssistant},
	})

	accumulator := newAssistantAccumulator()
	var usage ai.Usage
	var stopReason ai.StopReason
	sawDone := false

	for {
		if err := ctx.Err(); err != nil {
			return ai.Message{}, stopReason, usage, err
		}
		event, nextErr := stream.Next()
		if errors.Is(nextErr, io.EOF) {
			break
		}
		if nextErr != nil {
			return ai.Message{}, stopReason, usage, nextErr
		}

		switch event.Type {
		case ai.EventStart:
		case ai.EventTextDelta:
			if event.TextDelta != "" {
				accumulator.addText(event.TextDelta)
				emit(Event{Type: EventMessageDelta, Step: step, Text: event.TextDelta})
			}
		case ai.EventToolCallDelta:
			if event.ToolCall == nil {
				return ai.Message{}, stopReason, usage, errors.New("tool call delta is missing its payload")
			}
			if err := accumulator.addToolCall(*event.ToolCall); err != nil {
				return ai.Message{}, stopReason, usage, err
			}
		case ai.EventUsage:
			if event.Usage != nil {
				usage.Add(*event.Usage)
			}
		case ai.EventDone:
			sawDone = true
			stopReason = event.StopReason
		default:
			return ai.Message{}, stopReason, usage, fmt.Errorf("unknown model event type %q", event.Type)
		}
	}

	if !sawDone {
		stopReason = ai.StopReasonStop
		if len(accumulator.toolIndexes) > 0 {
			stopReason = ai.StopReasonToolCalls
		}
	}
	assistant, err := accumulator.message(step)
	if err != nil {
		return ai.Message{}, stopReason, usage, err
	}
	messageCopy := ai.CloneMessage(assistant)
	emit(Event{Type: EventMessageEnd, Step: step, Message: &messageCopy})
	return assistant, stopReason, usage, nil
}

func (s *Session) executeTool(ctx context.Context, call ai.ToolCall, step int, emit func(Event)) ToolResult {
	callCopy := ai.CloneToolCall(call)
	emit(Event{Type: EventToolStart, Step: step, ToolCall: &callCopy})

	result := ToolResult{CallID: call.ID}
	if s.agent.beforeToolCall != nil {
		if err := s.agent.beforeToolCall(ctx, callCopy); err != nil {
			result.Content = err.Error()
			result.IsError = true
			return s.finishTool(ctx, callCopy, result, step, emit)
		}
	}

	tool, ok := s.agent.tools.get(call.Name)
	if !ok {
		result.Content = fmt.Sprintf("unknown tool %q", call.Name)
		result.IsError = true
		return s.finishTool(ctx, callCopy, result, step, emit)
	}

	arguments := call.Arguments
	if len(strings.TrimSpace(string(arguments))) == 0 {
		arguments = json.RawMessage("{}")
	}
	callCopy.Arguments = append(json.RawMessage(nil), arguments...)

	func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				result.Content = fmt.Sprintf("tool %q panicked: %v", call.Name, recovered)
				result.IsError = true
			}
		}()
		var err error
		result, err = tool.Execute(ctx, callCopy)
		if result.CallID == "" {
			result.CallID = call.ID
		}
		if err != nil {
			result.Content = err.Error()
			result.IsError = true
		}
	}()

	return s.finishTool(ctx, callCopy, result, step, emit)
}

func (s *Session) finishTool(ctx context.Context, call ai.ToolCall, result ToolResult, step int, emit func(Event)) ToolResult {
	if s.agent.afterToolCall != nil {
		result = s.agent.afterToolCall(ctx, call, result)
	}
	if result.CallID == "" {
		result.CallID = call.ID
	}
	resultCopy := result
	callCopy := ai.CloneToolCall(call)
	emit(Event{
		Type:       EventToolEnd,
		Step:       step,
		ToolCall:   &callCopy,
		ToolResult: &resultCopy,
	})
	return result
}

type toolCallAccumulator struct {
	id        string
	name      string
	arguments strings.Builder
}

type assistantAccumulator struct {
	text        strings.Builder
	calls       map[int]*toolCallAccumulator
	toolIndexes []int
}

func newAssistantAccumulator() *assistantAccumulator {
	return &assistantAccumulator{calls: make(map[int]*toolCallAccumulator)}
}

func (a *assistantAccumulator) addText(text string) {
	a.text.WriteString(text)
}

func (a *assistantAccumulator) addToolCall(delta ai.ToolCallDelta) error {
	if delta.Index < 0 {
		return errors.New("tool call index must not be negative")
	}
	call, exists := a.calls[delta.Index]
	if !exists {
		call = &toolCallAccumulator{}
		a.calls[delta.Index] = call
		a.toolIndexes = append(a.toolIndexes, delta.Index)
	}
	if delta.ID != "" {
		call.id = delta.ID
	}
	call.name += delta.Name
	call.arguments.WriteString(delta.ArgumentsDelta)
	return nil
}

func (a *assistantAccumulator) message(step int) (ai.Message, error) {
	message := ai.Message{Role: ai.RoleAssistant}
	if a.text.Len() > 0 {
		message.Content = append(message.Content, ai.ContentBlock{
			Type: ai.BlockText,
			Text: a.text.String(),
		})
	}
	sort.Ints(a.toolIndexes)
	seenIDs := make(map[string]struct{}, len(a.toolIndexes))
	for _, index := range a.toolIndexes {
		call := a.calls[index]
		if call.name == "" {
			return ai.Message{}, fmt.Errorf("tool call %d is missing a name", index)
		}
		if call.id == "" {
			call.id = fmt.Sprintf("call_%d_%d", step, index)
		}
		if _, exists := seenIDs[call.id]; exists {
			return ai.Message{}, fmt.Errorf("duplicate tool call id %q", call.id)
		}
		seenIDs[call.id] = struct{}{}
		arguments := call.arguments.String()
		if strings.TrimSpace(arguments) == "" {
			arguments = "{}"
		}
		toolCall := ai.ToolCall{
			ID:        call.id,
			Name:      call.name,
			Arguments: json.RawMessage(arguments),
		}
		message.Content = append(message.Content, ai.ContentBlock{
			Type:     ai.BlockToolCall,
			ToolCall: &toolCall,
		})
	}
	return message, nil
}

func cloneResult(result Result) Result {
	result.Messages = ai.CloneMessages(result.Messages)
	return result
}
