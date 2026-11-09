package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
	"sync"

	"github.com/mrsibe/goat/ai"
)

const defaultMaxSteps = 10

var (
	// ErrMaxSteps is returned when a run exhausts its configured model turns.
	ErrMaxSteps = errors.New("agent: maximum steps reached")
	// ErrOutputLimit is returned when the model reaches its output length limit.
	ErrOutputLimit = errors.New("agent: model output length limit reached")
	// ErrModelStop is returned when a provider reports an unsuccessful terminal
	// stop reason without a more specific stream error.
	ErrModelStop = errors.New("agent: model stopped with an error")
)

// ToolExecutionMode controls how tool calls in one assistant message execute.
type ToolExecutionMode string

const (
	ToolExecutionSequential ToolExecutionMode = "sequential"
	ToolExecutionParallel   ToolExecutionMode = "parallel"
)

// MessageTransformer can inject, prune, or rewrite messages before every model
// request.
type MessageTransformer func(context.Context, []ai.Message) ([]ai.Message, error)

// BeforeToolCall can block a tool call. Returning an error prevents execution
// and sends the error text to the model as an error result.
type BeforeToolCall func(context.Context, ai.ToolCall) error

// AfterToolCall can replace the result of an executed tool call.
type AfterToolCall func(context.Context, ai.ToolCall, ToolResult) ToolResult

// Config controls the reusable agent runtime.
type Config struct {
	Model             ai.Model
	Tools             []Tool
	SystemPrompt      string
	MaxSteps          int
	RequestOptions    ai.RequestOptions
	ToolExecution     ToolExecutionMode
	TransformMessages MessageTransformer
	BeforeToolCall    BeforeToolCall
	AfterToolCall     AfterToolCall
}

// Agent coordinates model turns and tool execution.
type Agent struct {
	model             ai.Model
	tools             toolRegistry
	systemPrompt      string
	maxSteps          int
	requestOptions    ai.RequestOptions
	toolExecution     ToolExecutionMode
	transformMessages MessageTransformer
	beforeToolCall    BeforeToolCall
	afterToolCall     AfterToolCall
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
	toolExecution := config.ToolExecution
	if toolExecution == "" {
		toolExecution = ToolExecutionSequential
	}
	if toolExecution != ToolExecutionSequential && toolExecution != ToolExecutionParallel {
		return nil, fmt.Errorf("agent: unsupported tool execution mode %q", toolExecution)
	}
	return &Agent{
		model:             config.Model,
		tools:             tools,
		systemPrompt:      config.SystemPrompt,
		maxSteps:          maxSteps,
		requestOptions:    config.RequestOptions,
		toolExecution:     toolExecution,
		transformMessages: config.TransformMessages,
		beforeToolCall:    config.BeforeToolCall,
		afterToolCall:     config.AfterToolCall,
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

// NewSessionWithHistory restores a conversation. The agent's configured system
// prompt is prepended when present; system messages in history are ignored so
// that a restored session uses the agent's current prompt.
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
	requestOptions := mergeRequestOptions(s.agent.requestOptions, runConfig.requestOptions)

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
		assistant, stopReason, usage, err := s.runModelTurn(ctx, step, requestOptions, emit)
		result.Usage.Add(usage)
		if err != nil {
			if stopReason == ai.StopReasonError || len(assistant.Content) > 0 {
				result.StopReason = stopReason
				result.Answer = assistant.Text()
				s.messages = append(s.messages, ai.CloneMessage(assistant))
				emitTurnEnd(emit, step, assistant, nil)
			}
			return fail(fmt.Errorf("agent: model call %d: %w", step, err))
		}

		result.StopReason = stopReason
		result.Answer = assistant.Text()
		s.messages = append(s.messages, ai.CloneMessage(assistant))
		if stopReason == ai.StopReasonError {
			emitTurnEnd(emit, step, assistant, nil)
			return fail(ErrModelStop)
		}
		if stopReason == ai.StopReasonAborted {
			emitTurnEnd(emit, step, assistant, nil)
			return fail(context.Canceled)
		}

		toolCalls := assistant.ToolCalls()
		if len(toolCalls) == 0 {
			if stopReason == ai.StopReasonLength {
				return fail(ErrOutputLimit)
			}
			emitTurnEnd(emit, step, assistant, nil)
			result.Messages = ai.CloneMessages(s.messages)
			eventResult := cloneResult(result)
			emit(Event{Type: EventAgentEnd, Step: step, Result: &eventResult})
			return result, nil
		}

		if stopReason == ai.StopReasonLength {
			toolResults := make([]ToolResult, 0, len(toolCalls))
			for _, call := range toolCalls {
				toolResults = append(toolResults, s.rejectTruncatedToolCall(call, step, emit))
			}
			s.appendToolResults(toolCalls, toolResults, step, emit)
			emitTurnEnd(emit, step, assistant, toolResults)
			if step == s.agent.maxSteps {
				return fail(ErrOutputLimit)
			}
			continue
		}

		toolResults := s.executeToolCalls(ctx, toolCalls, step, emit)
		s.appendToolResults(toolCalls, toolResults, step, emit)
		emitTurnEnd(emit, step, assistant, toolResults)
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

func (s *Session) appendToolResults(calls []ai.ToolCall, results []ToolResult, step int, emit func(Event)) {
	for index, result := range results {
		call := calls[index]
		s.appendMessage(
			ai.ToolResultMessage(call.ID, call.Name, result.Content, result.IsError),
			step,
			emit,
		)
	}
}

func emitTurnEnd(emit func(Event), step int, assistant ai.Message, toolResults []ToolResult) {
	message := ai.CloneMessage(assistant)
	results := cloneToolResults(toolResults)
	emit(Event{
		Type:        EventTurnEnd,
		Step:        step,
		Message:     &message,
		ToolResults: results,
	})
}

func cloneToolResults(results []ToolResult) []ToolResult {
	if results == nil {
		return nil
	}
	cloned := make([]ToolResult, len(results))
	for index, result := range results {
		cloned[index] = ToolResult{
			Content: ai.CloneContentBlocks(result.Content),
			IsError: result.IsError,
		}
	}
	return cloned
}

func (s *Session) runModelTurn(
	ctx context.Context,
	step int,
	options ai.RequestOptions,
	emit func(Event),
) (ai.Message, ai.StopReason, ai.Usage, error) {
	messages := ai.CloneMessages(s.messages)
	if s.agent.transformMessages != nil {
		transformed, err := s.agent.transformMessages(ctx, messages)
		if err != nil {
			return ai.Message{}, "", ai.Usage{}, fmt.Errorf("transform messages: %w", err)
		}
		messages = ai.CloneMessages(transformed)
	}

	stream, err := s.agent.model.Stream(ctx, ai.Request{
		Messages: messages,
		Tools:    s.agent.tools.list(),
		Options:  options,
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
	var finalMessage *ai.Message
	sawUsage := false
	sawDone := false

	for {
		if err := ctx.Err(); err != nil {
			assistant, reason, finishErr := s.finishFailedModelTurn(
				accumulator,
				finalMessage,
				step,
				ai.StopReasonAborted,
				emit,
				err,
			)
			return assistant, reason, usage, finishErr
		}
		event, nextErr := stream.Next()
		if errors.Is(nextErr, io.EOF) {
			break
		}
		if nextErr != nil {
			assistant, reason, finishErr := s.finishFailedModelTurn(
				accumulator,
				finalMessage,
				step,
				ai.StopReasonError,
				emit,
				nextErr,
			)
			return assistant, reason, usage, finishErr
		}

		switch event.Type {
		case ai.EventStart, ai.EventTextStart, ai.EventTextEnd:
		case ai.EventTextDelta:
			if event.TextDelta != "" {
				accumulator.addText(event.TextDelta)
				text := event.TextDelta
				partial := cloneMessagePointer(event.Partial)
				emit(Event{Type: EventMessageDelta, Step: step, Text: text, Message: partial})
			}
		case ai.EventThinkingStart:
			partial := cloneMessagePointer(event.Partial)
			emit(Event{Type: EventThinkingStart, Step: step, Message: partial})
		case ai.EventThinkingDelta:
			if event.ThinkingDelta != "" {
				accumulator.addThinking(event.ThinkingDelta)
				partial := cloneMessagePointer(event.Partial)
				emit(Event{
					Type:     EventThinkingDelta,
					Step:     step,
					Thinking: event.ThinkingDelta,
					Message:  partial,
				})
			}
		case ai.EventThinkingEnd:
			partial := cloneMessagePointer(event.Partial)
			emit(Event{Type: EventThinkingEnd, Step: step, Thinking: event.Thinking, Message: partial})
		case ai.EventToolCallStart:
			if event.ToolCall == nil {
				return ai.Message{}, stopReason, usage, errors.New("tool call start is missing its payload")
			}
			accumulator.startToolCall(event.ContentIndex, *event.ToolCall)
		case ai.EventToolCallDelta:
			if event.ToolCallDelta == nil {
				return ai.Message{}, stopReason, usage, errors.New("tool call delta is missing its payload")
			}
			if err := accumulator.addToolCall(event.ContentIndex, *event.ToolCallDelta); err != nil {
				return ai.Message{}, stopReason, usage, err
			}
		case ai.EventToolCallEnd:
			if event.ToolCall == nil {
				return ai.Message{}, stopReason, usage, errors.New("tool call end is missing its payload")
			}
			accumulator.finishToolCall(event.ContentIndex, *event.ToolCall)
		case ai.EventUsage:
			if event.Usage != nil {
				usage.Add(*event.Usage)
				sawUsage = true
			}
		case ai.EventDone:
			sawDone = true
			stopReason = event.StopReason
			if event.Message != nil {
				message := ai.CloneMessage(*event.Message)
				finalMessage = &message
			}
			if !sawUsage && event.Usage != nil {
				usage = *event.Usage
			}
		case ai.EventError:
			if event.StopReason == "" {
				event.StopReason = ai.StopReasonError
			}
			if event.Message != nil {
				message := ai.CloneMessage(*event.Message)
				finalMessage = &message
			} else if event.Partial != nil {
				message := ai.CloneMessage(*event.Partial)
				finalMessage = &message
			}
			if !sawUsage && event.Usage != nil {
				usage = *event.Usage
			}
			var streamErr error
			if event.Err != nil {
				streamErr = event.Err
			} else if event.Error != "" {
				streamErr = errors.New(event.Error)
			} else {
				streamErr = errors.New("model stream reported an error")
			}
			assistant, reason, finishErr := s.finishFailedModelTurn(
				accumulator,
				finalMessage,
				step,
				event.StopReason,
				emit,
				streamErr,
			)
			return assistant, reason, usage, finishErr
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
	assistant, err := assistantMessage(accumulator, finalMessage, step)
	if err != nil {
		return ai.Message{}, stopReason, usage, err
	}
	messageCopy := ai.CloneMessage(assistant)
	emit(Event{Type: EventMessageEnd, Step: step, Message: &messageCopy})
	return assistant, stopReason, usage, nil
}

func cloneMessagePointer(message *ai.Message) *ai.Message {
	if message == nil {
		return nil
	}
	cloned := ai.CloneMessage(*message)
	return &cloned
}

func (s *Session) finishFailedModelTurn(
	accumulator *assistantAccumulator,
	finalMessage *ai.Message,
	step int,
	stopReason ai.StopReason,
	emit func(Event),
	err error,
) (ai.Message, ai.StopReason, error) {
	if stopReason == "" {
		stopReason = ai.StopReasonError
	}
	assistant, buildErr := assistantMessage(accumulator, finalMessage, step)
	if buildErr != nil {
		return ai.Message{}, stopReason, errors.Join(err, buildErr)
	}
	messageCopy := ai.CloneMessage(assistant)
	emit(Event{Type: EventMessageEnd, Step: step, Message: &messageCopy})
	return assistant, stopReason, err
}

func assistantMessage(
	accumulator *assistantAccumulator,
	finalMessage *ai.Message,
	step int,
) (ai.Message, error) {
	if finalMessage != nil {
		return ai.CloneMessage(*finalMessage), nil
	}
	return accumulator.message(step)
}

type preparedToolCall struct {
	call      ai.ToolCall
	tool      Tool
	immediate *ToolResult
}

func (s *Session) executeToolCalls(
	ctx context.Context,
	calls []ai.ToolCall,
	step int,
	emit func(Event),
) []ToolResult {
	if s.agent.toolExecution == ToolExecutionParallel {
		return s.executeToolCallsParallel(ctx, calls, step, emit)
	}
	results := make([]ToolResult, len(calls))
	for index, call := range calls {
		results[index] = s.executeTool(ctx, call, step, emit)
	}
	return results
}

func (s *Session) executeToolCallsParallel(
	ctx context.Context,
	calls []ai.ToolCall,
	step int,
	emit func(Event),
) []ToolResult {
	prepared := make([]preparedToolCall, len(calls))
	for index, call := range calls {
		prepared[index] = s.prepareTool(ctx, call, step, emit)
	}

	type completion struct {
		index  int
		result ToolResult
	}
	completed := make(chan completion, len(calls))
	var wait sync.WaitGroup
	for index, item := range prepared {
		if item.immediate != nil {
			completed <- completion{index: index, result: *item.immediate}
			continue
		}
		wait.Add(1)
		go func(index int, item preparedToolCall) {
			defer wait.Done()
			completed <- completion{
				index:  index,
				result: s.executePreparedTool(ctx, item),
			}
		}(index, item)
	}
	wait.Wait()
	close(completed)

	results := make([]ToolResult, len(calls))
	for item := range completed {
		results[item.index] = s.finishTool(ctx, prepared[item.index].call, item.result, step, emit)
	}
	return results
}

func (s *Session) executeTool(ctx context.Context, call ai.ToolCall, step int, emit func(Event)) ToolResult {
	prepared := s.prepareTool(ctx, call, step, emit)
	if prepared.immediate != nil {
		return s.finishTool(ctx, prepared.call, *prepared.immediate, step, emit)
	}
	result := s.executePreparedTool(ctx, prepared)
	return s.finishTool(ctx, prepared.call, result, step, emit)
}

func (s *Session) prepareTool(ctx context.Context, call ai.ToolCall, step int, emit func(Event)) preparedToolCall {
	callCopy := ai.CloneToolCall(call)
	emit(Event{Type: EventToolStart, Step: step, ToolCall: &callCopy})

	tool, spec, ok := s.agent.tools.get(call.Name)
	if !ok {
		result := textErrorResult(fmt.Sprintf("unknown tool %q", call.Name))
		return preparedToolCall{call: callCopy, immediate: &result}
	}

	arguments := call.Arguments
	if len(strings.TrimSpace(string(arguments))) == 0 {
		arguments = json.RawMessage("{}")
	}
	callCopy.Arguments = append(json.RawMessage(nil), arguments...)
	if call.ArgumentsError != "" {
		result := textErrorResult(fmt.Sprintf(
			"invalid arguments for tool %q: %s",
			call.Name,
			call.ArgumentsError,
		))
		return preparedToolCall{call: callCopy, immediate: &result}
	}
	if err := validateToolArguments(spec, callCopy.Arguments); err != nil {
		result := textErrorResult(fmt.Sprintf(
			"invalid arguments for tool %q: %s",
			call.Name,
			err,
		))
		return preparedToolCall{call: callCopy, immediate: &result}
	}
	if s.agent.beforeToolCall != nil {
		if err := s.agent.beforeToolCall(ctx, callCopy); err != nil {
			result := textErrorResult(err.Error())
			return preparedToolCall{call: callCopy, immediate: &result}
		}
	}
	return preparedToolCall{call: callCopy, tool: tool}
}

func (s *Session) executePreparedTool(ctx context.Context, prepared preparedToolCall) (result ToolResult) {
	defer func() {
		if recovered := recover(); recovered != nil {
			result = textErrorResult(fmt.Sprintf(
				"tool %q panicked: %v",
				prepared.call.Name,
				recovered,
			))
		}
	}()
	result, err := prepared.tool.Execute(ctx, prepared.call)
	if err != nil {
		return textErrorResult(err.Error())
	}
	if len(result.Content) == 0 {
		result.Content = []ai.ContentBlock{{Type: ai.BlockText, Text: ""}}
	}
	result.Content = ai.CloneContentBlocks(result.Content)
	if err := validateToolResult(result); err != nil {
		return textErrorResult(fmt.Sprintf("tool %q returned invalid content: %s", prepared.call.Name, err))
	}
	return result
}

func textErrorResult(text string) ToolResult {
	return ToolResult{
		Content: []ai.ContentBlock{{Type: ai.BlockText, Text: text}},
		IsError: true,
	}
}

func (s *Session) rejectTruncatedToolCall(call ai.ToolCall, step int, emit func(Event)) ToolResult {
	callCopy := ai.CloneToolCall(call)
	emit(Event{Type: EventToolStart, Step: step, ToolCall: &callCopy})
	content := fmt.Sprintf(
		"tool call %q was not executed because the response hit the output token limit; its arguments may be truncated. Re-issue the call with complete arguments.",
		call.Name,
	)
	result := textErrorResult(content)
	s.emitToolEnd(callCopy, result, step, emit)
	return result
}

func (s *Session) finishTool(
	ctx context.Context,
	call ai.ToolCall,
	result ToolResult,
	step int,
	emit func(Event),
) ToolResult {
	if s.agent.afterToolCall != nil {
		result = s.agent.afterToolCall(ctx, call, result)
	}
	result.Content = ai.CloneContentBlocks(result.Content)
	if err := validateToolResult(result); err != nil {
		result = textErrorResult(fmt.Sprintf("tool %q returned invalid content: %s", call.Name, err))
	}
	s.emitToolEnd(call, result, step, emit)
	return result
}

func (s *Session) emitToolEnd(call ai.ToolCall, result ToolResult, step int, emit func(Event)) {
	callCopy := ai.CloneToolCall(call)
	resultCopy := ToolResult{
		Content: ai.CloneContentBlocks(result.Content),
		IsError: result.IsError,
	}
	emit(Event{
		Type:       EventToolEnd,
		Step:       step,
		ToolCall:   &callCopy,
		ToolResult: &resultCopy,
	})
}

type toolCallAccumulator struct {
	id             string
	name           string
	arguments      strings.Builder
	argumentsError string
	final          *ai.ToolCall
}

type assistantAccumulator struct {
	text        strings.Builder
	thinking    strings.Builder
	calls       map[int]*toolCallAccumulator
	toolIndexes []int
}

func newAssistantAccumulator() *assistantAccumulator {
	return &assistantAccumulator{calls: make(map[int]*toolCallAccumulator)}
}

func (a *assistantAccumulator) addText(text string) {
	a.text.WriteString(text)
}

func (a *assistantAccumulator) addThinking(thinking string) {
	a.thinking.WriteString(thinking)
}

func (a *assistantAccumulator) startToolCall(index int, call ai.ToolCall) {
	acc := a.ensureToolCall(index)
	if call.ID != "" {
		acc.id = call.ID
	}
	if call.Name != "" {
		acc.name = call.Name
	}
}

func (a *assistantAccumulator) finishToolCall(index int, call ai.ToolCall) {
	acc := a.ensureToolCall(index)
	if call.ID != "" {
		acc.id = call.ID
	}
	if call.Name != "" {
		acc.name = call.Name
	}
	if call.ArgumentsError != "" {
		acc.argumentsError = call.ArgumentsError
	}
	cloned := ai.CloneToolCall(call)
	acc.final = &cloned
}

func (a *assistantAccumulator) addToolCall(index int, delta ai.ToolCallDelta) error {
	if index < 0 {
		return errors.New("tool call index must not be negative")
	}
	call := a.ensureToolCall(index)
	if delta.ID != "" {
		call.id = delta.ID
	}
	if delta.Name != "" {
		call.name = delta.Name
	}
	call.arguments.WriteString(delta.ArgumentsDelta)
	return nil
}

func (a *assistantAccumulator) ensureToolCall(index int) *toolCallAccumulator {
	call, exists := a.calls[index]
	if !exists {
		call = &toolCallAccumulator{}
		a.calls[index] = call
		a.toolIndexes = append(a.toolIndexes, index)
	}
	return call
}

func (a *assistantAccumulator) message(step int) (ai.Message, error) {
	message := ai.Message{Role: ai.RoleAssistant}
	if a.text.Len() > 0 {
		message.Content = append(message.Content, ai.ContentBlock{
			Type: ai.BlockText,
			Text: a.text.String(),
		})
	}
	if a.thinking.Len() > 0 {
		message.Content = append(message.Content, ai.ContentBlock{
			Type:     ai.BlockThinking,
			Thinking: a.thinking.String(),
		})
	}
	sort.Ints(a.toolIndexes)
	seenIDs := make(map[string]struct{}, len(a.toolIndexes))
	for _, index := range a.toolIndexes {
		call := a.calls[index]
		id := call.id
		name := call.name
		arguments := call.arguments.String()
		argumentsError := call.argumentsError
		if call.final != nil {
			if call.final.ID != "" {
				id = call.final.ID
			}
			if call.final.Name != "" {
				name = call.final.Name
			}
			if len(call.final.Arguments) > 0 {
				arguments = string(call.final.Arguments)
			}
			if call.final.ArgumentsError != "" {
				argumentsError = call.final.ArgumentsError
			}
		}
		if name == "" {
			return ai.Message{}, fmt.Errorf("tool call %d is missing a name", index)
		}
		if id == "" {
			id = fmt.Sprintf("call_%d_%d", step, index)
		}
		if _, exists := seenIDs[id]; exists {
			return ai.Message{}, fmt.Errorf("duplicate tool call id %q", id)
		}
		seenIDs[id] = struct{}{}
		if strings.TrimSpace(arguments) == "" {
			arguments = "{}"
		}
		toolCall := ai.ToolCall{
			ID:             id,
			Name:           name,
			Arguments:      json.RawMessage(arguments),
			ArgumentsError: argumentsError,
		}
		message.Content = append(message.Content, ai.ContentBlock{
			Type:     ai.BlockToolCall,
			ToolCall: &toolCall,
		})
	}
	return message, nil
}

func mergeRequestOptions(base ai.RequestOptions, override *ai.RequestOptions) ai.RequestOptions {
	if override == nil {
		return base
	}
	merged := base
	if override.Temperature != nil {
		merged.Temperature = override.Temperature
	}
	if override.TopP != nil {
		merged.TopP = override.TopP
	}
	if override.MaxTokens > 0 {
		merged.MaxTokens = override.MaxTokens
	}
	if override.ToolChoice.Mode != "" {
		merged.ToolChoice = override.ToolChoice
	}
	if override.SessionID != "" {
		merged.SessionID = override.SessionID
	}
	if override.ReasoningEffort != "" {
		merged.ReasoningEffort = override.ReasoningEffort
	}
	if override.ThinkingBudgetTokens > 0 {
		merged.ThinkingBudgetTokens = override.ThinkingBudgetTokens
	}
	return merged
}

func cloneResult(result Result) Result {
	result.Messages = ai.CloneMessages(result.Messages)
	return result
}
