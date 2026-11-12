package agent

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/mrsibe/goat/ai"
)

var (
	// ErrMaxSteps is returned when a run exhausts its configured model turns.
	ErrMaxSteps = errors.New("agent: maximum steps reached")
	// ErrBusy is returned when a prompt, continuation, or reset races an active run.
	ErrBusy = errors.New("agent: agent is busy")
	// ErrNoMessages is returned when there is no conversation to continue.
	ErrNoMessages = errors.New("agent: no messages to continue from")
	// ErrCannotContinue is returned when the transcript tail is an assistant message
	// and no queued message can be used instead.
	ErrCannotContinue = errors.New("agent: cannot continue from an assistant message")
	// ErrAborted is returned when a provider reports an aborted turn without the
	// caller's context being canceled.
	ErrAborted = errors.New("agent: run aborted")
)

// ModelError reports a provider or stream failure after the assistant message
// has been finalized and appended to the transcript.
type ModelError struct {
	Message ai.Message
	Err     error
}

func (e *ModelError) Error() string {
	if e == nil {
		return "<nil>"
	}
	if e.Err != nil {
		return e.Err.Error()
	}
	if e.Message.ErrorMessage != "" {
		return e.Message.ErrorMessage
	}
	return "agent: model stopped with an error"
}

func (e *ModelError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// Config controls the reusable, stateful agent runtime.
type Config struct {
	Model             ai.Model
	SystemPrompt      string
	Messages          []ai.Message
	Tools             []Tool
	MaxSteps          int
	RequestOptions    ai.RequestOptions
	ToolExecution     ToolExecutionMode
	SteeringMode      QueueMode
	FollowUpMode      QueueMode
	SessionID         string
	ThinkingLevel     ThinkingLevel
	ThinkingBudgets   map[ThinkingLevel]int
	TransformMessages MessageTransformer
	BeforeToolCall    BeforeToolCall
	AfterToolCall     AfterToolCall
	PrepareRequest    PrepareRequest
	PrepareNextTurn   PrepareNextTurn
	FinishTurn        FinishTurn
}

// Agent owns a mutable transcript, tool set, model, and pending input queues.
// Only one run may be active at a time.
type Agent struct {
	mu      sync.Mutex
	eventMu sync.Mutex

	model             ai.Model
	tools             toolRegistry
	messages          []ai.Message
	baselineSystem    *ai.Message
	requestOptions    ai.RequestOptions
	maxSteps          int
	toolExecution     ToolExecutionMode
	steeringMode      QueueMode
	followUpMode      QueueMode
	steeringQueue     []ai.Message
	followUpQueue     []ai.Message
	transformMessages MessageTransformer
	beforeToolCall    BeforeToolCall
	afterToolCall     AfterToolCall
	prepareRequest    PrepareRequest
	prepareNextTurn   PrepareNextTurn
	finishTurn        FinishTurn
	thinkingLevel     ThinkingLevel
	thinkingBudgets   map[ThinkingLevel]int
	sessionID         string

	listeners      map[uint64]func(Event)
	nextListenerID uint64

	active           *activeRun
	isStreaming      bool
	streamingMessage *ai.Message
	pendingToolCalls map[string]struct{}
	errorMessage     string
}

type activeRun struct {
	cancel context.CancelFunc
	done   chan struct{}
}

// New creates a stateful agent.
func New(config Config) (*Agent, error) {
	if config.Model == nil {
		return nil, errors.New("agent: model must not be nil")
	}
	tools, err := newToolRegistry(config.Tools)
	if err != nil {
		return nil, err
	}
	if config.MaxSteps < 0 {
		return nil, errors.New("agent: max steps must not be negative")
	}
	toolExecution := config.ToolExecution
	if toolExecution == "" {
		toolExecution = ToolExecutionParallel
	}
	if toolExecution != ToolExecutionSequential && toolExecution != ToolExecutionParallel {
		return nil, fmt.Errorf("agent: unsupported tool execution mode %q", toolExecution)
	}
	steeringMode := config.SteeringMode
	if steeringMode == "" {
		steeringMode = QueueOneAtATime
	}
	if steeringMode != QueueAll && steeringMode != QueueOneAtATime {
		return nil, fmt.Errorf("agent: unsupported steering mode %q", steeringMode)
	}
	followUpMode := config.FollowUpMode
	if followUpMode == "" {
		followUpMode = QueueOneAtATime
	}
	if followUpMode != QueueAll && followUpMode != QueueOneAtATime {
		return nil, fmt.Errorf("agent: unsupported follow-up mode %q", followUpMode)
	}
	thinkingLevel := config.ThinkingLevel
	if thinkingLevel == "" {
		thinkingLevel = ThinkingOff
	}
	if !validThinkingLevel(thinkingLevel) {
		return nil, fmt.Errorf("agent: unsupported thinking level %q", thinkingLevel)
	}

	messages := ai.CloneMessages(config.Messages)
	var baseline *ai.Message
	if config.SystemPrompt != "" {
		system := ai.TextMessage(ai.RoleSystem, config.SystemPrompt)
		baseline = &system
	}
	if len(messages) > 0 && messages[0].Role == ai.RoleSystem {
		system := ai.CloneMessage(messages[0])
		baseline = &system
	} else if baseline != nil {
		messages = append([]ai.Message{*baseline}, messages...)
	}

	budgets := make(map[ThinkingLevel]int, len(config.ThinkingBudgets))
	for level, budget := range config.ThinkingBudgets {
		if budget > 0 {
			budgets[level] = budget
		}
	}

	return &Agent{
		model:             config.Model,
		tools:             tools,
		messages:          messages,
		baselineSystem:    baseline,
		requestOptions:    config.RequestOptions,
		maxSteps:          config.MaxSteps,
		toolExecution:     toolExecution,
		steeringMode:      steeringMode,
		followUpMode:      followUpMode,
		transformMessages: config.TransformMessages,
		beforeToolCall:    config.BeforeToolCall,
		afterToolCall:     config.AfterToolCall,
		prepareRequest:    config.PrepareRequest,
		prepareNextTurn:   config.PrepareNextTurn,
		finishTurn:        config.FinishTurn,
		thinkingLevel:     thinkingLevel,
		thinkingBudgets:   budgets,
		sessionID:         config.SessionID,
		listeners:         make(map[uint64]func(Event)),
		pendingToolCalls:  make(map[string]struct{}),
	}, nil
}

func validThinkingLevel(level ThinkingLevel) bool {
	switch level {
	case ThinkingOff, ThinkingMinimal, ThinkingLow, ThinkingMedium, ThinkingHigh, ThinkingXHigh, ThinkingMax:
		return true
	default:
		return false
	}
}

// Subscribe registers a synchronous event listener and returns an unsubscribe
// function. Listeners run in registration order and are part of run settlement.
func (a *Agent) Subscribe(listener func(Event)) func() {
	if listener == nil {
		return func() {}
	}
	a.mu.Lock()
	id := a.nextListenerID
	a.nextListenerID++
	a.listeners[id] = listener
	a.mu.Unlock()
	return func() {
		a.mu.Lock()
		delete(a.listeners, id)
		a.mu.Unlock()
	}
}

// Prompt starts a run with one user text message and optional images.
func (a *Agent) Prompt(ctx context.Context, input string, images ...ai.ImageContent) (Result, error) {
	content := []ai.ContentBlock{{Type: ai.BlockText, Text: input}}
	for i := range images {
		image := images[i]
		content = append(content, ai.ContentBlock{Type: ai.BlockImage, Image: &image})
	}
	return a.PromptMessages(ctx, ai.Message{
		Role:      ai.RoleUser,
		Content:   content,
		Timestamp: time.Now().UnixMilli(),
	})
}

// PromptMessages starts a run with one or more messages.
func (a *Agent) PromptMessages(ctx context.Context, messages ...ai.Message) (Result, error) {
	if len(messages) == 0 {
		return Result{}, ErrNoMessages
	}
	return a.runMessages(ctx, ai.CloneMessages(messages), false, false, false)
}

// Continue resumes from the current transcript without adding a new user message.
func (a *Agent) Continue(ctx context.Context) (Result, error) {
	a.mu.Lock()
	messages := ai.CloneMessages(a.messages)
	busy := a.active != nil
	a.mu.Unlock()
	if busy {
		return Result{}, ErrBusy
	}
	if len(messages) == 0 {
		return Result{}, ErrNoMessages
	}
	allSystem := true
	for _, message := range messages {
		if message.Role != ai.RoleSystem {
			allSystem = false
			break
		}
	}
	if allSystem {
		return Result{}, ErrNoMessages
	}
	last := messages[len(messages)-1]
	if last.Role != ai.RoleAssistant {
		return a.runMessages(ctx, nil, true, false, false)
	}
	return a.runMessages(ctx, nil, false, false, true)
}

// Run is a convenience alias for Prompt.
func (a *Agent) Run(ctx context.Context, input string, images ...ai.ImageContent) (Result, error) {
	return a.Prompt(ctx, input, images...)
}

func (a *Agent) runMessages(
	ctx context.Context,
	prompts []ai.Message,
	continuation bool,
	skipInitialSteering bool,
	continueFromAssistant bool,
) (result Result, err error) {
	a.mu.Lock()
	if a.active != nil {
		a.mu.Unlock()
		return Result{}, ErrBusy
	}
	if continueFromAssistant {
		steering := peekQueue(a.steeringQueue, a.steeringMode)
		if len(steering) > 0 {
			a.steeringQueue = a.steeringQueue[len(steering):]
			prompts = steering
			continuation = false
			skipInitialSteering = true
		} else {
			followUp := peekQueue(a.followUpQueue, a.followUpMode)
			if len(followUp) > 0 {
				a.followUpQueue = a.followUpQueue[len(followUp):]
				prompts = followUp
				continuation = false
			} else {
				a.mu.Unlock()
				return Result{}, ErrCannotContinue
			}
		}
	}
	runCtx, cancel := context.WithCancel(ctx)
	active := &activeRun{cancel: cancel, done: make(chan struct{})}
	a.active = active
	a.isStreaming = true
	a.streamingMessage = nil
	a.errorMessage = ""
	a.mu.Unlock()

	defer func() {
		cancel()
		a.mu.Lock()
		a.isStreaming = false
		a.streamingMessage = nil
		a.pendingToolCalls = make(map[string]struct{})
		a.active = nil
		close(active.done)
		a.mu.Unlock()
	}()

	return a.runLoop(runCtx, prompts, continuation, skipInitialSteering)
}

// Abort cancels the active run. It is safe to call while idle.
func (a *Agent) Abort() {
	a.mu.Lock()
	active := a.active
	a.mu.Unlock()
	if active != nil {
		active.cancel()
	}
}

// WaitForIdle waits for the active run and its event listeners to finish.
func (a *Agent) WaitForIdle(ctx context.Context) error {
	a.mu.Lock()
	active := a.active
	a.mu.Unlock()
	if active == nil {
		return nil
	}
	select {
	case <-active.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Reset clears the transcript and queues, restoring the initial system prompt.
func (a *Agent) Reset() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.active != nil {
		return ErrBusy
	}
	a.messages = nil
	if a.baselineSystem != nil {
		a.messages = []ai.Message{ai.CloneMessage(*a.baselineSystem)}
	}
	a.steeringQueue = nil
	a.followUpQueue = nil
	a.isStreaming = false
	a.streamingMessage = nil
	a.pendingToolCalls = make(map[string]struct{})
	a.errorMessage = ""
	return nil
}

// State returns a deep-copied snapshot of the current conversation state.
func (a *Agent) State() State {
	a.mu.Lock()
	defer a.mu.Unlock()
	state := State{
		SystemPrompt:     systemPromptFromMessages(a.messages),
		Model:            a.model,
		ThinkingLevel:    a.thinkingLevel,
		Tools:            a.tools.listTools(),
		Messages:         ai.CloneMessages(a.messages),
		IsStreaming:      a.isStreaming,
		ErrorMessage:     a.errorMessage,
		PendingToolCalls: make([]string, 0, len(a.pendingToolCalls)),
	}
	if a.streamingMessage != nil {
		message := ai.CloneMessage(*a.streamingMessage)
		state.StreamingMessage = &message
	}
	for id := range a.pendingToolCalls {
		state.PendingToolCalls = append(state.PendingToolCalls, id)
	}
	sort.Strings(state.PendingToolCalls)
	return state
}

// SetModel replaces the model used by future requests.
func (a *Agent) SetModel(model ai.Model) error {
	if model == nil {
		return errors.New("agent: model must not be nil")
	}
	a.mu.Lock()
	a.model = model
	a.mu.Unlock()
	return nil
}

// SetThinkingLevel changes the reasoning level for future requests.
func (a *Agent) SetThinkingLevel(level ThinkingLevel) error {
	if level == "" {
		level = ThinkingOff
	}
	if !validThinkingLevel(level) {
		return fmt.Errorf("agent: unsupported thinking level %q", level)
	}
	a.mu.Lock()
	a.thinkingLevel = level
	a.mu.Unlock()
	return nil
}

// SetRequestOptions replaces default request options for future requests.
func (a *Agent) SetRequestOptions(options ai.RequestOptions) {
	a.mu.Lock()
	a.requestOptions = options
	a.mu.Unlock()
}

// SetTools replaces the executable tools for future runs.
func (a *Agent) SetTools(tools []Tool) error {
	registry, err := newToolRegistry(tools)
	if err != nil {
		return err
	}
	a.mu.Lock()
	a.tools = registry
	a.mu.Unlock()
	return nil
}

// SetMessages replaces the transcript. The initial system prompt is not
// reinserted automatically; use Reset to restore the configured baseline.
func (a *Agent) SetMessages(messages []ai.Message) {
	a.mu.Lock()
	a.messages = ai.CloneMessages(messages)
	a.mu.Unlock()
}

// Messages returns a deep copy of the transcript.
func (a *Agent) Messages() []ai.Message {
	a.mu.Lock()
	defer a.mu.Unlock()
	return ai.CloneMessages(a.messages)
}

// Steer queues a message for injection after the current assistant turn.
func (a *Agent) Steer(message ai.Message) {
	a.mu.Lock()
	a.steeringQueue = append(a.steeringQueue, ai.CloneMessage(message))
	a.mu.Unlock()
}

// FollowUp queues a message to run after the agent would otherwise stop.
func (a *Agent) FollowUp(message ai.Message) {
	a.mu.Lock()
	a.followUpQueue = append(a.followUpQueue, ai.CloneMessage(message))
	a.mu.Unlock()
}

// ClearSteeringQueue removes queued steering messages.
func (a *Agent) ClearSteeringQueue() {
	a.mu.Lock()
	a.steeringQueue = nil
	a.mu.Unlock()
}

// ClearFollowUpQueue removes queued follow-up messages.
func (a *Agent) ClearFollowUpQueue() {
	a.mu.Lock()
	a.followUpQueue = nil
	a.mu.Unlock()
}

// ClearAllQueues removes both queued steering and follow-up messages.
func (a *Agent) ClearAllQueues() {
	a.ClearSteeringQueue()
	a.ClearFollowUpQueue()
}

// HasQueuedMessages reports whether either input queue is non-empty.
func (a *Agent) HasQueuedMessages() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return len(a.steeringQueue) > 0 || len(a.followUpQueue) > 0
}

// PeekQueuedMessages previews the next messages without consuming them.
func (a *Agent) PeekQueuedMessages() []ai.Message {
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.steeringQueue) > 0 {
		return peekQueue(a.steeringQueue, a.steeringMode)
	}
	return peekQueue(a.followUpQueue, a.followUpMode)
}

func (a *Agent) drainSteering() []ai.Message {
	a.mu.Lock()
	defer a.mu.Unlock()
	drained := peekQueue(a.steeringQueue, a.steeringMode)
	a.steeringQueue = a.steeringQueue[len(drained):]
	return drained
}

func (a *Agent) drainFollowUp() []ai.Message {
	a.mu.Lock()
	defer a.mu.Unlock()
	drained := peekQueue(a.followUpQueue, a.followUpMode)
	a.followUpQueue = a.followUpQueue[len(drained):]
	return drained
}

func peekQueue(queue []ai.Message, mode QueueMode) []ai.Message {
	if len(queue) == 0 {
		return nil
	}
	count := 1
	if mode == QueueAll {
		count = len(queue)
	}
	return ai.CloneMessages(queue[:count])
}

func (a *Agent) contextSnapshot() AgentContext {
	a.mu.Lock()
	defer a.mu.Unlock()
	return AgentContext{
		Messages: ai.CloneMessages(a.messages),
		Tools:    a.tools.listTools(),
	}
}

func (a *Agent) currentModel() ai.Model {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.model
}

func (a *Agent) currentThinkingLevel() ThinkingLevel {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.thinkingLevel
}

func (a *Agent) requestOptionsFor(level ThinkingLevel) ai.RequestOptions {
	a.mu.Lock()
	defer a.mu.Unlock()
	options := a.requestOptions
	if a.sessionID != "" {
		options.SessionID = a.sessionID
	}
	if level != ThinkingOff {
		if options.ReasoningEffort == "" {
			options.ReasoningEffort = string(level)
		}
		if options.ThinkingBudgetTokens == 0 {
			options.ThinkingBudgetTokens = a.thinkingBudgets[level]
		}
	}
	return options
}

func (a *Agent) emit(event Event) {
	a.eventMu.Lock()
	defer a.eventMu.Unlock()

	a.mu.Lock()
	switch event.Type {
	case EventMessageStart:
		if event.Message != nil {
			message := ai.CloneMessage(*event.Message)
			a.streamingMessage = &message
		}
	case EventMessageUpdate, EventMessageDelta, EventThinkingStart, EventThinkingDelta, EventThinkingEnd:
		if event.Message != nil {
			message := ai.CloneMessage(*event.Message)
			a.streamingMessage = &message
		}
	case EventMessageEnd:
		if event.Message != nil {
			a.messages = append(a.messages, ai.CloneMessage(*event.Message))
		}
		a.streamingMessage = nil
	case EventToolStart:
		if event.ToolCall != nil {
			a.pendingToolCalls[event.ToolCall.ID] = struct{}{}
		}
	case EventToolEnd:
		if event.ToolCall != nil {
			delete(a.pendingToolCalls, event.ToolCall.ID)
		}
	case EventTurnEnd:
		if event.Message != nil && event.Message.Role == ai.RoleAssistant && event.Message.ErrorMessage != "" {
			a.errorMessage = event.Message.ErrorMessage
		}
	case EventAgentError:
		if event.Error != "" {
			a.errorMessage = event.Error
		}
	case EventAgentEnd:
		a.streamingMessage = nil
	}
	ids := make([]uint64, 0, len(a.listeners))
	for id := range a.listeners {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	listeners := make([]func(Event), 0, len(ids))
	for _, id := range ids {
		listeners = append(listeners, a.listeners[id])
	}
	a.mu.Unlock()

	for _, listener := range listeners {
		listener(event)
	}
}

func (a *Agent) emitMessage(message ai.Message, step int) {
	start := ai.CloneMessage(message)
	a.emit(Event{Type: EventMessageStart, Step: step, Message: &start})
	end := ai.CloneMessage(message)
	a.emit(Event{Type: EventMessageEnd, Step: step, Message: &end})
}

func (a *Agent) runLoop(
	ctx context.Context,
	prompts []ai.Message,
	continuation bool,
	skipInitialSteering bool,
) (Result, error) {
	result := Result{}
	newMessages := make([]ai.Message, 0, len(prompts)+2)
	finish := func() (Result, error) {
		result.NewMessages = ai.CloneMessages(newMessages)
		return a.finishResult(result)
	}
	fail := func(err error, synthesize bool) (Result, error) {
		result.NewMessages = ai.CloneMessages(newMessages)
		return a.failResult(result, err, synthesize)
	}
	currentContext := a.contextSnapshot()
	model := a.currentModel()
	thinking := a.currentThinkingLevel()

	a.emit(Event{Type: EventAgentStart})

	var pending []ai.Message
	if !skipInitialSteering {
		pending = a.drainSteering()
	}
	first := true
	var lastTurn *TurnContext
	explicitContinuation := false
	var assistantEmitted bool

	for {
		assistantEmitted = false
		if first {
			a.emit(Event{Type: EventTurnStart, Step: 1})
			if !continuation {
				for _, message := range prompts {
					a.emitMessage(message, 1)
					currentContext.Messages = append(currentContext.Messages, ai.CloneMessage(message))
					newMessages = append(newMessages, ai.CloneMessage(message))
				}
			}
			for _, message := range pending {
				a.emitMessage(message, 1)
				currentContext.Messages = append(currentContext.Messages, ai.CloneMessage(message))
				newMessages = append(newMessages, ai.CloneMessage(message))
			}
			pending = nil
			first = false
		} else {
			var preparedMessages []ai.Message
			if lastTurn != nil && a.prepareNextTurn != nil {
				update, err := a.prepareNextTurn(ctx, cloneTurnContext(*lastTurn))
				if err != nil {
					return fail(fmt.Errorf("agent: prepare next turn: %w", err), !assistantEmitted)
				}
				preparedMessages, err = a.applyTurnUpdate(update, &currentContext, &model, &thinking)
				if err != nil {
					return fail(err, !assistantEmitted)
				}
			}
			if lastTurn != nil && len(pending) == 0 {
				pending = a.drainSteering()
			}
			a.emit(Event{Type: EventTurnStart, Step: result.Steps + 1})
			for _, message := range append(preparedMessages, pending...) {
				a.emitMessage(message, result.Steps+1)
				currentContext.Messages = append(currentContext.Messages, ai.CloneMessage(message))
				newMessages = append(newMessages, ai.CloneMessage(message))
			}
			pending = nil
		}

		if err := ctx.Err(); err != nil {
			return fail(err, !assistantEmitted)
		}
		if a.maxSteps > 0 && result.Steps >= a.maxSteps {
			return fail(ErrMaxSteps, !assistantEmitted)
		}

		requestContext := cloneAgentContext(currentContext)
		if a.prepareRequest != nil {
			update, err := a.prepareRequest(ctx, PrepareRequestContext{
				Context:       cloneAgentContext(requestContext),
				Model:         model,
				ThinkingLevel: thinking,
			})
			if err != nil {
				return fail(fmt.Errorf("agent: prepare request: %w", err), !assistantEmitted)
			}
			if err := a.applyRequestUpdate(update, &requestContext, &model, &thinking); err != nil {
				return fail(err, !assistantEmitted)
			}
		}
		// PrepareRequest replacements apply to this and later requests in the run.
		// The persistent transcript still only receives messages emitted below.
		currentContext = cloneAgentContext(requestContext)

		requestMessages := ai.CloneMessages(requestContext.Messages)
		if a.transformMessages != nil {
			transformed, err := a.transformMessages(ctx, ai.CloneMessages(requestMessages))
			if err != nil {
				return fail(fmt.Errorf("agent: transform messages: %w", err), !assistantEmitted)
			}
			requestMessages = ai.CloneMessages(transformed)
		}
		specs, err := toolSpecs(requestContext.Tools)
		if err != nil {
			return fail(err, !assistantEmitted)
		}

		result.Steps++
		assistant, stopReason, usage, streamErr := a.runModelTurn(
			ctx,
			result.Steps,
			model,
			requestMessages,
			specs,
			a.requestOptionsFor(thinking),
		)
		assistantEmitted = true
		result.Usage.Add(usage)
		result.StopReason = stopReason
		result.Answer = assistant.Text()
		if assistant.ErrorMessage != "" {
			result.ErrorMessage = assistant.ErrorMessage
		}
		currentContext.Messages = append(currentContext.Messages, ai.CloneMessage(assistant))
		newMessages = append(newMessages, ai.CloneMessage(assistant))

		if streamErr != nil || stopReason == ai.StopReasonError || stopReason == ai.StopReasonAborted {
			turn := TurnContext{
				Message:     ai.CloneMessage(assistant),
				ToolResults: nil,
				Context:     cloneAgentContext(currentContext),
				NewMessages: ai.CloneMessages(newMessages),
			}
			if a.finishTurn != nil {
				if _, err := a.finishTurn(ctx, cloneTurnContext(turn)); err != nil {
					a.emit(Event{Type: EventTurnEnd, Step: result.Steps, Message: &turn.Message})
					return fail(fmt.Errorf("agent: finish turn: %w", err), false)
				}
			}
			lastTurn = &turn
			a.emit(Event{Type: EventTurnEnd, Step: result.Steps, Message: &turn.Message})
			if streamErr != nil {
				return fail(&ModelError{Message: assistant, Err: streamErr}, false)
			}
			if stopReason == ai.StopReasonAborted {
				if err := ctx.Err(); err != nil {
					return fail(err, false)
				}
				return fail(ErrAborted, false)
			}
			return fail(&ModelError{Message: assistant}, false)
		}

		calls := assistant.ToolCalls()
		var toolResults []ai.Message
		terminate := false
		if len(calls) > 0 {
			if stopReason == ai.StopReasonLength {
				toolResults = a.failTruncatedToolCalls(calls, result.Steps)
			} else {
				toolResults, terminate = a.executeToolCalls(ctx, currentContext, assistant, calls, result.Steps)
			}
			for _, toolResult := range toolResults {
				a.emitMessage(toolResult, result.Steps)
				currentContext.Messages = append(currentContext.Messages, ai.CloneMessage(toolResult))
				newMessages = append(newMessages, ai.CloneMessage(toolResult))
			}
		}

		turn := TurnContext{
			Message:     ai.CloneMessage(assistant),
			ToolResults: ai.CloneMessages(toolResults),
			Context:     cloneAgentContext(currentContext),
			NewMessages: ai.CloneMessages(newMessages),
		}
		decision := TurnDefault
		if a.finishTurn != nil {
			decision, err = a.finishTurn(ctx, cloneTurnContext(turn))
			if err != nil {
				a.emit(Event{
					Type:        EventTurnEnd,
					Step:        result.Steps,
					Message:     &turn.Message,
					ToolResults: cloneToolResultSlice(turn.ToolResults),
				})
				return fail(fmt.Errorf("agent: finish turn: %w", err), false)
			}
		}
		lastTurn = &turn
		a.emit(Event{
			Type:        EventTurnEnd,
			Step:        result.Steps,
			Message:     &turn.Message,
			ToolResults: cloneToolResultSlice(turn.ToolResults),
		})

		if decision == TurnEnd {
			return finish()
		}
		explicitContinuation = decision == TurnContinue
		if len(calls) > 0 && !terminate {
			explicitContinuation = false
			continue
		}
		pending = a.drainSteering()
		if len(pending) > 0 {
			explicitContinuation = false
			continue
		}
		followUp := a.drainFollowUp()
		if len(followUp) > 0 {
			pending = followUp
			explicitContinuation = false
			continue
		}
		if explicitContinuation {
			explicitContinuation = false
			continue
		}
		return finish()
	}
}

func (a *Agent) applyRequestUpdate(
	update *RequestUpdate,
	context *AgentContext,
	model *ai.Model,
	thinking *ThinkingLevel,
) error {
	if update == nil {
		return nil
	}
	if update.Context != nil {
		*context = cloneAgentContext(*update.Context)
	}
	if update.Model != nil {
		*model = update.Model
	}
	if update.ThinkingLevel != nil {
		if !validThinkingLevel(*update.ThinkingLevel) {
			return fmt.Errorf("agent: unsupported thinking level %q", *update.ThinkingLevel)
		}
		*thinking = *update.ThinkingLevel
	}
	return nil
}

func (a *Agent) applyTurnUpdate(
	update *TurnUpdate,
	context *AgentContext,
	model *ai.Model,
	thinking *ThinkingLevel,
) ([]ai.Message, error) {
	if update == nil {
		return nil, nil
	}
	if update.Context != nil {
		*context = cloneAgentContext(*update.Context)
	}
	if update.Model != nil {
		*model = update.Model
	}
	if update.ThinkingLevel != nil {
		if !validThinkingLevel(*update.ThinkingLevel) {
			return nil, fmt.Errorf("agent: unsupported thinking level %q", *update.ThinkingLevel)
		}
		*thinking = *update.ThinkingLevel
	}
	return ai.CloneMessages(update.Messages), nil
}

func (a *Agent) finishResult(result Result) (Result, error) {
	result.Messages = a.messagesCopy()
	eventResult := cloneResult(result)
	a.emit(Event{Type: EventAgentEnd, Step: result.Steps, Result: &eventResult})
	return result, nil
}

func (a *Agent) failResult(result Result, err error, synthesize bool) (Result, error) {
	if synthesize {
		stopReason := ai.StopReasonError
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			stopReason = ai.StopReasonAborted
		}
		message := failedAssistantMessage(a.currentModel().Info(), time.Now(), stopReason, err)
		step := result.Steps + 1
		a.emitMessage(message, step)
		result.NewMessages = append(result.NewMessages, ai.CloneMessage(message))
		end := ai.CloneMessage(message)
		a.emit(Event{Type: EventTurnEnd, Step: step, Message: &end})
		result.StopReason = stopReason
		if result.ErrorMessage == "" {
			result.ErrorMessage = err.Error()
		}
	}
	result.Messages = a.messagesCopy()
	if result.ErrorMessage == "" && err != nil {
		result.ErrorMessage = err.Error()
	}
	eventResult := cloneResult(result)
	a.emit(Event{
		Type:   EventAgentError,
		Step:   result.Steps,
		Result: &eventResult,
		Err:    err,
		Error:  err.Error(),
	})
	eventResult = cloneResult(result)
	a.emit(Event{Type: EventAgentEnd, Step: result.Steps, Result: &eventResult})
	return result, err
}

func (a *Agent) messagesCopy() []ai.Message {
	a.mu.Lock()
	defer a.mu.Unlock()
	return ai.CloneMessages(a.messages)
}

func cloneResult(result Result) Result {
	result.Messages = ai.CloneMessages(result.Messages)
	return result
}

func cloneAgentContext(context AgentContext) AgentContext {
	return AgentContext{
		Messages: ai.CloneMessages(context.Messages),
		Tools:    append([]Tool(nil), context.Tools...),
	}
}

func cloneTurnContext(turn TurnContext) TurnContext {
	return TurnContext{
		Message:     ai.CloneMessage(turn.Message),
		ToolResults: ai.CloneMessages(turn.ToolResults),
		Context:     cloneAgentContext(turn.Context),
		NewMessages: ai.CloneMessages(turn.NewMessages),
	}
}

func cloneToolResultSlice(messages []ai.Message) []ToolResult {
	results := make([]ToolResult, 0, len(messages))
	for _, message := range messages {
		result := ToolResult{
			Content:           ai.CloneContentBlocks(message.Content),
			Details:           ai.CloneRawMessage(message.Details),
			StructuredContent: ai.CloneRawMessage(message.StructuredContent),
			IsError:           message.IsError,
		}
		if message.Usage != nil {
			usage := *message.Usage
			result.Usage = &usage
		}
		results = append(results, result)
	}
	return results
}

func systemPromptFromMessages(messages []ai.Message) string {
	for _, message := range messages {
		if message.Role == ai.RoleSystem {
			return message.Text()
		}
	}
	return ""
}
