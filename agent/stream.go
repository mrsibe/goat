package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/mrsibe/goat/ai"
)

func (a *Agent) runModelTurn(
	ctx context.Context,
	step int,
	model ai.Model,
	messages []ai.Message,
	tools []ai.ToolSpec,
	options ai.RequestOptions,
	thinking ai.ThinkingLevel,
) (*ai.AssistantMessage, ai.StopReason, ai.Usage, error) {
	info := model.Info()
	started := time.Now()
	stream, err := model.Stream(ctx, ai.Request{
		Messages: ai.CloneMessages(messages),
		Tools:    append([]ai.ToolSpec(nil), tools...),
		Options:  options,
	})
	if err != nil {
		assistant := failedAssistantMessage(info, started, ai.StopReasonError, err)
		a.emitMessage(assistant, step)
		return assistant, ai.StopReasonError, ai.Usage{}, err
	}
	if stream == nil {
		err := errors.New("agent: model returned a nil stream")
		assistant := failedAssistantMessage(info, started, ai.StopReasonError, err)
		a.emitMessage(assistant, step)
		return assistant, ai.StopReasonError, ai.Usage{}, err
	}
	defer stream.Close()

	start := &ai.AssistantMessage{
		Provider: info.Provider,
		Model:    info.Model,
		API:      info.API,
		At:       started.UnixMilli(),
	}
	a.emit(Event{Type: EventMessageStart, Step: step, Message: ai.CloneMessage(start)})

	accumulator := newAssistantAccumulator()
	var usage ai.Usage
	var stopReason ai.StopReason
	var finalMessage *ai.AssistantMessage
	sawUsage := false
	sawDone := false

	for {
		if err := ctx.Err(); err != nil {
			assistant := finalizeAssistant(accumulator, finalMessage, info, started, ai.StopReasonAborted, usage, thinking)
			a.emit(Event{Type: EventMessageEnd, Step: step, Message: ai.CloneMessage(assistant)})
			return assistant, ai.StopReasonAborted, usage, err
		}
		event, nextErr := stream.Next()
		if errors.Is(nextErr, io.EOF) {
			break
		}
		if nextErr != nil {
			assistant := finalizeAssistant(accumulator, finalMessage, info, started, ai.StopReasonError, usage, thinking)
			assistant.ErrorMessage = nextErr.Error()
			a.emit(Event{Type: EventMessageEnd, Step: step, Message: ai.CloneMessage(assistant)})
			return assistant, ai.StopReasonError, usage, nextErr
		}

		switch event.Type {
		case ai.EventStart:
		case ai.EventTextStart:
			accumulator.ensureBlock(event.ContentIndex, ai.BlockText)
			a.emitStreamUpdate(step, info, started, event, accumulator, "", "")
		case ai.EventTextDelta:
			if event.TextDelta != "" {
				accumulator.addText(event.ContentIndex, event.TextDelta)
			}
			a.emitStreamUpdate(step, info, started, event, accumulator, event.TextDelta, "")
		case ai.EventTextEnd:
			block := accumulator.ensureBlock(event.ContentIndex, ai.BlockText)
			if block.text.Len() == 0 && event.Text != "" {
				block.text.WriteString(event.Text)
			}
			a.emitStreamUpdate(step, info, started, event, accumulator, "", "")
		case ai.EventThinkingStart:
			accumulator.ensureBlock(event.ContentIndex, ai.BlockThinking)
			a.emitStreamUpdate(step, info, started, event, accumulator, "", "")
		case ai.EventThinkingDelta:
			if event.ThinkingDelta != "" {
				accumulator.addThinking(event.ContentIndex, event.ThinkingDelta)
			}
			a.emitStreamUpdate(step, info, started, event, accumulator, "", event.ThinkingDelta)
		case ai.EventThinkingEnd:
			block := accumulator.ensureBlock(event.ContentIndex, ai.BlockThinking)
			if block.thinking.Len() == 0 && event.Thinking != "" {
				block.thinking.WriteString(event.Thinking)
			}
			a.emitStreamUpdate(step, info, started, event, accumulator, "", event.Thinking)
		case ai.EventToolCallStart:
			if event.ToolCall == nil {
				err := errors.New("agent: tool call start is missing its payload")
				assistant := finalizeAssistant(accumulator, finalMessage, info, started, ai.StopReasonError, usage, thinking)
				assistant.ErrorMessage = err.Error()
				a.emit(Event{Type: EventMessageEnd, Step: step, Message: ai.CloneMessage(assistant)})
				return assistant, ai.StopReasonError, usage, err
			}
			accumulator.startToolCall(event.ContentIndex, *event.ToolCall)
			a.emitStreamUpdate(step, info, started, event, accumulator, "", "")
		case ai.EventToolCallDelta:
			if event.ToolCallDelta == nil {
				err := errors.New("agent: tool call delta is missing its payload")
				assistant := finalizeAssistant(accumulator, finalMessage, info, started, ai.StopReasonError, usage, thinking)
				assistant.ErrorMessage = err.Error()
				a.emit(Event{Type: EventMessageEnd, Step: step, Message: ai.CloneMessage(assistant)})
				return assistant, ai.StopReasonError, usage, err
			}
			if err := accumulator.addToolCall(event.ContentIndex, *event.ToolCallDelta); err != nil {
				assistant := finalizeAssistant(accumulator, finalMessage, info, started, ai.StopReasonError, usage, thinking)
				assistant.ErrorMessage = err.Error()
				a.emit(Event{Type: EventMessageEnd, Step: step, Message: ai.CloneMessage(assistant)})
				return assistant, ai.StopReasonError, usage, err
			}
			a.emitStreamUpdate(step, info, started, event, accumulator, "", "")
		case ai.EventToolCallEnd:
			if event.ToolCall == nil {
				err := errors.New("agent: tool call end is missing its payload")
				assistant := finalizeAssistant(accumulator, finalMessage, info, started, ai.StopReasonError, usage, thinking)
				assistant.ErrorMessage = err.Error()
				a.emit(Event{Type: EventMessageEnd, Step: step, Message: ai.CloneMessage(assistant)})
				return assistant, ai.StopReasonError, usage, err
			}
			accumulator.finishToolCall(event.ContentIndex, *event.ToolCall)
			a.emitStreamUpdate(step, info, started, event, accumulator, "", "")
		case ai.EventUsage:
			if event.Usage != nil {
				usage.Add(*event.Usage)
				sawUsage = true
			}
		case ai.EventDone:
			sawDone = true
			stopReason = event.StopReason
			if event.Message != nil {
				finalMessage = asAssistantMessage(event.Message)
			} else if event.Partial != nil {
				finalMessage = asAssistantMessage(event.Partial)
			}
			if !sawUsage && event.Usage != nil {
				usage = *event.Usage
			}
		case ai.EventError:
			if event.StopReason == "" {
				event.StopReason = ai.StopReasonError
			}
			if event.Message != nil {
				finalMessage = asAssistantMessage(event.Message)
			} else if event.Partial != nil {
				finalMessage = asAssistantMessage(event.Partial)
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
				streamErr = errors.New("agent: model stream reported an error")
			}
			assistant := finalizeAssistant(accumulator, finalMessage, info, started, event.StopReason, usage, thinking)
			if assistant.ErrorMessage == "" {
				assistant.ErrorMessage = streamErr.Error()
			}
			a.emit(Event{Type: EventMessageEnd, Step: step, Message: ai.CloneMessage(assistant)})
			return assistant, event.StopReason, usage, streamErr
		default:
			err := fmt.Errorf("agent: unknown model event type %q", event.Type)
			assistant := finalizeAssistant(accumulator, finalMessage, info, started, ai.StopReasonError, usage, thinking)
			assistant.ErrorMessage = err.Error()
			a.emit(Event{Type: EventMessageEnd, Step: step, Message: ai.CloneMessage(assistant)})
			return assistant, ai.StopReasonError, usage, err
		}
	}

	if !sawDone {
		stopReason = ai.StopReasonStop
		if accumulator.hasToolCalls() {
			stopReason = ai.StopReasonToolUse
		}
	}
	assistant := finalizeAssistant(accumulator, finalMessage, info, started, stopReason, usage, thinking)
	a.emit(Event{Type: EventMessageEnd, Step: step, Message: ai.CloneMessage(assistant)})
	return assistant, stopReason, usage, nil
}

func failedAssistantMessage(info ai.ModelInfo, started time.Time, stopReason ai.StopReason, err error) *ai.AssistantMessage {
	return &ai.AssistantMessage{
		Content:      []ai.ContentBlock{{Type: ai.BlockText, Text: ""}},
		Provider:     info.Provider,
		Model:        info.Model,
		API:          info.API,
		At:           started.UnixMilli(),
		DurationMs:   time.Since(started).Milliseconds(),
		StopReason:   stopReason,
		ErrorMessage: err.Error(),
	}
}

// asAssistantMessage projects any message onto an assistant message.
func asAssistantMessage(message ai.Message) *ai.AssistantMessage {
	if assistant, ok := message.(*ai.AssistantMessage); ok {
		cloned := *assistant
		cloned.Content = ai.CloneContentBlocks(assistant.Content)
		return &cloned
	}
	return &ai.AssistantMessage{Content: ai.CloneContentBlocks(contentOf(message))}
}

func contentOf(message ai.Message) []ai.ContentBlock {
	switch typed := message.(type) {
	case *ai.SystemMessage:
		if typed.Content == "" {
			return nil
		}
		return []ai.ContentBlock{{Type: ai.BlockText, Text: typed.Content}}
	case *ai.UserMessage:
		return typed.Content
	case *ai.AssistantMessage:
		return typed.Content
	case *ai.ToolResultMessage:
		return typed.Content
	default:
		return nil
	}
}

func finalizeAssistant(
	accumulator *assistantAccumulator,
	finalMessage *ai.AssistantMessage,
	info ai.ModelInfo,
	started time.Time,
	stopReason ai.StopReason,
	usage ai.Usage,
	thinking ai.ThinkingLevel,
) *ai.AssistantMessage {
	var message *ai.AssistantMessage
	if finalMessage != nil {
		message = asAssistantMessage(finalMessage)
	} else {
		built, err := accumulator.message()
		message = built
		if err != nil {
			message = &ai.AssistantMessage{
				StopReason:   ai.StopReasonError,
				ErrorMessage: err.Error(),
			}
		}
	}
	if message.At == 0 {
		message.At = started.UnixMilli()
	}
	if message.DurationMs == 0 {
		message.DurationMs = time.Since(started).Milliseconds()
	}
	if message.Provider == "" {
		message.Provider = info.Provider
	}
	if message.Model == "" {
		message.Model = info.Model
	}
	if message.API == "" {
		message.API = info.API
	}
	if message.StopReason == "" {
		message.StopReason = stopReason
	}
	if message.Usage == (ai.Usage{}) {
		message.Usage = usage
	}
	if message.ThinkingLevel == "" {
		message.ThinkingLevel = thinking
	}
	return message
}

func cloneMessageValue(message ai.Message) ai.Message {
	if message == nil {
		return nil
	}
	return ai.CloneMessage(message)
}

func cloneModelEvent(event ai.Event) ai.Event {
	event.Partial = cloneMessageValue(event.Partial)
	event.Message = cloneMessageValue(event.Message)
	if event.ToolCall != nil {
		call := ai.CloneToolCall(*event.ToolCall)
		event.ToolCall = &call
	}
	if event.ToolCallDelta != nil {
		delta := *event.ToolCallDelta
		event.ToolCallDelta = &delta
	}
	if event.Usage != nil {
		usage := *event.Usage
		event.Usage = &usage
	}
	return event
}

func (a *Agent) emitStreamUpdate(
	step int,
	info ai.ModelInfo,
	started time.Time,
	modelEvent ai.Event,
	accumulator *assistantAccumulator,
	text string,
	thinking string,
) {
	partial := asAssistantMessage(modelEvent.Partial)
	if modelEvent.Partial == nil {
		partial = accumulator.partialMessage(info, started)
	}
	providerEvent := cloneModelEvent(modelEvent)
	providerEvent.Partial = ai.CloneMessage(partial)
	a.emit(Event{
		Type:                  EventMessageUpdate,
		Step:                  step,
		Text:                  text,
		Thinking:              thinking,
		Message:               ai.CloneMessage(partial),
		AssistantMessageEvent: &providerEvent,
	})

	switch modelEvent.Type {
	case ai.EventTextDelta:
		if text != "" {
			a.emit(Event{
				Type:    EventMessageDelta,
				Step:    step,
				Text:    text,
				Message: ai.CloneMessage(partial),
			})
		}
	case ai.EventThinkingStart:
		a.emit(Event{
			Type:    EventThinkingStart,
			Step:    step,
			Message: ai.CloneMessage(partial),
		})
	case ai.EventThinkingDelta:
		if thinking != "" {
			a.emit(Event{
				Type:     EventThinkingDelta,
				Step:     step,
				Thinking: thinking,
				Message:  ai.CloneMessage(partial),
			})
		}
	case ai.EventThinkingEnd:
		a.emit(Event{
			Type:     EventThinkingEnd,
			Step:     step,
			Thinking: thinking,
			Message:  ai.CloneMessage(partial),
		})
	}
}

type toolCallAccumulator struct {
	id               string
	itemID           string
	name             string
	arguments        strings.Builder
	argumentsError   string
	thoughtSignature string
	namespace        string
	final            *ai.ToolCall
}

type assistantBlock struct {
	kind     ai.BlockType
	text     strings.Builder
	thinking strings.Builder
	call     *toolCallAccumulator
}

type assistantBlockKey struct {
	index int
	kind  ai.BlockType
}

type assistantAccumulator struct {
	blocks map[assistantBlockKey]*assistantBlock
	order  []assistantBlockKey
}

func newAssistantAccumulator() *assistantAccumulator {
	return &assistantAccumulator{blocks: make(map[assistantBlockKey]*assistantBlock)}
}

func (a *assistantAccumulator) addText(index int, text string) {
	block := a.ensureBlock(index, ai.BlockText)
	block.text.WriteString(text)
}

func (a *assistantAccumulator) addThinking(index int, thinking string) {
	block := a.ensureBlock(index, ai.BlockThinking)
	block.thinking.WriteString(thinking)
}

func (a *assistantAccumulator) startToolCall(index int, call ai.ToolCall) {
	acc := a.ensureToolCall(index)
	if call.ID != "" {
		acc.id = call.ID
	}
	if call.ItemID != "" {
		acc.itemID = call.ItemID
	}
	if call.Name != "" {
		acc.name = call.Name
	}
	if call.ThoughtSignature != "" {
		acc.thoughtSignature = call.ThoughtSignature
	}
	if call.Namespace != "" {
		acc.namespace = call.Namespace
	}
}

func (a *assistantAccumulator) finishToolCall(index int, call ai.ToolCall) {
	acc := a.ensureToolCall(index)
	if call.ID != "" {
		acc.id = call.ID
	}
	if call.ItemID != "" {
		acc.itemID = call.ItemID
	}
	if call.Name != "" {
		acc.name = call.Name
	}
	if call.ArgumentsError != "" {
		acc.argumentsError = call.ArgumentsError
	}
	if call.ThoughtSignature != "" {
		acc.thoughtSignature = call.ThoughtSignature
	}
	if call.Namespace != "" {
		acc.namespace = call.Namespace
	}
	cloned := ai.CloneToolCall(call)
	acc.final = &cloned
}

func (a *assistantAccumulator) addToolCall(index int, delta ai.ToolCallDelta) error {
	if index < 0 {
		return errors.New("agent: tool call index must not be negative")
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
	block := a.ensureBlock(index, ai.BlockToolCall)
	if block.call == nil {
		block.call = &toolCallAccumulator{}
	}
	return block.call
}

func (a *assistantAccumulator) hasToolCalls() bool {
	for _, block := range a.blocks {
		if block.kind == ai.BlockToolCall {
			return true
		}
	}
	return false
}

func (a *assistantAccumulator) partialMessage(info ai.ModelInfo, started time.Time) *ai.AssistantMessage {
	message := &ai.AssistantMessage{
		Provider: info.Provider,
		Model:    info.Model,
		API:      info.API,
		At:       started.UnixMilli(),
	}
	for _, key := range a.order {
		block := a.blocks[key]
		switch block.kind {
		case ai.BlockText:
			message.Content = append(message.Content, ai.ContentBlock{
				Type: ai.BlockText,
				Text: block.text.String(),
			})
		case ai.BlockThinking:
			message.Content = append(message.Content, ai.ContentBlock{
				Type:     ai.BlockThinking,
				Thinking: block.thinking.String(),
			})
		case ai.BlockToolCall:
			if block.call == nil {
				continue
			}
			call := block.call.partial(key.index)
			message.Content = append(message.Content, ai.ContentBlock{
				Type:     ai.BlockToolCall,
				ToolCall: &call,
			})
		}
	}
	return message
}

func (a *toolCallAccumulator) partial(index int) ai.ToolCall {
	id := a.id
	itemID := a.itemID
	name := a.name
	arguments := a.arguments.String()
	argumentsError := a.argumentsError
	thoughtSignature := a.thoughtSignature
	namespace := a.namespace
	if a.final != nil {
		if a.final.ID != "" {
			id = a.final.ID
		}
		if a.final.ItemID != "" {
			itemID = a.final.ItemID
		}
		if a.final.Name != "" {
			name = a.final.Name
		}
		if len(a.final.Arguments) > 0 {
			arguments = string(a.final.Arguments)
		}
		if a.final.ArgumentsError != "" {
			argumentsError = a.final.ArgumentsError
		}
		if a.final.ThoughtSignature != "" {
			thoughtSignature = a.final.ThoughtSignature
		}
		if a.final.Namespace != "" {
			namespace = a.final.Namespace
		}
	}
	if id == "" {
		id = fmt.Sprintf("call_%d", index)
	}
	if strings.TrimSpace(arguments) == "" {
		arguments = "{}"
	}
	return ai.ToolCall{
		ID:               id,
		ItemID:           itemID,
		Name:             name,
		Arguments:        json.RawMessage(arguments),
		ArgumentsError:   argumentsError,
		ThoughtSignature: thoughtSignature,
		Namespace:        namespace,
	}
}

func (a *assistantAccumulator) message() (*ai.AssistantMessage, error) {
	message := &ai.AssistantMessage{}
	seenIDs := make(map[string]struct{})
	for _, key := range a.order {
		block := a.blocks[key]
		switch block.kind {
		case ai.BlockText:
			if block.text.Len() > 0 {
				message.Content = append(message.Content, ai.ContentBlock{
					Type: ai.BlockText,
					Text: block.text.String(),
				})
			}
			continue
		case ai.BlockThinking:
			if block.thinking.Len() > 0 {
				message.Content = append(message.Content, ai.ContentBlock{
					Type:     ai.BlockThinking,
					Thinking: block.thinking.String(),
				})
			}
			continue
		case ai.BlockToolCall:
		default:
			continue
		}
		call := block.call
		if call == nil {
			continue
		}
		id := call.id
		itemID := call.itemID
		name := call.name
		arguments := call.arguments.String()
		argumentsError := call.argumentsError
		thoughtSignature := call.thoughtSignature
		namespace := call.namespace
		if call.final != nil {
			if call.final.ID != "" {
				id = call.final.ID
			}
			if call.final.ItemID != "" {
				itemID = call.final.ItemID
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
			if call.final.ThoughtSignature != "" {
				thoughtSignature = call.final.ThoughtSignature
			}
			if call.final.Namespace != "" {
				namespace = call.final.Namespace
			}
		}
		if name == "" {
			return nil, fmt.Errorf("agent: tool call %d is missing a name", key.index)
		}
		if id == "" {
			id = fmt.Sprintf("call_%d", key.index)
		}
		if _, exists := seenIDs[id]; exists {
			return nil, fmt.Errorf("agent: duplicate tool call id %q", id)
		}
		seenIDs[id] = struct{}{}
		if strings.TrimSpace(arguments) == "" {
			arguments = "{}"
		}
		toolCall := ai.ToolCall{
			ID:               id,
			ItemID:           itemID,
			Name:             name,
			Arguments:        json.RawMessage(arguments),
			ArgumentsError:   argumentsError,
			ThoughtSignature: thoughtSignature,
			Namespace:        namespace,
		}
		message.Content = append(message.Content, ai.ContentBlock{
			Type:     ai.BlockToolCall,
			ToolCall: &toolCall,
		})
	}
	return message, nil
}

func (a *assistantAccumulator) ensureBlock(index int, kind ai.BlockType) *assistantBlock {
	key := assistantBlockKey{index: index, kind: kind}
	block, exists := a.blocks[key]
	if !exists {
		block = &assistantBlock{kind: kind}
		a.blocks[key] = block
		a.order = append(a.order, key)
	}
	return block
}
