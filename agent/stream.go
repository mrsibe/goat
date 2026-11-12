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
) (ai.Message, ai.StopReason, ai.Usage, error) {
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

	start := ai.Message{
		Role:      ai.RoleAssistant,
		Provider:  info.Provider,
		Model:     info.Model,
		API:       info.API,
		Timestamp: started.UnixMilli(),
	}
	startCopy := ai.CloneMessage(start)
	a.emit(Event{Type: EventMessageStart, Step: step, Message: &startCopy})

	accumulator := newAssistantAccumulator()
	var usage ai.Usage
	var stopReason ai.StopReason
	var finalMessage *ai.Message
	sawUsage := false
	sawDone := false

	for {
		if err := ctx.Err(); err != nil {
			assistant := finalizeAssistant(accumulator, finalMessage, info, started, ai.StopReasonAborted, usage)
			end := ai.CloneMessage(assistant)
			a.emit(Event{Type: EventMessageEnd, Step: step, Message: &end})
			return assistant, ai.StopReasonAborted, usage, err
		}
		event, nextErr := stream.Next()
		if errors.Is(nextErr, io.EOF) {
			break
		}
		if nextErr != nil {
			assistant := finalizeAssistant(accumulator, finalMessage, info, started, ai.StopReasonError, usage)
			assistant.ErrorMessage = nextErr.Error()
			end := ai.CloneMessage(assistant)
			a.emit(Event{Type: EventMessageEnd, Step: step, Message: &end})
			return assistant, ai.StopReasonError, usage, nextErr
		}

		switch event.Type {
		case ai.EventStart, ai.EventTextStart, ai.EventTextEnd:
		case ai.EventTextDelta:
			if event.TextDelta != "" {
				accumulator.addText(event.ContentIndex, event.TextDelta)
				partial := cloneMessagePointer(event.Partial)
				a.emit(Event{
					Type:    EventMessageDelta,
					Step:    step,
					Text:    event.TextDelta,
					Message: partial,
				})
			}
		case ai.EventThinkingStart:
			partial := cloneMessagePointer(event.Partial)
			a.emit(Event{Type: EventThinkingStart, Step: step, Message: partial})
		case ai.EventThinkingDelta:
			if event.ThinkingDelta != "" {
				accumulator.addThinking(event.ContentIndex, event.ThinkingDelta)
				partial := cloneMessagePointer(event.Partial)
				a.emit(Event{
					Type:     EventThinkingDelta,
					Step:     step,
					Thinking: event.ThinkingDelta,
					Message:  partial,
				})
			}
		case ai.EventThinkingEnd:
			partial := cloneMessagePointer(event.Partial)
			a.emit(Event{
				Type:     EventThinkingEnd,
				Step:     step,
				Thinking: event.Thinking,
				Message:  partial,
			})
		case ai.EventToolCallStart:
			if event.ToolCall == nil {
				err := errors.New("agent: tool call start is missing its payload")
				assistant := finalizeAssistant(accumulator, finalMessage, info, started, ai.StopReasonError, usage)
				assistant.ErrorMessage = err.Error()
				end := ai.CloneMessage(assistant)
				a.emit(Event{Type: EventMessageEnd, Step: step, Message: &end})
				return assistant, ai.StopReasonError, usage, err
			}
			accumulator.startToolCall(event.ContentIndex, *event.ToolCall)
		case ai.EventToolCallDelta:
			if event.ToolCallDelta == nil {
				err := errors.New("agent: tool call delta is missing its payload")
				assistant := finalizeAssistant(accumulator, finalMessage, info, started, ai.StopReasonError, usage)
				assistant.ErrorMessage = err.Error()
				end := ai.CloneMessage(assistant)
				a.emit(Event{Type: EventMessageEnd, Step: step, Message: &end})
				return assistant, ai.StopReasonError, usage, err
			}
			if err := accumulator.addToolCall(event.ContentIndex, *event.ToolCallDelta); err != nil {
				assistant := finalizeAssistant(accumulator, finalMessage, info, started, ai.StopReasonError, usage)
				assistant.ErrorMessage = err.Error()
				end := ai.CloneMessage(assistant)
				a.emit(Event{Type: EventMessageEnd, Step: step, Message: &end})
				return assistant, ai.StopReasonError, usage, err
			}
		case ai.EventToolCallEnd:
			if event.ToolCall == nil {
				err := errors.New("agent: tool call end is missing its payload")
				assistant := finalizeAssistant(accumulator, finalMessage, info, started, ai.StopReasonError, usage)
				assistant.ErrorMessage = err.Error()
				end := ai.CloneMessage(assistant)
				a.emit(Event{Type: EventMessageEnd, Step: step, Message: &end})
				return assistant, ai.StopReasonError, usage, err
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
			} else if event.Partial != nil {
				message := ai.CloneMessage(*event.Partial)
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
				streamErr = errors.New("agent: model stream reported an error")
			}
			assistant := finalizeAssistant(accumulator, finalMessage, info, started, event.StopReason, usage)
			if assistant.ErrorMessage == "" {
				assistant.ErrorMessage = streamErr.Error()
			}
			end := ai.CloneMessage(assistant)
			a.emit(Event{Type: EventMessageEnd, Step: step, Message: &end})
			return assistant, event.StopReason, usage, streamErr
		default:
			err := fmt.Errorf("agent: unknown model event type %q", event.Type)
			assistant := finalizeAssistant(accumulator, finalMessage, info, started, ai.StopReasonError, usage)
			assistant.ErrorMessage = err.Error()
			end := ai.CloneMessage(assistant)
			a.emit(Event{Type: EventMessageEnd, Step: step, Message: &end})
			return assistant, ai.StopReasonError, usage, err
		}
	}

	if !sawDone {
		stopReason = ai.StopReasonStop
		if accumulator.hasToolCalls() {
			stopReason = ai.StopReasonToolCalls
		}
	}
	assistant := finalizeAssistant(accumulator, finalMessage, info, started, stopReason, usage)
	end := ai.CloneMessage(assistant)
	a.emit(Event{Type: EventMessageEnd, Step: step, Message: &end})
	return assistant, stopReason, usage, nil
}

func failedAssistantMessage(info ai.ModelInfo, started time.Time, stopReason ai.StopReason, err error) ai.Message {
	return ai.Message{
		Role:         ai.RoleAssistant,
		Content:      []ai.ContentBlock{{Type: ai.BlockText, Text: ""}},
		Provider:     info.Provider,
		Model:        info.Model,
		API:          info.API,
		Timestamp:    started.UnixMilli(),
		DurationMs:   time.Since(started).Milliseconds(),
		StopReason:   stopReason,
		ErrorMessage: err.Error(),
	}
}

func finalizeAssistant(
	accumulator *assistantAccumulator,
	finalMessage *ai.Message,
	info ai.ModelInfo,
	started time.Time,
	stopReason ai.StopReason,
	usage ai.Usage,
) ai.Message {
	var message ai.Message
	if finalMessage != nil {
		message = ai.CloneMessage(*finalMessage)
	} else {
		built, err := accumulator.message()
		message = built
		if err != nil {
			message = ai.Message{
				Role:         ai.RoleAssistant,
				StopReason:   ai.StopReasonError,
				ErrorMessage: err.Error(),
			}
		}
	}
	if message.Role == "" {
		message.Role = ai.RoleAssistant
	}
	if message.Timestamp == 0 {
		message.Timestamp = started.UnixMilli()
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
	if message.Usage == nil {
		message.Usage = &usage
	}
	return message
}

func cloneMessagePointer(message *ai.Message) *ai.Message {
	if message == nil {
		return nil
	}
	cloned := ai.CloneMessage(*message)
	return &cloned
}

type toolCallAccumulator struct {
	id               string
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

func (a *assistantAccumulator) message() (ai.Message, error) {
	message := ai.Message{Role: ai.RoleAssistant}
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
		name := call.name
		arguments := call.arguments.String()
		argumentsError := call.argumentsError
		thoughtSignature := call.thoughtSignature
		namespace := call.namespace
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
			if call.final.ThoughtSignature != "" {
				thoughtSignature = call.final.ThoughtSignature
			}
			if call.final.Namespace != "" {
				namespace = call.final.Namespace
			}
		}
		if name == "" {
			return ai.Message{}, fmt.Errorf("agent: tool call %d is missing a name", key.index)
		}
		if id == "" {
			id = fmt.Sprintf("call_%d", key.index)
		}
		if _, exists := seenIDs[id]; exists {
			return ai.Message{}, fmt.Errorf("agent: duplicate tool call id %q", id)
		}
		seenIDs[id] = struct{}{}
		if strings.TrimSpace(arguments) == "" {
			arguments = "{}"
		}
		toolCall := ai.ToolCall{
			ID:               id,
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
