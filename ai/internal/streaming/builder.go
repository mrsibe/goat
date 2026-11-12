package streaming

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/mrsibe/goat/ai"
)

type textBlock struct {
	index     int
	text      strings.Builder
	signature string
}

type thinkingBlock struct {
	index     int
	thinking  strings.Builder
	signature string
	id        string
	redacted  bool
}

type toolBlock struct {
	index          int
	id             string
	itemID         string
	name           string
	arguments      strings.Builder
	argumentsError string
	signature      string
}

// Builder accumulates one assistant stream and emits normalized ai.Event
// snapshots. Adapters own provider parsing; Builder owns message assembly and
// event ordering.
type Builder struct {
	info       ai.ModelInfo
	message    ai.Message
	texts      map[string]*textBlock
	thinkings  map[string]*thinkingBlock
	tools      map[string]*toolBlock
	usage      ai.Usage
	started    bool
	terminated bool
}

func New(info ai.ModelInfo) *Builder {
	return &Builder{
		info:      info,
		message:   ai.Message{Role: ai.RoleAssistant},
		texts:     make(map[string]*textBlock),
		thinkings: make(map[string]*thinkingBlock),
		tools:     make(map[string]*toolBlock),
	}
}

func (b *Builder) Start() ai.Event {
	b.started = true
	return ai.Event{
		Type:    ai.EventStart,
		Partial: b.partial(),
	}
}

func (b *Builder) TextStart(key string) ai.Event {
	block := b.ensureText(key)
	return ai.Event{
		Type:         ai.EventTextStart,
		ContentIndex: block.index,
		Partial:      b.partial(),
	}
}

func (b *Builder) TextDelta(key, delta string) ai.Event {
	block := b.ensureText(key)
	block.text.WriteString(delta)
	b.syncText(block)
	return ai.Event{
		Type:         ai.EventTextDelta,
		ContentIndex: block.index,
		TextDelta:    delta,
		Partial:      b.partial(),
	}
}

func (b *Builder) TextEnd(key, text string) ai.Event {
	block := b.ensureText(key)
	if text == "" {
		text = block.text.String()
	} else {
		block.text.Reset()
		block.text.WriteString(text)
	}
	b.syncText(block)
	return ai.Event{
		Type:         ai.EventTextEnd,
		ContentIndex: block.index,
		Text:         text,
		Partial:      b.partial(),
	}
}

// SetTextMeta records provider replay metadata for one text block.
func (b *Builder) SetTextMeta(key, signature string) {
	block := b.ensureText(key)
	if signature != "" {
		block.signature = signature
	}
	b.syncText(block)
}

func (b *Builder) ThinkingStart(key string) ai.Event {
	block := b.ensureThinking(key)
	return ai.Event{
		Type:         ai.EventThinkingStart,
		ContentIndex: block.index,
		Partial:      b.partial(),
	}
}

func (b *Builder) ThinkingDelta(key, delta string) ai.Event {
	block := b.ensureThinking(key)
	block.thinking.WriteString(delta)
	b.syncThinking(block)
	return ai.Event{
		Type:          ai.EventThinkingDelta,
		ContentIndex:  block.index,
		ThinkingDelta: delta,
		Partial:       b.partial(),
	}
}

func (b *Builder) SetThinkingMeta(key, id, signature string, redacted bool) {
	block := b.ensureThinking(key)
	if id != "" {
		block.id = id
	}
	if signature != "" {
		block.signature = signature
	}
	block.redacted = block.redacted || redacted
	b.syncThinking(block)
}

func (b *Builder) ThinkingEnd(key string) ai.Event {
	block := b.ensureThinking(key)
	b.syncThinking(block)
	return ai.Event{
		Type:         ai.EventThinkingEnd,
		ContentIndex: block.index,
		Thinking:     block.thinking.String(),
		Partial:      b.partial(),
	}
}

func (b *Builder) ToolStart(key, id, name string) ai.Event {
	block := b.ensureTool(key)
	if id != "" {
		block.id = id
	}
	if name != "" {
		block.name = name
	}
	b.syncTool(block)
	call := b.toolCall(block)
	return ai.Event{
		Type:         ai.EventToolCallStart,
		ContentIndex: block.index,
		ToolCall:     &call,
		Partial:      b.partial(),
	}
}

// SetToolItemID records the provider item id for one tool call.
func (b *Builder) SetToolItemID(key, itemID string) {
	block := b.ensureTool(key)
	if itemID != "" {
		block.itemID = itemID
	}
	b.syncTool(block)
}

func (b *Builder) ToolDelta(key, delta string) ai.Event {
	block := b.ensureTool(key)
	block.arguments.WriteString(delta)
	b.syncTool(block)
	return ai.Event{
		Type:         ai.EventToolCallDelta,
		ContentIndex: block.index,
		ToolCallDelta: &ai.ToolCallDelta{
			ID:             block.id,
			Name:           block.name,
			ArgumentsDelta: delta,
		},
		Partial: b.partial(),
	}
}

func (b *Builder) UpdateToolInfo(key, id, name string) {
	block := b.ensureTool(key)
	if id != "" {
		block.id = id
	}
	if name != "" {
		block.name = name
	}
	b.syncTool(block)
}

func (b *Builder) ToolEnd(key string, final *ai.ToolCall) (ai.Event, error) {
	block := b.ensureTool(key)
	if final != nil {
		if final.ID != "" {
			block.id = final.ID
		}
		if final.ItemID != "" {
			block.itemID = final.ItemID
		}
		if final.Name != "" {
			block.name = final.Name
		}
		if len(final.Arguments) > 0 {
			block.arguments.Reset()
			block.arguments.Write(final.Arguments)
		}
		if final.ArgumentsError != "" {
			block.argumentsError = final.ArgumentsError
		}
		if final.ThoughtSignature != "" {
			block.signature = final.ThoughtSignature
		}
	}
	if block.name == "" {
		return ai.Event{}, fmt.Errorf("tool call at content index %d is missing a name", block.index)
	}
	if block.id == "" {
		block.id = fmt.Sprintf("call_%d", block.index)
	}
	arguments := block.arguments.String()
	if strings.TrimSpace(arguments) == "" {
		arguments = "{}"
		block.arguments.Reset()
		block.arguments.WriteString(arguments)
	}
	if !json.Valid([]byte(arguments)) {
		block.argumentsError = fmt.Sprintf(
			"tool call %q arguments are not valid JSON",
			block.name,
		)
		arguments = "{}"
		block.arguments.Reset()
		block.arguments.WriteString(arguments)
	}
	b.syncTool(block)
	call := b.toolCall(block)
	return ai.Event{
		Type:         ai.EventToolCallEnd,
		ContentIndex: block.index,
		ToolCall:     &call,
		Partial:      b.partial(),
	}, nil
}

func (b *Builder) AddUsage(usage ai.Usage) ai.Event {
	b.usage.Add(usage)
	snapshot := b.usage
	return ai.Event{
		Type:    ai.EventUsage,
		Usage:   &snapshot,
		Partial: b.partial(),
	}
}

func (b *Builder) SetUsage(usage ai.Usage) ai.Event {
	b.usage = usage
	snapshot := b.usage
	return ai.Event{
		Type:    ai.EventUsage,
		Usage:   &snapshot,
		Partial: b.partial(),
	}
}

func (b *Builder) Done(stopReason ai.StopReason) ai.Event {
	b.terminated = true
	message := b.Message()
	usage := b.usage
	return ai.Event{
		Type:       ai.EventDone,
		Message:    &message,
		Usage:      &usage,
		StopReason: stopReason,
	}
}

func (b *Builder) Fail(err error, stopReason ai.StopReason) ai.Event {
	b.terminated = true
	message := b.Message()
	if stopReason == "" {
		stopReason = ai.StopReasonError
	}
	return ai.Event{
		Type:       ai.EventError,
		Message:    &message,
		Partial:    &message,
		Usage:      b.usagePtr(),
		StopReason: stopReason,
		Err:        err,
		Error:      errorString(err),
	}
}

func (b *Builder) Terminated() bool {
	return b.terminated
}

func (b *Builder) Message() ai.Message {
	return ai.CloneMessage(b.message)
}

func (b *Builder) ensureText(key string) *textBlock {
	if block, ok := b.texts[key]; ok {
		return block
	}
	block := &textBlock{index: len(b.message.Content)}
	b.texts[key] = block
	b.message.Content = append(b.message.Content, ai.ContentBlock{
		Type: ai.BlockText,
	})
	return block
}

func (b *Builder) ensureThinking(key string) *thinkingBlock {
	if block, ok := b.thinkings[key]; ok {
		return block
	}
	block := &thinkingBlock{index: len(b.message.Content)}
	b.thinkings[key] = block
	b.message.Content = append(b.message.Content, ai.ContentBlock{
		Type: ai.BlockThinking,
	})
	return block
}

func (b *Builder) ensureTool(key string) *toolBlock {
	if block, ok := b.tools[key]; ok {
		return block
	}
	block := &toolBlock{index: len(b.message.Content)}
	b.tools[key] = block
	b.message.Content = append(b.message.Content, ai.ContentBlock{
		Type: ai.BlockToolCall,
		ToolCall: &ai.ToolCall{
			Arguments: json.RawMessage("{}"),
		},
	})
	return block
}

func (b *Builder) syncText(block *textBlock) {
	b.message.Content[block.index].Text = block.text.String()
	b.message.Content[block.index].TextSignature = block.signature
}

func (b *Builder) syncThinking(block *thinkingBlock) {
	b.message.Content[block.index] = ai.ContentBlock{
		Type:              ai.BlockThinking,
		Thinking:          block.thinking.String(),
		ThinkingSignature: block.signature,
		ID:                block.id,
		Redacted:          block.redacted,
	}
}

func (b *Builder) syncTool(block *toolBlock) {
	arguments := block.arguments.String()
	if arguments == "" || !json.Valid([]byte(arguments)) {
		arguments = "{}"
	}
	b.message.Content[block.index].ToolCall = &ai.ToolCall{
		ID:               block.id,
		ItemID:           block.itemID,
		Name:             block.name,
		Arguments:        json.RawMessage(arguments),
		ArgumentsError:   block.argumentsError,
		ThoughtSignature: block.signature,
	}
}

func (b *Builder) toolCall(block *toolBlock) ai.ToolCall {
	return ai.CloneToolCall(*b.message.Content[block.index].ToolCall)
}

func (b *Builder) partial() *ai.Message {
	if !b.started {
		return nil
	}
	message := b.Message()
	return &message
}

func (b *Builder) usagePtr() *ai.Usage {
	if b.usage == (ai.Usage{}) {
		return nil
	}
	usage := b.usage
	return &usage
}

func errorString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

var ErrInvalidEvent = errors.New("invalid model stream event")
