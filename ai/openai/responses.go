package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/ai/internal/streaming"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	"github.com/openai/openai-go/v3/shared/constant"
)

const minResponsesOutputTokens = 16

// ResponsesProvider implements ai.Model using the OpenAI Responses API.
type ResponsesProvider struct {
	core providerCore
}

// NewResponses creates an OpenAI Responses provider.
func NewResponses(config Config) (*ResponsesProvider, error) {
	core, err := newProviderCore(config, ai.APIOpenAIResponses)
	if err != nil {
		return nil, err
	}
	return &ResponsesProvider{core: core}, nil
}

func (p *ResponsesProvider) Info() ai.ModelInfo {
	if p == nil {
		return ai.ModelInfo{}
	}
	return p.core.info
}

func (p *ResponsesProvider) Stream(ctx context.Context, request ai.Request) (ai.Stream, error) {
	if p == nil {
		return nil, errors.New("openai: provider is nil")
	}
	params, err := p.params(request)
	if err != nil {
		return nil, err
	}
	raw := p.core.client.Responses.NewStreaming(ctx, params)
	return p.core.validateStream(&responsesStream{
		raw:     raw,
		info:    p.core.info,
		builder: streaming.New(p.core.info),
	}, nil)
}

func (p *ResponsesProvider) params(request ai.Request) (responses.ResponseNewParams, error) {
	var instructions []string
	input := make([]responses.ResponseInputItemUnionParam, 0, len(request.Messages))
	for _, message := range request.Messages {
		switch message.Role {
		case ai.RoleSystem:
			if text := message.Text(); text != "" {
				instructions = append(instructions, text)
			}
		case ai.RoleUser:
			content, err := responsesUserContent(message)
			if err != nil {
				return responses.ResponseNewParams{}, err
			}
			input = append(input, responses.ResponseInputItemParamOfMessage(
				content,
				responses.EasyInputMessageRoleUser,
			))
		case ai.RoleAssistant:
			assistantInput, err := responsesAssistantInput(message)
			if err != nil {
				return responses.ResponseNewParams{}, err
			}
			input = append(input, assistantInput...)
		case ai.RoleTool:
			if message.ToolCallID == "" {
				return responses.ResponseNewParams{}, errors.New("openai: tool message is missing its tool call id")
			}
			output, err := responsesToolOutput(message)
			if err != nil {
				return responses.ResponseNewParams{}, err
			}
			item := responses.ResponseInputItemParamOfFunctionCallOutput(output)
			item.OfFunctionCallOutput.CallID = param.NewOpt(message.ToolCallID)
			input = append(input, item)
		default:
			return responses.ResponseNewParams{}, fmt.Errorf("openai: unsupported message role %q", message.Role)
		}
	}

	tools := make([]responses.ToolUnionParam, 0, len(request.Tools))
	for _, spec := range request.Tools {
		parameters := spec.Parameters
		if parameters == nil {
			parameters = map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			}
		}
		tools = append(tools, responses.ToolUnionParam{
			OfFunction: &responses.FunctionToolParam{
				Name:        spec.Name,
				Description: param.NewOpt(spec.Description),
				Parameters:  parameters,
			},
		})
	}

	params := responses.ResponseNewParams{
		Model: shared.ResponsesModel(p.core.info.Model),
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: input,
		},
		Tools: tools,
		Store: param.NewOpt(false),
	}
	if len(instructions) > 0 {
		params.Instructions = param.NewOpt(strings.Join(instructions, "\n\n"))
	}
	if temperature := request.Options.Temperature; temperature != nil {
		params.Temperature = param.NewOpt(*temperature)
	}
	if topP := request.Options.TopP; topP != nil {
		params.TopP = param.NewOpt(*topP)
	}
	if request.Options.MaxTokens > 0 {
		maxTokens := request.Options.MaxTokens
		if maxTokens < minResponsesOutputTokens {
			maxTokens = minResponsesOutputTokens
		}
		params.MaxOutputTokens = param.NewOpt(int64(maxTokens))
	}
	if request.Options.ReasoningEffort != "" {
		params.Reasoning.Effort = shared.ReasoningEffort(request.Options.ReasoningEffort)
		params.Reasoning.Summary = shared.ReasoningSummaryAuto
	}
	if request.Options.SessionID != "" {
		params.PromptCacheKey = param.NewOpt(request.Options.SessionID)
	}
	if choice, ok := responsesToolChoice(request.Options.ToolChoice); ok {
		params.ToolChoice = choice
	}
	params.Include = []responses.ResponseIncludable{
		responses.ResponseIncludableReasoningEncryptedContent,
	}
	return params, nil
}

func responsesUserContent(message ai.Message) (responses.ResponseInputMessageContentListParam, error) {
	content := make(responses.ResponseInputMessageContentListParam, 0, len(message.Content))
	for _, block := range message.Content {
		switch block.Type {
		case ai.BlockText:
			content = append(content, responses.ResponseInputContentParamOfInputText(block.Text))
		case ai.BlockImage:
			url, err := imageURL(block.Image)
			if err != nil {
				return nil, err
			}
			image := responses.ResponseInputContentParamOfInputImage(
				responses.ResponseInputImageDetailAuto,
			)
			image.OfInputImage.ImageURL = param.NewOpt(url)
			content = append(content, image)
		case ai.BlockThinking:
			return nil, errors.New("openai: thinking blocks are not valid user content")
		default:
			return nil, fmt.Errorf("openai: unsupported user content block %q", block.Type)
		}
	}
	if len(content) == 0 {
		content = append(content, responses.ResponseInputContentParamOfInputText(""))
	}
	return content, nil
}

func responsesAssistantInput(message ai.Message) ([]responses.ResponseInputItemUnionParam, error) {
	input := make([]responses.ResponseInputItemUnionParam, 0, len(message.Content))
	for _, block := range message.Content {
		switch block.Type {
		case ai.BlockText:
			if block.Text == "" {
				continue
			}
			signatureID, signaturePhase := parseTextSignature(block.TextSignature)
			if signatureID != "" {
				text := responses.ResponseOutputTextParam{Text: block.Text}
				item := responses.ResponseOutputMessageParam{
					ID:      signatureID,
					Content: []responses.ResponseOutputMessageContentUnionParam{{OfOutputText: &text}},
					Status:  responses.ResponseOutputMessageStatusCompleted,
					Role:    constant.Assistant("assistant"),
					Type:    constant.Message("message"),
					Phase:   responses.ResponseOutputMessagePhase(signaturePhase),
				}
				input = append(input, responses.ResponseInputItemUnionParam{OfOutputMessage: &item})
				continue
			}
			input = append(input, responses.ResponseInputItemParamOfMessage(
				block.Text,
				responses.EasyInputMessageRoleAssistant,
			))
		case ai.BlockThinking:
			if block.ID == "" {
				continue
			}
			summary := []responses.ResponseReasoningItemSummaryParam{}
			if block.Thinking != "" {
				summary = append(summary, responses.ResponseReasoningItemSummaryParam{
					Text: block.Thinking,
				})
			}
			item := responses.ResponseInputItemParamOfReasoning(block.ID, summary)
			if block.ThinkingSignature != "" {
				item.OfReasoning.EncryptedContent = param.NewOpt(block.ThinkingSignature)
			}
			input = append(input, item)
		case ai.BlockToolCall:
			if block.ToolCall == nil {
				return nil, errors.New("openai: assistant tool-call block has no payload")
			}
			arguments := string(block.ToolCall.Arguments)
			if strings.TrimSpace(arguments) == "" {
				arguments = "{}"
			}
			item := responses.ResponseInputItemParamOfFunctionCall(
				arguments,
				block.ToolCall.ID,
				block.ToolCall.Name,
			)
			if block.ToolCall.ItemID != "" {
				item.OfFunctionCall.ID = param.NewOpt(block.ToolCall.ItemID)
			}
			input = append(input, item)
		case ai.BlockImage:
			return nil, errors.New("openai: assistant image blocks are not supported")
		default:
			return nil, fmt.Errorf("openai: unsupported assistant content block %q", block.Type)
		}
	}
	return input, nil
}

func responsesToolOutput(message ai.Message) (responses.ResponseFunctionCallOutputItemListParam, error) {
	items := make(responses.ResponseFunctionCallOutputItemListParam, 0, len(message.Content))
	for _, block := range message.Content {
		switch block.Type {
		case ai.BlockText:
			items = append(items, responses.ResponseFunctionCallOutputItemParamOfInputText(block.Text))
		case ai.BlockImage:
			url, err := imageURL(block.Image)
			if err != nil {
				return nil, err
			}
			image := responses.ResponseInputImageContentParam{
				ImageURL: param.NewOpt(url),
			}
			items = append(items, responses.ResponseFunctionCallOutputItemUnionParam{
				OfInputImage: &image,
			})
		default:
			return nil, fmt.Errorf("openai: unsupported tool result block %q", block.Type)
		}
	}
	if len(items) == 0 {
		items = append(items, responses.ResponseFunctionCallOutputItemParamOfInputText(""))
	}
	return items, nil
}

func responsesToolChoice(choice ai.ToolChoice) (responses.ResponseNewParamsToolChoiceUnion, bool) {
	switch choice.Mode {
	case "":
		return responses.ResponseNewParamsToolChoiceUnion{}, false
	case ai.ToolChoiceAuto:
		return responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsAuto),
		}, true
	case ai.ToolChoiceNone:
		return responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsNone),
		}, true
	case ai.ToolChoiceRequired:
		return responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsRequired),
		}, true
	case ai.ToolChoiceNamed:
		if choice.Name == "" {
			return responses.ResponseNewParamsToolChoiceUnion{}, false
		}
		return responses.ResponseNewParamsToolChoiceUnion{
			OfFunctionTool: &responses.ToolChoiceFunctionParam{Name: choice.Name},
		}, true
	default:
		return responses.ResponseNewParamsToolChoiceUnion{}, false
	}
}

type responsesStream struct {
	raw           responsesEventStream
	info          ai.ModelInfo
	builder       *streaming.Builder
	pending       []ai.Event
	started       bool
	done          bool
	terminal      bool
	textOrder     []string
	textOpen      map[string]bool
	textItems     map[string]string
	messagePhases map[string]string
	toolOrder     []string
	toolOpen      map[string]bool
	thinkOrder    []string
	thinkOpen     map[string]bool
	thinkSeen     map[string]bool
	sawCall       bool
	stopReason    ai.StopReason
}

type responsesEventStream interface {
	Next() bool
	Current() responses.ResponseStreamEventUnion
	Err() error
	Close() error
}

func (s *responsesStream) Next() (ai.Event, error) {
	if !s.started {
		s.started = true
		s.textOpen = make(map[string]bool)
		s.textItems = make(map[string]string)
		s.messagePhases = make(map[string]string)
		s.toolOpen = make(map[string]bool)
		s.thinkOpen = make(map[string]bool)
		s.thinkSeen = make(map[string]bool)
		return s.builder.Start(), nil
	}
	for {
		if len(s.pending) > 0 {
			event := s.pending[0]
			s.pending = s.pending[1:]
			return event, nil
		}
		if s.done {
			return ai.Event{}, io.EOF
		}
		if !s.raw.Next() {
			if err := s.raw.Err(); err != nil {
				s.done = true
				return s.builder.Fail(s.normalizeError(err), ai.StopReasonError), nil
			}
			if !s.terminal {
				s.done = true
				return s.builder.Fail(errors.New("openai responses stream ended without a terminal event"), ai.StopReasonError), nil
			}
			s.finalize()
			continue
		}
		s.consume(s.raw.Current())
	}
}

func (s *responsesStream) consume(event responses.ResponseStreamEventUnion) {
	switch event.Type {
	case "response.created", "response.in_progress", "response.queued":
	case "response.reasoning_summary_text.delta":
		delta := event.AsResponseReasoningSummaryTextDelta()
		key := reasoningKey(delta.ItemID, delta.OutputIndex)
		s.ensureThinking(key)
		if delta.Delta != "" {
			s.thinkSeen[key] = true
			s.pending = append(s.pending, s.builder.ThinkingDelta(key, delta.Delta))
		}
	case "response.reasoning_summary_text.done":
		done := event.AsResponseReasoningSummaryTextDone()
		key := reasoningKey(done.ItemID, done.OutputIndex)
		s.ensureThinking(key)
		if !s.thinkSeen[key] && done.Text != "" {
			s.thinkSeen[key] = true
			s.pending = append(s.pending, s.builder.ThinkingDelta(key, done.Text))
		}
	case "response.reasoning_text.delta":
		delta := event.AsResponseReasoningTextDelta()
		key := reasoningKey(delta.ItemID, delta.OutputIndex)
		s.ensureThinking(key)
		if delta.Delta != "" {
			s.thinkSeen[key] = true
			s.pending = append(s.pending, s.builder.ThinkingDelta(key, delta.Delta))
		}
	case "response.reasoning_text.done":
		done := event.AsResponseReasoningTextDone()
		key := reasoningKey(done.ItemID, done.OutputIndex)
		s.ensureThinking(key)
		if !s.thinkSeen[key] && done.Text != "" {
			s.thinkSeen[key] = true
			s.pending = append(s.pending, s.builder.ThinkingDelta(key, done.Text))
		}
	case "response.output_text.delta":
		delta := event.AsResponseOutputTextDelta()
		key := textKey(delta.ItemID, delta.OutputIndex, delta.ContentIndex)
		s.ensureText(key)
		s.pending = append(s.pending, s.builder.TextDelta(key, delta.Delta))
	case "response.output_text.done":
		done := event.AsResponseOutputTextDone()
		key := textKey(done.ItemID, done.OutputIndex, done.ContentIndex)
		s.ensureText(key)
		s.textItems[key] = done.ItemID
		s.builder.SetTextMeta(key, encodeTextSignature(done.ItemID, s.messagePhases[done.ItemID]))
		s.pending = append(s.pending, s.builder.TextEnd(key, done.Text))
		delete(s.textOpen, key)
	case "response.output_item.added":
		item := event.AsResponseOutputItemAdded()
		if item.Item.Type == "reasoning" {
			key := reasoningKey(item.Item.ID, item.OutputIndex)
			s.ensureThinking(key)
			s.builder.SetThinkingMeta(key, item.Item.ID, "", false)
			return
		}
		if item.Item.Type == "message" {
			message := item.Item.AsMessage()
			id := message.ID
			if id == "" {
				id = item.Item.ID
			}
			s.messagePhases[id] = string(message.Phase)
			return
		}
		if item.Item.Type == "function_call" {
			call := item.Item.AsFunctionCall()
			key := outputItemKey(item.Item.ID, item.OutputIndex)
			s.sawCall = true
			s.builder.SetToolItemID(key, item.Item.ID)
			if s.toolOpen[key] {
				s.builder.UpdateToolInfo(key, call.CallID, call.Name)
				return
			}
			s.toolOrder = append(s.toolOrder, key)
			s.toolOpen[key] = true
			s.pending = append(s.pending, s.builder.ToolStart(key, call.CallID, call.Name))
		}
	case "response.output_item.done":
		item := event.AsResponseOutputItemDone()
		if item.Item.Type == "reasoning" {
			reasoning := item.Item.AsReasoning()
			key := reasoningKey(reasoning.ID, item.OutputIndex)
			s.ensureThinking(key)
			if !s.thinkSeen[key] {
				if text := reasoningText(reasoning); text != "" {
					s.thinkSeen[key] = true
					s.pending = append(s.pending, s.builder.ThinkingDelta(key, text))
				}
			}
			s.builder.SetThinkingMeta(key, reasoning.ID, reasoning.EncryptedContent, false)
			s.endThinking(key)
			return
		}
		if item.Item.Type == "message" {
			message := item.Item.AsMessage()
			id := message.ID
			if id == "" {
				id = item.Item.ID
			}
			s.messagePhases[id] = string(message.Phase)
			for key, itemID := range s.textItems {
				if itemID == id {
					s.builder.SetTextMeta(key, encodeTextSignature(id, string(message.Phase)))
				}
			}
			return
		}
		if item.Item.Type != "function_call" {
			return
		}
		call := item.Item.AsFunctionCall()
		key := outputItemKey(item.Item.ID, item.OutputIndex)
		s.ensureTool(key)
		s.builder.SetToolItemID(key, item.Item.ID)
		toolCall := ai.ToolCall{
			ID:        call.CallID,
			ItemID:    item.Item.ID,
			Name:      call.Name,
			Arguments: json.RawMessage(call.Arguments),
		}
		toolEvent, err := s.builder.ToolEnd(key, &toolCall)
		if err != nil {
			s.pending = append(s.pending, s.builder.Fail(err, ai.StopReasonError))
			s.done = true
			return
		}
		s.pending = append(s.pending, toolEvent)
		delete(s.toolOpen, key)
	case "response.function_call_arguments.delta":
		delta := event.AsResponseFunctionCallArgumentsDelta()
		key := outputItemKey(delta.ItemID, delta.OutputIndex)
		s.ensureTool(key)
		s.pending = append(s.pending, s.builder.ToolDelta(key, delta.Delta))
	case "response.function_call_arguments.done":
		done := event.AsResponseFunctionCallArgumentsDone()
		key := outputItemKey(done.ItemID, done.OutputIndex)
		s.ensureTool(key)
		s.builder.SetToolItemID(key, done.ItemID)
		toolCall := ai.ToolCall{ItemID: done.ItemID, Arguments: json.RawMessage(done.Arguments)}
		toolEvent, err := s.builder.ToolEnd(key, &toolCall)
		if err != nil {
			s.pending = append(s.pending, s.builder.Fail(err, ai.StopReasonError))
			s.done = true
			return
		}
		s.pending = append(s.pending, toolEvent)
		delete(s.toolOpen, key)
	case "response.completed":
		completed := event.AsResponseCompleted()
		s.sawCall = s.sawCall || responseHasFunctionCall(completed.Response)
		s.stopReason = ai.StopReasonStop
		if s.sawCall {
			s.stopReason = ai.StopReasonToolCalls
		}
		if hasResponseUsage(completed.Response.Usage) {
			s.pending = append(s.pending, s.builder.SetUsage(responseUsage(completed.Response.Usage)))
		}
		s.terminal = true
		s.finalize()
	case "response.incomplete":
		incomplete := event.AsResponseIncomplete()
		switch incomplete.Response.IncompleteDetails.Reason {
		case "content_filter":
			s.stopReason = ai.StopReasonError
		case "steered":
			s.stopReason = ai.StopReasonStop
		default:
			s.stopReason = ai.StopReasonLength
		}
		if hasResponseUsage(incomplete.Response.Usage) {
			s.pending = append(s.pending, s.builder.SetUsage(responseUsage(incomplete.Response.Usage)))
		}
		s.terminal = true
		s.finalize()
	case "response.failed":
		failed := event.AsResponseFailed()
		s.terminal = true
		s.done = true
		s.stopReason = ai.StopReasonError
		err := &ai.APIError{
			Provider: s.info.Provider,
			Code:     string(failed.Response.Error.Code),
			Message:  failed.Response.Error.Message,
			Retryable: failed.Response.Error.Code == responses.ResponseErrorCodeRateLimitExceeded ||
				failed.Response.Error.Code == responses.ResponseErrorCodeServerError,
		}
		s.pending = append(s.pending, s.builder.Fail(err, ai.StopReasonError))
	case "error":
		errEvent := event.AsError()
		s.terminal = true
		s.done = true
		s.stopReason = ai.StopReasonError
		err := &ai.APIError{
			Provider:  s.info.Provider,
			Code:      errEvent.Code,
			Message:   errEvent.Message,
			Retryable: false,
		}
		s.pending = append(s.pending, s.builder.Fail(err, ai.StopReasonError))
	default:
		// Responses emits many provider-side tool events that the text/tool
		// contract does not expose. Ignore them rather than failing the run.
	}
}

func (s *responsesStream) ensureText(key string) {
	if s.textOpen[key] {
		return
	}
	s.textOpen[key] = true
	s.textOrder = append(s.textOrder, key)
	s.pending = append(s.pending, s.builder.TextStart(key))
}

func (s *responsesStream) ensureTool(key string) {
	if s.toolOpen[key] {
		return
	}
	s.toolOpen[key] = true
	s.toolOrder = append(s.toolOrder, key)
	s.pending = append(s.pending, s.builder.ToolStart(key, "", ""))
}

func (s *responsesStream) ensureThinking(key string) {
	if s.thinkOpen[key] {
		return
	}
	s.thinkOpen[key] = true
	s.thinkOrder = append(s.thinkOrder, key)
	s.pending = append(s.pending, s.builder.ThinkingStart(key))
}

func (s *responsesStream) endThinking(key string) {
	if !s.thinkOpen[key] {
		return
	}
	delete(s.thinkOpen, key)
	s.pending = append(s.pending, s.builder.ThinkingEnd(key))
}

func (s *responsesStream) finalize() {
	s.done = true
	for _, key := range s.thinkOrder {
		s.endThinking(key)
	}
	for _, key := range s.textOrder {
		if s.textOpen[key] {
			s.pending = append(s.pending, s.builder.TextEnd(key, ""))
			delete(s.textOpen, key)
		}
	}
	for _, key := range s.toolOrder {
		if s.toolOpen[key] {
			event, err := s.builder.ToolEnd(key, nil)
			if err != nil {
				s.pending = append(s.pending, s.builder.Fail(err, ai.StopReasonError))
				return
			}
			s.pending = append(s.pending, event)
			delete(s.toolOpen, key)
		}
	}
	stopReason := s.stopReason
	if stopReason == "" {
		stopReason = ai.StopReasonStop
	}
	s.pending = append(s.pending, s.builder.Done(stopReason))
}

func reasoningKey(itemID string, outputIndex int64) string {
	if itemID != "" {
		return "reasoning:" + itemID
	}
	return fmt.Sprintf("reasoning:output:%d", outputIndex)
}

func reasoningText(item responses.ResponseReasoningItem) string {
	var parts []string
	for _, summary := range item.Summary {
		if summary.Text != "" {
			parts = append(parts, summary.Text)
		}
	}
	for _, content := range item.Content {
		if content.Text != "" {
			parts = append(parts, content.Text)
		}
	}
	return strings.Join(parts, "\n")
}

func (s *responsesStream) Close() error {
	if s == nil || s.raw == nil {
		return nil
	}
	return s.raw.Close()
}

func (s *responsesStream) normalizeError(err error) error {
	return normalizeOpenAIError(s.info.Provider, err)
}

func responseUsage(usage responses.ResponseUsage) ai.Usage {
	return ai.Usage{
		PromptTokens:     usage.InputTokens,
		CompletionTokens: usage.OutputTokens,
		CacheReadTokens:  usage.InputTokensDetails.CachedTokens,
		CacheWriteTokens: usage.InputTokensDetails.CacheWriteTokens,
		ReasoningTokens:  usage.OutputTokensDetails.ReasoningTokens,
		TotalTokens:      usage.TotalTokens,
	}
}

func hasResponseUsage(usage responses.ResponseUsage) bool {
	return usage.InputTokens != 0 ||
		usage.OutputTokens != 0 ||
		usage.TotalTokens != 0 ||
		usage.InputTokensDetails.CachedTokens != 0 ||
		usage.InputTokensDetails.CacheWriteTokens != 0 ||
		usage.OutputTokensDetails.ReasoningTokens != 0
}

func responseHasFunctionCall(response responses.Response) bool {
	for _, item := range response.Output {
		if item.Type == "function_call" {
			return true
		}
	}
	return false
}

func textKey(itemID string, outputIndex, contentIndex int64) string {
	id := itemID
	if id == "" {
		id = strconv.FormatInt(outputIndex, 10)
	}
	return "text:" + id + ":" + strconv.FormatInt(contentIndex, 10)
}

func outputItemKey(itemID string, outputIndex int64) string {
	if itemID != "" {
		return "tool:" + itemID
	}
	return "tool:" + strconv.FormatInt(outputIndex, 10)
}

type responseTextSignature struct {
	ID    string `json:"id"`
	Phase string `json:"phase,omitempty"`
}

func encodeTextSignature(id, phase string) string {
	if id == "" && phase == "" {
		return ""
	}
	encoded, err := json.Marshal(responseTextSignature{ID: id, Phase: phase})
	if err != nil {
		return id
	}
	return string(encoded)
}

func parseTextSignature(signature string) (string, string) {
	if signature == "" {
		return "", ""
	}
	var parsed responseTextSignature
	if err := json.Unmarshal([]byte(signature), &parsed); err == nil && parsed.ID != "" {
		return parsed.ID, parsed.Phase
	}
	return signature, ""
}
