package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/ai/internal/streaming"
	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/shared"
)

// CompletionsProvider implements ai.Model using Chat Completions streaming.
type CompletionsProvider struct {
	core providerCore
}

// NewCompletions creates an OpenAI-compatible Chat Completions provider.
func NewCompletions(config Config) (*CompletionsProvider, error) {
	core, err := newProviderCore(config, ai.APIOpenAICompletions)
	if err != nil {
		return nil, err
	}
	return &CompletionsProvider{core: core}, nil
}

func (p *CompletionsProvider) Info() ai.ModelInfo {
	if p == nil {
		return ai.ModelInfo{}
	}
	return p.core.info
}

func (p *CompletionsProvider) Stream(ctx context.Context, request ai.Request) (ai.Stream, error) {
	if p == nil {
		return nil, errors.New("openai: provider is nil")
	}
	params, err := p.params(request)
	if err != nil {
		return nil, err
	}
	raw := p.core.client.Chat.Completions.NewStreaming(ctx, params)
	return p.core.validateStream(&completionsStream{
		raw:     raw,
		info:    p.core.info,
		builder: streaming.New(p.core.info),
	}, nil)
}

func (p *CompletionsProvider) params(request ai.Request) (openaisdk.ChatCompletionNewParams, error) {
	messages := make([]openaisdk.ChatCompletionMessageParamUnion, 0, len(request.Messages))
	for _, message := range request.Messages {
		converted, err := convertCompletionsMessage(message, p.core.reasoningFormat)
		if err != nil {
			return openaisdk.ChatCompletionNewParams{}, err
		}
		messages = append(messages, converted)
	}

	tools := make([]openaisdk.ChatCompletionToolUnionParam, 0, len(request.Tools))
	for _, spec := range request.Tools {
		parameters := spec.Parameters
		if parameters == nil {
			parameters = map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			}
		}
		tools = append(tools, openaisdk.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
			Name:        spec.Name,
			Description: openaisdk.String(spec.Description),
			Parameters:  shared.FunctionParameters(parameters),
		}))
	}

	params := openaisdk.ChatCompletionNewParams{
		Model:    shared.ChatModel(p.core.info.Model),
		Messages: messages,
		Tools:    tools,
		StreamOptions: openaisdk.ChatCompletionStreamOptionsParam{
			IncludeUsage: param.NewOpt(true),
		},
	}
	if temperature := request.Options.Temperature; temperature != nil {
		params.Temperature = param.NewOpt(*temperature)
	}
	if topP := request.Options.TopP; topP != nil {
		params.TopP = param.NewOpt(*topP)
	}
	if request.Options.MaxTokens > 0 {
		params.MaxTokens = param.NewOpt(int64(request.Options.MaxTokens))
	}
	applyCompletionsReasoning(&params, p.core.reasoningFormat, request.Options.ReasoningEffort)
	if request.Options.SessionID != "" {
		params.PromptCacheKey = param.NewOpt(request.Options.SessionID)
	}
	if choice, ok := completionsToolChoice(request.Options.ToolChoice); ok {
		params.ToolChoice = choice
	}
	return params, nil
}

func applyCompletionsReasoning(
	params *openaisdk.ChatCompletionNewParams,
	format ReasoningFormat,
	effort string,
) {
	effort = strings.TrimSpace(effort)
	switch format {
	case ReasoningFormatDeepSeek:
		switch effort {
		case "":
			return
		case "none":
			params.SetExtraFields(map[string]any{
				"thinking": map[string]any{"type": "disabled"},
			})
		default:
			params.SetExtraFields(map[string]any{
				"thinking":         map[string]any{"type": "enabled"},
				"reasoning_effort": effort,
			})
		}
	case ReasoningFormatOpenRouter:
		if effort == "" {
			return
		}
		params.SetExtraFields(map[string]any{
			"reasoning": map[string]any{"effort": effort},
		})
	default:
		if effort != "" {
			params.ReasoningEffort = shared.ReasoningEffort(effort)
		}
	}
}

func convertCompletionsMessage(
	message ai.Message,
	format ReasoningFormat,
) (openaisdk.ChatCompletionMessageParamUnion, error) {
	switch message.Role {
	case ai.RoleSystem:
		return openaisdk.SystemMessage(message.Text()), nil
	case ai.RoleUser:
		return convertCompletionsUserMessage(message)
	case ai.RoleAssistant:
		assistant := openaisdk.ChatCompletionAssistantMessageParam{}
		text := message.Text()
		if text != "" || len(message.ToolCalls()) == 0 {
			assistant.Content.OfString = param.NewOpt(text)
		}
		for _, call := range message.ToolCalls() {
			arguments := string(call.Arguments)
			if strings.TrimSpace(arguments) == "" {
				arguments = "{}"
			}
			assistant.ToolCalls = append(assistant.ToolCalls, openaisdk.ChatCompletionMessageToolCallUnionParam{
				OfFunction: &openaisdk.ChatCompletionMessageFunctionToolCallParam{
					ID: call.ID,
					Function: openaisdk.ChatCompletionMessageFunctionToolCallFunctionParam{
						Name:      call.Name,
						Arguments: arguments,
					},
				},
			})
		}
		applyCompletionsReasoningReplay(&assistant, message, format)
		return openaisdk.ChatCompletionMessageParamUnion{OfAssistant: &assistant}, nil
	case ai.RoleTool:
		if message.ToolCallID == "" {
			return openaisdk.ChatCompletionMessageParamUnion{}, errors.New("openai: tool message is missing its tool call id")
		}
		for _, block := range message.Content {
			switch block.Type {
			case ai.BlockText:
			case ai.BlockImage:
				return openaisdk.ChatCompletionMessageParamUnion{}, errors.New(
					"openai: chat completions does not support image tool results",
				)
			default:
				return openaisdk.ChatCompletionMessageParamUnion{}, fmt.Errorf(
					"openai: unsupported tool result block type %q",
					block.Type,
				)
			}
		}
		return openaisdk.ToolMessage(message.Text(), message.ToolCallID), nil
	default:
		return openaisdk.ChatCompletionMessageParamUnion{}, fmt.Errorf("openai: unsupported message role %q", message.Role)
	}
}

func applyCompletionsReasoningReplay(
	assistant *openaisdk.ChatCompletionAssistantMessageParam,
	message ai.Message,
	format ReasoningFormat,
) {
	var thinkingBlocks []ai.ContentBlock
	var thinkingText []string
	for _, block := range message.Content {
		if block.Type != ai.BlockThinking {
			continue
		}
		thinkingBlocks = append(thinkingBlocks, block)
		if strings.TrimSpace(block.Thinking) != "" {
			thinkingText = append(thinkingText, block.Thinking)
		}
	}
	text := strings.Join(thinkingText, "\n")

	if format == ReasoningFormatDeepSeek {
		assistant.SetExtraFields(map[string]any{"reasoning_content": text})
		return
	}

	for _, block := range thinkingBlocks {
		if format == ReasoningFormatOpenRouter {
			if details, ok := parseReasoningDetails(block.ThinkingSignature); ok {
				assistant.SetExtraFields(map[string]any{"reasoning_details": details})
				return
			}
		}
		if isCompletionsReasoningField(block.ThinkingSignature) {
			assistant.SetExtraFields(map[string]any{block.ThinkingSignature: text})
			return
		}
	}
}

const completionsReasoningKey = "thinking"

var completionsReasoningFields = []string{"reasoning_content", "reasoning", "reasoning_text"}

func isCompletionsReasoningField(field string) bool {
	for _, candidate := range completionsReasoningFields {
		if field == candidate {
			return true
		}
	}
	return false
}

func convertCompletionsUserMessage(message ai.Message) (openaisdk.ChatCompletionMessageParamUnion, error) {
	hasImage := false
	parts := make([]openaisdk.ChatCompletionContentPartUnionParam, 0, len(message.Content))
	for _, block := range message.Content {
		switch block.Type {
		case ai.BlockText:
			parts = append(parts, openaisdk.ChatCompletionContentPartUnionParam{
				OfText: &openaisdk.ChatCompletionContentPartTextParam{Text: block.Text},
			})
		case ai.BlockImage:
			url, err := imageURL(block.Image)
			if err != nil {
				return openaisdk.ChatCompletionMessageParamUnion{}, err
			}
			hasImage = true
			parts = append(parts, openaisdk.ChatCompletionContentPartUnionParam{
				OfImageURL: &openaisdk.ChatCompletionContentPartImageParam{
					ImageURL: openaisdk.ChatCompletionContentPartImageImageURLParam{URL: url},
				},
			})
		case ai.BlockThinking:
			return openaisdk.ChatCompletionMessageParamUnion{}, errors.New(
				"openai: chat completions does not support thinking blocks",
			)
		default:
			return openaisdk.ChatCompletionMessageParamUnion{}, fmt.Errorf(
				"openai: unsupported user block type %q",
				block.Type,
			)
		}
	}
	if !hasImage {
		return openaisdk.UserMessage(message.Text()), nil
	}
	return openaisdk.UserMessage(parts), nil
}

func completionsToolChoice(choice ai.ToolChoice) (openaisdk.ChatCompletionToolChoiceOptionUnionParam, bool) {
	switch choice.Mode {
	case "":
		return openaisdk.ChatCompletionToolChoiceOptionUnionParam{}, false
	case ai.ToolChoiceAuto:
		return openaisdk.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: param.NewOpt(string(openaisdk.ChatCompletionToolChoiceOptionAutoAuto)),
		}, true
	case ai.ToolChoiceNone:
		return openaisdk.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: param.NewOpt(string(openaisdk.ChatCompletionToolChoiceOptionAutoNone)),
		}, true
	case ai.ToolChoiceRequired:
		return openaisdk.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: param.NewOpt(string(openaisdk.ChatCompletionToolChoiceOptionAutoRequired)),
		}, true
	case ai.ToolChoiceNamed:
		if choice.Name == "" {
			return openaisdk.ChatCompletionToolChoiceOptionUnionParam{}, false
		}
		return openaisdk.ToolChoiceOptionFunctionToolChoice(openaisdk.ChatCompletionNamedToolChoiceFunctionParam{
			Name: choice.Name,
		}), true
	default:
		return openaisdk.ChatCompletionToolChoiceOptionUnionParam{}, false
	}
}

type completionsStream struct {
	raw              streamReader[openaisdk.ChatCompletionChunk]
	info             ai.ModelInfo
	builder          *streaming.Builder
	pending          []ai.Event
	started          bool
	done             bool
	finish           ai.StopReason
	sawCall          bool
	textOpen         bool
	thinkingOpen     bool
	reasoningField   string
	reasoningDetails []map[string]any
	toolOrder        []int
	toolSeen         map[int]bool
}

type streamReader[T any] interface {
	Next() bool
	Current() T
	Err() error
	Close() error
}

func (s *completionsStream) Next() (ai.Event, error) {
	if !s.started {
		s.started = true
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
			s.finishStream()
			continue
		}
		s.consume(s.raw.Current())
	}
}

func (s *completionsStream) normalizeError(err error) error {
	return normalizeOpenAIError(s.info.Provider, err)
}

func (s *completionsStream) consume(chunk openaisdk.ChatCompletionChunk) {
	if chunk.JSON.Usage.Valid() {
		s.pending = append(s.pending, s.builder.SetUsage(completionsUsage(chunk.Usage)))
	}
	if len(chunk.Choices) == 0 {
		return
	}
	choice := chunk.Choices[0]
	s.consumeReasoning(choice.Delta)
	if choice.Delta.Content != "" {
		if !s.textOpen {
			s.textOpen = true
			s.pending = append(s.pending, s.builder.TextStart("text"))
		}
		s.pending = append(s.pending, s.builder.TextDelta("text", choice.Delta.Content))
	}
	for _, call := range choice.Delta.ToolCalls {
		index := int(call.Index)
		if !s.toolSeen[index] {
			if s.toolSeen == nil {
				s.toolSeen = make(map[int]bool)
			}
			s.toolSeen[index] = true
			s.toolOrder = append(s.toolOrder, index)
			s.sawCall = true
			s.pending = append(s.pending, s.builder.ToolStart(toolKey(index), call.ID, call.Function.Name))
		} else {
			s.builder.UpdateToolInfo(toolKey(index), call.ID, call.Function.Name)
		}
		if call.Function.Arguments != "" {
			s.pending = append(s.pending, s.builder.ToolDelta(toolKey(index), call.Function.Arguments))
		}
	}
	if choice.FinishReason != "" {
		s.finish = mapCompletionsStopReason(choice.FinishReason, s.sawCall)
	}
}

func (s *completionsStream) consumeReasoning(delta openaisdk.ChatCompletionChunkChoiceDelta) {
	var reasoning struct {
		ReasoningContent string            `json:"reasoning_content"`
		Reasoning        string            `json:"reasoning"`
		ReasoningText    string            `json:"reasoning_text"`
		ReasoningDetails []json.RawMessage `json:"reasoning_details"`
	}
	if raw := delta.RawJSON(); raw != "" {
		if err := json.Unmarshal([]byte(raw), &reasoning); err != nil {
			return
		}
	}

	fields := []struct {
		name  string
		value string
	}{
		{name: "reasoning_content", value: reasoning.ReasoningContent},
		{name: "reasoning", value: reasoning.Reasoning},
		{name: "reasoning_text", value: reasoning.ReasoningText},
	}
	for _, field := range fields {
		if field.value == "" {
			continue
		}
		s.ensureThinkingOpen(field.name)
		s.pending = append(s.pending, s.builder.ThinkingDelta(completionsReasoningKey, field.value))
		break
	}

	for _, raw := range reasoning.ReasoningDetails {
		detail, ok := decodeReasoningDetail(raw)
		if !ok {
			continue
		}
		s.ensureThinkingOpen("")
		s.reasoningDetails = appendReasoningDetail(s.reasoningDetails, detail)
	}
}

func (s *completionsStream) ensureThinkingOpen(field string) {
	if !s.thinkingOpen {
		s.thinkingOpen = true
		s.pending = append(s.pending, s.builder.ThinkingStart(completionsReasoningKey))
	}
	if s.reasoningField == "" && isCompletionsReasoningField(field) {
		s.reasoningField = field
		s.builder.SetThinkingMeta(completionsReasoningKey, "", field, false)
	}
}

func (s *completionsStream) finishStream() {
	s.done = true
	if s.thinkingOpen {
		if details, err := json.Marshal(s.reasoningDetails); err == nil && len(s.reasoningDetails) > 0 {
			s.builder.SetThinkingMeta(completionsReasoningKey, "", string(details), false)
		}
		s.pending = append(s.pending, s.builder.ThinkingEnd(completionsReasoningKey))
	}
	if s.textOpen {
		s.pending = append(s.pending, s.builder.TextEnd("text", ""))
	}
	for _, index := range s.toolOrder {
		event, err := s.builder.ToolEnd(toolKey(index), nil)
		if err != nil {
			s.pending = append(s.pending, s.builder.Fail(err, ai.StopReasonError))
			return
		}
		s.pending = append(s.pending, event)
	}
	stopReason := s.finish
	if stopReason == "" {
		stopReason = ai.StopReasonStop
		if s.sawCall {
			stopReason = ai.StopReasonToolCalls
		}
	}
	s.pending = append(s.pending, s.builder.Done(stopReason))
}

func (s *completionsStream) Close() error {
	if s == nil || s.raw == nil {
		return nil
	}
	return s.raw.Close()
}

func toolKey(index int) string {
	return fmt.Sprintf("tool:%d", index)
}

func decodeReasoningDetail(raw json.RawMessage) (map[string]any, bool) {
	var detail map[string]any
	if err := json.Unmarshal(raw, &detail); err != nil || !isValidReasoningDetail(detail) {
		return nil, false
	}
	return detail, true
}

func isValidReasoningDetail(detail map[string]any) bool {
	if detail == nil {
		return false
	}
	if value, ok := detail["id"]; ok && value != nil {
		if _, ok := value.(string); !ok {
			return false
		}
	}
	if value, ok := detail["format"]; ok && value != nil {
		if _, ok := value.(string); !ok {
			return false
		}
	}
	if value, ok := detail["index"]; ok && value != nil {
		if _, ok := value.(float64); !ok {
			return false
		}
	}
	switch detail["type"] {
	case "reasoning.text":
		if _, ok := detail["text"].(string); !ok {
			return false
		}
		if value, ok := detail["signature"]; ok && value != nil {
			if _, ok := value.(string); !ok {
				return false
			}
		}
		return true
	case "reasoning.summary":
		_, ok := detail["summary"].(string)
		return ok
	case "reasoning.encrypted":
		_, ok := detail["data"].(string)
		return ok
	default:
		return false
	}
}

func appendReasoningDetail(details []map[string]any, detail map[string]any) []map[string]any {
	if len(details) == 0 {
		return append(details, detail)
	}
	last := details[len(details)-1]
	switch detail["type"] {
	case "reasoning.text":
		if last["type"] != "reasoning.text" {
			return append(details, detail)
		}
		last["text"] = last["text"].(string) + detail["text"].(string)
		fillMissingReasoningDetailFields(last, detail)
		if current, _ := last["signature"].(string); current == "" {
			if signature, _ := detail["signature"].(string); signature != "" {
				last["signature"] = signature
			}
		}
		return details
	case "reasoning.summary":
		if last["type"] != "reasoning.summary" {
			return append(details, detail)
		}
		last["summary"] = last["summary"].(string) + detail["summary"].(string)
		fillMissingReasoningDetailFields(last, detail)
		return details
	default:
		return append(details, detail)
	}
}

func fillMissingReasoningDetailFields(target, source map[string]any) {
	if value, ok := target["id"]; !ok || value == nil {
		if sourceID, ok := source["id"]; ok {
			target["id"] = sourceID
		}
	} else if value, _ := value.(string); value == "" {
		if sourceID, ok := source["id"]; ok {
			target["id"] = sourceID
		}
	}
	if value, _ := target["format"].(string); value == "" {
		if format, ok := source["format"]; ok {
			target["format"] = format
		}
	}
	if value, ok := target["index"]; !ok || value == nil {
		if index, ok := source["index"]; ok {
			target["index"] = index
		}
	}
}

func parseReasoningDetails(signature string) ([]map[string]any, bool) {
	if strings.TrimSpace(signature) == "" {
		return nil, false
	}
	var details []map[string]any
	if err := json.Unmarshal([]byte(signature), &details); err != nil || len(details) == 0 {
		return nil, false
	}
	for _, detail := range details {
		if !isValidReasoningDetail(detail) {
			return nil, false
		}
	}
	return details, true
}

func completionsUsage(raw openaisdk.CompletionUsage) ai.Usage {
	return ai.Usage{
		PromptTokens:     raw.PromptTokens,
		CompletionTokens: raw.CompletionTokens,
		CacheReadTokens:  raw.PromptTokensDetails.CachedTokens,
		ReasoningTokens:  raw.CompletionTokensDetails.ReasoningTokens,
		TotalTokens:      raw.TotalTokens,
	}
}

func mapCompletionsStopReason(reason string, sawToolCall bool) ai.StopReason {
	switch reason {
	case "length":
		return ai.StopReasonLength
	case "tool_calls", "function_call":
		return ai.StopReasonToolCalls
	case "content_filter":
		return ai.StopReasonError
	case "stop":
		if sawToolCall {
			return ai.StopReasonToolCalls
		}
		return ai.StopReasonStop
	default:
		if sawToolCall {
			return ai.StopReasonToolCalls
		}
		return ai.StopReasonStop
	}
}
