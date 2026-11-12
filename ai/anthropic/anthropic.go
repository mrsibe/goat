// Package anthropic adapts the Anthropic Messages API to the ai model
// interface. It uses the official Anthropic Go SDK for requests and streaming.
package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/ai/internal/streaming"
)

const defaultMaxTokens int64 = 4096

// Config configures an Anthropic Messages model.
type Config struct {
	APIKey          string
	AuthToken       string
	BaseURL         string
	Model           string
	Provider        string
	Headers         map[string]string
	HTTPClient      *http.Client
	MaxRetries      *int
	ContextWindow   int
	MaxOutputTokens int
}

// Provider implements ai.Model using Anthropic Messages streaming.
type Provider struct {
	client          anthropicsdk.Client
	info            ai.ModelInfo
	defaultMaxToken int64
}

// New creates an Anthropic Messages provider.
func New(config Config) (*Provider, error) {
	if strings.TrimSpace(config.Model) == "" {
		return nil, errors.New("anthropic: model must not be empty")
	}
	if config.APIKey != "" && config.AuthToken != "" {
		return nil, errors.New("anthropic: set only one of API key or auth token")
	}

	var options []option.RequestOption
	if config.AuthToken != "" {
		options = append(options, option.WithAuthToken(config.AuthToken))
	} else if config.APIKey != "" {
		options = append(options, option.WithAPIKey(config.APIKey))
	}
	if config.BaseURL != "" {
		options = append(options, option.WithBaseURL(config.BaseURL))
	}
	if config.HTTPClient != nil {
		options = append(options, option.WithHTTPClient(config.HTTPClient))
	}
	if config.MaxRetries != nil {
		options = append(options, option.WithMaxRetries(*config.MaxRetries))
	}
	for key, value := range config.Headers {
		if strings.TrimSpace(key) == "" {
			continue
		}
		options = append(options, option.WithHeader(key, value))
	}
	if config.AuthToken != "" {
		options = append(options, option.WithHeaderDel("x-api-key"))
	}

	provider := strings.TrimSpace(config.Provider)
	if provider == "" {
		provider = "anthropic"
	}
	maxTokens := int64(config.MaxOutputTokens)
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
	}
	return &Provider{
		client: anthropicsdk.NewClient(options...),
		info: ai.ModelInfo{
			Provider:                 provider,
			Model:                    config.Model,
			API:                      ai.APIAnthropicMessages,
			ContextWindow:            config.ContextWindow,
			MaxOutputTokens:          int(maxTokens),
			SupportsTools:            true,
			SupportsImages:           true,
			SupportsToolResultImages: true,
			SupportsThinking:         true,
		},
		defaultMaxToken: maxTokens,
	}, nil
}

// Info implements ai.Model.
func (p *Provider) Info() ai.ModelInfo {
	if p == nil {
		return ai.ModelInfo{}
	}
	return p.info
}

var _ ai.Model = (*Provider)(nil)

// Stream starts one streamed Messages request.
func (p *Provider) Stream(ctx context.Context, request ai.Request) (ai.Stream, error) {
	if p == nil {
		return nil, errors.New("anthropic: provider is nil")
	}
	params, err := p.params(request)
	if err != nil {
		return nil, err
	}
	raw := p.client.Messages.NewStreaming(ctx, params)
	if raw == nil {
		return nil, errors.New("anthropic: provider returned a nil stream")
	}
	return &messageStream{
		raw:     raw,
		info:    p.info,
		builder: streaming.New(p.info),
		blocks:  make(map[int64]*contentBlock),
	}, nil
}

func (p *Provider) params(request ai.Request) (anthropicsdk.MessageNewParams, error) {
	maxTokens := int64(request.Options.MaxTokens)
	if maxTokens <= 0 {
		maxTokens = p.defaultMaxToken
	}
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
	}

	system, messages, err := convertMessages(request.Messages)
	if err != nil {
		return anthropicsdk.MessageNewParams{}, err
	}
	tools, err := convertTools(request.Tools)
	if err != nil {
		return anthropicsdk.MessageNewParams{}, err
	}

	params := anthropicsdk.MessageNewParams{
		Model:     anthropicsdk.Model(p.info.Model),
		MaxTokens: maxTokens,
		Messages:  messages,
		System:    system,
		Tools:     tools,
	}
	if request.Options.Temperature != nil {
		params.Temperature = param.NewOpt(*request.Options.Temperature)
	}
	if request.Options.TopP != nil {
		params.TopP = param.NewOpt(*request.Options.TopP)
	}
	if request.Options.ThinkingBudgetTokens > 0 {
		params.Thinking = anthropicsdk.ThinkingConfigParamOfEnabled(
			int64(request.Options.ThinkingBudgetTokens),
		)
	}
	if choice, ok := convertToolChoice(request.Options.ToolChoice); ok {
		params.ToolChoice = choice
	}
	return params, nil
}

func convertMessages(messages []ai.Message) ([]anthropicsdk.TextBlockParam, []anthropicsdk.MessageParam, error) {
	var system []anthropicsdk.TextBlockParam
	converted := make([]anthropicsdk.MessageParam, 0, len(messages))
	lastWasToolResult := false

	for _, message := range messages {
		switch message.Role {
		case ai.RoleSystem:
			text := message.Text()
			if text != "" {
				system = append(system, anthropicsdk.TextBlockParam{Text: text})
			}
			lastWasToolResult = false
		case ai.RoleUser:
			blocks, err := userBlocks(message)
			if err != nil {
				return nil, nil, err
			}
			if len(blocks) == 0 {
				blocks = []anthropicsdk.ContentBlockParamUnion{anthropicsdk.NewTextBlock("")}
			}
			converted = append(converted, anthropicsdk.NewUserMessage(blocks...))
			lastWasToolResult = false
		case ai.RoleAssistant:
			blocks := make([]anthropicsdk.ContentBlockParamUnion, 0, len(message.Content))
			for _, block := range message.Content {
				switch block.Type {
				case ai.BlockText:
					blocks = append(blocks, anthropicsdk.NewTextBlock(block.Text))
				case ai.BlockThinking:
					if block.Redacted {
						blocks = append(blocks, anthropicsdk.NewRedactedThinkingBlock(block.ThinkingSignature))
						continue
					}
					blocks = append(blocks, anthropicsdk.NewThinkingBlock(
						block.ThinkingSignature,
						block.Thinking,
					))
				case ai.BlockImage:
					return nil, nil, errors.New("anthropic: assistant image blocks are not supported")
				case ai.BlockToolCall:
					if block.ToolCall == nil {
						return nil, nil, errors.New("anthropic: assistant tool-call block has no payload")
					}
					input := any(json.RawMessage("{}"))
					if len(strings.TrimSpace(string(block.ToolCall.Arguments))) > 0 {
						input = json.RawMessage(block.ToolCall.Arguments)
					}
					blocks = append(blocks, anthropicsdk.NewToolUseBlock(
						block.ToolCall.ID,
						input,
						block.ToolCall.Name,
					))
				default:
					return nil, nil, fmt.Errorf("anthropic: unsupported assistant block type %q", block.Type)
				}
			}
			converted = append(converted, anthropicsdk.NewAssistantMessage(blocks...))
			lastWasToolResult = false
		case ai.RoleTool:
			if message.ToolCallID == "" {
				return nil, nil, errors.New("anthropic: tool message is missing its tool call id")
			}
			block, err := toolResultBlock(message)
			if err != nil {
				return nil, nil, err
			}
			if lastWasToolResult {
				last := len(converted) - 1
				converted[last].Content = append(converted[last].Content, block)
			} else {
				converted = append(converted, anthropicsdk.NewUserMessage(block))
				lastWasToolResult = true
			}
		default:
			return nil, nil, fmt.Errorf("anthropic: unsupported message role %q", message.Role)
		}
	}
	return system, converted, nil
}

func userBlocks(message ai.Message) ([]anthropicsdk.ContentBlockParamUnion, error) {
	blocks := make([]anthropicsdk.ContentBlockParamUnion, 0, len(message.Content))
	for _, block := range message.Content {
		switch block.Type {
		case ai.BlockText:
			blocks = append(blocks, anthropicsdk.NewTextBlock(block.Text))
		case ai.BlockImage:
			image, err := imageBlock(block.Image)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, image)
		case ai.BlockThinking:
			return nil, errors.New("anthropic: thinking blocks are not valid user content")
		default:
			return nil, fmt.Errorf("anthropic: unsupported user content block %q", block.Type)
		}
	}
	return blocks, nil
}

func toolResultBlock(message ai.Message) (anthropicsdk.ContentBlockParamUnion, error) {
	result := anthropicsdk.ToolResultBlockParam{
		ToolUseID: message.ToolCallID,
		IsError:   param.NewOpt(message.IsError),
	}
	for _, block := range message.Content {
		switch block.Type {
		case ai.BlockText:
			result.Content = append(result.Content, anthropicsdk.ToolResultBlockParamContentUnion{
				OfText: &anthropicsdk.TextBlockParam{Text: block.Text},
			})
		case ai.BlockImage:
			image, err := imageBlock(block.Image)
			if err != nil {
				return anthropicsdk.ContentBlockParamUnion{}, err
			}
			result.Content = append(result.Content, anthropicsdk.ToolResultBlockParamContentUnion{
				OfImage: image.OfImage,
			})
		default:
			return anthropicsdk.ContentBlockParamUnion{}, fmt.Errorf(
				"anthropic: unsupported tool result block %q",
				block.Type,
			)
		}
	}
	if len(result.Content) == 0 {
		result.Content = append(result.Content, anthropicsdk.ToolResultBlockParamContentUnion{
			OfText: &anthropicsdk.TextBlockParam{Text: ""},
		})
	}
	return anthropicsdk.ContentBlockParamUnion{OfToolResult: &result}, nil
}

func imageBlock(image *ai.ImageContent) (anthropicsdk.ContentBlockParamUnion, error) {
	if image == nil {
		return anthropicsdk.ContentBlockParamUnion{}, errors.New("anthropic: image block is missing its payload")
	}
	data := strings.TrimSpace(image.Data)
	url := strings.TrimSpace(image.URL)
	if (data == "") == (url == "") {
		return anthropicsdk.ContentBlockParamUnion{}, errors.New(
			"anthropic: image must set exactly one of data or url",
		)
	}
	if data != "" {
		if strings.TrimSpace(image.MediaType) == "" {
			return anthropicsdk.ContentBlockParamUnion{}, errors.New(
				"anthropic: inline image requires a media type",
			)
		}
		return anthropicsdk.NewImageBlockBase64(image.MediaType, data), nil
	}
	return anthropicsdk.NewImageBlock(anthropicsdk.URLImageSourceParam{URL: url}), nil
}

func convertTools(specs []ai.ToolSpec) ([]anthropicsdk.ToolUnionParam, error) {
	tools := make([]anthropicsdk.ToolUnionParam, 0, len(specs))
	for _, spec := range specs {
		if strings.TrimSpace(spec.Name) == "" {
			return nil, errors.New("anthropic: tool name must not be empty")
		}
		schema, err := convertSchema(spec.Parameters)
		if err != nil {
			return nil, fmt.Errorf("anthropic: tool %q: %w", spec.Name, err)
		}
		tool := &anthropicsdk.ToolParam{
			Name:        spec.Name,
			Description: param.NewOpt(spec.Description),
			InputSchema: schema,
		}
		tools = append(tools, anthropicsdk.ToolUnionParam{OfTool: tool})
	}
	return tools, nil
}

func convertSchema(parameters map[string]any) (anthropicsdk.ToolInputSchemaParam, error) {
	schema := anthropicsdk.ToolInputSchemaParam{
		ExtraFields: make(map[string]any),
	}
	if parameters == nil {
		schema.Properties = map[string]any{}
		return schema, nil
	}
	for key, value := range parameters {
		switch key {
		case "type":
			if value != "object" {
				return anthropicsdk.ToolInputSchemaParam{}, fmt.Errorf("input schema type must be %q", "object")
			}
		case "properties":
			schema.Properties = value
		case "required":
			required, err := stringsFromAny(value)
			if err != nil {
				return anthropicsdk.ToolInputSchemaParam{}, fmt.Errorf("required: %w", err)
			}
			schema.Required = required
		default:
			schema.ExtraFields[key] = value
		}
	}
	if schema.Properties == nil {
		schema.Properties = map[string]any{}
	}
	return schema, nil
}

func stringsFromAny(value any) ([]string, error) {
	switch typed := value.(type) {
	case nil:
		return nil, nil
	case []string:
		return append([]string(nil), typed...), nil
	case []any:
		result := make([]string, 0, len(typed))
		for _, item := range typed {
			text, ok := item.(string)
			if !ok {
				return nil, fmt.Errorf("expected strings, got %T", item)
			}
			result = append(result, text)
		}
		return result, nil
	default:
		return nil, fmt.Errorf("expected an array of strings, got %T", value)
	}
}

func convertToolChoice(choice ai.ToolChoice) (anthropicsdk.ToolChoiceUnionParam, bool) {
	switch choice.Mode {
	case "":
		return anthropicsdk.ToolChoiceUnionParam{}, false
	case ai.ToolChoiceAuto:
		return anthropicsdk.ToolChoiceUnionParam{
			OfAuto: &anthropicsdk.ToolChoiceAutoParam{},
		}, true
	case ai.ToolChoiceNone:
		none := anthropicsdk.NewToolChoiceNoneParam()
		return anthropicsdk.ToolChoiceUnionParam{OfNone: &none}, true
	case ai.ToolChoiceRequired:
		return anthropicsdk.ToolChoiceUnionParam{
			OfAny: &anthropicsdk.ToolChoiceAnyParam{},
		}, true
	case ai.ToolChoiceNamed:
		if choice.Name == "" {
			return anthropicsdk.ToolChoiceUnionParam{}, false
		}
		return anthropicsdk.ToolChoiceParamOfTool(choice.Name), true
	default:
		return anthropicsdk.ToolChoiceUnionParam{}, false
	}
}

type contentBlock struct {
	kind         string
	key          string
	id           string
	name         string
	signature    string
	redacted     bool
	textOpen     bool
	thinkingOpen bool
	toolOpen     bool
}

type messageStream struct {
	raw     messageEventStream
	info    ai.ModelInfo
	builder *streaming.Builder
	pending []ai.Event
	started bool
	done    bool
	blocks  map[int64]*contentBlock
	order   []int64
	usage   usageTotals
	stop    ai.StopReason
}

type messageEventStream interface {
	Next() bool
	Current() anthropicsdk.MessageStreamEventUnion
	Err() error
	Close() error
}

func (s *messageStream) Next() (ai.Event, error) {
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
			s.done = true
			return s.builder.Fail(
				errors.New("anthropic: stream ended without a message_stop event"),
				ai.StopReasonError,
			), nil
		}
		s.consume(s.raw.Current())
	}
}

func (s *messageStream) consume(event anthropicsdk.MessageStreamEventUnion) {
	switch event.Type {
	case "message_start":
		start := event.AsMessageStart()
		s.usage.setMessage(start.Message.Usage)
	case "content_block_start":
		start := event.AsContentBlockStart()
		block := s.ensureBlock(start.Index)
		switch start.ContentBlock.Type {
		case "text":
			block.kind = "text"
			block.key = "text:" + strconv.FormatInt(start.Index, 10)
			if !block.textOpen {
				block.textOpen = true
				s.pending = append(s.pending, s.builder.TextStart(block.key))
			}
			if start.ContentBlock.Text != "" {
				s.pending = append(s.pending, s.builder.TextDelta(block.key, start.ContentBlock.Text))
			}
		case "thinking":
			block.kind = "thinking"
			block.key = "thinking:" + strconv.FormatInt(start.Index, 10)
			block.signature = start.ContentBlock.Signature
			if !block.thinkingOpen {
				block.thinkingOpen = true
				s.pending = append(s.pending, s.builder.ThinkingStart(block.key))
			}
			if start.ContentBlock.Thinking != "" {
				s.pending = append(s.pending, s.builder.ThinkingDelta(
					block.key,
					start.ContentBlock.Thinking,
				))
			}
			if block.signature != "" {
				s.builder.SetThinkingMeta(block.key, "", block.signature, false)
			}
		case "redacted_thinking":
			block.kind = "thinking"
			block.key = "thinking:" + strconv.FormatInt(start.Index, 10)
			block.redacted = true
			block.signature = start.ContentBlock.Data
			if !block.thinkingOpen {
				block.thinkingOpen = true
				s.pending = append(s.pending, s.builder.ThinkingStart(block.key))
			}
			s.builder.SetThinkingMeta(block.key, "", block.signature, true)
		case "tool_use":
			block.kind = "tool"
			block.key = "tool:" + strconv.FormatInt(start.Index, 10)
			block.id = start.ContentBlock.ID
			block.name = start.ContentBlock.Name
			if !block.toolOpen {
				block.toolOpen = true
				s.pending = append(s.pending, s.builder.ToolStart(block.key, block.id, block.name))
			}
		default:
			block.kind = start.ContentBlock.Type
		}
	case "content_block_delta":
		delta := event.AsContentBlockDelta()
		block := s.ensureBlock(delta.Index)
		switch delta.Delta.Type {
		case "text_delta":
			if block.kind == "" {
				block.kind = "text"
				block.key = "text:" + strconv.FormatInt(delta.Index, 10)
			}
			if block.kind == "text" {
				if !block.textOpen {
					block.textOpen = true
					s.pending = append(s.pending, s.builder.TextStart(block.key))
				}
				if delta.Delta.Text != "" {
					s.pending = append(s.pending, s.builder.TextDelta(block.key, delta.Delta.Text))
				}
			}
		case "thinking_delta":
			if block.kind == "" {
				block.kind = "thinking"
				block.key = "thinking:" + strconv.FormatInt(delta.Index, 10)
			}
			if block.kind == "thinking" {
				if !block.thinkingOpen {
					block.thinkingOpen = true
					s.pending = append(s.pending, s.builder.ThinkingStart(block.key))
				}
				if delta.Delta.Thinking != "" {
					s.pending = append(s.pending, s.builder.ThinkingDelta(
						block.key,
						delta.Delta.Thinking,
					))
				}
			}
		case "signature_delta":
			if block.kind == "" {
				block.kind = "thinking"
				block.key = "thinking:" + strconv.FormatInt(delta.Index, 10)
			}
			if block.kind == "thinking" {
				block.signature += delta.Delta.Signature
				s.builder.SetThinkingMeta(block.key, "", block.signature, block.redacted)
			}
		case "input_json_delta":
			if block.kind == "" {
				block.kind = "tool"
				block.key = "tool:" + strconv.FormatInt(delta.Index, 10)
			}
			if block.kind == "tool" {
				if !block.toolOpen {
					block.toolOpen = true
					s.pending = append(s.pending, s.builder.ToolStart(block.key, block.id, block.name))
				}
				if delta.Delta.PartialJSON != "" {
					s.pending = append(s.pending, s.builder.ToolDelta(block.key, delta.Delta.PartialJSON))
				}
			}
		}
	case "content_block_stop":
		stop := event.AsContentBlockStop()
		block := s.blocks[stop.Index]
		if block == nil {
			return
		}
		switch block.kind {
		case "text":
			if block.textOpen {
				block.textOpen = false
				s.pending = append(s.pending, s.builder.TextEnd(block.key, ""))
			}
		case "thinking":
			if block.thinkingOpen {
				block.thinkingOpen = false
				s.pending = append(s.pending, s.builder.ThinkingEnd(block.key))
			}
		case "tool":
			if block.toolOpen {
				block.toolOpen = false
				end, err := s.builder.ToolEnd(block.key, nil)
				if err != nil {
					s.pending = append(s.pending, s.builder.Fail(err, ai.StopReasonError))
					s.done = true
					return
				}
				s.pending = append(s.pending, end)
			}
		}
	case "message_delta":
		delta := event.AsMessageDelta()
		s.stop = stopReason(delta.Delta.StopReason)
		s.usage.applyDelta(delta.Usage)
	case "message_stop":
		s.finalize()
	case "error":
		s.done = true
		s.pending = append(s.pending, s.builder.Fail(
			fmt.Errorf("anthropic: %s", event.RawJSON()),
			ai.StopReasonError,
		))
	}
}

func (s *messageStream) ensureBlock(index int64) *contentBlock {
	if block, ok := s.blocks[index]; ok {
		return block
	}
	block := &contentBlock{}
	s.blocks[index] = block
	s.order = append(s.order, index)
	return block
}

func (s *messageStream) finalize() {
	if s.done {
		return
	}
	for _, index := range s.order {
		block := s.blocks[index]
		switch block.kind {
		case "text":
			if block.textOpen {
				block.textOpen = false
				s.pending = append(s.pending, s.builder.TextEnd(block.key, ""))
			}
		case "thinking":
			if block.thinkingOpen {
				block.thinkingOpen = false
				s.pending = append(s.pending, s.builder.ThinkingEnd(block.key))
			}
		case "tool":
			if block.toolOpen {
				block.toolOpen = false
				end, err := s.builder.ToolEnd(block.key, nil)
				if err != nil {
					s.pending = append(s.pending, s.builder.Fail(err, ai.StopReasonError))
					s.done = true
					return
				}
				s.pending = append(s.pending, end)
			}
		}
	}
	if usage := s.usage.normalized(); usage != (ai.Usage{}) {
		s.pending = append(s.pending, s.builder.SetUsage(usage))
	}
	reason := s.stop
	if reason == "" {
		reason = ai.StopReasonStop
	}
	s.pending = append(s.pending, s.builder.Done(reason))
	s.done = true
}

func (s *messageStream) Close() error {
	if s == nil || s.raw == nil {
		return nil
	}
	return s.raw.Close()
}

func (s *messageStream) normalizeError(err error) error {
	if err == nil {
		return nil
	}
	var apiErr *anthropicsdk.Error
	if errors.As(err, &apiErr) {
		message := apiErr.RawJSON()
		if message == "" {
			message = http.StatusText(apiErr.StatusCode)
		}
		return &ai.APIError{
			Provider:   s.info.Provider,
			StatusCode: apiErr.StatusCode,
			Code:       string(apiErr.Type()),
			Message:    message,
			RequestID:  apiErr.RequestID,
			Retryable:  apiErr.StatusCode == http.StatusTooManyRequests || apiErr.StatusCode >= 500,
			Err:        err,
		}
	}
	return err
}

type usageTotals struct {
	input      int64
	output     int64
	cacheRead  int64
	cacheWrite int64
	reasoning  int64
}

func (u *usageTotals) setMessage(usage anthropicsdk.Usage) {
	u.input = usage.InputTokens
	u.output = usage.OutputTokens
	u.cacheRead = usage.CacheReadInputTokens
	u.cacheWrite = usage.CacheCreationInputTokens
	u.reasoning = usage.OutputTokensDetails.ThinkingTokens
}

func (u *usageTotals) applyDelta(usage anthropicsdk.MessageDeltaUsage) {
	if usage.JSON.InputTokens.Valid() {
		u.input = usage.InputTokens
	}
	if usage.JSON.OutputTokens.Valid() {
		u.output = usage.OutputTokens
	}
	if usage.JSON.CacheReadInputTokens.Valid() {
		u.cacheRead = usage.CacheReadInputTokens
	}
	if usage.JSON.CacheCreationInputTokens.Valid() {
		u.cacheWrite = usage.CacheCreationInputTokens
	}
	if usage.JSON.OutputTokensDetails.Valid() {
		u.reasoning = usage.OutputTokensDetails.ThinkingTokens
	}
}

func (u usageTotals) normalized() ai.Usage {
	prompt := u.input + u.cacheRead + u.cacheWrite
	return ai.Usage{
		PromptTokens:     prompt,
		CompletionTokens: u.output,
		CacheReadTokens:  u.cacheRead,
		CacheWriteTokens: u.cacheWrite,
		ReasoningTokens:  u.reasoning,
		TotalTokens:      prompt + u.output,
	}
}

func stopReason(reason anthropicsdk.StopReason) ai.StopReason {
	switch reason {
	case "tool_use":
		return ai.StopReasonToolCalls
	case "max_tokens", "model_context_window_exceeded":
		return ai.StopReasonLength
	case "refusal":
		return ai.StopReasonError
	case "end_turn", "stop_sequence", "pause_turn", "":
		return ai.StopReasonStop
	default:
		return ai.StopReasonStop
	}
}
