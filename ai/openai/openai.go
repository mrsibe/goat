// Package openai adapts OpenAI-compatible Chat Completions APIs to the ai model
// interface.
package openai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/mrsibe/goat/ai"
	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/shared"
)

// Config configures an OpenAI-compatible provider. APIKey may be empty for
// local servers that do not require authentication.
type Config struct {
	APIKey     string
	BaseURL    string
	Model      string
	HTTPClient *http.Client
}

// Provider implements ai.Model using Chat Completions streaming.
type Provider struct {
	client openaisdk.Client
	model  string
}

// New creates an OpenAI-compatible provider.
func New(config Config) (*Provider, error) {
	if strings.TrimSpace(config.Model) == "" {
		return nil, errors.New("openai: model must not be empty")
	}

	var options []option.RequestOption
	if config.APIKey != "" {
		options = append(options, option.WithAPIKey(config.APIKey))
	}
	if config.BaseURL != "" {
		options = append(options, option.WithBaseURL(config.BaseURL))
	}
	if config.HTTPClient != nil {
		options = append(options, option.WithHTTPClient(config.HTTPClient))
	}

	return &Provider{
		client: openaisdk.NewClient(options...),
		model:  config.Model,
	}, nil
}

// Stream starts one streamed Chat Completions request.
func (p *Provider) Stream(ctx context.Context, request ai.Request) (ai.Stream, error) {
	if p == nil {
		return nil, errors.New("openai: provider is nil")
	}

	params, err := p.params(request)
	if err != nil {
		return nil, err
	}

	return &streamAdapter{
		stream: p.client.Chat.Completions.NewStreaming(ctx, params),
	}, nil
}

func (p *Provider) params(request ai.Request) (openaisdk.ChatCompletionNewParams, error) {
	messages := make([]openaisdk.ChatCompletionMessageParamUnion, 0, len(request.Messages))
	for _, message := range request.Messages {
		converted, err := convertMessage(message)
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

	return openaisdk.ChatCompletionNewParams{
		Model:    shared.ChatModel(p.model),
		Messages: messages,
		Tools:    tools,
		StreamOptions: openaisdk.ChatCompletionStreamOptionsParam{
			IncludeUsage: param.NewOpt(true),
		},
	}, nil
}

func convertMessage(message ai.Message) (openaisdk.ChatCompletionMessageParamUnion, error) {
	switch message.Role {
	case ai.RoleSystem:
		return openaisdk.SystemMessage(message.Text()), nil
	case ai.RoleUser:
		return openaisdk.UserMessage(message.Text()), nil
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
		return openaisdk.ChatCompletionMessageParamUnion{OfAssistant: &assistant}, nil
	case ai.RoleTool:
		if message.ToolCallID == "" {
			return openaisdk.ChatCompletionMessageParamUnion{}, errors.New("openai: tool message is missing its tool call id")
		}
		return openaisdk.ToolMessage(message.Text(), message.ToolCallID), nil
	default:
		return openaisdk.ChatCompletionMessageParamUnion{}, fmt.Errorf("openai: unsupported message role %q", message.Role)
	}
}

type rawStream interface {
	Next() bool
	Current() openaisdk.ChatCompletionChunk
	Err() error
	Close() error
}

type streamAdapter struct {
	stream      rawStream
	pending     []ai.Event
	started     bool
	sawToolCall bool
	done        bool
	finishStop  ai.StopReason
}

func (s *streamAdapter) Next() (ai.Event, error) {
	if !s.started {
		s.started = true
		return ai.Event{Type: ai.EventStart}, nil
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
		if !s.stream.Next() {
			if err := s.stream.Err(); err != nil {
				return ai.Event{}, err
			}
			s.done = true
			stopReason := ai.StopReasonStop
			if s.finishStop != "" {
				stopReason = s.finishStop
			} else if s.sawToolCall {
				stopReason = ai.StopReasonToolCalls
			}
			return ai.Event{Type: ai.EventDone, StopReason: stopReason}, nil
		}

		s.consume(s.stream.Current())
	}
}

func (s *streamAdapter) consume(chunk openaisdk.ChatCompletionChunk) {
	if chunk.JSON.Usage.Valid() {
		usage := ai.Usage{
			PromptTokens:     chunk.Usage.PromptTokens,
			CompletionTokens: chunk.Usage.CompletionTokens,
			TotalTokens:      chunk.Usage.TotalTokens,
		}
		s.pending = append(s.pending, ai.Event{Type: ai.EventUsage, Usage: &usage})
	}

	if len(chunk.Choices) == 0 {
		return
	}

	choice := chunk.Choices[0]
	if choice.Delta.Content != "" {
		s.pending = append(s.pending, ai.Event{
			Type:      ai.EventTextDelta,
			TextDelta: choice.Delta.Content,
		})
	}
	for _, call := range choice.Delta.ToolCalls {
		s.sawToolCall = true
		delta := &ai.ToolCallDelta{
			Index:          int(call.Index),
			ID:             call.ID,
			Name:           call.Function.Name,
			ArgumentsDelta: call.Function.Arguments,
		}
		s.pending = append(s.pending, ai.Event{Type: ai.EventToolCallDelta, ToolCall: delta})
	}

	if choice.FinishReason != "" {
		s.finishStop = mapStopReason(choice.FinishReason, s.sawToolCall)
	}
}

func mapStopReason(reason string, sawToolCall bool) ai.StopReason {
	switch reason {
	case "length":
		return ai.StopReasonLength
	case "tool_calls", "function_call":
		return ai.StopReasonToolCalls
	case "stop", "content_filter":
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

func (s *streamAdapter) Close() error {
	if s == nil || s.stream == nil {
		return nil
	}
	return s.stream.Close()
}
