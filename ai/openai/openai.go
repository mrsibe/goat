// Package openai adapts OpenAI Chat Completions and Responses APIs to the ai
// model interface.
package openai

import (
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/mrsibe/goat/ai"
	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// ReasoningFormat selects the request and replay wire format used by
// OpenAI-compatible Chat Completions endpoints.
type ReasoningFormat string

const (
	// ReasoningFormatOpenAI uses the standard top-level reasoning_effort field.
	ReasoningFormatOpenAI ReasoningFormat = "openai"
	// ReasoningFormatDeepSeek uses DeepSeek's thinking object and
	// reasoning_content replay field.
	ReasoningFormatDeepSeek ReasoningFormat = "deepseek"
	// ReasoningFormatOpenRouter uses OpenRouter's nested reasoning object and
	// reasoning_details replay metadata.
	ReasoningFormatOpenRouter ReasoningFormat = "openrouter"
)

// Config configures an OpenAI-compatible model. APIKey may be empty for local
// servers that do not require authentication.
type Config struct {
	APIKey           string
	BaseURL          string
	Model            string
	Provider         string
	Headers          map[string]string
	Query            map[string]string
	ReasoningFormat  ReasoningFormat
	SupportsThinking bool
	// DisableEnvAuth prevents the SDK from reading OPENAI_API_KEY from the
	// environment. Adapters that authenticate with another header use this to
	// avoid sending an unrelated Authorization header.
	DisableEnvAuth  bool
	HTTPClient      *http.Client
	MaxRetries      *int
	ContextWindow   int
	MaxOutputTokens int
}

type providerCore struct {
	client          openaisdk.Client
	info            ai.ModelInfo
	reasoningFormat ReasoningFormat
}

func newProviderCore(config Config, api ai.API) (providerCore, error) {
	if strings.TrimSpace(config.Model) == "" {
		return providerCore{}, errors.New("openai: model must not be empty")
	}
	reasoningFormat := ReasoningFormat(strings.TrimSpace(string(config.ReasoningFormat)))
	if reasoningFormat == "" {
		reasoningFormat = ReasoningFormatOpenAI
	}
	switch reasoningFormat {
	case ReasoningFormatOpenAI, ReasoningFormatDeepSeek, ReasoningFormatOpenRouter:
	default:
		return providerCore{}, fmt.Errorf("openai: unsupported reasoning format %q", config.ReasoningFormat)
	}

	var options []option.RequestOption
	if config.DisableEnvAuth {
		options = append(options, option.WithAPIKey(""))
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
	for key, value := range config.Query {
		if strings.TrimSpace(key) == "" {
			continue
		}
		options = append(options, option.WithQueryAdd(key, value))
	}
	provider := strings.TrimSpace(config.Provider)
	if provider == "" {
		provider = "openai"
	}

	info := ai.ModelInfo{
		Provider:        provider,
		Model:           config.Model,
		API:             api,
		ContextWindow:   config.ContextWindow,
		MaxOutputTokens: config.MaxOutputTokens,
		SupportsTools:   true,
	}
	switch api {
	case ai.APIOpenAICompletions:
		info.SupportsImages = true
		info.SupportsThinking = config.SupportsThinking
	case ai.APIOpenAIResponses:
		info.SupportsImages = true
		info.SupportsToolResultImages = true
		info.SupportsThinking = true
	}

	return providerCore{
		client:          openaisdk.NewClient(options...),
		info:            info,
		reasoningFormat: reasoningFormat,
	}, nil
}

// New creates a Chat Completions provider. It is equivalent to
// [NewCompletions] and preserves the original GOAT entry point.
func New(config Config) (*CompletionsProvider, error) {
	return NewCompletions(config)
}

// Compile-time check for the shared model interface.
var _ ai.Model = (*CompletionsProvider)(nil)

var _ ai.Model = (*ResponsesProvider)(nil)

// Provider is the original name for a Chat Completions model. New code should
// use [CompletionsProvider] or [ResponsesProvider] explicitly.
type Provider = CompletionsProvider

func (p *providerCore) Info() ai.ModelInfo {
	if p == nil {
		return ai.ModelInfo{}
	}
	return p.info
}

func (p *providerCore) normalizeError(err error) error {
	return normalizeOpenAIError(p.info.Provider, err)
}

func normalizeOpenAIError(provider string, err error) error {
	if err == nil {
		return nil
	}
	var apiErr *openaisdk.Error
	if errors.As(err, &apiErr) {
		requestID := ""
		if apiErr.Response != nil {
			requestID = apiErr.Response.Header.Get("x-request-id")
		}
		return &ai.APIError{
			Provider:   provider,
			StatusCode: apiErr.StatusCode,
			Code:       apiErr.Code,
			Message:    apiErr.Message,
			RequestID:  requestID,
			Retryable:  apiErr.StatusCode == http.StatusTooManyRequests || apiErr.StatusCode >= 500,
			Err:        err,
		}
	}
	return err
}

func (p *providerCore) validateStream(stream ai.Stream, err error) (ai.Stream, error) {
	if err != nil {
		return nil, p.normalizeError(err)
	}
	if stream == nil {
		return nil, fmt.Errorf("openai: provider returned a nil stream")
	}
	return stream, nil
}

func imageURL(image *ai.ImageContent) (string, error) {
	if image == nil {
		return "", errors.New("openai: image block is missing its payload")
	}
	data := strings.TrimSpace(image.Data)
	url := strings.TrimSpace(image.URL)
	if (data == "") == (url == "") {
		return "", errors.New("openai: image must set exactly one of data or url")
	}
	if url != "" {
		return url, nil
	}
	mediaType := strings.TrimSpace(image.MediaType)
	if mediaType == "" {
		return "", errors.New("openai: inline image requires a media type")
	}
	return "data:" + mediaType + ";base64," + data, nil
}
