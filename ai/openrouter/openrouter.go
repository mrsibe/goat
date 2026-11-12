// Package openrouter adapts OpenRouter's OpenAI-compatible Chat Completions
// and Anthropic-compatible Messages endpoints to the ai model interface.
package openrouter

import (
	"errors"
	"net/http"
	"os"
	"strings"

	"github.com/mrsibe/goat/ai/anthropic"
	"github.com/mrsibe/goat/ai/openai"
)

// DefaultBaseURL is the OpenRouter API base URL.
const DefaultBaseURL = "https://openrouter.ai/api/v1"

// Config configures an OpenRouter model. APIKey falls back to
// OPENROUTER_API_KEY when empty.
type Config struct {
	APIKey          string
	BaseURL         string
	Model           string
	Provider        string
	Headers         map[string]string
	HTTPClient      *http.Client
	MaxRetries      *int
	ContextWindow   int
	MaxOutputTokens int
}

// NewCompletions creates an OpenRouter Chat Completions provider.
func NewCompletions(config Config) (*openai.CompletionsProvider, error) {
	resolved, err := resolveConfig(config)
	if err != nil {
		return nil, err
	}
	return openai.NewCompletions(openai.Config{
		APIKey:           resolved.apiKey,
		BaseURL:          resolved.baseURL,
		Model:            resolved.model,
		Provider:         resolved.provider,
		Headers:          resolved.headers,
		ReasoningFormat:  openai.ReasoningFormatOpenRouter,
		SupportsThinking: true,
		HTTPClient:       resolved.httpClient,
		MaxRetries:       resolved.maxRetries,
		ContextWindow:    resolved.contextWindow,
		MaxOutputTokens:  resolved.maxOutputTokens,
	})
}

// NewMessages creates an OpenRouter Anthropic Messages provider.
func NewMessages(config Config) (*anthropic.Provider, error) {
	resolved, err := resolveConfig(config)
	if err != nil {
		return nil, err
	}
	return anthropic.New(anthropic.Config{
		AuthToken:       resolved.apiKey,
		BaseURL:         messagesBaseURL(resolved.baseURL),
		Model:           resolved.model,
		Provider:        resolved.provider,
		Headers:         resolved.headers,
		HTTPClient:      resolved.httpClient,
		MaxRetries:      resolved.maxRetries,
		ContextWindow:   resolved.contextWindow,
		MaxOutputTokens: resolved.maxOutputTokens,
	})
}

func messagesBaseURL(baseURL string) string {
	baseURL = strings.TrimRight(baseURL, "/")
	if strings.HasSuffix(baseURL, "/v1/messages") {
		return strings.TrimSuffix(baseURL, "/v1/messages")
	}
	if strings.HasSuffix(baseURL, "/v1") {
		return strings.TrimSuffix(baseURL, "/v1")
	}
	return baseURL
}

type resolvedConfig struct {
	apiKey          string
	baseURL         string
	model           string
	provider        string
	headers         map[string]string
	httpClient      *http.Client
	maxRetries      *int
	contextWindow   int
	maxOutputTokens int
}

func resolveConfig(config Config) (resolvedConfig, error) {
	model := strings.TrimSpace(config.Model)
	if model == "" {
		return resolvedConfig{}, errors.New("openrouter: model must not be empty")
	}
	apiKey := strings.TrimSpace(config.APIKey)
	if apiKey == "" {
		apiKey = strings.TrimSpace(os.Getenv("OPENROUTER_API_KEY"))
	}
	if apiKey == "" {
		return resolvedConfig{}, errors.New("openrouter: API key must not be empty")
	}
	baseURL := strings.TrimRight(strings.TrimSpace(config.BaseURL), "/")
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	provider := strings.TrimSpace(config.Provider)
	if provider == "" {
		provider = "openrouter"
	}
	return resolvedConfig{
		apiKey:          apiKey,
		baseURL:         baseURL,
		model:           model,
		provider:        provider,
		headers:         config.Headers,
		httpClient:      config.HTTPClient,
		maxRetries:      config.MaxRetries,
		contextWindow:   config.ContextWindow,
		maxOutputTokens: config.MaxOutputTokens,
	}, nil
}
