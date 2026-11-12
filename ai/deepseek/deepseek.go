// Package deepseek adapts DeepSeek's OpenAI-compatible Chat Completions API to
// the ai model interface, including its thinking request and replay format.
package deepseek

import (
	"context"
	"errors"
	"net/http"
	"os"
	"strings"

	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/ai/openai"
)

// DefaultBaseURL is the DeepSeek API base URL.
const DefaultBaseURL = "https://api.deepseek.com"

// Config configures a DeepSeek model. APIKey falls back to DEEPSEEK_API_KEY
// when empty.
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

// Provider is a DeepSeek Chat Completions model.
type Provider struct {
	*openai.CompletionsProvider
}

// New creates a DeepSeek Chat Completions provider.
func New(config Config) (*Provider, error) {
	model := strings.TrimSpace(config.Model)
	if model == "" {
		return nil, errors.New("deepseek: model must not be empty")
	}
	apiKey := strings.TrimSpace(config.APIKey)
	if apiKey == "" {
		apiKey = strings.TrimSpace(os.Getenv("DEEPSEEK_API_KEY"))
	}
	if apiKey == "" {
		return nil, errors.New("deepseek: API key must not be empty")
	}
	baseURL := strings.TrimRight(strings.TrimSpace(config.BaseURL), "/")
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	provider := strings.TrimSpace(config.Provider)
	if provider == "" {
		provider = "deepseek"
	}

	completions, err := openai.NewCompletions(openai.Config{
		APIKey:           apiKey,
		BaseURL:          baseURL,
		Model:            model,
		Provider:         provider,
		Headers:          config.Headers,
		ReasoningFormat:  openai.ReasoningFormatDeepSeek,
		SupportsThinking: true,
		HTTPClient:       config.HTTPClient,
		MaxRetries:       config.MaxRetries,
		ContextWindow:    config.ContextWindow,
		MaxOutputTokens:  config.MaxOutputTokens,
	})
	if err != nil {
		return nil, err
	}
	return &Provider{CompletionsProvider: completions}, nil
}

var _ ai.Model = (*Provider)(nil)

// Info returns DeepSeek model metadata. DeepSeek Chat Completions accepts text
// and tools, but not image input.
func (p *Provider) Info() ai.ModelInfo {
	if p == nil || p.CompletionsProvider == nil {
		return ai.ModelInfo{}
	}
	info := p.CompletionsProvider.Info()
	info.SupportsImages = false
	return info
}

// Stream sends one normalized turn to DeepSeek.
func (p *Provider) Stream(ctx context.Context, request ai.Request) (ai.Stream, error) {
	if p == nil || p.CompletionsProvider == nil {
		return nil, errors.New("deepseek: provider is nil")
	}
	return p.CompletionsProvider.Stream(ctx, request)
}
