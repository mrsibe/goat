// Package xai adapts xAI's OpenAI-compatible Responses API to the ai model
// interface.
package xai

import (
	"errors"
	"net/http"
	"os"
	"strings"

	"github.com/mrsibe/goat/ai/openai"
)

// DefaultBaseURL is the xAI API base URL.
const DefaultBaseURL = "https://api.x.ai/v1"

// Config configures an xAI model. APIKey falls back to XAI_API_KEY when empty.
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

// New creates an xAI Responses provider.
func New(config Config) (*openai.ResponsesProvider, error) {
	model := strings.TrimSpace(config.Model)
	if model == "" {
		return nil, errors.New("xai: model must not be empty")
	}
	apiKey := strings.TrimSpace(config.APIKey)
	if apiKey == "" {
		apiKey = strings.TrimSpace(os.Getenv("XAI_API_KEY"))
	}
	if apiKey == "" {
		return nil, errors.New("xai: API key must not be empty")
	}
	baseURL := strings.TrimRight(strings.TrimSpace(config.BaseURL), "/")
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	provider := strings.TrimSpace(config.Provider)
	if provider == "" {
		provider = "xai"
	}
	return openai.NewResponses(openai.Config{
		APIKey:          apiKey,
		BaseURL:         baseURL,
		Model:           model,
		Provider:        provider,
		Headers:         config.Headers,
		HTTPClient:      config.HTTPClient,
		MaxRetries:      config.MaxRetries,
		ContextWindow:   config.ContextWindow,
		MaxOutputTokens: config.MaxOutputTokens,
	})
}
