// Package azure adapts Azure OpenAI deployments to the ai model interface.
//
// Azure's v1 API is OpenAI-compatible, so this package reuses the Chat
// Completions and Responses adapters from ai/openai. It resolves Azure
// endpoints, authenticates with the api-key header, and adds the
// api-version query parameter.
package azure

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/ai/openai"
)

// DefaultAPIVersion is the Azure OpenAI API version used when APIVersion is
// empty.
const DefaultAPIVersion = "v1"

// Config configures an Azure OpenAI model. Endpoint takes precedence over
// ResourceName. Deployment takes precedence over Model; Model is otherwise
// treated as the deployment name.
type Config struct {
	APIKey          string
	Endpoint        string
	ResourceName    string
	APIVersion      string
	Model           string
	Deployment      string
	Provider        string
	Headers         map[string]string
	HTTPClient      *http.Client
	MaxRetries      *int
	ContextWindow   int
	MaxOutputTokens int
}

// NewCompletions creates an Azure OpenAI Chat Completions provider.
func NewCompletions(config Config) (*openai.CompletionsProvider, error) {
	resolved, err := resolveConfig(config)
	if err != nil {
		return nil, err
	}
	return openai.NewCompletions(resolved.openaiConfig())
}

// ResponsesProvider is an Azure OpenAI Responses model. It reports the Azure
// Responses API while sharing streaming and message conversion with ai/openai.
type ResponsesProvider struct {
	*openai.ResponsesProvider
}

// NewResponses creates an Azure OpenAI Responses provider.
func NewResponses(config Config) (*ResponsesProvider, error) {
	resolved, err := resolveConfig(config)
	if err != nil {
		return nil, err
	}
	provider, err := openai.NewResponses(resolved.openaiConfig())
	if err != nil {
		return nil, err
	}
	return &ResponsesProvider{ResponsesProvider: provider}, nil
}

var _ ai.Model = (*ResponsesProvider)(nil)

// Info returns the Azure model metadata.
func (p *ResponsesProvider) Info() ai.ModelInfo {
	if p == nil || p.ResponsesProvider == nil {
		return ai.ModelInfo{}
	}
	info := p.ResponsesProvider.Info()
	info.API = ai.APIAzureOpenAIResponses
	return info
}

// Stream sends one normalized turn to the Azure Responses API.
func (p *ResponsesProvider) Stream(ctx context.Context, request ai.Request) (ai.Stream, error) {
	if p == nil || p.ResponsesProvider == nil {
		return nil, errors.New("azure: provider is nil")
	}
	return p.ResponsesProvider.Stream(ctx, request)
}

type resolvedConfig struct {
	baseURL         string
	apiVersion      string
	model           string
	provider        string
	headers         map[string]string
	httpClient      *http.Client
	maxRetries      *int
	contextWindow   int
	maxOutputTokens int
}

func (r resolvedConfig) openaiConfig() openai.Config {
	return openai.Config{
		BaseURL:         r.baseURL,
		Model:           r.model,
		Provider:        r.provider,
		Headers:         r.headers,
		Query:           map[string]string{"api-version": r.apiVersion},
		DisableEnvAuth:  true,
		HTTPClient:      r.httpClient,
		MaxRetries:      r.maxRetries,
		ContextWindow:   r.contextWindow,
		MaxOutputTokens: r.maxOutputTokens,
	}
}

func resolveConfig(config Config) (resolvedConfig, error) {
	baseURL, err := resolveBaseURL(config.Endpoint, config.ResourceName)
	if err != nil {
		return resolvedConfig{}, err
	}

	model := strings.TrimSpace(config.Deployment)
	if model == "" {
		model = strings.TrimSpace(config.Model)
	}
	if model == "" {
		return resolvedConfig{}, errors.New("azure: model or deployment must not be empty")
	}

	apiVersion := strings.TrimSpace(config.APIVersion)
	if apiVersion == "" {
		apiVersion = DefaultAPIVersion
	}

	headers := cloneHeaders(config.Headers)
	if config.APIKey != "" {
		headers["api-key"] = config.APIKey
	} else if !hasHeader(headers, "api-key") && !hasHeader(headers, "authorization") {
		return resolvedConfig{}, errors.New("azure: api key must not be empty")
	}

	provider := strings.TrimSpace(config.Provider)
	if provider == "" {
		provider = "azure"
	}

	return resolvedConfig{
		baseURL:         baseURL,
		apiVersion:      apiVersion,
		model:           model,
		provider:        provider,
		headers:         headers,
		httpClient:      config.HTTPClient,
		maxRetries:      config.MaxRetries,
		contextWindow:   config.ContextWindow,
		maxOutputTokens: config.MaxOutputTokens,
	}, nil
}

func resolveBaseURL(endpoint, resourceName string) (string, error) {
	raw := strings.TrimSpace(endpoint)
	if raw == "" {
		resourceName = strings.TrimSpace(resourceName)
		if resourceName == "" {
			return "", errors.New("azure: endpoint or resource name is required")
		}
		raw = "https://" + resourceName + ".openai.azure.com/openai/v1"
	}
	return normalizeBaseURL(raw)
}

func normalizeBaseURL(raw string) (string, error) {
	trimmed := strings.TrimRight(strings.TrimSpace(raw), "/")
	parsed, err := url.Parse(trimmed)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return "", fmt.Errorf("azure: invalid base URL %q", raw)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "", fmt.Errorf("azure: invalid base URL %q: unsupported scheme", raw)
	}

	host := strings.ToLower(parsed.Hostname())
	isAzureHost := strings.HasSuffix(host, ".openai.azure.com") ||
		strings.HasSuffix(host, ".cognitiveservices.azure.com") ||
		strings.HasSuffix(host, ".ai.azure.com")
	if isAzureHost {
		switch parsed.Path {
		case "", "/", "/openai", "/openai/v1/responses":
			parsed.Path = "/openai/v1"
			parsed.RawPath = ""
			parsed.RawQuery = ""
			parsed.Fragment = ""
		}
	}

	return strings.TrimRight(parsed.String(), "/"), nil
}

func cloneHeaders(headers map[string]string) map[string]string {
	cloned := make(map[string]string, len(headers)+1)
	for key, value := range headers {
		if strings.TrimSpace(key) == "" {
			continue
		}
		cloned[key] = value
	}
	return cloned
}

func hasHeader(headers map[string]string, name string) bool {
	for key, value := range headers {
		if strings.EqualFold(key, name) && strings.TrimSpace(value) != "" {
			return true
		}
	}
	return false
}
