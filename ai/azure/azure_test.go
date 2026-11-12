package azure

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/ai/internal/modeltest"
)

type capturedRequest struct {
	path          string
	query         string
	apiKey        string
	authorization string
	body          []byte
}

type requestBody struct {
	Model string `json:"model"`
}

func TestNormalizeBaseURL(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"azure root", "https://my-resource.openai.azure.com", "https://my-resource.openai.azure.com/openai/v1"},
		{"cognitive services root", "https://my-resource.cognitiveservices.azure.com/", "https://my-resource.cognitiveservices.azure.com/openai/v1"},
		{"foundry root", "https://my-resource.ai.azure.com", "https://my-resource.ai.azure.com/openai/v1"},
		{"openai path", "https://my-resource.cognitiveservices.azure.com/openai", "https://my-resource.cognitiveservices.azure.com/openai/v1"},
		{"responses path", "https://my-resource.services.ai.azure.com/openai/v1/responses", "https://my-resource.services.ai.azure.com/openai/v1"},
		{"preserves v1", "https://my-resource.openai.azure.com/openai/v1", "https://my-resource.openai.azure.com/openai/v1"},
		{"strips azure query", "https://my-resource.openai.azure.com/openai?api-version=2024-12-01", "https://my-resource.openai.azure.com/openai/v1"},
		{"proxy path", "https://my-proxy.example.com/v1", "https://my-proxy.example.com/v1"},
		{"proxy query", "https://my-proxy.example.com/v1?custom=true", "https://my-proxy.example.com/v1?custom=true"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := normalizeBaseURL(test.in)
			if err != nil {
				t.Fatal(err)
			}
			if got != test.want {
				t.Fatalf("normalizeBaseURL(%q) = %q, want %q", test.in, got, test.want)
			}
		})
	}
}

func TestNormalizeBaseURLRejectsInvalidURL(t *testing.T) {
	for _, value := range []string{"not-a-url", "ftp://example.com", "https://"} {
		if _, err := normalizeBaseURL(value); err == nil {
			t.Fatalf("normalizeBaseURL(%q) returned no error", value)
		}
	}
}

func TestResolveConfigBuildsResourceEndpoint(t *testing.T) {
	resolved, err := resolveConfig(Config{APIKey: "key", ResourceName: "my-resource", Model: "gpt-4o-mini"})
	if err != nil {
		t.Fatal(err)
	}
	if resolved.baseURL != "https://my-resource.openai.azure.com/openai/v1" {
		t.Fatalf("base URL = %q", resolved.baseURL)
	}
	if resolved.apiVersion != "v1" {
		t.Fatalf("api version = %q", resolved.apiVersion)
	}
	if resolved.provider != "azure" {
		t.Fatalf("provider = %q", resolved.provider)
	}
}

func TestResolveConfigUsesDeploymentOverride(t *testing.T) {
	resolved, err := resolveConfig(Config{
		APIKey:     "key",
		Endpoint:   "https://example.test/openai/v1",
		Model:      "gpt-4o-mini",
		Deployment: "my-deployment",
		APIVersion: "2025-01-01",
	})
	if err != nil {
		t.Fatal(err)
	}
	if resolved.model != "my-deployment" {
		t.Fatalf("model = %q", resolved.model)
	}
	if resolved.apiVersion != "2025-01-01" {
		t.Fatalf("api version = %q", resolved.apiVersion)
	}
}

func TestResolveConfigRequiresEndpointAndAuth(t *testing.T) {
	if _, err := resolveConfig(Config{APIKey: "key", Model: "gpt-4o-mini"}); err == nil {
		t.Fatal("expected missing endpoint error")
	}
	if _, err := resolveConfig(Config{Endpoint: "https://example.test/openai/v1", Model: "gpt-4o-mini"}); err == nil {
		t.Fatal("expected missing API key error")
	}
	if _, err := resolveConfig(Config{
		Endpoint: "https://example.test/openai/v1",
		Model:    "gpt-4o-mini",
		Headers:  map[string]string{"Authorization": "Bearer token"},
	}); err != nil {
		t.Fatalf("authorization header should be accepted: %v", err)
	}
}

func TestCompletionsSendsAzureAuthPathAndQuery(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "env-key")
	requests := make(chan capturedRequest, 1)
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			return nil, err
		}
		requests <- capturedRequest{
			path:          request.URL.Path,
			query:         request.URL.RawQuery,
			apiKey:        request.Header.Get("api-key"),
			authorization: request.Header.Get("Authorization"),
			body:          body,
		}
		return sseResponse(request, `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"deployment-1","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`), nil
	})}

	provider, err := NewCompletions(Config{
		APIKey:     "azure-key",
		Endpoint:   "https://example.test/openai/v1",
		Model:      "gpt-4o-mini",
		Deployment: "deployment-1",
		HTTPClient: client,
	})
	if err != nil {
		t.Fatal(err)
	}
	if provider.Info().Provider != "azure" || provider.Info().Model != "deployment-1" {
		t.Fatalf("info = %+v", provider.Info())
	}

	if _, err := modeltest.Collect(context.Background(), provider, ai.Request{
		Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hi")},
	}); err != nil {
		t.Fatal(err)
	}

	captured := <-requests
	if captured.path != "/openai/v1/chat/completions" {
		t.Fatalf("path = %q", captured.path)
	}
	if captured.query != "api-version=v1" {
		t.Fatalf("query = %q", captured.query)
	}
	if captured.apiKey != "azure-key" {
		t.Fatalf("api-key = %q", captured.apiKey)
	}
	if captured.authorization != "" {
		t.Fatalf("unexpected Authorization header %q", captured.authorization)
	}
	var body requestBody
	if err := json.Unmarshal(captured.body, &body); err != nil {
		t.Fatal(err)
	}
	if body.Model != "deployment-1" {
		t.Fatalf("request model = %q", body.Model)
	}
}

func TestResponsesSendsAzurePathAndReportsAPI(t *testing.T) {
	requests := make(chan capturedRequest, 1)
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			return nil, err
		}
		requests <- capturedRequest{
			path:   request.URL.Path,
			query:  request.URL.RawQuery,
			apiKey: request.Header.Get("api-key"),
			body:   body,
		}
		return sseResponse(request, `{"type":"response.completed","response":{"id":"resp_1","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`), nil
	})}

	provider, err := NewResponses(Config{
		APIKey:     "azure-key",
		Endpoint:   "https://example.test/openai/v1",
		Model:      "gpt-4o-mini",
		Deployment: "deployment-1",
		HTTPClient: client,
	})
	if err != nil {
		t.Fatal(err)
	}
	if provider.Info().API != ai.APIAzureOpenAIResponses {
		t.Fatalf("info = %+v", provider.Info())
	}

	if _, err := modeltest.Collect(context.Background(), provider, ai.Request{
		Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hi")},
	}); err != nil {
		t.Fatal(err)
	}

	captured := <-requests
	if captured.path != "/openai/v1/responses" {
		t.Fatalf("path = %q", captured.path)
	}
	if captured.query != "api-version=v1" {
		t.Fatalf("query = %q", captured.query)
	}
	if captured.apiKey != "azure-key" {
		t.Fatalf("api-key = %q", captured.apiKey)
	}
	var body requestBody
	if err := json.Unmarshal(captured.body, &body); err != nil {
		t.Fatal(err)
	}
	if body.Model != "deployment-1" {
		t.Fatalf("request model = %q", body.Model)
	}
}

func TestResponsesNilProvider(t *testing.T) {
	var provider *ResponsesProvider
	if provider.Info() != (ai.ModelInfo{}) {
		t.Fatalf("info = %+v", provider.Info())
	}
	if _, err := provider.Stream(context.Background(), ai.Request{}); err == nil {
		t.Fatal("expected nil provider error")
	}
}

func sseResponse(request *http.Request, chunks ...string) *http.Response {
	var body strings.Builder
	for _, chunk := range chunks {
		body.WriteString("data: ")
		body.WriteString(chunk)
		body.WriteString("\n\n")
	}
	body.WriteString("data: [DONE]\n\n")
	return &http.Response{
		StatusCode: http.StatusOK,
		Status:     "200 OK",
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader(body.String())),
		Request:    request,
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return f(request)
}
