package openrouter_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/ai/openrouter"
)

type capturedRequest struct {
	path          string
	authorization string
	apiKey        string
	body          []byte
}

func TestCompletionsUsesDefaultsAndEnvironmentKey(t *testing.T) {
	t.Setenv("OPENROUTER_API_KEY", "env-key")
	requests := make(chan capturedRequest, 1)
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			return nil, err
		}
		requests <- capturedRequest{
			path:          request.URL.Path,
			authorization: request.Header.Get("Authorization"),
			body:          body,
		}
		return completionSSE(request, "hi"), nil
	})}

	provider, err := openrouter.NewCompletions(openrouter.Config{
		Model:      "anthropic/claude-sonnet-4",
		HTTPClient: client,
	})
	if err != nil {
		t.Fatal(err)
	}
	info := provider.Info()
	if info.Provider != "openrouter" || info.API != ai.APIOpenAICompletions ||
		!info.SupportsThinking || !info.SupportsImages {
		t.Fatalf("info = %+v", info)
	}
	stream, err := provider.Stream(context.Background(), ai.Request{
		Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hi")},
	})
	if err != nil {
		t.Fatal(err)
	}
	drainStream(t, stream)

	captured := <-requests
	if captured.path != "/api/v1/chat/completions" {
		t.Fatalf("path = %q", captured.path)
	}
	if captured.authorization != "Bearer env-key" {
		t.Fatalf("Authorization = %q", captured.authorization)
	}
	var body map[string]any
	if err := json.Unmarshal(captured.body, &body); err != nil {
		t.Fatal(err)
	}
	if body["model"] != "anthropic/claude-sonnet-4" {
		t.Fatalf("model = %+v", body["model"])
	}
}

func TestMessagesUsesBearerAuthentication(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "anthropic-env-key")
	requests := make(chan capturedRequest, 1)
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		requests <- capturedRequest{
			path:          request.URL.Path,
			authorization: request.Header.Get("Authorization"),
			apiKey:        request.Header.Get("x-api-key"),
		}
		return anthropicSSE(request), nil
	})}

	provider, err := openrouter.NewMessages(openrouter.Config{
		APIKey:     "openrouter-key",
		Model:      "anthropic/claude-sonnet-4",
		HTTPClient: client,
	})
	if err != nil {
		t.Fatal(err)
	}
	info := provider.Info()
	if info.Provider != "openrouter" || info.API != ai.APIAnthropicMessages || !info.SupportsThinking {
		t.Fatalf("info = %+v", info)
	}
	stream, err := provider.Stream(context.Background(), ai.Request{
		Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hi")},
	})
	if err != nil {
		t.Fatal(err)
	}
	drainStream(t, stream)

	captured := <-requests
	if captured.path != "/api/v1/messages" {
		t.Fatalf("path = %q", captured.path)
	}
	if captured.authorization != "Bearer openrouter-key" {
		t.Fatalf("Authorization = %q", captured.authorization)
	}
	if captured.apiKey != "" {
		t.Fatalf("x-api-key = %q", captured.apiKey)
	}
}

func TestProviderRequiresAPIKey(t *testing.T) {
	t.Setenv("OPENROUTER_API_KEY", "")
	if _, err := openrouter.NewCompletions(openrouter.Config{Model: "model"}); err == nil ||
		!strings.Contains(err.Error(), "API key") {
		t.Fatalf("error = %v", err)
	}
}

func completionSSE(request *http.Request, content string) *http.Response {
	var body strings.Builder
	fmt.Fprintf(&body, "data: %s\n\n", `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"model","choices":[{"index":0,"delta":{"role":"assistant","content":"`+content+`"},"finish_reason":"stop"}]}`)
	body.WriteString("data: [DONE]\n\n")
	return &http.Response{
		StatusCode: http.StatusOK,
		Status:     "200 OK",
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader(body.String())),
		Request:    request,
	}
}

func anthropicSSE(request *http.Request) *http.Response {
	var body strings.Builder
	body.WriteString("event: message_start\ndata: ")
	body.WriteString(`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"usage":{"input_tokens":1,"output_tokens":1}}}`)
	body.WriteString("\n\n")
	body.WriteString("event: message_stop\ndata: ")
	body.WriteString(`{"type":"message_stop"}`)
	body.WriteString("\n\n")
	return &http.Response{
		StatusCode: http.StatusOK,
		Status:     "200 OK",
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader(body.String())),
		Request:    request,
	}
}

func drainStream(t *testing.T, stream ai.Stream) {
	t.Helper()
	defer stream.Close()
	for {
		event, err := stream.Next()
		if err == io.EOF {
			return
		}
		if err != nil {
			t.Fatal(err)
		}
		if event.Type == ai.EventError {
			t.Fatalf("stream error event: %s", event.Error)
		}
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return fn(request)
}
