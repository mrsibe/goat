package anthropic_test

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/ai/anthropic"
)

func TestProviderSendsBearerAuthToken(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "env-api-key")
	headersCh := make(chan http.Header, 1)
	var responseBody strings.Builder
	writeSSE(&responseBody, map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test",
			"content": []any{},
			"usage":   map[string]any{"input_tokens": 1, "output_tokens": 1},
		},
	})
	writeSSE(&responseBody, map[string]any{"type": "message_stop"})

	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		headersCh <- request.Header.Clone()
		return &http.Response{
			StatusCode: http.StatusOK,
			Status:     "200 OK",
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:       io.NopCloser(strings.NewReader(responseBody.String())),
			Request:    request,
		}, nil
	})}

	provider, err := anthropic.New(anthropic.Config{
		AuthToken:  "bearer-token",
		BaseURL:    "https://example.test",
		Model:      "claude-test",
		HTTPClient: client,
	})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := provider.Stream(context.Background(), ai.Request{
		Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hi")},
	})
	if err != nil {
		t.Fatal(err)
	}
	drainStream(t, stream)

	headers := <-headersCh
	if got := headers.Get("Authorization"); got != "Bearer bearer-token" {
		t.Fatalf("Authorization = %q", got)
	}
	if got := headers.Get("x-api-key"); got != "" {
		t.Fatalf("x-api-key = %q", got)
	}
}

func TestProviderRejectsBothAPIKeyAndAuthToken(t *testing.T) {
	if _, err := anthropic.New(anthropic.Config{
		APIKey:    "api-key",
		AuthToken: "auth-token",
		Model:     "claude-test",
	}); err == nil || !strings.Contains(err.Error(), "only one") {
		t.Fatalf("error = %v", err)
	}
}
