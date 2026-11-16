package openai_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/ai/internal/modeltest"
	"github.com/mrsibe/goat/ai/openai"
)

func TestRequestHooksAndIdentity(t *testing.T) {
	var captured map[string]any
	headers := http.Header{}
	responseStatus := 0
	streamEvents := 0

	var responseBody strings.Builder
	writeChunk(&responseBody, map[string]any{
		"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "test-model",
		"choices": []any{map[string]any{
			"index": 0, "delta": map[string]any{"content": "hi"}, "finish_reason": "stop",
		}},
	})
	responseBody.WriteString("data: [DONE]\n\n")

	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		headers = request.Header.Clone()
		body, err := io.ReadAll(request.Body)
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(body, &captured); err != nil {
			return nil, err
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Status:     "200 OK",
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:       io.NopCloser(strings.NewReader(responseBody.String())),
			Request:    request,
		}, nil
	})}

	provider, err := openai.New(openai.Config{
		APIKey:     "base-key",
		BaseURL:    "https://example.test",
		Model:      "test-model",
		HTTPClient: client,
	})
	if err != nil {
		t.Fatal(err)
	}

	options := ai.RequestOptions{
		APIKey: "override-key",
		OnPayload: func(payload map[string]any) (map[string]any, error) {
			payload["custom_marker"] = "x"
			return payload, nil
		},
		OnResponse: func(status int, _ http.Header) error {
			responseStatus = status
			return nil
		},
		OnProviderStreamEvent: func(any) error {
			streamEvents++
			return nil
		},
		Identity: &ai.Identity{
			UserAgent:   "goat-test/1.0",
			Attribution: &ai.Attribution{Name: "goat", URL: "https://example.test"},
		},
	}

	if _, err := modeltest.Collect(context.Background(), provider, ai.Request{
		Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hi")},
		Options:  options,
	}); err != nil {
		t.Fatal(err)
	}

	if captured["custom_marker"] != "x" {
		t.Fatalf("payload hook not applied: %+v", captured)
	}
	if got := headers.Get("Authorization"); got != "Bearer override-key" {
		t.Fatalf("authorization = %q", got)
	}
	if got := headers.Get("User-Agent"); got != "goat-test/1.0" {
		t.Fatalf("user agent = %q", got)
	}
	if responseStatus != http.StatusOK {
		t.Fatalf("onResponse status = %d", responseStatus)
	}
	if streamEvents == 0 {
		t.Fatal("provider stream events were not observed")
	}
}

func TestWebSocketTransportIsRejected(t *testing.T) {
	provider, err := openai.New(openai.Config{Model: "test-model", BaseURL: "https://example.test"})
	if err != nil {
		t.Fatal(err)
	}
	_, err = provider.Stream(context.Background(), ai.Request{
		Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hi")},
		Options:  ai.RequestOptions{Transport: ai.TransportWebSocket},
	})
	if err == nil {
		t.Fatal("websocket transport must be rejected")
	}
}
