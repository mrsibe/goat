package xai_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/ai/xai"
)

func TestProviderUsesResponsesDefaults(t *testing.T) {
	t.Setenv("XAI_API_KEY", "xai-key")
	requests := make(chan struct {
		path          string
		authorization string
		body          []byte
	}, 1)
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			return nil, err
		}
		authorization := request.Header.Get("Authorization")
		requests <- struct {
			path          string
			authorization string
			body          []byte
		}{
			path:          request.URL.Path,
			authorization: authorization,
			body:          body,
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Status:     "200 OK",
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body: io.NopCloser(strings.NewReader(
				"event: response.completed\ndata: " +
					`{"type":"response.completed","response":{"id":"resp_1","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}` +
					"\n\n",
			)),
			Request: request,
		}, nil
	})}

	provider, err := xai.New(xai.Config{
		Model:      "grok-4",
		HTTPClient: client,
	})
	if err != nil {
		t.Fatal(err)
	}
	info := provider.Info()
	if info.Provider != "xai" || info.API != ai.APIOpenAIResponses ||
		!info.SupportsThinking || !info.SupportsToolResultImages {
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
	if captured.path != "/v1/responses" {
		t.Fatalf("path = %q", captured.path)
	}
	if captured.authorization != "Bearer xai-key" {
		t.Fatalf("Authorization = %q", captured.authorization)
	}
	var body map[string]any
	if err := json.Unmarshal(captured.body, &body); err != nil {
		t.Fatal(err)
	}
	include, ok := body["include"].([]any)
	if !ok || len(include) != 1 || include[0] != "reasoning.encrypted_content" {
		t.Fatalf("include = %+v", body["include"])
	}
}

func TestProviderRequiresAPIKey(t *testing.T) {
	t.Setenv("XAI_API_KEY", "")
	if _, err := xai.New(xai.Config{Model: "grok-4"}); err == nil ||
		!strings.Contains(err.Error(), "API key") {
		t.Fatalf("error = %v", err)
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
