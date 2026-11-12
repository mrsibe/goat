package deepseek_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/ai/deepseek"
)

func TestProviderUsesDefaultsAndDeepSeekReasoning(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "deepseek-key")
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
		requests <- struct {
			path          string
			authorization string
			body          []byte
		}{
			path:          request.URL.Path,
			authorization: request.Header.Get("Authorization"),
			body:          body,
		}
		return completionSSE(request), nil
	})}

	provider, err := deepseek.New(deepseek.Config{
		Model:      "deepseek-reasoner",
		HTTPClient: client,
	})
	if err != nil {
		t.Fatal(err)
	}
	info := provider.Info()
	if info.Provider != "deepseek" || info.API != ai.APIOpenAICompletions ||
		!info.SupportsThinking || info.SupportsImages {
		t.Fatalf("info = %+v", info)
	}
	assistant := ai.Message{
		Role: ai.RoleAssistant,
		Content: []ai.ContentBlock{
			{Type: ai.BlockThinking, Thinking: "plan", ThinkingSignature: "reasoning_content"},
			{Type: ai.BlockText, Text: "done"},
		},
	}
	stream, err := provider.Stream(context.Background(), ai.Request{
		Messages: []ai.Message{assistant, ai.TextMessage(ai.RoleUser, "continue")},
		Options:  ai.RequestOptions{ReasoningEffort: "high"},
	})
	if err != nil {
		t.Fatal(err)
	}
	drainStream(t, stream)

	captured := <-requests
	if captured.path != "/chat/completions" {
		t.Fatalf("path = %q", captured.path)
	}
	if captured.authorization != "Bearer deepseek-key" {
		t.Fatalf("Authorization = %q", captured.authorization)
	}
	var body map[string]any
	if err := json.Unmarshal(captured.body, &body); err != nil {
		t.Fatal(err)
	}
	thinking, ok := body["thinking"].(map[string]any)
	if !ok || thinking["type"] != "enabled" || body["reasoning_effort"] != "high" {
		t.Fatalf("reasoning request = %+v", body)
	}
	messages := body["messages"].([]any)
	requestAssistant := messages[0].(map[string]any)
	if requestAssistant["reasoning_content"] != "plan" {
		t.Fatalf("assistant = %+v", requestAssistant)
	}
}

func TestProviderRequiresAPIKey(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "")
	if _, err := deepseek.New(deepseek.Config{Model: "model"}); err == nil ||
		!strings.Contains(err.Error(), "API key") {
		t.Fatalf("error = %v", err)
	}
}

func completionSSE(request *http.Request) *http.Response {
	var body strings.Builder
	fmt.Fprintf(&body, "data: %s\n\n", `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"model","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"plan","content":"hi"},"finish_reason":"stop"}]}`)
	body.WriteString("data: [DONE]\n\n")
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
