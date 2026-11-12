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

func TestCompletionsMapsReasoningContent(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		return chatSSE(request,
			map[string]any{
				"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "model",
				"choices": []any{map[string]any{
					"index":         0,
					"delta":         map[string]any{"role": "assistant", "reasoning_content": "plan "},
					"finish_reason": nil,
				}},
			},
			map[string]any{
				"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "model",
				"choices": []any{map[string]any{
					"index":         0,
					"delta":         map[string]any{"reasoning_content": "more", "content": "answer"},
					"finish_reason": nil,
				}},
			},
			map[string]any{
				"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "model",
				"choices": []any{map[string]any{
					"index":         0,
					"delta":         map[string]any{},
					"finish_reason": "stop",
				}},
			},
		), nil
	})}

	provider, err := openai.NewCompletions(openai.Config{
		BaseURL:          "https://example.test",
		Model:            "model",
		SupportsThinking: true,
		HTTPClient:       client,
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := modeltest.Collect(context.Background(), provider, ai.Request{
		Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hi")},
	})
	if err != nil {
		t.Fatal(err)
	}
	modeltest.AssertEventTypes(t, result.Events, []ai.EventType{
		ai.EventStart,
		ai.EventThinkingStart,
		ai.EventThinkingDelta,
		ai.EventThinkingDelta,
		ai.EventTextStart,
		ai.EventTextDelta,
		ai.EventThinkingEnd,
		ai.EventTextEnd,
		ai.EventDone,
	})
	if result.Message == nil || len(result.Message.Content) != 2 {
		t.Fatalf("message = %+v", result.Message)
	}
	thinking := result.Message.Content[0]
	if thinking.Type != ai.BlockThinking || thinking.Thinking != "plan more" ||
		thinking.ThinkingSignature != "reasoning_content" {
		t.Fatalf("thinking block = %+v", thinking)
	}
	if result.Message.Content[1].Text != "answer" {
		t.Fatalf("text block = %+v", result.Message.Content[1])
	}
}

func TestCompletionsReasoningFieldPriority(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		return chatSSE(request,
			map[string]any{
				"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "model",
				"choices": []any{map[string]any{
					"index": 0,
					"delta": map[string]any{
						"reasoning_content": "content",
						"reasoning":         "reasoning",
						"reasoning_text":    "text",
					},
					"finish_reason": "stop",
				}},
			},
		), nil
	})}

	provider, err := openai.NewCompletions(openai.Config{
		BaseURL:          "https://example.test",
		Model:            "model",
		SupportsThinking: true,
		HTTPClient:       client,
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := modeltest.Collect(context.Background(), provider, ai.Request{
		Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hi")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Message == nil || len(result.Message.Content) != 1 {
		t.Fatalf("message = %+v", result.Message)
	}
	thinking := result.Message.Content[0]
	if thinking.Thinking != "content" || thinking.ThinkingSignature != "reasoning_content" {
		t.Fatalf("thinking block = %+v", thinking)
	}
}

func TestCompletionsOpenRouterReasoningDetailsReplay(t *testing.T) {
	requests := make(chan []byte, 2)
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			return nil, err
		}
		requests <- body
		return chatSSE(request,
			map[string]any{
				"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "model",
				"choices": []any{map[string]any{
					"index": 0,
					"delta": map[string]any{
						"reasoning_details": []any{map[string]any{
							"type": "reasoning.text", "text": "a", "id": "r1", "index": 0,
						}},
					},
					"finish_reason": nil,
				}},
			},
			map[string]any{
				"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "model",
				"choices": []any{map[string]any{
					"index": 0,
					"delta": map[string]any{
						"reasoning_details": []any{map[string]any{
							"type": "reasoning.text", "text": "b",
						}},
						"content": "answer",
					},
					"finish_reason": "stop",
				}},
			},
		), nil
	})}

	provider, err := openai.NewCompletions(openai.Config{
		BaseURL:          "https://example.test",
		Model:            "model",
		Provider:         "openrouter",
		ReasoningFormat:  openai.ReasoningFormatOpenRouter,
		SupportsThinking: true,
		HTTPClient:       client,
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := modeltest.Collect(context.Background(), provider, ai.Request{
		Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hi")},
	})
	if err != nil {
		t.Fatal(err)
	}
	<-requests
	if result.Message == nil || len(result.Message.Content) != 2 {
		t.Fatalf("message = %+v", result.Message)
	}
	thinking := result.Message.Content[0]
	if thinking.Type != ai.BlockThinking || thinking.Thinking != "" {
		t.Fatalf("thinking block = %+v", thinking)
	}
	var details []map[string]any
	if err := json.Unmarshal([]byte(thinking.ThinkingSignature), &details); err != nil {
		t.Fatalf("thinking signature = %q: %v", thinking.ThinkingSignature, err)
	}
	if len(details) != 1 || details[0]["text"] != "ab" ||
		details[0]["id"] != "r1" || details[0]["index"] != float64(0) {
		t.Fatalf("reasoning details = %+v", details)
	}

	stream, err := provider.Stream(context.Background(), ai.Request{
		Messages: []ai.Message{
			*result.Message,
			ai.TextMessage(ai.RoleUser, "continue"),
		},
		Options: ai.RequestOptions{ReasoningEffort: "high"},
	})
	if err != nil {
		t.Fatal(err)
	}
	drainStream(t, stream)
	body := <-requests
	var requestBody map[string]any
	if err := json.Unmarshal(body, &requestBody); err != nil {
		t.Fatal(err)
	}
	reasoning, ok := requestBody["reasoning"].(map[string]any)
	if !ok || reasoning["effort"] != "high" {
		t.Fatalf("reasoning request = %+v", requestBody["reasoning"])
	}
	if _, ok := requestBody["reasoning_effort"]; ok {
		t.Fatalf("unexpected top-level reasoning_effort: %+v", requestBody)
	}
	assistant := assistantRequestMessage(t, requestBody)
	if _, ok := assistant["reasoning_details"]; !ok {
		t.Fatalf("assistant reasoning_details missing: %+v", assistant)
	}
	if _, ok := assistant["reasoning_content"]; ok {
		t.Fatalf("unexpected reasoning_content: %+v", assistant)
	}
}

func TestCompletionsDeepSeekReasoningRequestAndReplay(t *testing.T) {
	requests := make(chan []byte, 3)
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			return nil, err
		}
		requests <- body
		return &http.Response{
			StatusCode: http.StatusOK,
			Status:     "200 OK",
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:       io.NopCloser(strings.NewReader("data: [DONE]\n\n")),
			Request:    request,
		}, nil
	})}

	provider, err := openai.NewCompletions(openai.Config{
		BaseURL:          "https://example.test",
		Model:            "deepseek-reasoner",
		Provider:         "deepseek",
		ReasoningFormat:  openai.ReasoningFormatDeepSeek,
		SupportsThinking: true,
		HTTPClient:       client,
	})
	if err != nil {
		t.Fatal(err)
	}
	assistant := ai.Message{
		Role: ai.RoleAssistant,
		Content: []ai.ContentBlock{
			{Type: ai.BlockThinking, Thinking: "plan", ThinkingSignature: "reasoning_content"},
			{Type: ai.BlockText, Text: "done"},
		},
	}
	tests := []struct {
		name             string
		effort           string
		wantThinkingType string
		wantEffort       string
	}{
		{name: "enabled", effort: "high", wantThinkingType: "enabled", wantEffort: "high"},
		{name: "disabled", effort: "none", wantThinkingType: "disabled"},
		{name: "default", effort: ""},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			stream, err := provider.Stream(context.Background(), ai.Request{
				Messages: []ai.Message{
					assistant,
					ai.TextMessage(ai.RoleUser, "continue"),
				},
				Options: ai.RequestOptions{ReasoningEffort: test.effort},
			})
			if err != nil {
				t.Fatal(err)
			}
			drainStream(t, stream)

			var requestBody map[string]any
			if err := json.Unmarshal(<-requests, &requestBody); err != nil {
				t.Fatal(err)
			}
			if test.wantThinkingType == "" {
				if _, ok := requestBody["thinking"]; ok {
					t.Fatalf("unexpected thinking field: %+v", requestBody)
				}
			} else {
				thinking, ok := requestBody["thinking"].(map[string]any)
				if !ok || thinking["type"] != test.wantThinkingType {
					t.Fatalf("thinking = %+v", requestBody["thinking"])
				}
			}
			if test.wantEffort == "" {
				if _, ok := requestBody["reasoning_effort"]; ok {
					t.Fatalf("unexpected reasoning_effort: %+v", requestBody)
				}
			} else if requestBody["reasoning_effort"] != test.wantEffort {
				t.Fatalf("reasoning_effort = %+v", requestBody["reasoning_effort"])
			}
			requestAssistant := assistantRequestMessage(t, requestBody)
			if requestAssistant["reasoning_content"] != "plan" {
				t.Fatalf("assistant reasoning_content = %+v", requestAssistant["reasoning_content"])
			}
		})
	}
}

func TestCompletionsReplaysStandardReasoningField(t *testing.T) {
	requests := make(chan []byte, 1)
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			return nil, err
		}
		requests <- body
		return &http.Response{
			StatusCode: http.StatusOK,
			Status:     "200 OK",
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:       io.NopCloser(strings.NewReader("data: [DONE]\n\n")),
			Request:    request,
		}, nil
	})}
	provider, err := openai.NewCompletions(openai.Config{
		BaseURL:          "https://example.test",
		Model:            "model",
		SupportsThinking: true,
		HTTPClient:       client,
	})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := provider.Stream(context.Background(), ai.Request{
		Messages: []ai.Message{{
			Role: ai.RoleAssistant,
			Content: []ai.ContentBlock{
				{Type: ai.BlockThinking, Thinking: "plan", ThinkingSignature: "reasoning_text"},
				{Type: ai.BlockText, Text: "done"},
			},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	drainStream(t, stream)

	var requestBody map[string]any
	if err := json.Unmarshal(<-requests, &requestBody); err != nil {
		t.Fatal(err)
	}
	assistant := assistantRequestMessage(t, requestBody)
	if assistant["reasoning_text"] != "plan" {
		t.Fatalf("assistant = %+v", assistant)
	}
}

func TestCompletionsRejectsUnknownReasoningFormat(t *testing.T) {
	if _, err := openai.NewCompletions(openai.Config{
		Model:           "model",
		ReasoningFormat: "unknown",
	}); err == nil || !strings.Contains(err.Error(), "unsupported reasoning format") {
		t.Fatalf("error = %v", err)
	}
}

func chatSSE(request *http.Request, chunks ...map[string]any) *http.Response {
	var body strings.Builder
	for _, chunk := range chunks {
		writeChunk(&body, chunk)
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

func assistantRequestMessage(t *testing.T, body map[string]any) map[string]any {
	t.Helper()
	messages, ok := body["messages"].([]any)
	if !ok {
		t.Fatalf("messages = %+v", body["messages"])
	}
	for _, item := range messages {
		message, ok := item.(map[string]any)
		if ok && message["role"] == "assistant" {
			return message
		}
	}
	t.Fatalf("assistant message not found: %+v", messages)
	return nil
}
