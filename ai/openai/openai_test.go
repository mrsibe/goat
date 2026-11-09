package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/ai/openai"
)

type capturedRequest struct {
	path string
	body []byte
}

type requestBody struct {
	Model    string `json:"model"`
	Stream   bool   `json:"stream"`
	Messages []struct {
		Role       string `json:"role"`
		Content    string `json:"content"`
		ToolCallID string `json:"tool_call_id"`
		ToolCalls  []struct {
			ID       string `json:"id"`
			Type     string `json:"type"`
			Function struct {
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			} `json:"function"`
		} `json:"tool_calls"`
	} `json:"messages"`
	Tools []struct {
		Type     string `json:"type"`
		Function struct {
			Name        string         `json:"name"`
			Description string         `json:"description"`
			Parameters  map[string]any `json:"parameters"`
		} `json:"function"`
	} `json:"tools"`
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
}

func TestProviderMapsHistoryToolsAndStreaming(t *testing.T) {
	requests := make(chan capturedRequest, 1)
	var responseBody strings.Builder
	writeChunk(&responseBody, map[string]any{
		"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "test-model",
		"choices": []any{map[string]any{
			"index":         0,
			"delta":         map[string]any{"role": "assistant", "content": "你好"},
			"finish_reason": nil,
		}},
	})
	writeChunk(&responseBody, map[string]any{
		"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "test-model",
		"choices": []any{map[string]any{
			"index": 0,
			"delta": map[string]any{
				"tool_calls": []any{map[string]any{
					"index": 0,
					"id":    "call_1",
					"type":  "function",
					"function": map[string]any{
						"name":      "get_weather",
						"arguments": `{"city":`,
					},
				}},
			},
			"finish_reason": nil,
		}},
	})
	writeChunk(&responseBody, map[string]any{
		"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "test-model",
		"choices": []any{map[string]any{
			"index":         0,
			"delta":         map[string]any{},
			"finish_reason": "tool_calls",
		}},
	})
	writeChunk(&responseBody, map[string]any{
		"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "test-model",
		"choices": []any{},
		"usage": map[string]any{
			"prompt_tokens":     10,
			"completion_tokens": 3,
			"total_tokens":      13,
		},
	})
	responseBody.WriteString("data: [DONE]\n\n")

	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			return nil, err
		}
		requests <- capturedRequest{path: request.URL.Path, body: body}
		return &http.Response{
			StatusCode: http.StatusOK,
			Status:     "200 OK",
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:       io.NopCloser(strings.NewReader(responseBody.String())),
			Request:    request,
		}, nil
	})}

	provider, err := openai.New(openai.Config{
		APIKey:     "",
		BaseURL:    "https://example.test",
		Model:      "test-model",
		HTTPClient: client,
	})
	if err != nil {
		t.Fatal(err)
	}
	call := ai.ToolCall{ID: "call_1", Name: "get_weather", Arguments: json.RawMessage(`{"city":"北京"}`)}
	assistant := ai.Message{
		Role: ai.RoleAssistant,
		Content: []ai.ContentBlock{
			{Type: ai.BlockText, Text: "checking"},
			{Type: ai.BlockToolCall, ToolCall: &call},
		},
	}
	stream, err := provider.Stream(context.Background(), ai.Request{
		Messages: []ai.Message{
			ai.TextMessage(ai.RoleSystem, "sys"),
			ai.TextMessage(ai.RoleUser, "hi"),
			assistant,
			ai.ToolMessage("call_1", "sunny"),
		},
		Tools: []ai.ToolSpec{{
			Name:        "get_weather",
			Description: "weather",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{"type": "string"},
				},
			},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()

	var events []ai.Event
	for {
		event, nextErr := stream.Next()
		if errors.Is(nextErr, io.EOF) {
			break
		}
		if nextErr != nil {
			t.Fatal(nextErr)
		}
		events = append(events, event)
	}
	wantTypes := []ai.EventType{
		ai.EventStart,
		ai.EventTextDelta,
		ai.EventToolCallDelta,
		ai.EventUsage,
		ai.EventDone,
	}
	gotTypes := make([]ai.EventType, len(events))
	for i, event := range events {
		gotTypes[i] = event.Type
	}
	if !reflect.DeepEqual(gotTypes, wantTypes) {
		t.Fatalf("event types = %v, want %v", gotTypes, wantTypes)
	}
	if events[1].TextDelta != "你好" {
		t.Fatalf("text delta = %q", events[1].TextDelta)
	}
	if events[2].ToolCall == nil || events[2].ToolCall.ID != "call_1" ||
		events[2].ToolCall.Name != "get_weather" || events[2].ToolCall.ArgumentsDelta != `{"city":` {
		t.Fatalf("unexpected tool delta: %+v", events[2].ToolCall)
	}
	if events[3].Usage == nil || events[3].Usage.TotalTokens != 13 {
		t.Fatalf("usage = %+v", events[3].Usage)
	}
	if events[4].StopReason != ai.StopReasonToolCalls {
		t.Fatalf("stop reason = %q", events[4].StopReason)
	}

	captured := <-requests
	if captured.path != "/chat/completions" {
		t.Fatalf("path = %q", captured.path)
	}
	var body requestBody
	if err := json.Unmarshal(captured.body, &body); err != nil {
		t.Fatal(err)
	}
	if body.Model != "test-model" || !body.Stream || !body.StreamOptions.IncludeUsage {
		t.Fatalf("unexpected request controls: %+v", body)
	}
	wantRoles := []string{"system", "user", "assistant", "tool"}
	gotRoles := make([]string, len(body.Messages))
	for i, message := range body.Messages {
		gotRoles[i] = message.Role
	}
	if !reflect.DeepEqual(gotRoles, wantRoles) {
		t.Fatalf("roles = %v, want %v", gotRoles, wantRoles)
	}
	if got := body.Messages[2].ToolCalls; len(got) != 1 || got[0].Function.Arguments != `{"city":"北京"}` {
		t.Fatalf("request tool calls = %+v", got)
	}
	if body.Messages[3].ToolCallID != "call_1" {
		t.Fatalf("tool_call_id = %q", body.Messages[3].ToolCallID)
	}
	if len(body.Tools) != 1 || body.Tools[0].Function.Name != "get_weather" {
		t.Fatalf("tools = %+v", body.Tools)
	}
}

func TestProviderValidationAndConversionErrors(t *testing.T) {
	if _, err := openai.New(openai.Config{}); err == nil {
		t.Fatal("expected an error for an empty model")
	}
	provider, err := openai.New(openai.Config{Model: "model"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := provider.Stream(context.Background(), ai.Request{
		Messages: []ai.Message{{Role: "invalid"}},
	}); err == nil {
		t.Fatal("expected an unsupported role error")
	}
	if _, err := provider.Stream(context.Background(), ai.Request{
		Messages: []ai.Message{ai.TextMessage(ai.RoleTool, "result")},
	}); err == nil {
		t.Fatal("expected a missing tool call id error")
	}
}

func writeChunk(writer io.Writer, chunk map[string]any) {
	data, err := json.Marshal(chunk)
	if err != nil {
		panic(err)
	}
	fmt.Fprintf(writer, "data: %s\n\n", data)
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return fn(request)
}
