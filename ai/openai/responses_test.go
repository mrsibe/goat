package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/ai/openai"
)

type responsesRequestBody struct {
	Model           string           `json:"model"`
	Stream          bool             `json:"stream"`
	Instructions    string           `json:"instructions"`
	Input           []map[string]any `json:"input"`
	Tools           []map[string]any `json:"tools"`
	ToolChoice      map[string]any   `json:"tool_choice"`
	MaxOutputTokens int64            `json:"max_output_tokens"`
	PromptCacheKey  string           `json:"prompt_cache_key"`
	Store           bool             `json:"store"`
}

func TestResponsesMapsHistoryToolsAndStreaming(t *testing.T) {
	requests := make(chan capturedRequest, 1)
	var responseBody strings.Builder
	writeChunk(&responseBody, map[string]any{
		"type": "response.created", "sequence_number": 1,
		"response": map[string]any{"id": "resp_1"},
	})
	writeChunk(&responseBody, map[string]any{
		"type": "response.output_text.delta", "sequence_number": 2,
		"item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": "你好",
	})
	writeChunk(&responseBody, map[string]any{
		"type": "response.output_text.done", "sequence_number": 3,
		"item_id": "msg_1", "output_index": 0, "content_index": 0, "text": "你好",
	})
	writeChunk(&responseBody, map[string]any{
		"type": "response.output_item.added", "sequence_number": 4, "output_index": 1,
		"item": map[string]any{
			"id": "fc_1", "type": "function_call", "call_id": "call_1",
			"name": "get_weather", "arguments": "",
		},
	})
	writeChunk(&responseBody, map[string]any{
		"type": "response.function_call_arguments.delta", "sequence_number": 5,
		"item_id": "fc_1", "output_index": 1, "delta": `{"city":"北京"}`,
	})
	writeChunk(&responseBody, map[string]any{
		"type": "response.function_call_arguments.done", "sequence_number": 6,
		"item_id": "fc_1", "output_index": 1, "arguments": `{"city":"北京"}`,
	})
	writeChunk(&responseBody, map[string]any{
		"type": "response.completed", "sequence_number": 7,
		"response": map[string]any{
			"id": "resp_1",
			"output": []any{map[string]any{
				"id": "fc_1", "type": "function_call", "call_id": "call_1",
				"name": "get_weather", "arguments": `{"city":"北京"}`,
			}},
			"usage": map[string]any{
				"input_tokens":  10,
				"output_tokens": 5,
				"total_tokens":  15,
				"input_tokens_details": map[string]any{
					"cached_tokens":      2,
					"cache_write_tokens": 3,
				},
				"output_tokens_details": map[string]any{"reasoning_tokens": 1},
			},
		},
	})

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

	provider, err := openai.NewResponses(openai.Config{
		APIKey:     "test-key",
		BaseURL:    "https://example.test",
		Model:      "gpt-test",
		HTTPClient: client,
	})
	if err != nil {
		t.Fatal(err)
	}
	if provider.Info().API != ai.APIOpenAIResponses {
		t.Fatalf("info = %+v", provider.Info())
	}

	call := ai.ToolCall{ID: "call_1", Name: "get_weather", Arguments: json.RawMessage(`{"city":"北京"}`)}
	assistant := ai.Message{
		Role: ai.RoleAssistant,
		Content: []ai.ContentBlock{
			{Type: ai.BlockText, Text: "checking"},
			{Type: ai.BlockToolCall, ToolCall: &call},
		},
	}
	maxTokens := 1024
	temperature := 0.2
	stream, err := provider.Stream(context.Background(), ai.Request{
		Messages: []ai.Message{
			ai.TextMessage(ai.RoleSystem, "sys"),
			ai.TextMessage(ai.RoleUser, "hi"),
			assistant,
			ai.ToolMessage("call_1", "get_weather", "sunny"),
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
		Options: ai.RequestOptions{
			Temperature: &temperature,
			MaxTokens:   maxTokens,
			SessionID:   "session-1",
			ToolChoice:  ai.ToolChoice{Mode: ai.ToolChoiceNamed, Name: "get_weather"},
		},
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
		ai.EventTextStart,
		ai.EventTextDelta,
		ai.EventTextEnd,
		ai.EventToolCallStart,
		ai.EventToolCallDelta,
		ai.EventToolCallEnd,
		ai.EventUsage,
		ai.EventDone,
	}
	gotTypes := make([]ai.EventType, len(events))
	for i, event := range events {
		gotTypes[i] = event.Type
	}
	if !reflect.DeepEqual(gotTypes, wantTypes) {
		t.Fatalf("event types = %v, want %v (events: %+v)", gotTypes, wantTypes, events)
	}
	if events[2].TextDelta != "你好" || events[2].ContentIndex != 0 {
		t.Fatalf("text delta = %+v", events[2])
	}
	if events[5].ContentIndex != 1 || events[5].ToolCallDelta.ArgumentsDelta != `{"city":"北京"}` {
		t.Fatalf("tool delta = %+v", events[5])
	}
	usage := events[7].Usage
	if usage == nil || usage.PromptTokens != 10 || usage.CompletionTokens != 5 ||
		usage.CacheReadTokens != 2 || usage.CacheWriteTokens != 3 ||
		usage.ReasoningTokens != 1 || usage.TotalTokens != 15 {
		t.Fatalf("usage = %+v", usage)
	}
	if events[8].StopReason != ai.StopReasonToolCalls || events[8].Message == nil {
		t.Fatalf("done = %+v", events[8])
	}
	calls := events[8].Message.ToolCalls()
	if len(calls) != 1 || calls[0].ID != "call_1" || calls[0].Name != "get_weather" ||
		string(calls[0].Arguments) != `{"city":"北京"}` {
		t.Fatalf("done tool calls = %+v", calls)
	}

	captured := <-requests
	if captured.path != "/responses" {
		t.Fatalf("path = %q", captured.path)
	}
	var body responsesRequestBody
	if err := json.Unmarshal(captured.body, &body); err != nil {
		t.Fatal(err)
	}
	if body.Model != "gpt-test" || !body.Stream || body.Instructions != "sys" ||
		body.MaxOutputTokens != 1024 || body.PromptCacheKey != "session-1" || body.Store {
		t.Fatalf("request controls = %+v", body)
	}
	if len(body.Input) != 4 {
		t.Fatalf("input = %+v", body.Input)
	}
	if body.Input[0]["role"] != "user" || body.Input[1]["role"] != "assistant" {
		t.Fatalf("message input = %+v", body.Input)
	}
	if body.Input[2]["type"] != "function_call" || body.Input[2]["call_id"] != "call_1" ||
		body.Input[2]["name"] != "get_weather" {
		t.Fatalf("function call input = %+v", body.Input[2])
	}
	if body.Input[3]["type"] != "function_call_output" || body.Input[3]["call_id"] != "call_1" {
		t.Fatalf("function output input = %+v", body.Input[3])
	}
	if len(body.Tools) != 1 || body.Tools[0]["name"] != "get_weather" {
		t.Fatalf("tools = %+v", body.Tools)
	}
	if body.ToolChoice["type"] != "function" || body.ToolChoice["name"] != "get_weather" {
		t.Fatalf("tool choice = %+v", body.ToolChoice)
	}
}

func TestResponsesMapsUserAndToolResultImages(t *testing.T) {
	requests := make(chan capturedRequest, 1)
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
			Body: io.NopCloser(strings.NewReader(
				"event: response.completed\ndata: " +
					`{"type":"response.completed","response":{"id":"resp_1","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}` +
					"\n\n",
			)),
			Request: request,
		}, nil
	})}

	provider, err := openai.NewResponses(openai.Config{
		BaseURL:    "https://example.test",
		Model:      "gpt-test",
		HTTPClient: client,
	})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := provider.Stream(context.Background(), ai.Request{
		Messages: []ai.Message{
			{
				Role: ai.RoleUser,
				Content: []ai.ContentBlock{
					{Type: ai.BlockText, Text: "look"},
					{Type: ai.BlockImage, Image: &ai.ImageContent{Data: "aGVsbG8=", MediaType: "image/png"}},
				},
			},
			ai.ToolResultMessage(
				"call_1",
				"screenshot",
				[]ai.ContentBlock{{
					Type:  ai.BlockImage,
					Image: &ai.ImageContent{Data: "d29ybGQ=", MediaType: "image/jpeg"},
				}},
				false,
			),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	drainStream(t, stream)

	captured := <-requests
	var body responsesRequestBody
	if err := json.Unmarshal(captured.body, &body); err != nil {
		t.Fatal(err)
	}
	if len(body.Input) != 2 {
		t.Fatalf("input = %+v", body.Input)
	}
	userContent := body.Input[0]["content"].([]any)
	if len(userContent) != 2 {
		t.Fatalf("user content = %+v", userContent)
	}
	userImage := userContent[1].(map[string]any)
	if userImage["type"] != "input_image" || userImage["image_url"] != "data:image/png;base64,aGVsbG8=" {
		t.Fatalf("user image = %+v", userImage)
	}
	output := body.Input[1]
	if output["type"] != "function_call_output" || output["call_id"] != "call_1" {
		t.Fatalf("tool output = %+v", output)
	}
	outputContent := output["output"].([]any)
	toolImage := outputContent[0].(map[string]any)
	if toolImage["type"] != "input_image" || toolImage["image_url"] != "data:image/jpeg;base64,d29ybGQ=" {
		t.Fatalf("tool image = %+v", toolImage)
	}
}
