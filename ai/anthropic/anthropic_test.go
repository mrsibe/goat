package anthropic_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/ai/anthropic"
	"github.com/mrsibe/goat/ai/internal/modeltest"
)

type capturedRequest struct {
	path string
	body []byte
}

type requestBody struct {
	Model     string `json:"model"`
	MaxTokens int64  `json:"max_tokens"`
	Stream    bool   `json:"stream"`
	System    []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"system"`
	Messages []struct {
		Role    string `json:"role"`
		Content []struct {
			Type      string          `json:"type"`
			Text      string          `json:"text"`
			ID        string          `json:"id"`
			Name      string          `json:"name"`
			Input     json.RawMessage `json:"input"`
			ToolUseID string          `json:"tool_use_id"`
			IsError   bool            `json:"is_error"`
		} `json:"content"`
	} `json:"messages"`
	Tools []struct {
		Name        string         `json:"name"`
		Description string         `json:"description"`
		InputSchema map[string]any `json:"input_schema"`
	} `json:"tools"`
	ToolChoice map[string]any `json:"tool_choice"`
}

func TestProviderMapsHistoryToolsAndStreaming(t *testing.T) {
	requests := make(chan capturedRequest, 1)
	var responseBody strings.Builder
	writeSSE(&responseBody, map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test",
			"content": []any{},
			"usage": map[string]any{
				"input_tokens":                10,
				"output_tokens":               1,
				"cache_read_input_tokens":     2,
				"cache_creation_input_tokens": 3,
				"output_tokens_details":       map[string]any{"thinking_tokens": 1},
			},
		},
	})
	writeSSE(&responseBody, map[string]any{
		"type": "content_block_start", "index": 0,
		"content_block": map[string]any{"type": "text", "text": ""},
	})
	writeSSE(&responseBody, map[string]any{
		"type": "content_block_delta", "index": 0,
		"delta": map[string]any{"type": "text_delta", "text": "你好"},
	})
	writeSSE(&responseBody, map[string]any{
		"type": "content_block_stop", "index": 0,
	})
	writeSSE(&responseBody, map[string]any{
		"type": "content_block_start", "index": 1,
		"content_block": map[string]any{
			"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": map[string]any{},
		},
	})
	writeSSE(&responseBody, map[string]any{
		"type": "content_block_delta", "index": 1,
		"delta": map[string]any{"type": "input_json_delta", "partial_json": `{"city":`},
	})
	writeSSE(&responseBody, map[string]any{
		"type": "content_block_delta", "index": 1,
		"delta": map[string]any{"type": "input_json_delta", "partial_json": `"北京"}`},
	})
	writeSSE(&responseBody, map[string]any{
		"type": "content_block_stop", "index": 1,
	})
	writeSSE(&responseBody, map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": "tool_use", "stop_sequence": nil},
		"usage": map[string]any{
			"output_tokens":         5,
			"output_tokens_details": map[string]any{"thinking_tokens": 2},
		},
	})
	writeSSE(&responseBody, map[string]any{"type": "message_stop"})

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

	provider, err := anthropic.New(anthropic.Config{
		APIKey:          "test-key",
		BaseURL:         "https://example.test",
		Model:           "claude-test",
		HTTPClient:      client,
		MaxOutputTokens: 2048,
	})
	if err != nil {
		t.Fatal(err)
	}
	if provider.Info().API != ai.APIAnthropicMessages || provider.Info().MaxOutputTokens != 2048 {
		t.Fatalf("info = %+v", provider.Info())
	}

	call := ai.ToolCall{ID: "toolu_1", Name: "get_weather", Arguments: json.RawMessage(`{"city":"北京"}`)}
	assistant := ai.Message{
		Role: ai.RoleAssistant,
		Content: []ai.ContentBlock{
			{Type: ai.BlockText, Text: "checking"},
			{Type: ai.BlockToolCall, ToolCall: &call},
		},
	}
	collected, err := modeltest.Collect(context.Background(), provider, ai.Request{
		Messages: []ai.Message{
			ai.TextMessage(ai.RoleSystem, "sys"),
			ai.TextMessage(ai.RoleUser, "hi"),
			assistant,
			ai.ToolTextMessage("toolu_1", "get_weather", "upstream failed", true),
		},
		Tools: []ai.ToolSpec{{
			Name:        "get_weather",
			Description: "weather",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{"type": "string"},
				},
				"required": []string{"city"},
			},
		}},
		Options: ai.RequestOptions{
			MaxTokens: 1024,
			ToolChoice: ai.ToolChoice{
				Mode: ai.ToolChoiceNamed,
				Name: "get_weather",
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	events := collected.Events
	wantTypes := []ai.EventType{
		ai.EventStart,
		ai.EventTextStart,
		ai.EventTextDelta,
		ai.EventTextEnd,
		ai.EventToolCallStart,
		ai.EventToolCallDelta,
		ai.EventToolCallDelta,
		ai.EventToolCallEnd,
		ai.EventUsage,
		ai.EventDone,
	}
	modeltest.AssertEventTypes(t, events, wantTypes)
	if events[2].TextDelta != "你好" || events[2].ContentIndex != 0 {
		t.Fatalf("text delta = %+v", events[2])
	}
	if events[5].ContentIndex != 1 || events[5].ToolCallDelta.ArgumentsDelta != `{"city":` {
		t.Fatalf("tool delta = %+v", events[5])
	}
	usage := events[8].Usage
	if usage == nil || usage.PromptTokens != 15 || usage.CompletionTokens != 5 ||
		usage.CacheReadTokens != 2 || usage.CacheWriteTokens != 3 ||
		usage.ReasoningTokens != 2 || usage.TotalTokens != 20 {
		t.Fatalf("usage = %+v", usage)
	}
	if events[9].StopReason != ai.StopReasonToolCalls || events[9].Message == nil {
		t.Fatalf("done = %+v", events[9])
	}
	if got := events[9].Message.Text(); got != "你好" {
		t.Fatalf("done text = %q", got)
	}
	calls := events[9].Message.ToolCalls()
	if len(calls) != 1 || calls[0].ID != "toolu_1" || calls[0].Name != "get_weather" ||
		string(calls[0].Arguments) != `{"city":"北京"}` {
		t.Fatalf("done tool calls = %+v", calls)
	}

	captured := <-requests
	if captured.path != "/v1/messages" {
		t.Fatalf("path = %q", captured.path)
	}
	var body requestBody
	if err := json.Unmarshal(captured.body, &body); err != nil {
		t.Fatal(err)
	}
	if body.Model != "claude-test" || body.MaxTokens != 1024 || !body.Stream {
		t.Fatalf("request controls = %+v", body)
	}
	if len(body.System) != 1 || body.System[0].Text != "sys" {
		t.Fatalf("system = %+v", body.System)
	}
	if len(body.Messages) != 3 {
		t.Fatalf("messages = %+v", body.Messages)
	}
	if body.Messages[0].Role != "user" || body.Messages[1].Role != "assistant" || body.Messages[2].Role != "user" {
		t.Fatalf("roles = %+v", body.Messages)
	}
	if len(body.Messages[1].Content) != 2 ||
		body.Messages[1].Content[0].Type != "text" ||
		body.Messages[1].Content[1].Type != "tool_use" ||
		body.Messages[1].Content[1].ID != "toolu_1" {
		t.Fatalf("assistant content = %+v", body.Messages[1].Content)
	}
	if len(body.Messages[2].Content) != 1 ||
		body.Messages[2].Content[0].Type != "tool_result" ||
		body.Messages[2].Content[0].ToolUseID != "toolu_1" ||
		!body.Messages[2].Content[0].IsError {
		t.Fatalf("tool result content = %+v", body.Messages[2].Content)
	}
	if len(body.Tools) != 1 || body.Tools[0].Name != "get_weather" {
		t.Fatalf("tools = %+v", body.Tools)
	}
	if body.Tools[0].InputSchema["type"] != "object" {
		t.Fatalf("input schema = %+v", body.Tools[0].InputSchema)
	}
	if body.ToolChoice["type"] != "tool" || body.ToolChoice["name"] != "get_weather" {
		t.Fatalf("tool choice = %+v", body.ToolChoice)
	}
}

func TestProviderValidationAndDefaultMaxTokens(t *testing.T) {
	if _, err := anthropic.New(anthropic.Config{}); err == nil {
		t.Fatal("expected an error for an empty model")
	}
	provider, err := anthropic.New(anthropic.Config{Model: "model"})
	if err != nil {
		t.Fatal(err)
	}
	if provider.Info().MaxOutputTokens != 4096 {
		t.Fatalf("max output tokens = %d", provider.Info().MaxOutputTokens)
	}
	info := provider.Info()
	if !info.SupportsImages || !info.SupportsToolResultImages || !info.SupportsThinking {
		t.Fatalf("info = %+v", info)
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

func TestProviderMapsThinkingAndImages(t *testing.T) {
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
				"event: message_start\ndata: " +
					`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"usage":{"input_tokens":1,"output_tokens":1}}}` +
					"\n\nevent: message_stop\ndata: " +
					`{"type":"message_stop"}` +
					"\n\n",
			)),
			Request: request,
		}, nil
	})}

	provider, err := anthropic.New(anthropic.Config{
		APIKey:     "test-key",
		BaseURL:    "https://example.test",
		Model:      "claude-test",
		HTTPClient: client,
	})
	if err != nil {
		t.Fatal(err)
	}
	call := ai.ToolCall{ID: "toolu_1", Name: "screenshot", Arguments: json.RawMessage(`{}`)}
	stream, err := provider.Stream(context.Background(), ai.Request{
		Messages: []ai.Message{
			{
				Role: ai.RoleUser,
				Content: []ai.ContentBlock{
					{Type: ai.BlockText, Text: "look"},
					{Type: ai.BlockImage, Image: &ai.ImageContent{Data: "aGVsbG8=", MediaType: "image/png"}},
				},
			},
			{
				Role: ai.RoleAssistant,
				Content: []ai.ContentBlock{
					{Type: ai.BlockThinking, Thinking: "plan", ThinkingSignature: "sig"},
					{Type: ai.BlockText, Text: "taking a screenshot"},
					{Type: ai.BlockToolCall, ToolCall: &call},
				},
			},
			ai.ToolResultMessage(
				"toolu_1",
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
	var body map[string]any
	if err := json.Unmarshal(captured.body, &body); err != nil {
		t.Fatal(err)
	}
	messages := body["messages"].([]any)
	userContent := messages[0].(map[string]any)["content"].([]any)
	userImage := userContent[1].(map[string]any)
	userSource := userImage["source"].(map[string]any)
	if userImage["type"] != "image" || userSource["type"] != "base64" ||
		userSource["media_type"] != "image/png" || userSource["data"] != "aGVsbG8=" {
		t.Fatalf("user image = %+v", userImage)
	}

	assistantContent := messages[1].(map[string]any)["content"].([]any)
	thinking := assistantContent[0].(map[string]any)
	if thinking["type"] != "thinking" || thinking["thinking"] != "plan" || thinking["signature"] != "sig" {
		t.Fatalf("thinking block = %+v", thinking)
	}

	toolResult := messages[2].(map[string]any)["content"].([]any)[0].(map[string]any)
	toolContent := toolResult["content"].([]any)
	toolImage := toolContent[0].(map[string]any)
	toolSource := toolImage["source"].(map[string]any)
	if toolResult["type"] != "tool_result" || toolImage["type"] != "image" ||
		toolSource["media_type"] != "image/jpeg" || toolSource["data"] != "d29ybGQ=" {
		t.Fatalf("tool result = %+v", toolResult)
	}
}

func TestProviderSendsCustomHeaders(t *testing.T) {
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
	writeSSE(&responseBody, map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": "end_turn"},
		"usage": map[string]any{"output_tokens": 1},
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
		APIKey:     "test-key",
		BaseURL:    "https://example.test",
		Model:      "claude-test",
		Provider:   "anthropic-compatible",
		Headers:    map[string]string{"X-Goat-Test": "anthropic"},
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
	if got := headers.Get("X-Goat-Test"); got != "anthropic" {
		t.Fatalf("X-Goat-Test = %q", got)
	}
	if got := provider.Info().Provider; got != "anthropic-compatible" {
		t.Fatalf("provider = %q", got)
	}
}

func writeSSE(writer io.Writer, event map[string]any) {
	data, err := json.Marshal(event)
	if err != nil {
		panic(err)
	}
	eventType, _ := event["type"].(string)
	fmt.Fprintf(writer, "event: %s\ndata: %s\n\n", eventType, data)
}

func drainStream(t *testing.T, stream ai.Stream) {
	t.Helper()
	defer stream.Close()
	for {
		event, err := stream.Next()
		if errors.Is(err, io.EOF) {
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
