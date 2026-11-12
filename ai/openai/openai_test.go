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
	"github.com/mrsibe/goat/ai/internal/modeltest"
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
			"index": 0,
			"delta": map[string]any{
				"tool_calls": []any{map[string]any{
					"index": 0,
					"function": map[string]any{
						"arguments": `"北京"}`,
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
	collected, err := modeltest.Collect(context.Background(), provider, ai.Request{
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
	})
	if err != nil {
		t.Fatal(err)
	}
	events := collected.Events
	wantTypes := []ai.EventType{
		ai.EventStart,
		ai.EventTextStart,
		ai.EventTextDelta,
		ai.EventToolCallStart,
		ai.EventToolCallDelta,
		ai.EventToolCallDelta,
		ai.EventUsage,
		ai.EventTextEnd,
		ai.EventToolCallEnd,
		ai.EventDone,
	}
	modeltest.AssertEventTypes(t, events, wantTypes)
	if events[2].TextDelta != "你好" {
		t.Fatalf("text delta = %q", events[2].TextDelta)
	}
	if events[4].ToolCallDelta == nil || events[4].ToolCallDelta.ID != "call_1" ||
		events[4].ToolCallDelta.Name != "get_weather" || events[4].ToolCallDelta.ArgumentsDelta != `{"city":` {
		t.Fatalf("unexpected tool delta: %+v", events[4].ToolCallDelta)
	}
	if events[5].ToolCallDelta == nil || events[5].ToolCallDelta.ArgumentsDelta != `"北京"}` {
		t.Fatalf("unexpected second tool delta: %+v", events[5].ToolCallDelta)
	}
	if events[6].Usage == nil || events[6].Usage.TotalTokens != 13 {
		t.Fatalf("usage = %+v", events[6].Usage)
	}
	if events[9].StopReason != ai.StopReasonToolCalls {
		t.Fatalf("stop reason = %q", events[9].StopReason)
	}
	if events[9].Message == nil || len(events[9].Message.ToolCalls()) != 1 {
		t.Fatalf("done message = %+v", events[9].Message)
	}
	if got := string(events[9].Message.ToolCalls()[0].Arguments); got != `{"city":"北京"}` {
		t.Fatalf("tool call arguments = %q", got)
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

func TestProvidersReportProtocolCapabilities(t *testing.T) {
	completions, err := openai.NewCompletions(openai.Config{Model: "model"})
	if err != nil {
		t.Fatal(err)
	}
	info := completions.Info()
	if !info.SupportsImages || info.SupportsToolResultImages || info.SupportsThinking {
		t.Fatalf("completions info = %+v", info)
	}

	responses, err := openai.NewResponses(openai.Config{Model: "model"})
	if err != nil {
		t.Fatal(err)
	}
	info = responses.Info()
	if !info.SupportsImages || !info.SupportsToolResultImages || !info.SupportsThinking {
		t.Fatalf("responses info = %+v", info)
	}
}

func TestCompletionsMapsUserImages(t *testing.T) {
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
			Body:       io.NopCloser(strings.NewReader("data: [DONE]\n\n")),
			Request:    request,
		}, nil
	})}

	provider, err := openai.NewCompletions(openai.Config{
		BaseURL:    "https://example.test",
		Model:      "test-model",
		HTTPClient: client,
	})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := provider.Stream(context.Background(), ai.Request{
		Messages: []ai.Message{{
			Role: ai.RoleUser,
			Content: []ai.ContentBlock{
				{Type: ai.BlockText, Text: "look"},
				{Type: ai.BlockImage, Image: &ai.ImageContent{Data: "aGVsbG8=", MediaType: "image/png"}},
			},
		}},
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
	message := messages[0].(map[string]any)
	content := message["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("content = %+v", content)
	}
	text := content[0].(map[string]any)
	if text["type"] != "text" || text["text"] != "look" {
		t.Fatalf("text part = %+v", text)
	}
	image := content[1].(map[string]any)
	imageURL := image["image_url"].(map[string]any)
	if image["type"] != "image_url" || imageURL["url"] != "data:image/png;base64,aGVsbG8=" {
		t.Fatalf("image part = %+v", image)
	}
}

func TestCompletionsRejectsImageToolResults(t *testing.T) {
	provider, err := openai.NewCompletions(openai.Config{Model: "model"})
	if err != nil {
		t.Fatal(err)
	}
	_, err = provider.Stream(context.Background(), ai.Request{
		Messages: []ai.Message{ai.ToolResultMessage(
			"call_1",
			"screenshot",
			[]ai.ContentBlock{{
				Type:  ai.BlockImage,
				Image: &ai.ImageContent{Data: "aGVsbG8=", MediaType: "image/png"},
			}},
			false,
		)},
	})
	if err == nil || !strings.Contains(err.Error(), "does not support image tool results") {
		t.Fatalf("error = %v", err)
	}
}

func TestProviderSendsCustomHeaders(t *testing.T) {
	headersCh := make(chan http.Header, 1)
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		headersCh <- request.Header.Clone()
		return &http.Response{
			StatusCode: http.StatusOK,
			Status:     "200 OK",
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:       io.NopCloser(strings.NewReader("data: [DONE]\n\n")),
			Request:    request,
		}, nil
	})}

	provider, err := openai.New(openai.Config{
		BaseURL:    "https://example.test",
		Model:      "test-model",
		Provider:   "compatible",
		Headers:    map[string]string{"X-Goat-Test": "openai"},
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
	if got := headers.Get("X-Goat-Test"); got != "openai" {
		t.Fatalf("X-Goat-Test = %q", got)
	}
	if got := provider.Info().Provider; got != "compatible" {
		t.Fatalf("provider = %q", got)
	}
}

func writeChunk(writer io.Writer, chunk map[string]any) {
	data, err := json.Marshal(chunk)
	if err != nil {
		panic(err)
	}
	fmt.Fprintf(writer, "data: %s\n\n", data)
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
