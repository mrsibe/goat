package google

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/ai/internal/modeltest"
	"google.golang.org/genai"
)

type capturedRequest struct {
	path   string
	query  string
	apiKey string
	body   map[string]any
}

func TestProviderMapsGeminiRequestsAndStreaming(t *testing.T) {
	requests := make(chan capturedRequest, 1)
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		raw, err := io.ReadAll(request.Body)
		if err != nil {
			return nil, err
		}
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			return nil, err
		}
		requests <- capturedRequest{
			path:   request.URL.Path,
			query:  request.URL.RawQuery,
			apiKey: request.Header.Get("x-goog-api-key"),
			body:   body,
		}
		return sseResponse(request, []string{
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"thinking","thought":true,"thoughtSignature":"c2ln"}]}}],"responseId":"resp_1"}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"hello","thoughtSignature":"c2ln"}]}}]}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"call_2","name":"get_weather","args":{"city":"Shanghai"}},"thoughtSignature":"c2ln"}]}}]}`,
			`{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":3,"thoughtsTokenCount":2,"cachedContentTokenCount":1,"totalTokenCount":15}}`,
		}), nil
	})}

	provider, err := New(Config{
		APIKey:     "test-key",
		BaseURL:    "https://example.test/v1beta",
		Model:      "gemini-3-flash",
		HTTPClient: client,
	})
	if err != nil {
		t.Fatal(err)
	}
	if provider.Info().API != ai.APIGoogleGenerativeAI {
		t.Fatalf("info = %+v", provider.Info())
	}

	temperature := 0.2
	call := ai.ToolCall{
		ID:               "call_1",
		Name:             "get_weather",
		Arguments:        json.RawMessage(`{"city":"Beijing"}`),
		ThoughtSignature: "c2ln",
	}
	request := ai.Request{
		Messages: []ai.Message{
			ai.TextMessage(ai.RoleSystem, "sys"),
			ai.TextMessage(ai.RoleUser, "hi"),
			{
				Role:     ai.RoleAssistant,
				Provider: "google",
				Model:    "gemini-3-flash",
				Content: []ai.ContentBlock{
					{Type: ai.BlockText, Text: "previous"},
					{Type: ai.BlockThinking, Thinking: "reasoning", ThinkingSignature: "c2ln"},
					{Type: ai.BlockToolCall, ToolCall: &call},
				},
			},
			{
				Role:       ai.RoleTool,
				ToolCallID: "call_1",
				ToolName:   "get_weather",
				Content: []ai.ContentBlock{
					{Type: ai.BlockText, Text: "sunny"},
					{Type: ai.BlockImage, Image: &ai.ImageContent{Data: "aGVsbG8=", MediaType: "image/png"}},
				},
			},
		},
		Tools: []ai.ToolSpec{{
			Name:        "get_weather",
			Description: "weather",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{"type": "string"},
					"unit": map[string]any{"const": "celsius"},
				},
			},
		}},
		Options: ai.RequestOptions{
			Temperature:     &temperature,
			MaxTokens:       100,
			ReasoningEffort: "medium",
			ToolChoice:      ai.ToolChoice{Mode: ai.ToolChoiceNamed, Name: "get_weather"},
		},
	}

	result, err := modeltest.Collect(context.Background(), provider, request)
	if err != nil {
		t.Fatal(err)
	}
	modeltest.AssertEventTypes(t, result.Events, []ai.EventType{
		ai.EventStart,
		ai.EventThinkingStart,
		ai.EventThinkingDelta,
		ai.EventThinkingEnd,
		ai.EventTextStart,
		ai.EventTextDelta,
		ai.EventTextEnd,
		ai.EventToolCallStart,
		ai.EventToolCallDelta,
		ai.EventToolCallEnd,
		ai.EventUsage,
		ai.EventDone,
	})
	if result.Events[2].ThinkingDelta != "thinking" {
		t.Fatalf("thinking delta = %q", result.Events[2].ThinkingDelta)
	}
	if result.Events[5].TextDelta != "hello" {
		t.Fatalf("text delta = %q", result.Events[5].TextDelta)
	}
	if result.StopReason != ai.StopReasonToolCalls {
		t.Fatalf("stop reason = %q", result.StopReason)
	}
	if result.Usage.PromptTokens != 9 || result.Usage.CompletionTokens != 5 ||
		result.Usage.CacheReadTokens != 1 || result.Usage.ReasoningTokens != 2 ||
		result.Usage.TotalTokens != 15 {
		t.Fatalf("usage = %+v", result.Usage)
	}
	if result.Message == nil {
		t.Fatal("missing terminal message")
	}
	text := result.Message.Text()
	if text != "hello" {
		t.Fatalf("message text = %q", text)
	}
	blocks := result.Message.Content
	if blocks[0].Thinking != "thinking" || blocks[0].ThinkingSignature != "c2ln" {
		t.Fatalf("thinking block = %+v", blocks[0])
	}
	if blocks[1].TextSignature != "c2ln" {
		t.Fatalf("text block = %+v", blocks[1])
	}
	toolCalls := modeltest.ToolCalls(result)
	if len(toolCalls) != 1 || toolCalls[0].ID != "call_2" ||
		toolCalls[0].Name != "get_weather" ||
		string(toolCalls[0].Arguments) != `{"city":"Shanghai"}` ||
		toolCalls[0].ThoughtSignature != "c2ln" {
		t.Fatalf("tool calls = %+v", toolCalls)
	}

	captured := <-requests
	if !strings.HasSuffix(captured.path, "/models/gemini-3-flash:streamGenerateContent") {
		t.Fatalf("path = %q", captured.path)
	}
	if captured.query != "alt=sse" {
		t.Fatalf("query = %q", captured.query)
	}
	if captured.apiKey != "test-key" {
		t.Fatalf("api key header = %q", captured.apiKey)
	}
	assertGeminiRequest(t, captured.body)
}

func assertGeminiRequest(t *testing.T, body map[string]any) {
	t.Helper()
	system := asMap(t, body["systemInstruction"])
	systemParts := asSlice(t, system["parts"])
	if len(systemParts) != 1 || asMap(t, systemParts[0])["text"] != "sys" {
		t.Fatalf("system instruction = %+v", system)
	}

	contents := asSlice(t, body["contents"])
	if len(contents) != 3 {
		t.Fatalf("contents = %+v", contents)
	}
	if asMap(t, contents[0])["role"] != "user" {
		t.Fatalf("first content = %+v", contents[0])
	}
	modelContent := asMap(t, contents[1])
	if modelContent["role"] != "model" {
		t.Fatalf("model content = %+v", modelContent)
	}
	modelParts := asSlice(t, modelContent["parts"])
	if len(modelParts) != 3 {
		t.Fatalf("model parts = %+v", modelParts)
	}
	thinking := asMap(t, modelParts[1])
	if thinking["thought"] != true || thinking["text"] != "reasoning" ||
		thinking["thoughtSignature"] != "c2ln" {
		t.Fatalf("thinking part = %+v", thinking)
	}
	functionCall := asMap(t, asMap(t, modelParts[2])["functionCall"])
	if functionCall["id"] != "call_1" || functionCall["name"] != "get_weather" {
		t.Fatalf("function call = %+v", functionCall)
	}

	toolContent := asMap(t, contents[2])
	toolParts := asSlice(t, toolContent["parts"])
	if len(toolParts) != 1 {
		t.Fatalf("tool parts = %+v", toolParts)
	}
	functionResponse := asMap(t, asMap(t, toolParts[0])["functionResponse"])
	if functionResponse["name"] != "get_weather" || functionResponse["id"] != "call_1" {
		t.Fatalf("function response = %+v", functionResponse)
	}
	if _, ok := functionResponse["parts"]; !ok {
		t.Fatalf("missing multimodal function response parts: %+v", functionResponse)
	}

	tools := asSlice(t, body["tools"])
	declarations := asSlice(t, asMap(t, tools[0])["functionDeclarations"])
	schema := asMap(t, asMap(t, declarations[0])["parametersJsonSchema"])
	properties := asMap(t, schema["properties"])
	unit := asMap(t, properties["unit"])
	if _, hasConst := unit["const"]; hasConst {
		t.Fatalf("const was not rewritten: %+v", unit)
	}
	enum := asSlice(t, unit["enum"])
	if len(enum) != 1 || enum[0] != "celsius" {
		t.Fatalf("enum = %+v", enum)
	}

	toolConfig := asMap(t, body["toolConfig"])
	calling := asMap(t, toolConfig["functionCallingConfig"])
	if calling["mode"] != "ANY" {
		t.Fatalf("tool mode = %+v", calling)
	}
	allowed := asSlice(t, calling["allowedFunctionNames"])
	if len(allowed) != 1 || allowed[0] != "get_weather" {
		t.Fatalf("allowed functions = %+v", allowed)
	}

	generation := asMap(t, body["generationConfig"])
	if generation["temperature"] != 0.2 || generation["maxOutputTokens"] != float64(100) {
		t.Fatalf("generation config = %+v", generation)
	}
	thinkingConfig := asMap(t, generation["thinkingConfig"])
	if thinkingConfig["thinkingLevel"] != "MEDIUM" || thinkingConfig["includeThoughts"] != true {
		t.Fatalf("thinking config = %+v", thinkingConfig)
	}
}

func TestProviderMapsMaxTokensStopReason(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		return sseResponse(request, []string{
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"partial"}]},"finishReason":"MAX_TOKENS"}]}`,
		}), nil
	})}

	provider, err := New(Config{
		APIKey:     "test-key",
		BaseURL:    "https://example.test/v1beta",
		Model:      "gemini-2.5-flash",
		HTTPClient: client,
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := modeltest.Collect(context.Background(), provider, ai.Request{
		Messages: []ai.Message{ai.TextMessage(ai.RoleUser, "hi")},
		Options:  ai.RequestOptions{ReasoningEffort: "low"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.StopReason != ai.StopReasonLength {
		t.Fatalf("stop reason = %q", result.StopReason)
	}
	if result.Message == nil || result.Message.Text() != "partial" {
		t.Fatalf("message = %+v", result.Message)
	}
}

func TestProviderNormalizesAPIErrors(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusTooManyRequests,
			Status:     "429 Too Many Requests",
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body: io.NopCloser(strings.NewReader(
				`{"error":{"code":429,"message":"quota exceeded","status":"RESOURCE_EXHAUSTED"}}`,
			)),
			Request: request,
		}, nil
	})}

	provider, err := New(Config{
		APIKey:     "test-key",
		BaseURL:    "https://example.test/v1beta",
		Model:      "gemini-2.5-flash",
		HTTPClient: client,
		MaxRetries: intPtr(0),
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
	if result.StopReason != ai.StopReasonError {
		t.Fatalf("stop reason = %q", result.StopReason)
	}
	last := result.Events[len(result.Events)-1]
	if last.Type != ai.EventError || last.Err == nil {
		t.Fatalf("last event = %+v", last)
	}
	var apiErr *ai.APIError
	if !errors.As(last.Err, &apiErr) {
		t.Fatalf("error = %T %v", last.Err, last.Err)
	}
	if apiErr.StatusCode != http.StatusTooManyRequests || !apiErr.Retryable || apiErr.Provider != "google" {
		t.Fatalf("api error = %+v", apiErr)
	}
}

func TestRewriteConstAsEnumPreservesPropertyNames(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"const": map[string]any{"type": "string", "const": "value"},
		},
	}
	rewritten := asMap(t, rewriteConstAsEnum(schema))
	properties := asMap(t, rewritten["properties"])
	property := asMap(t, properties["const"])
	if _, ok := property["const"]; ok {
		t.Fatalf("property const was not rewritten: %+v", property)
	}
	enum := asSlice(t, property["enum"])
	if len(enum) != 1 || enum[0] != "value" {
		t.Fatalf("enum = %+v", enum)
	}
}

func TestNormalizeSchemaHandlesTypedSlices(t *testing.T) {
	schema := map[string]any{
		"anyOf": []map[string]any{
			{"type": "string", "const": "first"},
			{"type": "string", "const": "second"},
		},
	}
	normalized, err := normalizeSchema(schema)
	if err != nil {
		t.Fatal(err)
	}
	rewritten := asMap(t, rewriteConstAsEnum(normalized))
	options := asSlice(t, rewritten["anyOf"])
	for index, option := range options {
		optionMap := asMap(t, option)
		if _, ok := optionMap["const"]; ok {
			t.Fatalf("option %d const was not rewritten: %+v", index, optionMap)
		}
		if _, ok := optionMap["enum"]; !ok {
			t.Fatalf("option %d missing enum: %+v", index, optionMap)
		}
	}
}

func TestThinkingConfigModes(t *testing.T) {
	if config := convertThinkingConfig("gemini-2.5-flash", ai.RequestOptions{}); config != nil {
		t.Fatalf("default thinking config = %+v", config)
	}
	config := convertThinkingConfig("gemini-2.5-flash", ai.RequestOptions{ReasoningEffort: "high"})
	if config == nil || config.ThinkingBudget == nil || *config.ThinkingBudget != 24576 || !config.IncludeThoughts {
		t.Fatalf("budget config = %+v", config)
	}
	config = convertThinkingConfig("gemini-3-flash", ai.RequestOptions{ReasoningEffort: "minimal"})
	if config == nil || config.ThinkingLevel != "MINIMAL" || !config.IncludeThoughts {
		t.Fatalf("level config = %+v", config)
	}
	config = convertThinkingConfig("gemini-3-flash", ai.RequestOptions{ReasoningEffort: "off"})
	if config == nil || config.ThinkingLevel != genai.ThinkingLevelMinimal || config.ThinkingBudget != nil {
		t.Fatalf("disabled config = %+v", config)
	}
	config = convertThinkingConfig("gemini-2.5-flash", ai.RequestOptions{ReasoningEffort: "off"})
	if config == nil || config.ThinkingBudget == nil || *config.ThinkingBudget != 0 {
		t.Fatalf("disabled budget config = %+v", config)
	}
	config = convertThinkingConfig("gemini-2.5-flash", ai.RequestOptions{ThinkingBudgetTokens: -1})
	if config == nil || config.ThinkingBudget == nil || *config.ThinkingBudget != -1 || !config.IncludeThoughts {
		t.Fatalf("dynamic budget config = %+v", config)
	}
}

func TestProviderRejectsEmptyModel(t *testing.T) {
	if _, err := New(Config{}); err == nil {
		t.Fatal("expected empty model error")
	}
}

func TestNormalizeToolCallID(t *testing.T) {
	if got := normalizeToolCallID("gemini-2.5-flash", "call/with spaces"); got != "call/with spaces" {
		t.Fatalf("gemini 2.5 id = %q", got)
	}
	if got := normalizeToolCallID("gemini-3-flash", "call/with spaces"); got != "call_with_spaces" {
		t.Fatalf("gemini 3 id = %q", got)
	}
	long := strings.Repeat("a", 80)
	if got := normalizeToolCallID("gemini-3-flash", long); len(got) != 64 {
		t.Fatalf("long id length = %d", len(got))
	}
}

func TestSignedFunctionCallDoesNotCreateEmptyTextBlock(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		return sseResponse(request, []string{
			`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"call_1","name":"get_weather","args":{"city":"Shanghai"}},"thoughtSignature":"c2ln"}]}}]}`,
			`{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP"}]}`,
		}), nil
	})}

	provider, err := New(Config{
		APIKey:     "test-key",
		BaseURL:    "https://example.test/v1beta",
		Model:      "gemini-3-flash",
		HTTPClient: client,
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
	if len(result.Message.Content) != 1 || result.Message.Content[0].Type != ai.BlockToolCall {
		t.Fatalf("content = %+v", result.Message.Content)
	}
	if result.Message.Content[0].ToolCall.ThoughtSignature != "c2ln" {
		t.Fatalf("tool call = %+v", result.Message.Content[0].ToolCall)
	}
}

func intPtr(value int) *int {
	return &value
}

func sseResponse(request *http.Request, chunks []string) *http.Response {
	var body strings.Builder
	for _, chunk := range chunks {
		body.WriteString("data: ")
		body.WriteString(chunk)
		body.WriteString("\n\n")
	}
	return &http.Response{
		StatusCode: http.StatusOK,
		Status:     "200 OK",
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader(body.String())),
		Request:    request,
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return f(request)
}

func asMap(t *testing.T, value any) map[string]any {
	t.Helper()
	result, ok := value.(map[string]any)
	if !ok {
		t.Fatalf("expected map, got %T (%v)", value, value)
	}
	return result
}

func asSlice(t *testing.T, value any) []any {
	t.Helper()
	result, ok := value.([]any)
	if !ok {
		t.Fatalf("expected slice, got %T (%v)", value, value)
	}
	return result
}
