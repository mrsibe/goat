// Package google adapts the Google Gemini API to the ai model interface.
//
// It uses the official google.golang.org/genai SDK and maps Gemini content,
// function calls, thought signatures, usage metadata, and streaming events to
// GOAT's provider-neutral model contract.
package google

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/ai/internal/streaming"
	"google.golang.org/genai"
)

// Config configures a Gemini API model. BaseURL is optional and should include
// the API version path when set (for example https://proxy.example.com/v1beta).
type Config struct {
	APIKey          string
	BaseURL         string
	Model           string
	Provider        string
	Headers         map[string]string
	HTTPClient      *http.Client
	MaxRetries      *int
	ContextWindow   int
	MaxOutputTokens int
}

// Provider implements ai.Model using Gemini streaming generation.
type Provider struct {
	client *genai.Client
	info   ai.ModelInfo
}

// New creates a Gemini API provider.
func New(config Config) (*Provider, error) {
	model := strings.TrimSpace(config.Model)
	if model == "" {
		return nil, errors.New("google: model must not be empty")
	}
	provider := strings.TrimSpace(config.Provider)
	if provider == "" {
		provider = "google"
	}

	httpOptions := genai.HTTPOptions{}
	if config.BaseURL != "" {
		httpOptions.BaseURL = strings.TrimRight(config.BaseURL, "/")
		httpOptions.APIVersion = ""
	}
	if len(config.Headers) > 0 {
		httpOptions.Headers = make(http.Header, len(config.Headers))
		for key, value := range config.Headers {
			if strings.TrimSpace(key) == "" {
				continue
			}
			httpOptions.Headers.Set(key, value)
		}
	}
	if config.MaxRetries != nil {
		attempts := int32(*config.MaxRetries + 1)
		if attempts < 1 {
			attempts = 1
		}
		httpOptions.RetryOptions = &genai.HTTPRetryOptions{Attempts: &attempts}
	}

	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:      config.APIKey,
		Backend:     genai.BackendGeminiAPI,
		HTTPClient:  config.HTTPClient,
		HTTPOptions: httpOptions,
	})
	if err != nil {
		return nil, fmt.Errorf("google: create client: %w", err)
	}

	return &Provider{
		client: client,
		info: ai.ModelInfo{
			Provider:                 provider,
			Model:                    model,
			API:                      ai.APIGoogleGenerativeAI,
			ContextWindow:            config.ContextWindow,
			MaxOutputTokens:          config.MaxOutputTokens,
			SupportsTools:            true,
			SupportsImages:           true,
			SupportsToolResultImages: supportsMultimodalFunctionResponse(model),
			SupportsThinking:         true,
		},
	}, nil
}

// Info implements ai.Model.
func (p *Provider) Info() ai.ModelInfo {
	if p == nil {
		return ai.ModelInfo{}
	}
	return p.info
}

var _ ai.Model = (*Provider)(nil)

// Stream starts one streamed Gemini generation request.
func (p *Provider) Stream(ctx context.Context, request ai.Request) (ai.Stream, error) {
	if p == nil || p.client == nil {
		return nil, errors.New("google: provider is nil")
	}
	contents, system, err := p.convertMessages(request.Messages)
	if err != nil {
		return nil, err
	}
	config, err := p.generationConfig(request, system)
	if err != nil {
		return nil, err
	}

	sequence := p.client.Models.GenerateContentStream(ctx, p.info.Model, contents, config)
	next, stop := iter.Pull2(sequence)
	return &responseStream{
		info:    p.info,
		builder: streaming.New(p.info),
		next:    next,
		stop:    stop,
		seenIDs: make(map[string]bool),
	}, nil
}

func (p *Provider) generationConfig(request ai.Request, system string) (*genai.GenerateContentConfig, error) {
	config := &genai.GenerateContentConfig{}
	if system != "" {
		config.SystemInstruction = &genai.Content{
			Parts: []*genai.Part{{Text: system}},
		}
	}
	if request.Options.Temperature != nil {
		value := float32(*request.Options.Temperature)
		config.Temperature = &value
	}
	if request.Options.TopP != nil {
		value := float32(*request.Options.TopP)
		config.TopP = &value
	}
	if request.Options.MaxTokens > 0 {
		config.MaxOutputTokens = int32(request.Options.MaxTokens)
	}

	tools, err := convertTools(request.Tools)
	if err != nil {
		return nil, err
	}
	if len(tools) > 0 {
		config.Tools = tools
		config.ToolConfig = convertToolChoice(request.Options.ToolChoice)
	}
	config.ThinkingConfig = convertThinkingConfig(p.info.Model, request.Options)
	return config, nil
}

var toolCallCounter atomic.Uint64

type responseStream struct {
	info    ai.ModelInfo
	builder *streaming.Builder
	next    func() (*genai.GenerateContentResponse, error, bool)
	stop    func()

	pending []ai.Event
	started bool
	done    bool

	finishReason string
	sawToolCall  bool
	blockSeq     int
	toolSeq      int
	currentKey   string
	currentType  ai.BlockType
	seenIDs      map[string]bool
}

func (s *responseStream) Next() (ai.Event, error) {
	if !s.started {
		s.started = true
		return s.builder.Start(), nil
	}
	for {
		if len(s.pending) > 0 {
			event := s.pending[0]
			s.pending = s.pending[1:]
			return event, nil
		}
		if s.done {
			return ai.Event{}, io.EOF
		}
		chunk, err, ok := s.next()
		if !ok {
			s.finish()
			continue
		}
		if err != nil {
			s.done = true
			return s.builder.Fail(normalizeError(s.info.Provider, err), ai.StopReasonError), nil
		}
		s.consume(chunk)
	}
}

func (s *responseStream) Close() error {
	if s != nil && s.stop != nil {
		s.stop()
	}
	return nil
}

func (s *responseStream) consume(chunk *genai.GenerateContentResponse) {
	if chunk == nil {
		return
	}
	if chunk.UsageMetadata != nil {
		s.pending = append(s.pending, s.builder.SetUsage(usageFromMetadata(*chunk.UsageMetadata)))
	}
	if len(chunk.Candidates) == 0 {
		return
	}
	candidate := chunk.Candidates[0]
	if candidate == nil {
		return
	}
	if candidate.Content != nil {
		for _, part := range candidate.Content.Parts {
			if part == nil {
				continue
			}
			if part.Text != "" || (part.FunctionCall == nil && len(part.ThoughtSignature) > 0) {
				s.consumeText(part)
			}
			if part.FunctionCall != nil {
				s.consumeFunctionCall(part)
			}
		}
	}
	if candidate.FinishReason != "" {
		s.finishReason = string(candidate.FinishReason)
	}
}

func (s *responseStream) consumeText(part *genai.Part) {
	blockType := ai.BlockText
	if part.Thought {
		blockType = ai.BlockThinking
	}
	if s.currentType != blockType {
		s.closeBlock()
		s.blockSeq++
		s.currentType = blockType
		s.currentKey = fmt.Sprintf("%s:%d", blockType, s.blockSeq)
		if blockType == ai.BlockThinking {
			s.pending = append(s.pending, s.builder.ThinkingStart(s.currentKey))
		} else {
			s.pending = append(s.pending, s.builder.TextStart(s.currentKey))
		}
	}
	signature := base64.StdEncoding.EncodeToString(part.ThoughtSignature)
	if blockType == ai.BlockThinking {
		if signature != "" {
			s.builder.SetThinkingMeta(s.currentKey, "", signature, false)
		}
		if part.Text != "" {
			s.pending = append(s.pending, s.builder.ThinkingDelta(s.currentKey, part.Text))
		}
		return
	}
	if signature != "" {
		s.builder.SetTextMeta(s.currentKey, signature)
	}
	if part.Text != "" {
		s.pending = append(s.pending, s.builder.TextDelta(s.currentKey, part.Text))
	}
}

func (s *responseStream) consumeFunctionCall(part *genai.Part) {
	s.closeBlock()
	call := part.FunctionCall
	name := call.Name
	arguments := call.Args
	if arguments == nil {
		arguments = map[string]any{}
	}
	raw, err := json.Marshal(arguments)
	if err != nil {
		s.done = true
		s.pending = append(s.pending, s.builder.Fail(
			fmt.Errorf("google: encode tool arguments: %w", err),
			ai.StopReasonError,
		))
		return
	}
	id := call.ID
	if id == "" || s.seenIDs[id] {
		id = generateToolCallID(name)
	}
	s.seenIDs[id] = true
	s.sawToolCall = true
	s.toolSeq++
	key := fmt.Sprintf("tool:%d", s.toolSeq)
	signature := base64.StdEncoding.EncodeToString(part.ThoughtSignature)

	s.pending = append(s.pending, s.builder.ToolStart(key, id, name))
	s.pending = append(s.pending, s.builder.ToolDelta(key, string(raw)))
	event, err := s.builder.ToolEnd(key, &ai.ToolCall{
		ID:               id,
		Name:             name,
		Arguments:        raw,
		ThoughtSignature: signature,
	})
	if err != nil {
		s.done = true
		s.pending = append(s.pending, s.builder.Fail(err, ai.StopReasonError))
		return
	}
	s.pending = append(s.pending, event)
}

func (s *responseStream) closeBlock() {
	if s.currentType == "" {
		return
	}
	if s.currentType == ai.BlockThinking {
		s.pending = append(s.pending, s.builder.ThinkingEnd(s.currentKey))
	} else {
		s.pending = append(s.pending, s.builder.TextEnd(s.currentKey, ""))
	}
	s.currentType = ""
	s.currentKey = ""
}

func (s *responseStream) finish() {
	s.done = true
	s.closeBlock()
	if s.finishReason == "" {
		s.pending = append(s.pending, s.builder.Fail(
			errors.New("google: stream ended without a finish reason"),
			ai.StopReasonError,
		))
		return
	}
	reason := mapStopReason(s.finishReason)
	if reason == ai.StopReasonError {
		s.pending = append(s.pending, s.builder.Fail(
			fmt.Errorf("google: provider stopped with: %s", s.finishReason),
			ai.StopReasonError,
		))
		return
	}
	if reason == ai.StopReasonStop && s.sawToolCall {
		reason = ai.StopReasonToolCalls
	}
	s.pending = append(s.pending, s.builder.Done(reason))
}

func generateToolCallID(name string) string {
	if name == "" {
		name = "tool"
	}
	return fmt.Sprintf("%s_%d_%d", name, time.Now().UnixNano(), toolCallCounter.Add(1))
}

func usageFromMetadata(metadata genai.GenerateContentResponseUsageMetadata) ai.Usage {
	prompt := int64(metadata.PromptTokenCount - metadata.CachedContentTokenCount)
	if prompt < 0 {
		prompt = 0
	}
	return ai.Usage{
		PromptTokens:     prompt,
		CompletionTokens: int64(metadata.CandidatesTokenCount + metadata.ThoughtsTokenCount),
		CacheReadTokens:  int64(metadata.CachedContentTokenCount),
		ReasoningTokens:  int64(metadata.ThoughtsTokenCount),
		TotalTokens:      int64(metadata.TotalTokenCount),
	}
}

func mapStopReason(reason string) ai.StopReason {
	switch reason {
	case string(genai.FinishReasonStop):
		return ai.StopReasonStop
	case string(genai.FinishReasonMaxTokens):
		return ai.StopReasonLength
	case string(genai.FinishReasonContinuation):
		return ai.StopReasonLength
	default:
		return ai.StopReasonError
	}
}

func normalizeError(provider string, err error) error {
	if err == nil {
		return nil
	}
	var apiErr genai.APIError
	if errors.As(err, &apiErr) {
		return newAPIError(provider, apiErr, err)
	}
	var apiErrPtr *genai.APIError
	if errors.As(err, &apiErrPtr) && apiErrPtr != nil {
		return newAPIError(provider, *apiErrPtr, err)
	}
	return err
}

func newAPIError(provider string, apiErr genai.APIError, err error) error {
	return &ai.APIError{
		Provider:   provider,
		StatusCode: apiErr.Code,
		Code:       apiErr.Status,
		Message:    apiErr.Message,
		Retryable:  apiErr.Code == http.StatusTooManyRequests || apiErr.Code >= 500,
		Err:        err,
	}
}
