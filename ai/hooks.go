package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
)

// Transport is a preferred model transport. GOAT currently serves every
// provider over HTTP; "websocket" is rejected by adapters.
type Transport string

const (
	TransportAuto      Transport = "auto"
	TransportSSE       Transport = "sse"
	TransportWebSocket Transport = "websocket"
)

// Attribution identifies the calling app to providers that support routing
// headers (currently OpenRouter).
type Attribution struct {
	Name       string
	URL        string
	Categories []string
}

// Identity overrides request identification headers.
type Identity struct {
	UserAgent   string
	Attribution *Attribution
}

type hooksKey struct{}

type requestHooks struct {
	options  RequestOptions
	provider string
}

// WithRequestHooks attaches per-request hooks and identity to the context used
// by provider adapters.
func WithRequestHooks(ctx context.Context, options RequestOptions, provider string) context.Context {
	if options.OnResponse == nil && options.Identity == nil {
		return ctx
	}
	return context.WithValue(ctx, hooksKey{}, requestHooks{options: options, provider: provider})
}

func requestHooksFromContext(ctx context.Context) (requestHooks, bool) {
	hooks, ok := ctx.Value(hooksKey{}).(requestHooks)
	return hooks, ok
}

// ValidateTransport rejects transports GOAT cannot serve.
func ValidateTransport(transport Transport) error {
	switch transport {
	case "", TransportAuto, TransportSSE:
		return nil
	case TransportWebSocket:
		return errors.New("ai: websocket transport is not supported")
	default:
		return fmt.Errorf("ai: unsupported transport %q", transport)
	}
}

// ApplyPayloadHook lets OnPayload inspect or replace a provider request body.
// The body is round-tripped through JSON and decoded back into params. Adapters
// that encode through HookHTTPClient get this automatically; this helper is for
// adapters that want to hook before encoding.
func ApplyPayloadHook(ctx context.Context, params any) error {
	hooks, ok := requestHooksFromContext(ctx)
	if !ok || hooks.options.OnPayload == nil {
		return nil
	}
	raw, err := json.Marshal(params)
	if err != nil {
		return err
	}
	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		return err
	}
	replacement, err := hooks.options.OnPayload(payload)
	if err != nil {
		return err
	}
	if replacement == nil {
		return nil
	}
	raw, err = json.Marshal(replacement)
	if err != nil {
		return err
	}
	return json.Unmarshal(raw, params)
}

// EmitProviderStreamEvent forwards one raw provider event to the observer.
func EmitProviderStreamEvent(ctx context.Context, event any) error {
	hooks, ok := requestHooksFromContext(ctx)
	if !ok || hooks.options.OnProviderStreamEvent == nil {
		return nil
	}
	return hooks.options.OnProviderStreamEvent(event)
}

// HookHTTPClient wraps a client's transport so identity headers are applied and
// OnResponse observes the raw HTTP response.
func HookHTTPClient(base *http.Client) *http.Client {
	if base == nil {
		base = &http.Client{}
	}
	cloned := *base
	cloned.Transport = observingTransport{base: base.Transport, original: base}
	return &cloned
}

type observingTransport struct {
	base     http.RoundTripper
	original *http.Client
}

func (t observingTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	hooks, _ := requestHooksFromContext(request.Context())
	if hooks.options.Identity != nil {
		applyIdentity(request, hooks.options.Identity, hooks.provider)
	}
	if hooks.options.OnPayload != nil || len(hooks.options.SamplingParams) > 0 {
		if err := applyBodyHook(request, hooks.options.OnPayload, hooks.options.SamplingParams); err != nil {
			return nil, err
		}
	}
	base := t.base
	if base == nil {
		base = http.DefaultTransport
	}
	response, err := base.RoundTrip(request)
	if err != nil || response == nil {
		return response, err
	}
	if hooks.options.OnResponse != nil {
		if hookErr := hooks.options.OnResponse(response.StatusCode, response.Header.Clone()); hookErr != nil {
			_ = response.Body.Close()
			return nil, hookErr
		}
	}
	return response, nil
}

func applyBodyHook(
	request *http.Request,
	hook func(map[string]any) (map[string]any, error),
	samplingParams map[string]any,
) error {
	if request.Body == nil {
		return nil
	}
	body, err := io.ReadAll(request.Body)
	if err != nil {
		return err
	}
	_ = request.Body.Close()
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		request.Body = io.NopCloser(bytes.NewReader(body))
		return nil
	}
	for key, value := range samplingParams {
		payload[key] = value
	}
	replacement := payload
	if hook != nil {
		replacement, err = hook(payload)
		if err != nil {
			request.Body = io.NopCloser(bytes.NewReader(body))
			return err
		}
		if replacement == nil {
			request.Body = io.NopCloser(bytes.NewReader(body))
			return nil
		}
	}
	encoded, err := json.Marshal(replacement)
	if err != nil {
		request.Body = io.NopCloser(bytes.NewReader(body))
		return err
	}
	request.Body = io.NopCloser(bytes.NewReader(encoded))
	request.ContentLength = int64(len(encoded))
	request.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(encoded)), nil
	}
	request.Header.Set("Content-Length", fmt.Sprintf("%d", len(encoded)))
	return nil
}

func applyIdentity(request *http.Request, identity *Identity, provider string) {
	if identity.UserAgent != "" {
		request.Header.Set("User-Agent", identity.UserAgent)
	}
	if identity.Attribution == nil {
		return
	}
	if provider == "openrouter" || provider == "" {
		if identity.Attribution.URL != "" {
			request.Header.Set("HTTP-Referer", identity.Attribution.URL)
		}
		if identity.Attribution.Name != "" {
			request.Header.Set("X-Title", identity.Attribution.Name)
		}
		if len(identity.Attribution.Categories) > 0 {
			request.Header.Set("X-OpenRouter-Categories", joinCategories(identity.Attribution.Categories))
		}
	}
}

func joinCategories(categories []string) string {
	result := ""
	for i, category := range categories {
		if i > 0 {
			result += ","
		}
		result += category
	}
	return result
}
