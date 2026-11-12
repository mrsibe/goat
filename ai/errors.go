package ai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"
)

// APIError is the provider-neutral error returned by model adapters.
type APIError struct {
	Provider   string
	StatusCode int
	Code       string
	Message    string
	RequestID  string
	Retryable  bool
	Err        error
}

func (e *APIError) Error() string {
	if e == nil {
		return "<nil>"
	}
	switch {
	case e.Provider != "" && e.StatusCode != 0:
		return fmt.Sprintf("%s api error (%d): %s", e.Provider, e.StatusCode, e.Message)
	case e.Provider != "":
		return fmt.Sprintf("%s api error: %s", e.Provider, e.Message)
	default:
		return e.Message
	}
}

func (e *APIError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// RetryPolicy controls bounded retries with exponential backoff. MaxRetries is
// the number of retries after the initial call.
type RetryPolicy struct {
	Enabled    bool
	MaxRetries int
	BaseDelay  time.Duration
	MaxDelay   time.Duration
}

// RetryDelay returns the delay before the given 1-indexed retry attempt.
func RetryDelay(policy RetryPolicy, attempt int) time.Duration {
	if attempt < 1 {
		attempt = 1
	}
	delay := policy.BaseDelay
	for i := 1; i < attempt; i++ {
		if policy.MaxDelay > 0 && delay >= policy.MaxDelay/2 {
			delay = policy.MaxDelay
			break
		}
		delay *= 2
	}
	if policy.MaxDelay > 0 && delay > policy.MaxDelay {
		delay = policy.MaxDelay
	}
	return delay
}

// IsRetryable reports whether an error represents a transient provider or
// transport failure. APIError.Retryable is authoritative when set.
func IsRetryable(err error) bool {
	if err == nil {
		return false
	}
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		if apiErr.Retryable {
			return true
		}
		return apiErr.StatusCode == http.StatusTooManyRequests || apiErr.StatusCode >= 500
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var netErr net.Error
	if errors.As(err, &netErr) && (netErr.Timeout() || netErr.Temporary()) {
		return true
	}
	if errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}
	message := strings.ToLower(err.Error())
	for _, fragment := range []string{
		"rate limit",
		"overloaded",
		"temporarily unavailable",
		"timeout",
		"timed out",
		"connection reset",
		"connection refused",
		"fetch failed",
	} {
		if strings.Contains(message, fragment) {
			return true
		}
	}
	return false
}

// Retry runs op until it succeeds, returns a non-retryable error, or exhausts
// the configured retry budget. Context cancellation interrupts backoff.
func Retry(ctx context.Context, policy RetryPolicy, op func(context.Context) error) error {
	if op == nil {
		return errors.New("ai: retry operation must not be nil")
	}
	attempts := 1
	if policy.Enabled && policy.MaxRetries > 0 {
		attempts += policy.MaxRetries
	}
	var lastErr error
	for attempt := 1; attempt <= attempts; attempt++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		lastErr = op(ctx)
		if lastErr == nil || !IsRetryable(lastErr) || attempt == attempts {
			return lastErr
		}
		delay := RetryDelay(policy, attempt)
		if delay <= 0 {
			continue
		}
		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
	return lastErr
}
