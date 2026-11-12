package ai_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/mrsibe/goat/ai"
)

func TestIsRetryable(t *testing.T) {
	if ai.IsRetryable(nil) {
		t.Fatal("nil error must not be retryable")
	}
	if !ai.IsRetryable(&ai.APIError{StatusCode: 429}) {
		t.Fatal("429 should be retryable")
	}
	if !ai.IsRetryable(&ai.APIError{StatusCode: 503}) {
		t.Fatal("503 should be retryable")
	}
	if ai.IsRetryable(&ai.APIError{StatusCode: 400}) {
		t.Fatal("400 should not be retryable")
	}
	if !ai.IsRetryable(&ai.APIError{Retryable: true, StatusCode: 400}) {
		t.Fatal("explicit Retryable should win")
	}
	if ai.IsRetryable(context.Canceled) {
		t.Fatal("context cancellation must not be retryable")
	}
}

func TestRetryDelayIsCapped(t *testing.T) {
	policy := ai.RetryPolicy{BaseDelay: 100 * time.Millisecond, MaxDelay: 250 * time.Millisecond}
	if got := ai.RetryDelay(policy, 1); got != 100*time.Millisecond {
		t.Fatalf("attempt 1 = %v", got)
	}
	if got := ai.RetryDelay(policy, 2); got != 200*time.Millisecond {
		t.Fatalf("attempt 2 = %v", got)
	}
	if got := ai.RetryDelay(policy, 3); got != 250*time.Millisecond {
		t.Fatalf("attempt 3 = %v", got)
	}
}

func TestRetryStopsAfterSuccess(t *testing.T) {
	attempts := 0
	err := ai.Retry(context.Background(), ai.RetryPolicy{
		Enabled:    true,
		MaxRetries: 3,
		BaseDelay:  time.Millisecond,
		MaxDelay:   time.Millisecond,
	}, func(context.Context) error {
		attempts++
		if attempts < 2 {
			return &ai.APIError{StatusCode: 500}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if attempts != 2 {
		t.Fatalf("attempts = %d", attempts)
	}
}

func TestRetryStopsOnNonRetryableAndCancellation(t *testing.T) {
	attempts := 0
	err := ai.Retry(context.Background(), ai.RetryPolicy{
		Enabled:    true,
		MaxRetries: 3,
		BaseDelay:  time.Millisecond,
	}, func(context.Context) error {
		attempts++
		return &ai.APIError{StatusCode: 400}
	})
	if err == nil || attempts != 1 {
		t.Fatalf("err = %v attempts = %d", err, attempts)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	attempts = 0
	err = ai.Retry(ctx, ai.RetryPolicy{Enabled: true, MaxRetries: 3}, func(context.Context) error {
		attempts++
		return nil
	})
	if !errors.Is(err, context.Canceled) || attempts != 0 {
		t.Fatalf("err = %v attempts = %d", err, attempts)
	}
}
