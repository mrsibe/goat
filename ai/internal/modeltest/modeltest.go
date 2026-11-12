// Package modeltest provides shared contract helpers for ai.Model adapters.
package modeltest

import (
	"context"
	"errors"
	"io"
	"reflect"
	"testing"

	"github.com/mrsibe/goat/ai"
)

// StreamResult is the normalized output of one adapter stream.
type StreamResult struct {
	Events     []ai.Event
	Message    *ai.Message
	Usage      ai.Usage
	StopReason ai.StopReason
}

// Collect drains a model stream and returns its normalized events and terminal
// message. Stream construction and Next errors are returned directly.
func Collect(ctx context.Context, model ai.Model, request ai.Request) (StreamResult, error) {
	stream, err := model.Stream(ctx, request)
	if err != nil {
		return StreamResult{}, err
	}
	if stream == nil {
		return StreamResult{}, errors.New("modeltest: model returned a nil stream")
	}
	defer stream.Close()

	var result StreamResult
	for {
		event, nextErr := stream.Next()
		if errors.Is(nextErr, io.EOF) {
			return result, nil
		}
		if nextErr != nil {
			return result, nextErr
		}
		result.Events = append(result.Events, event)
		switch event.Type {
		case ai.EventUsage:
			if event.Usage != nil {
				result.Usage = *event.Usage
			}
		case ai.EventDone, ai.EventError:
			result.StopReason = event.StopReason
			if event.Message != nil {
				message := ai.CloneMessage(*event.Message)
				result.Message = &message
			}
		}
	}
}

// EventTypes returns the event type sequence.
func EventTypes(events []ai.Event) []ai.EventType {
	types := make([]ai.EventType, len(events))
	for i, event := range events {
		types[i] = event.Type
	}
	return types
}

// AssertEventTypes fails the test when the normalized event sequence differs.
func AssertEventTypes(t testing.TB, events []ai.Event, want []ai.EventType) {
	t.Helper()
	if got := EventTypes(events); !reflect.DeepEqual(got, want) {
		t.Fatalf("event types = %v, want %v (events: %+v)", got, want, events)
	}
}

// ToolCalls returns the tool calls from the terminal assistant message.
func ToolCalls(result StreamResult) []ai.ToolCall {
	if result.Message == nil {
		return nil
	}
	return result.Message.ToolCalls()
}
