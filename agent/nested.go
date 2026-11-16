package agent

import (
	"context"
	"encoding/json"
	"sync"

	"github.com/mrsibe/goat/ai"
)

const (
	nestedCallLimit     = 32
	nestedArgumentLimit = 2048
)

type nestedRecorderKey struct{}

type nestedCallRecorder struct {
	mu      sync.Mutex
	calls   []ai.NestedToolCallRecord
	dropped bool
}

// WithNestedCallRecorder returns a context that records tool calls made by a
// tool. Agent.RunToolCall records into it automatically.
func WithNestedCallRecorder(ctx context.Context) (context.Context, *nestedCallRecorder) {
	recorder := &nestedCallRecorder{}
	return context.WithValue(ctx, nestedRecorderKey{}, recorder), recorder
}

func nestedRecorderFromContext(ctx context.Context) *nestedCallRecorder {
	recorder, _ := ctx.Value(nestedRecorderKey{}).(*nestedCallRecorder)
	return recorder
}

func (r *nestedCallRecorder) record(callID, name string, args json.RawMessage, status string, durationMs int64, message string) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.calls) >= nestedCallLimit {
		r.dropped = true
		return
	}
	entry := ai.NestedToolCallRecord{
		ID:         callID,
		Name:       name,
		Status:     status,
		DurationMs: durationMs,
		Error:      message,
	}
	if len(args) > nestedArgumentLimit {
		entry.ArgumentsBytes = len(args)
	} else if len(args) > 0 {
		entry.Arguments = ai.CloneRawMessage(args)
	}
	r.calls = append(r.calls, entry)
}

func (r *nestedCallRecorder) snapshot() *ai.NestedToolCalls {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.calls) == 0 {
		return nil
	}
	return &ai.NestedToolCalls{
		Calls:    append([]ai.NestedToolCallRecord(nil), r.calls...),
		Complete: !r.dropped,
	}
}
