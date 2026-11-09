// Package agent provides a provider-neutral tool-calling agent loop.
package agent

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/mrsibe/goat/ai"
)

// ToolResult is the observation returned to the model after a tool call.
type ToolResult struct {
	CallID  string `json:"call_id"`
	Content string `json:"content"`
	IsError bool   `json:"is_error"`
}

// Tool is a callable capability exposed to the model.
type Tool interface {
	Spec() ai.ToolSpec
	Execute(ctx context.Context, call ai.ToolCall) (ToolResult, error)
}

// ToolFunc is a function-backed tool implementation.
type ToolFunc func(context.Context, ai.ToolCall) (string, error)

type functionTool struct {
	spec ai.ToolSpec
	fn   ToolFunc
}

// NewTool adapts a function into a Tool.
func NewTool(spec ai.ToolSpec, fn ToolFunc) Tool {
	if fn == nil {
		fn = func(context.Context, ai.ToolCall) (string, error) {
			return "", errors.New("tool handler is nil")
		}
	}
	return &functionTool{spec: cloneToolSpec(spec), fn: fn}
}

func (t *functionTool) Spec() ai.ToolSpec {
	return cloneToolSpec(t.spec)
}

func (t *functionTool) Execute(ctx context.Context, call ai.ToolCall) (ToolResult, error) {
	content, err := t.fn(ctx, call)
	return ToolResult{CallID: call.ID, Content: content}, err
}

type toolRegistry struct {
	byName map[string]Tool
	specs  []ai.ToolSpec
}

func newToolRegistry(tools []Tool) (toolRegistry, error) {
	registry := toolRegistry{byName: make(map[string]Tool, len(tools))}
	for _, tool := range tools {
		if tool == nil {
			return toolRegistry{}, errors.New("agent: tool must not be nil")
		}
		spec := cloneToolSpec(tool.Spec())
		if strings.TrimSpace(spec.Name) == "" {
			return toolRegistry{}, errors.New("agent: tool name must not be empty")
		}
		if _, exists := registry.byName[spec.Name]; exists {
			return toolRegistry{}, fmt.Errorf("agent: duplicate tool name %q", spec.Name)
		}
		if spec.Parameters == nil {
			spec.Parameters = map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			}
		}
		registry.byName[spec.Name] = tool
		registry.specs = append(registry.specs, spec)
	}
	slices.SortFunc(registry.specs, func(a, b ai.ToolSpec) int {
		return strings.Compare(a.Name, b.Name)
	})
	return registry, nil
}

func (r toolRegistry) list() []ai.ToolSpec {
	specs := make([]ai.ToolSpec, len(r.specs))
	for i, spec := range r.specs {
		specs[i] = cloneToolSpec(spec)
	}
	return specs
}

func (r toolRegistry) get(name string) (Tool, bool) {
	tool, ok := r.byName[name]
	return tool, ok
}

func cloneToolSpec(spec ai.ToolSpec) ai.ToolSpec {
	spec.Parameters = maps.Clone(spec.Parameters)
	return spec
}
