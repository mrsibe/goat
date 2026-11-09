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
	Content []ai.ContentBlock `json:"content"`
	IsError bool              `json:"is_error"`
}

// TextResult creates a successful text tool result.
func TextResult(text string) ToolResult {
	return ToolResult{
		Content: []ai.ContentBlock{{Type: ai.BlockText, Text: text}},
	}
}

// ErrorResult creates a failed text tool result.
func ErrorResult(text string) ToolResult {
	return ToolResult{
		Content: []ai.ContentBlock{{Type: ai.BlockText, Text: text}},
		IsError: true,
	}
}

// Text returns all text blocks in the result joined without separators.
func (r ToolResult) Text() string {
	var text strings.Builder
	for _, block := range r.Content {
		if block.Type == ai.BlockText {
			text.WriteString(block.Text)
		}
	}
	return text.String()
}

// Tool is a callable capability exposed to the model.
type Tool interface {
	Spec() ai.ToolSpec
	Execute(ctx context.Context, call ai.ToolCall) (ToolResult, error)
}

// ToolFunc is a function-backed tool implementation.
type ToolFunc func(context.Context, ai.ToolCall) (ToolResult, error)

type functionTool struct {
	spec ai.ToolSpec
	fn   ToolFunc
}

// NewTool adapts a function into a Tool.
func NewTool(spec ai.ToolSpec, fn ToolFunc) Tool {
	if fn == nil {
		fn = func(context.Context, ai.ToolCall) (ToolResult, error) {
			return ToolResult{}, errors.New("tool handler is nil")
		}
	}
	return &functionTool{spec: cloneToolSpec(spec), fn: fn}
}

// NewTextTool adapts a text-returning function into a Tool.
func NewTextTool(spec ai.ToolSpec, fn func(context.Context, ai.ToolCall) (string, error)) Tool {
	if fn == nil {
		return NewTool(spec, nil)
	}
	return NewTool(spec, func(ctx context.Context, call ai.ToolCall) (ToolResult, error) {
		text, err := fn(ctx, call)
		if err != nil {
			return ToolResult{}, err
		}
		return TextResult(text), nil
	})
}

func (t *functionTool) Spec() ai.ToolSpec {
	return cloneToolSpec(t.spec)
}

func (t *functionTool) Execute(ctx context.Context, call ai.ToolCall) (ToolResult, error) {
	result, err := t.fn(ctx, call)
	result.Content = ai.CloneContentBlocks(result.Content)
	return result, err
}

type toolRegistry struct {
	byName     map[string]Tool
	specByName map[string]ai.ToolSpec
	specs      []ai.ToolSpec
}

func newToolRegistry(tools []Tool) (toolRegistry, error) {
	registry := toolRegistry{
		byName:     make(map[string]Tool, len(tools)),
		specByName: make(map[string]ai.ToolSpec, len(tools)),
	}
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
		registry.specByName[spec.Name] = spec
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

func (r toolRegistry) get(name string) (Tool, ai.ToolSpec, bool) {
	tool, ok := r.byName[name]
	if !ok {
		return nil, ai.ToolSpec{}, false
	}
	return tool, cloneToolSpec(r.specByName[name]), true
}

func cloneToolSpec(spec ai.ToolSpec) ai.ToolSpec {
	spec.Parameters = maps.Clone(spec.Parameters)
	return spec
}

func validateToolResult(result ToolResult) error {
	for index, block := range result.Content {
		switch block.Type {
		case ai.BlockText:
		case ai.BlockImage:
			if block.Image == nil {
				return fmt.Errorf("content block %d is an image without payload", index)
			}
			hasData := strings.TrimSpace(block.Image.Data) != ""
			hasURL := strings.TrimSpace(block.Image.URL) != ""
			if hasData == hasURL {
				return fmt.Errorf("content block %d image must set exactly one of data or url", index)
			}
			if hasData && strings.TrimSpace(block.Image.MediaType) == "" {
				return fmt.Errorf("content block %d image data requires a media type", index)
			}
		default:
			return fmt.Errorf("tool result content block %d has unsupported type %q", index, block.Type)
		}
	}
	return nil
}
