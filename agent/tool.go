package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/mrsibe/goat/ai"
)

// ToolResult is the observation returned to the model after a tool call.
type ToolResult struct {
	Content           []ai.ContentBlock   `json:"content"`
	Details           json.RawMessage     `json:"details,omitempty"`
	StructuredContent json.RawMessage     `json:"structured_content,omitempty"`
	Usage             *ai.Usage           `json:"usage,omitempty"`
	NestedCalls       *ai.NestedToolCalls `json:"nested_calls,omitempty"`
	IsError           bool                `json:"is_error"`
	Terminate         bool                `json:"terminate,omitempty"`
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

func (r ToolResult) clone() ToolResult {
	cloned := r
	cloned.Content = ai.CloneContentBlocks(r.Content)
	cloned.Details = ai.CloneRawMessage(r.Details)
	cloned.StructuredContent = ai.CloneRawMessage(r.StructuredContent)
	if r.Usage != nil {
		usage := *r.Usage
		cloned.Usage = &usage
	}
	if r.NestedCalls != nil {
		nested := *r.NestedCalls
		nested.Calls = append([]ai.NestedToolCallRecord(nil), r.NestedCalls.Calls...)
		cloned.NestedCalls = &nested
	}
	return cloned
}

// ToolReplay is the recovery policy for a tool effect whose durable intent
// exists but whose outcome is unknown.
type ToolReplay string

const (
	ToolReplayNever ToolReplay = "never"
	ToolReplaySafe  ToolReplay = "safe"
)

// ToolSpec is the model-facing declaration plus runtime execution hints.
type ToolSpec struct {
	Name             string
	Description      string
	Parameters       map[string]any
	Label            string
	OutputSchema     map[string]any
	ExecutionMode    ToolExecutionMode
	Replay           ToolReplay
	PrepareArguments func(json.RawMessage) (json.RawMessage, error)
}

// Tool is a callable capability exposed to the model. args is the validated
// JSON object for the call; onUpdate may be nil when the caller cannot consume
// partial results.
type Tool interface {
	Spec() ToolSpec
	Execute(ctx context.Context, callID string, args json.RawMessage, onUpdate func(ToolResult)) (ToolResult, error)
}

// ToolFunc is a non-streaming function-backed tool implementation.
type ToolFunc func(context.Context, string, json.RawMessage) (ToolResult, error)

// StreamingToolFunc is a function-backed streaming tool implementation.
type StreamingToolFunc func(context.Context, string, json.RawMessage, func(ToolResult)) (ToolResult, error)

type functionTool struct {
	spec   ToolSpec
	fn     ToolFunc
	stream StreamingToolFunc
}

// NewTool adapts a function into a Tool.
func NewTool(spec ToolSpec, fn ToolFunc) Tool {
	if fn == nil {
		fn = func(context.Context, string, json.RawMessage) (ToolResult, error) {
			return ToolResult{}, errors.New("tool handler is nil")
		}
	}
	return &functionTool{spec: cloneToolSpec(spec), fn: fn}
}

// NewTextTool adapts a text-returning function into a Tool.
func NewTextTool(spec ToolSpec, fn func(context.Context, string, json.RawMessage) (string, error)) Tool {
	if fn == nil {
		return NewTool(spec, nil)
	}
	return NewTool(spec, func(ctx context.Context, callID string, args json.RawMessage) (ToolResult, error) {
		text, err := fn(ctx, callID, args)
		if err != nil {
			return ToolResult{}, err
		}
		return TextResult(text), nil
	})
}

// NewStreamingTool adapts a streaming function into a Tool.
func NewStreamingTool(spec ToolSpec, fn StreamingToolFunc) Tool {
	if fn == nil {
		return NewTool(spec, nil)
	}
	return &functionTool{
		spec: cloneToolSpec(spec),
		fn: func(ctx context.Context, callID string, args json.RawMessage) (ToolResult, error) {
			return fn(ctx, callID, args, nil)
		},
		stream: fn,
	}
}

func (t *functionTool) Spec() ToolSpec {
	return cloneToolSpec(t.spec)
}

func (t *functionTool) Execute(ctx context.Context, callID string, args json.RawMessage, onUpdate func(ToolResult)) (ToolResult, error) {
	var result ToolResult
	var err error
	if t.stream != nil {
		result, err = t.stream(ctx, callID, args, func(partial ToolResult) {
			if onUpdate != nil {
				onUpdate(partial.clone())
			}
		})
	} else {
		result, err = t.fn(ctx, callID, args)
	}
	return result.clone(), err
}

type toolRegistry struct {
	byName map[string]Tool
	specs  []ToolSpec
	tools  []Tool
}

func newToolRegistry(tools []Tool) (toolRegistry, error) {
	registry := toolRegistry{
		byName: make(map[string]Tool, len(tools)),
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
		registry.specs = append(registry.specs, spec)
		registry.tools = append(registry.tools, tool)
	}
	slices.SortFunc(registry.specs, func(a, b ToolSpec) int {
		return strings.Compare(a.Name, b.Name)
	})
	return registry, nil
}

func (r toolRegistry) list() []ToolSpec {
	specs := make([]ToolSpec, len(r.specs))
	for i, spec := range r.specs {
		specs[i] = cloneToolSpec(spec)
	}
	return specs
}

func (r toolRegistry) listTools() []Tool {
	return append([]Tool(nil), r.tools...)
}

func (r toolRegistry) get(name string) (Tool, ToolSpec, bool) {
	tool, ok := r.byName[name]
	if !ok {
		return nil, ToolSpec{}, false
	}
	for _, spec := range r.specs {
		if spec.Name == name {
			return tool, cloneToolSpec(spec), true
		}
	}
	return nil, ToolSpec{}, false
}

func cloneToolSpec(spec ToolSpec) ToolSpec {
	spec.Parameters = maps.Clone(spec.Parameters)
	spec.OutputSchema = maps.Clone(spec.OutputSchema)
	return spec
}

func toolSpecsToAI(specs []ToolSpec) []ai.ToolSpec {
	if len(specs) == 0 {
		return nil
	}
	result := make([]ai.ToolSpec, len(specs))
	for i, spec := range specs {
		result[i] = spec.aiToolSpec()
	}
	return result
}

func (spec ToolSpec) aiToolSpec() ai.ToolSpec {
	return ai.ToolSpec{
		Name:        spec.Name,
		Description: spec.Description,
		Parameters:  maps.Clone(spec.Parameters),
	}
}

func toolSpecs(tools []Tool) ([]ai.ToolSpec, error) {
	registry, err := newToolRegistry(tools)
	if err != nil {
		return nil, err
	}
	specs := registry.list()
	result := make([]ai.ToolSpec, len(specs))
	for i, spec := range specs {
		result[i] = spec.aiToolSpec()
	}
	return result, nil
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
			if hasData && strings.TrimSpace(block.Image.MimeType) == "" {
				return fmt.Errorf("content block %d image data requires a media type", index)
			}
		default:
			return fmt.Errorf("tool result content block %d has unsupported type %q", index, block.Type)
		}
	}
	return nil
}
