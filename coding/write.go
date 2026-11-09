package coding

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/mrsibe/goat/agent"
	"github.com/mrsibe/goat/ai"
)

type writeInput struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

type writeTool struct {
	workspace string
}

func (t *writeTool) Spec() ai.ToolSpec {
	return ai.ToolSpec{
		Name:        "write",
		Description: "Write content to a file. Creates the file if it does not exist, overwrites it if it does, and creates parent directories.",
		Parameters: objectSchema(map[string]any{
			"path":    map[string]any{"type": "string", "description": "Path to the file to write (relative or absolute, confined to the workspace)"},
			"content": map[string]any{"type": "string", "description": "Content to write to the file"},
		}, "path", "content"),
	}
}

func (t *writeTool) Execute(ctx context.Context, call ai.ToolCall) (agent.ToolResult, error) {
	var input writeInput
	if err := decodeArguments(call, &input); err != nil {
		return agent.ToolResult{}, err
	}
	if err := ctx.Err(); err != nil {
		return agent.ToolResult{}, err
	}
	path, err := resolveWorkspacePath(t.workspace, input.Path, true)
	if err != nil {
		return agent.ToolResult{}, err
	}
	if info, err := os.Stat(path); err == nil && info.IsDir() {
		return agent.ToolResult{}, errors.New("cannot write to a directory")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return agent.ToolResult{}, fmt.Errorf("create parent directories: %w", err)
	}
	if err := os.WriteFile(path, []byte(input.Content), 0o644); err != nil {
		return agent.ToolResult{}, fmt.Errorf("write %s: %w", input.Path, err)
	}
	return agent.ToolResult{
		CallID:  call.ID,
		Content: fmt.Sprintf("Successfully wrote to %s", input.Path),
	}, nil
}
