package coding

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/mrsibe/goat/agent"
	"github.com/mrsibe/goat/ai"
)

type editReplacement struct {
	OldText string `json:"oldText"`
	NewText string `json:"newText"`
}

type editInput struct {
	Path  string            `json:"path"`
	Edits []editReplacement `json:"edits"`
}

type editTool struct {
	workspace string
}

func (t *editTool) Spec() ai.ToolSpec {
	return ai.ToolSpec{
		Name:        "edit",
		Description: "Edit one file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file.",
		Parameters: objectSchema(map[string]any{
			"path": map[string]any{"type": "string", "description": "Path to the file to edit (relative or absolute, confined to the workspace)"},
			"edits": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"oldText": map[string]any{"type": "string", "description": "Exact unique text to replace in the original file"},
						"newText": map[string]any{"type": "string", "description": "Replacement text"},
					},
					"required": []string{"oldText", "newText"},
				},
			},
		}, "path", "edits"),
	}
}

func (t *editTool) Execute(ctx context.Context, call ai.ToolCall) (agent.ToolResult, error) {
	var input editInput
	if err := decodeArguments(call, &input); err != nil {
		return agent.ToolResult{}, err
	}
	if len(input.Edits) == 0 {
		return agent.ToolResult{}, errors.New("edits must contain at least one replacement")
	}
	if err := ctx.Err(); err != nil {
		return agent.ToolResult{}, err
	}
	path, err := resolveWorkspacePath(t.workspace, input.Path, false)
	if err != nil {
		return agent.ToolResult{}, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return agent.ToolResult{}, fmt.Errorf("read %s: %w", input.Path, err)
	}
	final, err := applyEdits(raw, input.Edits)
	if err != nil {
		return agent.ToolResult{}, err
	}
	if err := os.WriteFile(path, final, 0o644); err != nil {
		return agent.ToolResult{}, fmt.Errorf("write %s: %w", input.Path, err)
	}
	return agent.ToolResult{
		CallID:  call.ID,
		Content: fmt.Sprintf("Successfully replaced %d block(s) in %s.", len(input.Edits), input.Path),
	}, nil
}

type replacementRange struct {
	start   int
	end     int
	newText string
}

func applyEdits(raw []byte, edits []editReplacement) ([]byte, error) {
	bom := []byte(nil)
	content := string(raw)
	if strings.HasPrefix(content, "\ufeff") {
		bom = []byte("\xef\xbb\xbf")
		content = strings.TrimPrefix(content, "\ufeff")
	}

	newline := "\n"
	if strings.Contains(content, "\r\n") {
		withoutCRLF := strings.ReplaceAll(content, "\r\n", "")
		if strings.ContainsAny(withoutCRLF, "\r\n") {
			return nil, errors.New("mixed line endings are not supported")
		}
		newline = "\r\n"
		content = strings.ReplaceAll(content, "\r\n", "\n")
	} else if strings.Contains(content, "\r") && strings.Contains(content, "\n") {
		return nil, errors.New("mixed line endings are not supported")
	} else if strings.Contains(content, "\r") {
		newline = "\r"
		content = strings.ReplaceAll(content, "\r", "\n")
	}

	ranges := make([]replacementRange, 0, len(edits))
	for i, edit := range edits {
		oldText := normalizeEditText(edit.OldText)
		newText := normalizeEditText(edit.NewText)
		if oldText == "" {
			return nil, fmt.Errorf("edit %d: oldText must not be empty", i+1)
		}
		count := strings.Count(content, oldText)
		if count == 0 {
			return nil, fmt.Errorf("edit %d: oldText was not found", i+1)
		}
		if count > 1 {
			return nil, fmt.Errorf("edit %d: oldText matches %d locations; it must be unique", i+1, count)
		}
		start := strings.Index(content, oldText)
		ranges = append(ranges, replacementRange{
			start:   start,
			end:     start + len(oldText),
			newText: newText,
		})
	}
	sort.Slice(ranges, func(i, j int) bool { return ranges[i].start < ranges[j].start })
	for i := 1; i < len(ranges); i++ {
		if ranges[i].start < ranges[i-1].end {
			return nil, fmt.Errorf("edits %d and %d overlap", i, i+1)
		}
	}

	var result strings.Builder
	cursor := 0
	for _, edit := range ranges {
		result.WriteString(content[cursor:edit.start])
		result.WriteString(edit.newText)
		cursor = edit.end
	}
	result.WriteString(content[cursor:])
	updated := result.String()
	if updated == content {
		return nil, errors.New("edits produced no content change")
	}
	if newline != "\n" {
		updated = strings.ReplaceAll(updated, "\n", newline)
	}
	return append(bom, []byte(updated)...), nil
}

func normalizeEditText(text string) string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	return strings.ReplaceAll(text, "\r", "\n")
}
