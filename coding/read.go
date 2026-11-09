package coding

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/mrsibe/goat/agent"
	"github.com/mrsibe/goat/ai"
)

type readInput struct {
	Path   string `json:"path"`
	Offset int    `json:"offset,omitempty"`
	Limit  int    `json:"limit,omitempty"`
}

type readTool struct {
	workspace string
}

func (t *readTool) Spec() ai.ToolSpec {
	return ai.ToolSpec{
		Name:        "read",
		Description: "Read the contents of a text file. Output is truncated to 2000 lines or 50KB, whichever is hit first. Use offset/limit for large files.",
		Parameters: objectSchema(map[string]any{
			"path":   map[string]any{"type": "string", "description": "Path to the file to read (relative or absolute, confined to the workspace)"},
			"offset": map[string]any{"type": "integer", "minimum": 1, "description": "Line number to start reading from (1-indexed)"},
			"limit":  map[string]any{"type": "integer", "minimum": 1, "description": "Maximum number of lines to read"},
		}, "path"),
	}
}

func (t *readTool) Execute(ctx context.Context, call ai.ToolCall) (agent.ToolResult, error) {
	var input readInput
	if err := decodeArguments(call, &input); err != nil {
		return agent.ToolResult{}, err
	}
	if input.Offset == 0 {
		input.Offset = 1
	}
	if input.Offset < 1 {
		return agent.ToolResult{}, errors.New("offset must be at least 1")
	}
	if input.Limit < 0 {
		return agent.ToolResult{}, errors.New("limit must not be negative")
	}
	path, err := resolveWorkspacePath(t.workspace, input.Path, false)
	if err != nil {
		return agent.ToolResult{}, err
	}
	file, err := os.Open(path)
	if err != nil {
		return agent.ToolResult{}, fmt.Errorf("read %s: %w", input.Path, err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return agent.ToolResult{}, err
	}
	if info.IsDir() {
		return agent.ToolResult{}, fmt.Errorf("%s is a directory", input.Path)
	}

	output, err := selectReadLines(ctx, file, input.Offset, input.Limit, input.Path)
	if err != nil {
		return agent.ToolResult{}, err
	}
	return agent.ToolResult{CallID: call.ID, Content: output}, nil
}

func selectReadLines(ctx context.Context, reader io.Reader, offset, limit int, displayPath string) (string, error) {
	buffered := bufio.NewReader(reader)
	lineNumber := 1
	totalLines := 0
	selectedLines := 0
	output := make([]string, 0, min(limitOrMax(limit), defaultMaxLines))
	outputBytes := 0
	firstLineExceeds := false
	truncatedByBytes := false
	var firstSelected string

	for {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		line, readErr := buffered.ReadString('\n')
		if len(line) == 0 && errors.Is(readErr, io.EOF) {
			break
		}
		if strings.IndexByte(line, 0) >= 0 {
			return "", errors.New("file appears to be binary or contains NUL bytes")
		}
		line = strings.TrimSuffix(line, "\n")
		line = strings.TrimSuffix(line, "\r")
		totalLines++

		if lineNumber >= offset && (limit == 0 || selectedLines < limit) {
			if selectedLines == 0 {
				firstSelected = line
				if len(line) > defaultMaxBytes {
					firstLineExceeds = true
				}
			}
			selectedLines++
			if !firstLineExceeds && !truncatedByBytes && len(output) < defaultMaxLines {
				lineBytes := len(line)
				if len(output) > 0 {
					lineBytes++
				}
				if outputBytes+lineBytes > defaultMaxBytes {
					truncatedByBytes = true
				} else {
					output = append(output, line)
					outputBytes += lineBytes
				}
			}
		}
		lineNumber++
		if readErr != nil {
			if !errors.Is(readErr, io.EOF) {
				return "", readErr
			}
			break
		}
	}

	if totalLines == 0 {
		if offset == 1 {
			return "", nil
		}
		return "", fmt.Errorf("offset %d is beyond end of file (0 lines total)", offset)
	}
	if offset > totalLines {
		return "", fmt.Errorf("offset %d is beyond end of file (%d lines total)", offset, totalLines)
	}

	startDisplay := offset
	if firstLineExceeds {
		return fmt.Sprintf("[Line %d is %s, exceeds %s limit. Use bash: sed -n '%dp' %s | head -c %d]",
			startDisplay, formatSize(len(firstSelected)), formatSize(defaultMaxBytes), startDisplay, displayPath, defaultMaxBytes), nil
	}
	content := strings.Join(output, "\n")
	if truncatedByBytes || selectedLines > len(output) {
		endDisplay := startDisplay + len(output) - 1
		nextOffset := endDisplay + 1
		if truncatedByBytes {
			return content + fmt.Sprintf("\n\n[Showing lines %d-%d of %d (%s limit). Use offset=%d to continue.]",
				startDisplay, endDisplay, totalLines, formatSize(defaultMaxBytes), nextOffset), nil
		}
		return content + fmt.Sprintf("\n\n[Showing lines %d-%d of %d. Use offset=%d to continue.]",
			startDisplay, endDisplay, totalLines, nextOffset), nil
	}
	remaining := totalLines - (offset + selectedLines - 1)
	if limit > 0 && remaining > 0 {
		return content + fmt.Sprintf("\n\n[%d more lines in file. Use offset=%d to continue.]",
			remaining, offset+selectedLines), nil
	}
	return content, nil
}

func limitOrMax(limit int) int {
	if limit <= 0 {
		return defaultMaxLines
	}
	return limit
}
