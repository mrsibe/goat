package coding

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/mrsibe/goat/agent"
	"github.com/mrsibe/goat/ai"
)

type bashInput struct {
	Command string  `json:"command"`
	Timeout float64 `json:"timeout,omitempty"`
}

type bashTool struct {
	workspace string
}

func (t *bashTool) Spec() ai.ToolSpec {
	return ai.ToolSpec{
		Name:        "bash",
		Description: "Execute a bash command in the workspace. Returns combined stdout and stderr, truncated to the last 2000 lines or 50KB. If truncated, the full output is saved to a temp file.",
		Parameters: objectSchema(map[string]any{
			"command": map[string]any{"type": "string", "description": "Shell command to execute"},
			"timeout": map[string]any{"type": "number", "exclusiveMinimum": 0, "description": "Timeout in seconds (optional, no default timeout)"},
		}, "command"),
	}
}

func (t *bashTool) Execute(ctx context.Context, call ai.ToolCall) (agent.ToolResult, error) {
	var input bashInput
	if err := decodeArguments(call, &input); err != nil {
		return agent.ToolResult{}, err
	}
	if strings.TrimSpace(input.Command) == "" {
		return agent.ToolResult{}, errors.New("command must not be empty")
	}
	if input.Timeout < 0 {
		return agent.ToolResult{}, errors.New("timeout must be greater than zero")
	}

	runCtx := ctx
	cancel := func() {}
	timedOut := false
	if input.Timeout > 0 {
		duration := time.Duration(input.Timeout * float64(time.Second))
		runCtx, cancel = context.WithTimeout(ctx, duration)
	}
	defer cancel()

	spool, err := os.CreateTemp("", "goding-bash-*.log")
	if err != nil {
		return agent.ToolResult{}, fmt.Errorf("create output spool: %w", err)
	}
	spoolPath := spool.Name()
	keepSpool := false
	defer func() {
		spool.Close()
		if !keepSpool {
			os.Remove(spoolPath)
		}
	}()

	tail := &tailBuffer{}
	output := io.MultiWriter(spool, tail)
	cmd := exec.CommandContext(runCtx, "bash", "-lc", input.Command)
	cmd.Dir = t.workspace
	cmd.Stdout = output
	cmd.Stderr = output
	configureCommand(cmd)
	cmd.Cancel = func() error {
		return killCommand(cmd)
	}

	runErr := cmd.Run()
	if runCtx.Err() == context.DeadlineExceeded {
		timedOut = true
	}
	if err := ctx.Err(); err != nil {
		return agent.ToolResult{}, err
	}

	truncated := tail.truncate()
	content := truncated.content
	if truncated.truncated {
		keepSpool = true
		if content != "" {
			content += "\n\n"
		}
		content += fmt.Sprintf("[Showing last %d lines of %d. Full output: %s]",
			truncated.outputLines, truncated.totalLines, spoolPath)
	}

	if timedOut {
		if content != "" {
			content += "\n\n"
		}
		content += fmt.Sprintf("Command timed out after %g seconds", input.Timeout)
		return agent.ToolResult{CallID: call.ID, Content: content, IsError: true}, nil
	}
	if runErr != nil {
		exitCode := 1
		var exitErr *exec.ExitError
		if errors.As(runErr, &exitErr) {
			exitCode = exitErr.ExitCode()
		}
		if content != "" {
			content += "\n\n"
		}
		content += fmt.Sprintf("Exit code: %d", exitCode)
		return agent.ToolResult{CallID: call.ID, Content: content, IsError: true}, nil
	}
	return agent.ToolResult{CallID: call.ID, Content: content}, nil
}
