package coding_test

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/mrsibe/goat/agent"
	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/coding"
)

func TestReadOffsetLimitTruncationAndErrors(t *testing.T) {
	workspace := t.TempDir()
	var content strings.Builder
	for i := 1; i <= 2500; i++ {
		fmt.Fprintf(&content, "line %d\n", i)
	}
	writeFile(t, filepath.Join(workspace, "big.txt"), content.String())
	read := toolByName(t, workspace, "read")

	result := execute(t, read, map[string]any{"path": "big.txt"})
	if !strings.Contains(result.Content, "line 1\n") || !strings.Contains(result.Content, "line 2000") {
		t.Fatalf("read output missing expected lines: %.200q", result.Content)
	}
	if !strings.Contains(result.Content, "Use offset=2001 to continue") {
		t.Fatalf("read output missing continuation: %.200q", result.Content)
	}

	result = execute(t, read, map[string]any{"path": "big.txt", "offset": 2500, "limit": 1})
	if result.Content != "line 2500" {
		t.Fatalf("offset output = %q", result.Content)
	}

	result = execute(t, read, map[string]any{"path": "big.txt", "offset": 1, "limit": 2})
	if !strings.Contains(result.Content, "line 1\nline 2") || !strings.Contains(result.Content, "2498 more lines") {
		t.Fatalf("limit output = %q", result.Content)
	}

	if _, err := read.Execute(context.Background(), call("read", map[string]any{"path": "big.txt", "offset": 9999})); err == nil {
		t.Fatal("expected offset beyond EOF error")
	}
	if _, err := read.Execute(context.Background(), call("read", map[string]any{"path": "../outside.txt"})); err == nil {
		t.Fatal("expected traversal error")
	}
}

func TestWriteEditAndPathConfinement(t *testing.T) {
	workspace := t.TempDir()
	write := toolByName(t, workspace, "write")
	edit := toolByName(t, workspace, "edit")

	result := execute(t, write, map[string]any{"path": "nested/file.txt", "content": "alpha\nbeta\ngamma\n"})
	if !strings.Contains(result.Content, "Successfully wrote") {
		t.Fatalf("write result = %q", result.Content)
	}
	if got := readFile(t, filepath.Join(workspace, "nested/file.txt")); got != "alpha\nbeta\ngamma\n" {
		t.Fatalf("file = %q", got)
	}

	result = execute(t, edit, map[string]any{
		"path": "nested/file.txt",
		"edits": []map[string]string{
			{"oldText": "alpha", "newText": "A"},
			{"oldText": "gamma", "newText": "G"},
		},
	})
	if !strings.Contains(result.Content, "Successfully replaced 2") {
		t.Fatalf("edit result = %q", result.Content)
	}
	if got := readFile(t, filepath.Join(workspace, "nested/file.txt")); got != "A\nbeta\nG\n" {
		t.Fatalf("edited file = %q", got)
	}

	if _, err := edit.Execute(context.Background(), call("edit", map[string]any{
		"path":  "nested/file.txt",
		"edits": []map[string]string{{"oldText": "missing", "newText": "x"}},
	})); err == nil {
		t.Fatal("expected missing oldText error")
	}
	writeFile(t, filepath.Join(workspace, "duplicate.txt"), "same\nsame\n")
	if _, err := edit.Execute(context.Background(), call("edit", map[string]any{
		"path":  "duplicate.txt",
		"edits": []map[string]string{{"oldText": "same", "newText": "x"}},
	})); err == nil || !strings.Contains(err.Error(), "matches 2") {
		t.Fatalf("duplicate edit error = %v", err)
	}
	writeFile(t, filepath.Join(workspace, "overlap.txt"), "abcdef")
	if _, err := edit.Execute(context.Background(), call("edit", map[string]any{
		"path": "overlap.txt",
		"edits": []map[string]string{
			{"oldText": "abc", "newText": "x"},
			{"oldText": "bcd", "newText": "y"},
		},
	})); err == nil || !strings.Contains(err.Error(), "matches 2") {
		// bcd is not unique after abc? It is unique, so this exercises overlap detection.
		if err == nil || !strings.Contains(err.Error(), "overlap") {
			t.Fatalf("overlap edit error = %v", err)
		}
	}
	if _, err := edit.Execute(context.Background(), call("edit", map[string]any{
		"path":  "nested/file.txt",
		"edits": []map[string]string{{"oldText": "A", "newText": "A"}},
	})); err == nil || !strings.Contains(err.Error(), "no content change") {
		t.Fatalf("no-change edit error = %v", err)
	}

	outside := filepath.Join(t.TempDir(), "outside.txt")
	writeFile(t, outside, "secret")
	if _, err := write.Execute(context.Background(), call("write", map[string]any{"path": "../outside.txt", "content": "x"})); err == nil {
		t.Fatal("expected outside workspace write error")
	}
	link := filepath.Join(workspace, "outside-link")
	if err := os.Symlink(outside, link); err != nil {
		t.Fatal(err)
	}
	if _, err := readTool(t, workspace).Execute(context.Background(), call("read", map[string]any{"path": "outside-link"})); err == nil {
		t.Fatal("expected symlink escape error")
	}
	if _, err := write.Execute(context.Background(), call("write", map[string]any{"path": "outside-link", "content": "x"})); err == nil {
		t.Fatal("expected symlink escape write error")
	}
}

func TestEditPreservesCRLFAndRejectsMixedEndings(t *testing.T) {
	workspace := t.TempDir()
	edit := toolByName(t, workspace, "edit")
	path := filepath.Join(workspace, "crlf.txt")
	writeFile(t, path, "alpha\r\nbeta\r\n")
	execute(t, edit, map[string]any{
		"path":  "crlf.txt",
		"edits": []map[string]string{{"oldText": "beta", "newText": "gamma\nline"}},
	})
	if got := readFile(t, path); got != "alpha\r\ngamma\r\nline\r\n" {
		t.Fatalf("CRLF result = %q", got)
	}

	writeFile(t, path, "alpha\r\nbeta\n")
	if _, err := edit.Execute(context.Background(), call("edit", map[string]any{
		"path":  "crlf.txt",
		"edits": []map[string]string{{"oldText": "beta", "newText": "gamma"}},
	})); err == nil || !strings.Contains(err.Error(), "mixed line endings") {
		t.Fatalf("mixed newline error = %v", err)
	}
}

func TestBashOutputExitTimeoutAndTruncation(t *testing.T) {
	workspace := t.TempDir()
	bash := toolByName(t, workspace, "bash")
	result := execute(t, bash, map[string]any{
		"command": `printf 'out\n'; printf 'err\n' >&2; exit 3`,
	})
	if !result.IsError || !strings.Contains(result.Content, "out") ||
		!strings.Contains(result.Content, "err") || !strings.Contains(result.Content, "Exit code: 3") {
		t.Fatalf("bash result = %+v", result)
	}

	result = execute(t, bash, map[string]any{"command": "sleep 2", "timeout": 0.1})
	if !result.IsError || !strings.Contains(result.Content, "timed out") {
		t.Fatalf("timeout result = %+v", result)
	}

	result = execute(t, bash, map[string]any{"command": "seq 1 2500"})
	if result.IsError || !strings.Contains(result.Content, "2500") ||
		!strings.Contains(result.Content, "Full output:") {
		t.Fatalf("truncated bash result = %.400q", result.Content)
	}
	match := regexp.MustCompile(`Full output: ([^\]]+)`).FindStringSubmatch(result.Content)
	if len(match) != 2 {
		t.Fatalf("missing full output path in %q", result.Content)
	}
	t.Cleanup(func() { _ = os.Remove(strings.TrimSpace(match[1])) })

	result = execute(t, bash, map[string]any{"command": "pwd"})
	if strings.TrimSpace(result.Content) != workspace {
		t.Fatalf("pwd = %q, want %q", result.Content, workspace)
	}
}

func TestProjectInstructionsAndPrompt(t *testing.T) {
	workspace := t.TempDir()
	writeFile(t, filepath.Join(workspace, "AGENTS.md"), "root rules")
	if err := os.MkdirAll(filepath.Join(workspace, "nested"), 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(workspace, "nested", "AGENTS.md"), "nested rules")
	instructions, err := coding.LoadProjectInstructions(workspace)
	if err != nil {
		t.Fatal(err)
	}
	if instructions != "root rules" {
		t.Fatalf("instructions = %q", instructions)
	}
	prompt := coding.BuildSystemPrompt(workspace, instructions)
	for _, expected := range []string{"<project_context>", "root rules", "<cwd>", filepath.ToSlash(workspace)} {
		if !strings.Contains(prompt, expected) {
			t.Fatalf("prompt missing %q:\n%s", expected, prompt)
		}
	}
	if strings.Contains(prompt, "nested rules") {
		t.Fatal("prompt unexpectedly loaded nested AGENTS.md")
	}
}

func toolByName(t *testing.T, workspace, name string) agent.Tool {
	t.Helper()
	tools, err := coding.NewTools(workspace)
	if err != nil {
		t.Fatal(err)
	}
	for _, tool := range tools {
		if tool.Spec().Name == name {
			return tool
		}
	}
	t.Fatalf("tool %q not found", name)
	return nil
}

func readTool(t *testing.T, workspace string) agent.Tool {
	return toolByName(t, workspace, "read")
}

func execute(t *testing.T, tool agent.Tool, arguments map[string]any) agent.ToolResult {
	t.Helper()
	result, err := tool.Execute(context.Background(), call(tool.Spec().Name, arguments))
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func call(name string, arguments map[string]any) ai.ToolCall {
	data, err := json.Marshal(arguments)
	if err != nil {
		panic(err)
	}
	return ai.ToolCall{ID: "call_1", Name: name, Arguments: data}
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func readFile(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}
