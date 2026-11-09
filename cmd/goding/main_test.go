package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/internal/faux"
)

func TestRunTextMode(t *testing.T) {
	t.Chdir(t.TempDir())
	model := directModel("answer")
	var stdout, stderr bytes.Buffer
	code := run(
		[]string{"--model", "test", "--no-session", "hello"},
		strings.NewReader(""),
		true,
		&stdout,
		&stderr,
		loader(model),
	)
	if code != 0 {
		t.Fatalf("code = %d, stderr = %s", code, stderr.String())
	}
	if stdout.String() != "answer\n" {
		t.Fatalf("stdout = %q", stdout.String())
	}
}

func TestRunJSONModeAndStdin(t *testing.T) {
	t.Chdir(t.TempDir())
	model := directModel("answer")
	var stdout, stderr bytes.Buffer
	code := run(
		[]string{"--model", "test", "--no-session", "--json"},
		strings.NewReader("from stdin"),
		false,
		&stdout,
		&stderr,
		loader(model),
	)
	if code != 0 {
		t.Fatalf("code = %d, stderr = %s", code, stderr.String())
	}
	lines := strings.Split(strings.TrimSpace(stdout.String()), "\n")
	if len(lines) < 3 {
		t.Fatalf("json lines = %v", lines)
	}
	var first, last map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &first); err != nil {
		t.Fatal(err)
	}
	if first["type"] != "session_start" || first["session_id"] == "" {
		t.Fatalf("first event = %+v", first)
	}
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &last); err != nil {
		t.Fatal(err)
	}
	if last["type"] != "agent_end" {
		t.Fatalf("last event = %+v", last)
	}
	if got := model.Requests[0].Messages[len(model.Requests[0].Messages)-1].Text(); got != "from stdin" {
		t.Fatalf("prompt = %q", got)
	}
}

func TestRunSessionContinueAndResume(t *testing.T) {
	t.Chdir(t.TempDir())
	sessionDir := t.TempDir()
	var stdout, stderr bytes.Buffer
	code := run(
		[]string{"--model", "test", "--session-dir", sessionDir, "--json", "first"},
		strings.NewReader(""),
		true,
		&stdout,
		&stderr,
		loader(directModel("one")),
	)
	if code != 0 {
		t.Fatalf("code = %d, stderr = %s", code, stderr.String())
	}
	sessionID := firstSessionID(t, stdout.String())

	stdout.Reset()
	stderr.Reset()
	model := directModel("two")
	code = run(
		[]string{"--model", "test", "--session-dir", sessionDir, "--resume", sessionID, "second"},
		strings.NewReader(""),
		true,
		&stdout,
		&stderr,
		loader(model),
	)
	if code != 0 {
		t.Fatalf("resume code = %d, stderr = %s", code, stderr.String())
	}
	if got := len(model.Requests[0].Messages); got != 4 {
		t.Fatalf("resumed request messages = %d, want 4", got)
	}

	stdout.Reset()
	stderr.Reset()
	code = run(
		[]string{"--model", "test", "--session-dir", sessionDir, "--continue", "third"},
		strings.NewReader(""),
		true,
		&stdout,
		&stderr,
		loader(directModel("three")),
	)
	if code != 0 {
		t.Fatalf("continue code = %d, stderr = %s", code, stderr.String())
	}
}

func TestRunExitCodesAndValidation(t *testing.T) {
	t.Chdir(t.TempDir())
	t.Setenv("OPENAI_MODEL", "")
	var stdout, stderr bytes.Buffer
	code := run(nil, strings.NewReader(""), true, &stdout, &stderr, loader(directModel("x")))
	if code != 2 || !strings.Contains(stderr.String(), "model is required") {
		t.Fatalf("missing model: code=%d stderr=%q", code, stderr.String())
	}

	stderr.Reset()
	code = run(
		[]string{"--model", "test", "--continue", "--resume", "id", "prompt"},
		strings.NewReader(""),
		true,
		&stdout,
		&stderr,
		loader(directModel("x")),
	)
	if code != 2 || !strings.Contains(stderr.String(), "mutually exclusive") {
		t.Fatalf("conflict: code=%d stderr=%q", code, stderr.String())
	}

	stderr.Reset()
	code = run(
		[]string{"--model", "test", "--no-session", "prompt"},
		strings.NewReader(""),
		true,
		&stdout,
		&stderr,
		func(string, string, string) (ai.Model, error) {
			return nil, errors.New("provider down")
		},
	)
	if code != 2 || !strings.Contains(stderr.String(), "provider down") {
		t.Fatalf("loader error: code=%d stderr=%q", code, stderr.String())
	}

	stderr.Reset()
	code = run(
		[]string{"--model", "test", "--no-session", "prompt"},
		strings.NewReader(""),
		true,
		&stdout,
		&stderr,
		loader(faux.New(faux.Turn{StartErr: errors.New("stream failed")})),
	)
	if code != 1 || !strings.Contains(stderr.String(), "stream failed") {
		t.Fatalf("run error: code=%d stderr=%q", code, stderr.String())
	}

	stdout.Reset()
	stderr.Reset()
	code = run([]string{"--version"}, strings.NewReader(""), true, &stdout, &stderr, nil)
	if code != 0 || !strings.Contains(stdout.String(), "goding ") {
		t.Fatalf("version: code=%d stdout=%q", code, stdout.String())
	}
}

func TestRunUsesEnvironmentAndSessionDirectory(t *testing.T) {
	workspace := t.TempDir()
	t.Chdir(workspace)
	sessionDir := t.TempDir()
	t.Setenv("OPENAI_MODEL", "env-model")
	t.Setenv("OPENAI_BASE_URL", "https://example.test")
	t.Setenv("OPENAI_API_KEY", "key")
	var stdout, stderr bytes.Buffer
	code := run(
		[]string{"--session-dir", sessionDir, "hello"},
		strings.NewReader(""),
		true,
		&stdout,
		&stderr,
		loader(directModel("answer")),
	)
	if code != 0 {
		t.Fatalf("code = %d, stderr = %s", code, stderr.String())
	}
	matches, err := filepath.Glob(filepath.Join(sessionDir, "*", "*.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) != 1 {
		t.Fatalf("session files = %v", matches)
	}
}

func loader(model ai.Model) modelLoader {
	return func(string, string, string) (ai.Model, error) {
		return model, nil
	}
}

func directModel(text string) *faux.Model {
	return faux.New(faux.Turn{Events: []ai.Event{
		{Type: ai.EventStart},
		{Type: ai.EventTextDelta, TextDelta: text},
		{Type: ai.EventDone, StopReason: ai.StopReasonStop},
	}})
}

func firstSessionID(t *testing.T, output string) string {
	t.Helper()
	decoder := json.NewDecoder(strings.NewReader(output))
	for {
		var event map[string]any
		if err := decoder.Decode(&event); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			t.Fatal(err)
		}
		if event["type"] == "session_start" {
			id, _ := event["session_id"].(string)
			if id != "" {
				return id
			}
		}
	}
	t.Fatal("session_start event not found")
	return ""
}
