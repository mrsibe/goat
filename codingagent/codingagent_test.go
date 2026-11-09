package codingagent_test

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/codingagent"
	"github.com/mrsibe/goat/internal/faux"
)

func TestSessionPersistsContinuesResumesAndLoadsAgentsMD(t *testing.T) {
	workspace := t.TempDir()
	sessionDir := t.TempDir()
	writeFile(t, filepath.Join(workspace, "AGENTS.md"), "always run tests")

	model := directModel("hello")
	assistant := newAgent(t, workspace, sessionDir, model, false)
	session, err := assistant.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	if matches, _ := filepath.Glob(filepath.Join(sessionDir, "*", "*.jsonl")); len(matches) != 0 {
		t.Fatalf("session file created before a message: %v", matches)
	}
	result, err := session.Run(context.Background(), "hi", nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Answer != "hello" {
		t.Fatalf("answer = %q", result.Answer)
	}
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	if len(model.Requests) != 1 {
		t.Fatalf("requests = %d", len(model.Requests))
	}
	systemText := model.Requests[0].Messages[0].Text()
	if !strings.Contains(systemText, "always run tests") {
		t.Fatalf("system prompt missing AGENTS.md:\n%s", systemText)
	}

	files := sessionFiles(t, sessionDir)
	if len(files) != 1 {
		t.Fatalf("session files = %v", files)
	}
	data, err := os.ReadFile(files[0])
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) < 3 {
		t.Fatalf("session lines = %q", lines)
	}
	var header map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &header); err != nil {
		t.Fatal(err)
	}
	if header["type"] != "session" || header["version"] != float64(1) {
		t.Fatalf("header = %+v", header)
	}

	continued, err := newAgent(t, workspace, sessionDir, directModel("next"), false).Continue()
	if err != nil {
		t.Fatal(err)
	}
	roles := messageRoles(continued.Messages())
	wantRoles := []ai.Role{ai.RoleSystem, ai.RoleUser, ai.RoleAssistant}
	if !equalRoles(roles, wantRoles) {
		t.Fatalf("continued roles = %v, want %v", roles, wantRoles)
	}
	if continued.ID() != session.ID() {
		t.Fatalf("continued id = %q, want %q", continued.ID(), session.ID())
	}
	_ = continued.Close()

	resumed, err := newAgent(t, workspace, sessionDir, directModel("third"), false).Resume(session.ID())
	if err != nil {
		t.Fatal(err)
	}
	if !equalRoles(messageRoles(resumed.Messages()), wantRoles) {
		t.Fatalf("resumed messages = %v", messageRoles(resumed.Messages()))
	}
	_ = resumed.Close()

	if _, err := newAgent(t, workspace, sessionDir, directModel("x"), false).Resume("../escape"); err == nil {
		t.Fatal("expected invalid session id error")
	}
}

func TestContinuePartialAndCorruptFiles(t *testing.T) {
	workspace := t.TempDir()
	sessionDir := t.TempDir()
	assistant := newAgent(t, workspace, sessionDir, directModel("hello"), false)
	session, err := assistant.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := session.Run(context.Background(), "hi", nil); err != nil {
		t.Fatal(err)
	}
	_ = session.Close()
	path := sessionFiles(t, sessionDir)[0]

	file, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.WriteString(`{"type":"message"`); err != nil {
		t.Fatal(err)
	}
	_ = file.Close()

	continued, err := newAgent(t, workspace, sessionDir, directModel("next"), false).Continue()
	if err != nil {
		t.Fatalf("partial final line should be ignored: %v", err)
	}
	if got := len(continued.Messages()); got != 3 {
		t.Fatalf("messages = %d, want 3", got)
	}
	_ = continued.Close()

	file, err = os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.WriteString("\nnot-json\n"); err != nil {
		t.Fatal(err)
	}
	_ = file.Close()
	if _, err := newAgent(t, workspace, sessionDir, directModel("next"), false).Continue(); err == nil {
		t.Fatal("expected interior corruption error")
	}
}

func TestWorkspaceIsolationAndDisabledPersistence(t *testing.T) {
	firstWorkspace := t.TempDir()
	secondWorkspace := t.TempDir()
	sessionDir := t.TempDir()
	first := newAgent(t, firstWorkspace, sessionDir, directModel("first"), false)
	session, err := first.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := session.Run(context.Background(), "hello", nil); err != nil {
		t.Fatal(err)
	}
	_ = session.Close()

	second := newAgent(t, secondWorkspace, sessionDir, directModel("second"), false)
	continued, err := second.Continue()
	if err != nil {
		t.Fatal(err)
	}
	if got := messageRoles(continued.Messages()); !equalRoles(got, []ai.Role{ai.RoleSystem}) {
		t.Fatalf("second workspace unexpectedly saw history: %v", got)
	}
	_ = continued.Close()

	noPersistDir := t.TempDir()
	noPersist := newAgent(t, firstWorkspace, noPersistDir, directModel("ephemeral"), true)
	ephemeral, err := noPersist.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ephemeral.Run(context.Background(), "hello", nil); err != nil {
		t.Fatal(err)
	}
	_ = ephemeral.Close()
	if matches, _ := filepath.Glob(filepath.Join(noPersistDir, "*", "*.jsonl")); len(matches) != 0 {
		t.Fatalf("persistence disabled but files exist: %v", matches)
	}
}

func newAgent(t *testing.T, workspace, sessionDir string, model ai.Model, disabled bool) *codingagent.Agent {
	t.Helper()
	assistant, err := codingagent.New(codingagent.Config{
		Model:              model,
		Workspace:          workspace,
		SessionDir:         sessionDir,
		DisablePersistence: disabled,
	})
	if err != nil {
		t.Fatal(err)
	}
	return assistant
}

func directModel(text string) *faux.Model {
	return faux.New(faux.Turn{Events: []ai.Event{
		{Type: ai.EventStart},
		{Type: ai.EventTextDelta, TextDelta: text},
		{Type: ai.EventUsage, Usage: &ai.Usage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2}},
		{Type: ai.EventDone, StopReason: ai.StopReasonStop},
	}})
}

func sessionFiles(t *testing.T, sessionDir string) []string {
	t.Helper()
	var files []string
	err := filepath.WalkDir(sessionDir, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !entry.IsDir() && strings.HasSuffix(path, ".jsonl") {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return files
}

func messageRoles(messages []ai.Message) []ai.Role {
	roles := make([]ai.Role, len(messages))
	for i, message := range messages {
		roles[i] = message.Role
	}
	return roles
}

func equalRoles(left, right []ai.Role) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
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
