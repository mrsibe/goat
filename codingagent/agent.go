// Package codingagent composes the model, generic agent, and coding tools into
// the goding coding assistant.
package codingagent

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/mrsibe/goat/agent"
	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/coding"
)

// Config controls the goding coding agent.
type Config struct {
	Model              ai.Model
	Workspace          string
	MaxSteps           int
	SessionDir         string
	DisablePersistence bool
}

// Agent composes a generic agent with coding tools and optional JSONL
// session persistence.
type Agent struct {
	core      *agent.Agent
	workspace string
	store     *sessionStore
}

// New creates a coding agent.
func New(config Config) (*Agent, error) {
	workspace, err := resolveWorkspace(config.Workspace)
	if err != nil {
		return nil, err
	}
	instructions, err := coding.LoadProjectInstructions(workspace)
	if err != nil {
		return nil, err
	}
	tools, err := coding.NewTools(workspace)
	if err != nil {
		return nil, err
	}
	core, err := agent.New(agent.Config{
		Model:        config.Model,
		Tools:        tools,
		SystemPrompt: coding.BuildSystemPrompt(workspace, instructions),
		MaxSteps:     config.MaxSteps,
	})
	if err != nil {
		return nil, err
	}
	var store *sessionStore
	if !config.DisablePersistence {
		store, err = newSessionStore(config.SessionDir, workspace)
		if err != nil {
			return nil, err
		}
	}
	return &Agent{core: core, workspace: workspace, store: store}, nil
}

// NewSession creates a conversation. The JSONL file is created lazily when the
// first message completes.
func (a *Agent) NewSession() (*Session, error) {
	return a.newSession(nil, "", nil)
}

// Continue resumes the most recently modified session for this workspace. If
// none exists it creates a new session.
func (a *Agent) Continue() (*Session, error) {
	if a.store == nil {
		return a.NewSession()
	}
	header, messages, path, ok, err := a.store.continueRecent()
	if err != nil {
		return nil, err
	}
	if !ok {
		return a.NewSession()
	}
	return a.newSession(messages, path, &header)
}

// Resume opens a specific session id for this workspace.
func (a *Agent) Resume(id string) (*Session, error) {
	if a.store == nil {
		return nil, errors.New("codingagent: session persistence is disabled")
	}
	header, messages, err := a.store.load(id)
	if err != nil {
		return nil, err
	}
	path, _ := a.store.pathForID(id)
	return a.newSession(messages, path, &header)
}

// Run executes one prompt in a fresh session.
func (a *Agent) Run(ctx context.Context, input string, onEvent func(agent.Event)) (agent.Result, error) {
	session, err := a.NewSession()
	if err != nil {
		return agent.Result{}, err
	}
	defer session.Close()
	return session.Run(ctx, input, onEvent)
}

func (a *Agent) newSession(history []ai.Message, path string, header *sessionHeader) (*Session, error) {
	var id string
	if header != nil {
		id = header.ID
	} else if a.store != nil {
		var err error
		id, err = a.store.newID()
		if err != nil {
			return nil, err
		}
		path = filepath.Join(a.store.dir, id+".jsonl")
	} else {
		var err error
		id, err = randomSessionID()
		if err != nil {
			return nil, err
		}
	}
	return &Session{
		id:        id,
		core:      a.core.NewSessionWithHistory(history),
		store:     a.store,
		workspace: a.workspace,
		path:      path,
		createdAt: time.Now().UTC(),
		existing:  header != nil,
	}, nil
}

// Session is one resumable goding conversation.
type Session struct {
	id        string
	core      *agent.Session
	store     *sessionStore
	workspace string
	path      string
	createdAt time.Time
	existing  bool

	mu         sync.Mutex
	file       *os.File
	persistErr error
	closed     bool
}

// ID returns the stable session id.
func (s *Session) ID() string {
	return s.id
}

// Messages returns a copy of the model-visible conversation including the
// current system prompt.
func (s *Session) Messages() []ai.Message {
	return s.core.Messages()
}

// Run adds one user prompt and runs the agent. Message completion events are
// appended to the session file before Run returns.
func (s *Session) Run(ctx context.Context, input string, onEvent func(agent.Event)) (agent.Result, error) {
	if s == nil || s.core == nil {
		return agent.Result{}, errors.New("codingagent: session is not initialized")
	}
	handler := func(event agent.Event) {
		if onEvent != nil {
			onEvent(event)
		}
		if event.Type == agent.EventMessageEnd && event.Message != nil && event.Message.Role != ai.RoleSystem {
			if err := s.appendMessage(*event.Message); err != nil {
				s.mu.Lock()
				if s.persistErr == nil {
					s.persistErr = err
				}
				s.mu.Unlock()
			}
		}
	}
	result, runErr := s.core.Run(ctx, input, agent.WithOnEvent(handler))
	if result.Usage != (ai.Usage{}) {
		if err := s.appendUsage(result.Usage); err != nil {
			s.mu.Lock()
			if s.persistErr == nil {
				s.persistErr = err
			}
			s.mu.Unlock()
		}
	}
	s.mu.Lock()
	persistErr := s.persistErr
	s.persistErr = nil
	s.mu.Unlock()
	if persistErr != nil {
		if runErr != nil {
			return result, errors.Join(runErr, persistErr)
		}
		return result, persistErr
	}
	return result, runErr
}

// Close flushes and closes the session file.
func (s *Session) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	if s.file != nil {
		return s.file.Close()
	}
	return nil
}

func (s *Session) appendMessage(message ai.Message) error {
	entry := sessionEntry{
		Type:      "message",
		Timestamp: time.Now().UTC().Format(time.RFC3339Nano),
		Message:   &message,
	}
	return s.appendEntry(entry)
}

func (s *Session) appendUsage(usage ai.Usage) error {
	if s.store == nil {
		return nil
	}
	entry := sessionEntry{
		Type:      "usage",
		Timestamp: time.Now().UTC().Format(time.RFC3339Nano),
		Usage:     &usage,
	}
	return s.appendEntry(entry)
}

func (s *Session) appendEntry(entry sessionEntry) error {
	if s.store == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return errors.New("codingagent: session is closed")
	}
	if s.persistErr != nil {
		return s.persistErr
	}
	if err := s.ensureOpenLocked(); err != nil {
		s.persistErr = err
		return err
	}
	data, err := json.Marshal(entry)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if _, err := s.file.Write(data); err != nil {
		s.persistErr = fmt.Errorf("append session: %w", err)
		return s.persistErr
	}
	if err := s.file.Sync(); err != nil {
		s.persistErr = fmt.Errorf("sync session: %w", err)
		return s.persistErr
	}
	return nil
}

func (s *Session) ensureOpenLocked() error {
	if s.file != nil {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o700); err != nil {
		return fmt.Errorf("create session directory: %w", err)
	}
	file, err := os.OpenFile(s.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return fmt.Errorf("open session: %w", err)
	}
	s.file = file
	if s.existing {
		return nil
	}
	header := sessionHeader{
		Type:      "session",
		Version:   sessionFileVersion,
		ID:        s.id,
		Timestamp: s.createdAt.Format(time.RFC3339Nano),
		CWD:       s.workspace,
	}
	data, err := json.Marshal(header)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if _, err := file.Write(data); err != nil {
		return fmt.Errorf("write session header: %w", err)
	}
	if err := file.Sync(); err != nil {
		return fmt.Errorf("sync session header: %w", err)
	}
	return nil
}

func resolveWorkspace(workspace string) (string, error) {
	if workspace == "" {
		workspace = "."
	}
	absolute, err := filepath.Abs(workspace)
	if err != nil {
		return "", err
	}
	resolved, err := filepath.EvalSymlinks(absolute)
	if err != nil {
		return "", fmt.Errorf("resolve workspace: %w", err)
	}
	info, err := os.Stat(resolved)
	if err != nil {
		return "", err
	}
	if !info.IsDir() {
		return "", fmt.Errorf("workspace %q is not a directory", workspace)
	}
	return resolved, nil
}

func randomSessionID() (string, error) {
	var random [8]byte
	if _, err := rand.Read(random[:]); err != nil {
		return "", err
	}
	return time.Now().UTC().Format(timeLayout) + "-" + fmt.Sprintf("%x", random), nil
}
