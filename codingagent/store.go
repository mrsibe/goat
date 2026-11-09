package codingagent

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/mrsibe/goat/ai"
)

const (
	sessionFileVersion = 1
	timeLayout         = "20060102T150405Z"
)

type sessionHeader struct {
	Type      string `json:"type"`
	Version   int    `json:"version"`
	ID        string `json:"id"`
	Timestamp string `json:"timestamp"`
	CWD       string `json:"cwd"`
}

type sessionEntry struct {
	Type      string      `json:"type"`
	Timestamp string      `json:"timestamp"`
	Message   *ai.Message `json:"message,omitempty"`
	Usage     *ai.Usage   `json:"usage,omitempty"`
}

type sessionStore struct {
	dir string
}

func newSessionStore(baseDir, workspace string) (*sessionStore, error) {
	if baseDir == "" {
		home := os.Getenv("GODING_HOME")
		if home == "" {
			userHome, err := os.UserHomeDir()
			if err != nil {
				return nil, fmt.Errorf("resolve user home: %w", err)
			}
			home = filepath.Join(userHome, ".goding")
		}
		baseDir = filepath.Join(home, "sessions")
	}
	absolute, err := filepath.Abs(baseDir)
	if err != nil {
		return nil, fmt.Errorf("resolve session directory: %w", err)
	}
	sum := sha256.Sum256([]byte(workspace))
	workspaceHash := hex.EncodeToString(sum[:8])
	return &sessionStore{
		dir: filepath.Join(absolute, workspaceHash),
	}, nil
}

func (s *sessionStore) newID() (string, error) {
	var random [8]byte
	if _, err := rand.Read(random[:]); err != nil {
		return "", fmt.Errorf("generate session id: %w", err)
	}
	return time.Now().UTC().Format(timeLayout) + "-" + hex.EncodeToString(random[:]), nil
}

func (s *sessionStore) load(id string) (sessionHeader, []ai.Message, error) {
	path, err := s.pathForID(id)
	if err != nil {
		return sessionHeader{}, nil, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return sessionHeader{}, nil, fmt.Errorf("read session %s: %w", id, err)
	}
	header, messages, _, err := parseSessionFile(data)
	if err != nil {
		return sessionHeader{}, nil, fmt.Errorf("load session %s: %w", id, err)
	}
	return header, messages, nil
}

func (s *sessionStore) continueRecent() (sessionHeader, []ai.Message, string, bool, error) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return sessionHeader{}, nil, "", false, nil
		}
		return sessionHeader{}, nil, "", false, fmt.Errorf("read session directory: %w", err)
	}
	type candidate struct {
		path    string
		modTime time.Time
	}
	var candidates []candidate
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".jsonl") {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			return sessionHeader{}, nil, "", false, err
		}
		candidates = append(candidates, candidate{
			path:    filepath.Join(s.dir, entry.Name()),
			modTime: info.ModTime(),
		})
	}
	if len(candidates) == 0 {
		return sessionHeader{}, nil, "", false, nil
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].modTime.Equal(candidates[j].modTime) {
			return candidates[i].path > candidates[j].path
		}
		return candidates[i].modTime.After(candidates[j].modTime)
	})
	data, err := os.ReadFile(candidates[0].path)
	if err != nil {
		return sessionHeader{}, nil, "", false, err
	}
	header, messages, _, err := parseSessionFile(data)
	if err != nil {
		return sessionHeader{}, nil, "", false, fmt.Errorf("load session %s: %w", header.ID, err)
	}
	return header, messages, candidates[0].path, true, nil
}

func (s *sessionStore) pathForID(id string) (string, error) {
	if id == "" || id == "." || id == ".." || strings.ContainsAny(id, `/\`) {
		return "", fmt.Errorf("invalid session id %q", id)
	}
	return filepath.Join(s.dir, id+".jsonl"), nil
}

func parseSessionFile(data []byte) (sessionHeader, []ai.Message, []ai.Usage, error) {
	var header sessionHeader
	var messages []ai.Message
	var usages []ai.Usage
	headerSeen := false
	endedWithNewline := len(data) == 0 || data[len(data)-1] == '\n'
	lines := strings.Split(string(data), "\n")

	for index, rawLine := range lines {
		lineNumber := index + 1
		line := strings.TrimSpace(rawLine)
		if line == "" {
			continue
		}
		var kind struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal([]byte(line), &kind); err != nil {
			isFinal := index == len(lines)-1 && !endedWithNewline
			if isFinal {
				break
			}
			return sessionHeader{}, nil, nil, fmt.Errorf("line %d is not valid JSON: %w", lineNumber, err)
		}
		if !headerSeen {
			if kind.Type != "session" {
				return sessionHeader{}, nil, nil, errors.New("first entry is not a session header")
			}
			if err := json.Unmarshal([]byte(line), &header); err != nil {
				return sessionHeader{}, nil, nil, fmt.Errorf("parse session header: %w", err)
			}
			if header.Version != sessionFileVersion {
				return sessionHeader{}, nil, nil, fmt.Errorf("unsupported session version %d", header.Version)
			}
			if header.ID == "" {
				return sessionHeader{}, nil, nil, errors.New("session header is missing an id")
			}
			headerSeen = true
			continue
		}

		switch kind.Type {
		case "message":
			var entry sessionEntry
			if err := json.Unmarshal([]byte(line), &entry); err != nil {
				return sessionHeader{}, nil, nil, fmt.Errorf("parse line %d: %w", lineNumber, err)
			}
			if entry.Message == nil {
				return sessionHeader{}, nil, nil, fmt.Errorf("line %d has no message", lineNumber)
			}
			messages = append(messages, *entry.Message)
		case "usage":
			var entry sessionEntry
			if err := json.Unmarshal([]byte(line), &entry); err == nil && entry.Usage != nil {
				usages = append(usages, *entry.Usage)
			}
		default:
			return sessionHeader{}, nil, nil, fmt.Errorf("line %d has unknown entry type %q", lineNumber, kind.Type)
		}
	}
	if !headerSeen {
		return sessionHeader{}, nil, nil, errors.New("session file is empty")
	}
	return header, messages, usages, nil
}
