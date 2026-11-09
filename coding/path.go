package coding

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func resolveWorkspacePath(workspace, input string, allowMissing bool) (string, error) {
	if strings.TrimSpace(input) == "" {
		return "", errors.New("path must not be empty")
	}
	root, err := normalizeWorkspace(workspace)
	if err != nil {
		return "", err
	}

	candidate := input
	if !filepath.IsAbs(candidate) {
		candidate = filepath.Join(root, candidate)
	}
	candidate, err = filepath.Abs(candidate)
	if err != nil {
		return "", fmt.Errorf("resolve path: %w", err)
	}
	candidate = filepath.Clean(candidate)
	if !pathWithin(root, candidate) {
		return "", fmt.Errorf("path %q is outside workspace %q", input, root)
	}

	if _, err := os.Lstat(candidate); err == nil {
		resolved, err := filepath.EvalSymlinks(candidate)
		if err != nil {
			return "", fmt.Errorf("resolve path symlinks: %w", err)
		}
		if !pathWithin(root, resolved) {
			return "", fmt.Errorf("path %q resolves outside workspace %q", input, root)
		}
		return resolved, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", fmt.Errorf("inspect path: %w", err)
	}
	if !allowMissing {
		return "", fmt.Errorf("path %q does not exist", input)
	}

	ancestor := candidate
	for {
		parent := filepath.Dir(ancestor)
		if parent == ancestor {
			break
		}
		ancestor = parent
		if _, err := os.Lstat(ancestor); err == nil {
			break
		} else if !errors.Is(err, os.ErrNotExist) {
			return "", fmt.Errorf("inspect path: %w", err)
		}
	}
	resolvedAncestor, err := filepath.EvalSymlinks(ancestor)
	if err != nil {
		return "", fmt.Errorf("resolve path ancestor symlinks: %w", err)
	}
	if !pathWithin(root, resolvedAncestor) {
		return "", fmt.Errorf("path %q resolves outside workspace %q", input, root)
	}
	return candidate, nil
}

func pathWithin(root, candidate string) bool {
	relative, err := filepath.Rel(root, candidate)
	if err != nil {
		return false
	}
	return relative == "." || (relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator)) && !filepath.IsAbs(relative))
}
