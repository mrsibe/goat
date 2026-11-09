// Package coding provides reusable coding-agent prompts and tools.
package coding

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

const projectInstructionsFile = "AGENTS.md"

// LoadProjectInstructions reads the workspace-root AGENTS.md file. A missing
// file is not an error. Nested AGENTS.md files are intentionally ignored in v1.
func LoadProjectInstructions(workspace string) (string, error) {
	root, err := normalizeWorkspace(workspace)
	if err != nil {
		return "", err
	}
	path := filepath.Join(root, projectInstructionsFile)
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", nil
		}
		return "", fmt.Errorf("coding: read %s: %w", path, err)
	}
	return string(data), nil
}

func normalizeWorkspace(workspace string) (string, error) {
	if workspace == "" {
		return "", errors.New("coding: workspace must not be empty")
	}
	absolute, err := filepath.Abs(workspace)
	if err != nil {
		return "", fmt.Errorf("coding: resolve workspace: %w", err)
	}
	resolved, err := filepath.EvalSymlinks(absolute)
	if err != nil {
		return "", fmt.Errorf("coding: resolve workspace symlinks: %w", err)
	}
	info, err := os.Stat(resolved)
	if err != nil {
		return "", fmt.Errorf("coding: stat workspace: %w", err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("coding: workspace %q is not a directory", workspace)
	}
	return resolved, nil
}
