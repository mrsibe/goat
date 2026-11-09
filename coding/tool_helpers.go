package coding

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/mrsibe/goat/agent"
	"github.com/mrsibe/goat/ai"
)

// NewTools returns the default coding tools in pi order: read, bash, edit, write.
func NewTools(workspace string) ([]agent.Tool, error) {
	root, err := normalizeWorkspace(workspace)
	if err != nil {
		return nil, err
	}
	return []agent.Tool{
		&readTool{workspace: root},
		&bashTool{workspace: root},
		&editTool{workspace: root},
		&writeTool{workspace: root},
	}, nil
}

func decodeArguments(call ai.ToolCall, destination any) error {
	arguments := call.Arguments
	if len(strings.TrimSpace(string(arguments))) == 0 {
		arguments = json.RawMessage("{}")
	}
	if err := json.Unmarshal(arguments, destination); err != nil {
		return fmt.Errorf("invalid arguments for tool %q: %w", call.Name, err)
	}
	return nil
}

func objectSchema(properties map[string]any, required ...string) map[string]any {
	schema := map[string]any{
		"type":       "object",
		"properties": properties,
	}
	if len(required) > 0 {
		schema["required"] = required
	}
	return schema
}
