package agent

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/mrsibe/goat/ai"
)

func TestValidateToolArguments(t *testing.T) {
	spec := ai.ToolSpec{
		Name: "weather",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city": map[string]any{"type": "string", "minLength": 2},
				"days": map[string]any{
					"type":       "integer",
					"minimum":    1,
					"maximum":    7,
					"multipleOf": 1,
				},
				"units": map[string]any{
					"type": "string",
					"enum": []string{"celsius", "fahrenheit"},
				},
			},
			"required":             []string{"city"},
			"additionalProperties": false,
		},
	}

	if err := validateToolArguments(spec, json.RawMessage(`{"city":"北京","days":3,"units":"celsius"}`)); err != nil {
		t.Fatalf("valid arguments: %v", err)
	}

	for _, testCase := range []struct {
		name      string
		arguments string
		contains  string
	}{
		{name: "missing required", arguments: `{"days":3}`, contains: "required property"},
		{name: "wrong type", arguments: `{"city":3}`, contains: "expected string"},
		{name: "invalid enum", arguments: `{"city":"北京","units":"kelvin"}`, contains: "must be one of"},
		{name: "number range", arguments: `{"city":"北京","days":9}`, contains: "must be at most"},
		{name: "additional property", arguments: `{"city":"北京","extra":true}`, contains: "additional property"},
		{name: "not an object", arguments: `[]`, contains: "JSON object"},
		{name: "invalid JSON", arguments: `{"city":`, contains: "invalid JSON"},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			err := validateToolArguments(spec, json.RawMessage(testCase.arguments))
			if err == nil || !strings.Contains(err.Error(), testCase.contains) {
				t.Fatalf("error = %v, want substring %q", err, testCase.contains)
			}
		})
	}
}

func TestValidateNestedToolArguments(t *testing.T) {
	spec := ai.ToolSpec{
		Name: "edit",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"edits": map[string]any{
					"type":     "array",
					"minItems": 1,
					"items": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"oldText": map[string]any{"type": "string"},
							"newText": map[string]any{"type": "string"},
						},
						"required": []string{"oldText", "newText"},
					},
				},
			},
			"required": []string{"edits"},
		},
	}

	if err := validateToolArguments(spec, json.RawMessage(
		`{"edits":[{"oldText":"a","newText":"b"}]}`,
	)); err != nil {
		t.Fatalf("valid nested arguments: %v", err)
	}
	if err := validateToolArguments(spec, json.RawMessage(
		`{"edits":[{"oldText":"a"}]}`,
	)); err == nil || !strings.Contains(err.Error(), "arguments.edits[0].newText") {
		t.Fatalf("error = %v", err)
	}
	if err := validateToolArguments(spec, json.RawMessage(`{"edits":[]}`)); err == nil ||
		!strings.Contains(err.Error(), "at least 1 items") {
		t.Fatalf("error = %v", err)
	}
}
