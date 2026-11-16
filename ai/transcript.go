package ai

import (
	"encoding/json"
	"sort"
	"strings"
)

// Context is request input accepted by the public entry points. SystemPrompt
// and Tools are shorthand for a leading system message; NormalizeContext folds
// them into one before a provider sees the request.
type Context struct {
	SystemPrompt string
	Messages     []Message
	Tools        []ToolSpec
}

// TranscriptContext is a normalized request context: the prompt and tool
// declarations live in the leading system message, never beside the messages.
type TranscriptContext struct {
	Messages []Message
}

// NormalizeContext folds SystemPrompt and Tools into a leading system message.
func NormalizeContext(input Context) TranscriptContext {
	messages := CloneMessages(input.Messages)
	if len(messages) > 0 {
		if _, ok := messages[0].(*SystemMessage); ok && input.SystemPrompt == "" && input.Tools == nil {
			return TranscriptContext{Messages: messages}
		}
	}
	if input.SystemPrompt == "" && input.Tools == nil {
		return TranscriptContext{Messages: messages}
	}
	system := &SystemMessage{Content: input.SystemPrompt, ToolsAdded: cloneToolSpecs(input.Tools)}
	messages = append([]Message{system}, messages...)
	return TranscriptContext{Messages: messages}
}

// GetInitialSystemMessage returns the leading system message, if any.
func GetInitialSystemMessage(messages []Message) *SystemMessage {
	if len(messages) == 0 {
		return nil
	}
	system, _ := messages[0].(*SystemMessage)
	return system
}

// GetCurrentSystemMessage replays every system message in order and returns the
// effective prompt: later content is appended, sections replace or remove named
// sections, and tool declarations accumulate.
func GetCurrentSystemMessage(messages []Message) *SystemMessage {
	current := &SystemMessage{}
	seen := false
	for _, message := range messages {
		system, ok := message.(*SystemMessage)
		if !ok {
			continue
		}
		if !seen {
			current.ToolsRemoved = nil
			seen = true
		}
		if strings.TrimSpace(system.Content) != "" {
			if strings.TrimSpace(current.Content) == "" {
				current.Content = system.Content
			} else {
				current.Content += "\n\n" + system.Content
			}
		}
		if len(system.Sections) > 0 {
			if current.Sections == nil {
				current.Sections = make(map[string]*string, len(system.Sections))
			}
			for name, value := range system.Sections {
				if value == nil {
					if _, exists := current.Sections[name]; exists {
						delete(current.Sections, name)
						current.SectionOrder = removeSectionName(current.SectionOrder, name)
					}
					continue
				}
				copied := *value
				current.SetSection(name, &copied)
			}
		}
		if len(system.ToolsAdded) > 0 {
			current.ToolsAdded = mergeToolSpecs(current.ToolsAdded, system.ToolsAdded)
		}
		if len(system.ToolsRemoved) > 0 {
			current.ToolsAdded = removeToolSpecs(current.ToolsAdded, system.ToolsRemoved)
		}
	}
	if !seen {
		return nil
	}
	return current
}

// GetCurrentSystemPrompt renders the replayed system message: content followed
// by named sections. Sections are rendered in name order so the result is
// deterministic.
func GetCurrentSystemPrompt(messages []Message) string {
	system := GetCurrentSystemMessage(messages)
	if system == nil {
		return ""
	}
	return RenderSystemMessage(*system)
}

// RenderSystemMessage renders a system message's content and sections.
func RenderSystemMessage(system SystemMessage) string {
	var text strings.Builder
	text.WriteString(system.Content)
	seen := make(map[string]bool, len(system.Sections))
	appendSection := func(name string) {
		if seen[name] {
			return
		}
		value := system.Sections[name]
		if value == nil {
			return
		}
		seen[name] = true
		if text.Len() > 0 {
			text.WriteString("\n\n")
		}
		text.WriteString(*value)
	}
	for _, name := range system.SectionOrder {
		appendSection(name)
	}
	if len(seen) < len(system.Sections) {
		remaining := make([]string, 0, len(system.Sections)-len(seen))
		for name := range system.Sections {
			if !seen[name] {
				remaining = append(remaining, name)
			}
		}
		sort.Strings(remaining)
		for _, name := range remaining {
			appendSection(name)
		}
	}
	return text.String()
}

// GetCurrentTools returns the tool set declared by the transcript.
func GetCurrentTools(messages []Message) []ToolSpec {
	var tools []ToolSpec
	for _, message := range messages {
		system, ok := message.(*SystemMessage)
		if !ok {
			continue
		}
		tools = mergeToolSpecs(tools, system.ToolsAdded)
		tools = removeToolSpecs(tools, system.ToolsRemoved)
	}
	if tools == nil {
		return nil
	}
	return cloneToolSpecs(tools)
}

// GetDeclaredTools is an alias for GetCurrentTools.
func GetDeclaredTools(messages []Message) []ToolSpec {
	return GetCurrentTools(messages)
}

// ToolStateChanges is the delta between two tool sets.
type ToolStateChanges struct {
	ToolsAdded   []ToolSpec
	ToolsRemoved []ToolReference
}

// GetToolStateChanges computes the declarations needed to turn previous into
// current.
func GetToolStateChanges(previous, current []ToolSpec) ToolStateChanges {
	previousByName := make(map[string]ToolSpec, len(previous))
	for _, spec := range previous {
		previousByName[spec.Name] = spec
	}
	currentByName := make(map[string]ToolSpec, len(current))
	for _, spec := range current {
		currentByName[spec.Name] = spec
	}

	var changes ToolStateChanges
	for _, spec := range current {
		existing, ok := previousByName[spec.Name]
		if !ok || !toolSpecsEqual(existing, spec) {
			changes.ToolsAdded = append(changes.ToolsAdded, spec)
		}
	}
	for _, spec := range previous {
		if _, ok := currentByName[spec.Name]; !ok {
			changes.ToolsRemoved = append(changes.ToolsRemoved, ToolReference{Name: spec.Name})
		}
	}
	return changes
}

// HasToolChanges reports whether the delta is empty.
func (c ToolStateChanges) HasToolChanges() bool {
	return len(c.ToolsAdded) > 0 || len(c.ToolsRemoved) > 0
}

func toolSpecsEqual(left, right ToolSpec) bool {
	if left.Name != right.Name || left.Description != right.Description {
		return false
	}
	leftJSON, leftErr := json.Marshal(left.Parameters)
	rightJSON, rightErr := json.Marshal(right.Parameters)
	return leftErr == nil && rightErr == nil && string(leftJSON) == string(rightJSON)
}

func mergeToolSpecs(base, added []ToolSpec) []ToolSpec {
	if len(added) == 0 {
		return base
	}
	merged := cloneToolSpecs(base)
	for _, spec := range added {
		replaced := false
		for i := range merged {
			if merged[i].Name == spec.Name {
				merged[i] = spec
				replaced = true
				break
			}
		}
		if !replaced {
			merged = append(merged, spec)
		}
	}
	return merged
}

func removeSectionName(order []string, name string) []string {
	for i, existing := range order {
		if existing == name {
			return append(order[:i], order[i+1:]...)
		}
	}
	return order
}

func removeToolSpecs(base []ToolSpec, removed []ToolReference) []ToolSpec {
	if len(base) == 0 || len(removed) == 0 {
		return base
	}
	dropped := make(map[string]struct{}, len(removed))
	for _, reference := range removed {
		dropped[reference.Name] = struct{}{}
	}
	kept := make([]ToolSpec, 0, len(base))
	for _, spec := range base {
		if _, ok := dropped[spec.Name]; ok {
			continue
		}
		kept = append(kept, spec)
	}
	return kept
}
