package coding

import (
	"fmt"
	"path/filepath"
	"strings"
)

// BuildSystemPrompt builds the structured coding-agent system prompt. The
// project instructions are inserted verbatim inside a tagged section.
func BuildSystemPrompt(workspace, projectInstructions string) string {
	cwd := filepath.ToSlash(workspace)
	var prompt strings.Builder
	prompt.WriteString("You are goding, a coding assistant running inside a small Go coding-agent harness. Help users inspect files, run commands, and modify code.\n")
	prompt.WriteString("\n<tools>\n")
	prompt.WriteString("- read: Read file contents\n")
	prompt.WriteString("- bash: Execute bash commands (ls, rg, find, tests, builds)\n")
	prompt.WriteString("- edit: Make precise file edits with exact text replacement\n")
	prompt.WriteString("- write: Create or overwrite files\n")
	prompt.WriteString("</tools>\n")
	prompt.WriteString("\n<rules>\n")
	prompt.WriteString("- Use read to examine files instead of cat or sed\n")
	prompt.WriteString("- Use edit for targeted changes and batch disjoint edits in one call\n")
	prompt.WriteString("- Use write only for new files or complete rewrites\n")
	prompt.WriteString("- Use bash for listing, searching, finding, tests, and builds\n")
	prompt.WriteString("- File tools are confined to the workspace; bash is unrestricted and is not a sandbox\n")
	prompt.WriteString("- Be concise in your responses\n")
	prompt.WriteString("- Show file paths clearly when working with files\n")
	prompt.WriteString("</rules>\n")
	if strings.TrimSpace(projectInstructions) != "" {
		prompt.WriteString("\n<project_context>\n")
		prompt.WriteString("Project-specific instructions and guidelines:\n\n")
		prompt.WriteString(fmt.Sprintf("<project_instructions path=%q>\n%s\n</project_instructions>", filepath.ToSlash(filepath.Join(workspace, projectInstructionsFile)), projectInstructions))
		prompt.WriteString("\n</project_context>\n")
	}
	prompt.WriteString("\n<cwd>\n")
	prompt.WriteString(cwd)
	prompt.WriteString("\n</cwd>")
	return prompt.String()
}
