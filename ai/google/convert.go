package google

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/mrsibe/goat/ai"
	"google.golang.org/genai"
)

func (p *Provider) convertMessages(messages []ai.Message) ([]*genai.Content, string, error) {
	var systemParts []string
	contents := make([]*genai.Content, 0, len(messages))
	for _, message := range messages {
		switch message.Role {
		case ai.RoleSystem:
			if text := message.Text(); text != "" {
				systemParts = append(systemParts, text)
			}
		case ai.RoleUser:
			parts, err := userParts(message)
			if err != nil {
				return nil, "", err
			}
			if len(parts) > 0 {
				contents = append(contents, &genai.Content{Role: "user", Parts: parts})
			}
		case ai.RoleAssistant:
			parts, err := p.assistantParts(message)
			if err != nil {
				return nil, "", err
			}
			if len(parts) > 0 {
				contents = append(contents, &genai.Content{Role: "model", Parts: parts})
			}
		case ai.RoleTool:
			part, images, err := toolResultPart(p.info.Model, message)
			if err != nil {
				return nil, "", err
			}
			if len(contents) > 0 && contents[len(contents)-1].Role == "user" &&
				hasFunctionResponse(contents[len(contents)-1]) {
				contents[len(contents)-1].Parts = append(contents[len(contents)-1].Parts, part)
			} else {
				contents = append(contents, &genai.Content{Role: "user", Parts: []*genai.Part{part}})
			}
			if len(images) > 0 && !supportsMultimodalFunctionResponse(p.info.Model) {
				parts := []*genai.Part{{Text: "Tool result image:"}}
				parts = append(parts, images...)
				contents = append(contents, &genai.Content{Role: "user", Parts: parts})
			}
		default:
			return nil, "", fmt.Errorf("google: unsupported message role %q", message.Role)
		}
	}
	return contents, strings.Join(systemParts, "\n\n"), nil
}

func userParts(message ai.Message) ([]*genai.Part, error) {
	parts := make([]*genai.Part, 0, len(message.Content))
	for _, block := range message.Content {
		switch block.Type {
		case ai.BlockText:
			parts = append(parts, &genai.Part{Text: block.Text})
		case ai.BlockImage:
			part, err := imagePart(block.Image)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)
		case ai.BlockThinking:
			return nil, errors.New("google: user messages do not support thinking blocks")
		case ai.BlockToolCall:
			return nil, errors.New("google: user messages do not support tool calls")
		default:
			return nil, fmt.Errorf("google: unsupported user block type %q", block.Type)
		}
	}
	return parts, nil
}

func (p *Provider) assistantParts(message ai.Message) ([]*genai.Part, error) {
	sameProviderModel := message.Provider == p.info.Provider && message.Model == p.info.Model
	parts := make([]*genai.Part, 0, len(message.Content))
	for _, block := range message.Content {
		switch block.Type {
		case ai.BlockText:
			signature := signatureBytes(block.TextSignature)
			if strings.TrimSpace(block.Text) == "" && len(signature) == 0 {
				continue
			}
			if !sameProviderModel {
				signature = nil
			}
			parts = append(parts, &genai.Part{Text: block.Text, ThoughtSignature: signature})
		case ai.BlockThinking:
			signature := signatureBytes(block.ThinkingSignature)
			if strings.TrimSpace(block.Thinking) == "" && len(signature) == 0 {
				continue
			}
			if sameProviderModel {
				parts = append(parts, &genai.Part{
					Text:             block.Thinking,
					Thought:          true,
					ThoughtSignature: signature,
				})
			} else {
				parts = append(parts, &genai.Part{Text: block.Thinking})
			}
		case ai.BlockToolCall:
			if block.ToolCall == nil {
				return nil, errors.New("google: tool call block is missing its payload")
			}
			arguments, err := argumentsMap(block.ToolCall.Arguments)
			if err != nil {
				return nil, fmt.Errorf("google: tool call %q: %w", block.ToolCall.Name, err)
			}
			call := &genai.FunctionCall{Name: block.ToolCall.Name, Args: arguments}
			if requiresToolCallID(p.info.Model) {
				call.ID = normalizeToolCallID(p.info.Model, block.ToolCall.ID)
			}
			signature := signatureBytes(block.ToolCall.ThoughtSignature)
			if !sameProviderModel {
				signature = nil
			}
			parts = append(parts, &genai.Part{FunctionCall: call, ThoughtSignature: signature})
		case ai.BlockImage:
			return nil, errors.New("google: assistant messages do not support images")
		default:
			return nil, fmt.Errorf("google: unsupported assistant block type %q", block.Type)
		}
	}
	return parts, nil
}

func toolResultPart(model string, message ai.Message) (*genai.Part, []*genai.Part, error) {
	if strings.TrimSpace(message.ToolName) == "" {
		return nil, nil, errors.New("google: tool message is missing its tool name")
	}
	var textParts []string
	var imageParts []*genai.Part
	var responseParts []*genai.FunctionResponsePart
	for _, block := range message.Content {
		switch block.Type {
		case ai.BlockText:
			if block.Text != "" {
				textParts = append(textParts, block.Text)
			}
		case ai.BlockImage:
			part, err := imagePart(block.Image)
			if err != nil {
				return nil, nil, err
			}
			imageParts = append(imageParts, part)
			responsePart, err := functionResponseImagePart(block.Image)
			if err != nil {
				return nil, nil, err
			}
			responseParts = append(responseParts, responsePart)
		default:
			return nil, nil, fmt.Errorf("google: unsupported tool result block type %q", block.Type)
		}
	}
	text := strings.Join(textParts, "\n")
	value := text
	if strings.TrimSpace(value) == "" && len(imageParts) > 0 {
		value = "(see attached image)"
	}
	response := map[string]any{"output": value}
	if message.IsError {
		response = map[string]any{"error": value}
	}
	responsePart := &genai.Part{FunctionResponse: &genai.FunctionResponse{
		Name:     message.ToolName,
		Response: response,
	}}
	if supportsMultimodalFunctionResponse(model) && len(responseParts) > 0 {
		responsePart.FunctionResponse.Parts = responseParts
	}
	if requiresToolCallID(model) {
		responsePart.FunctionResponse.ID = normalizeToolCallID(model, message.ToolCallID)
	}
	return responsePart, imageParts, nil
}

func hasFunctionResponse(content *genai.Content) bool {
	if content == nil {
		return false
	}
	for _, part := range content.Parts {
		if part != nil && part.FunctionResponse != nil {
			return true
		}
	}
	return false
}

func convertTools(specs []ai.ToolSpec) ([]*genai.Tool, error) {
	if len(specs) == 0 {
		return nil, nil
	}
	declarations := make([]*genai.FunctionDeclaration, 0, len(specs))
	for _, spec := range specs {
		if strings.TrimSpace(spec.Name) == "" {
			return nil, errors.New("google: tool name must not be empty")
		}
		parameters := spec.Parameters
		if parameters == nil {
			parameters = map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			}
		}
		normalized, err := normalizeSchema(parameters)
		if err != nil {
			return nil, fmt.Errorf("google: tool %q schema: %w", spec.Name, err)
		}
		declarations = append(declarations, &genai.FunctionDeclaration{
			Name:                 spec.Name,
			Description:          spec.Description,
			ParametersJsonSchema: rewriteConstAsEnum(normalized),
		})
	}
	return []*genai.Tool{{FunctionDeclarations: declarations}}, nil
}

func normalizeSchema(parameters map[string]any) (any, error) {
	raw, err := json.Marshal(parameters)
	if err != nil {
		return nil, err
	}
	var normalized any
	if err := json.Unmarshal(raw, &normalized); err != nil {
		return nil, err
	}
	return normalized, nil
}

func convertToolChoice(choice ai.ToolChoice) *genai.ToolConfig {
	if choice.Mode == "" {
		return nil
	}
	config := &genai.FunctionCallingConfig{}
	switch choice.Mode {
	case ai.ToolChoiceAuto:
		config.Mode = genai.FunctionCallingConfigModeAuto
	case ai.ToolChoiceNone:
		config.Mode = genai.FunctionCallingConfigModeNone
	case ai.ToolChoiceRequired:
		config.Mode = genai.FunctionCallingConfigModeAny
	case ai.ToolChoiceNamed:
		if strings.TrimSpace(choice.Name) == "" {
			return nil
		}
		config.Mode = genai.FunctionCallingConfigModeAny
		config.AllowedFunctionNames = []string{choice.Name}
	default:
		return nil
	}
	return &genai.ToolConfig{FunctionCallingConfig: config}
}

func convertThinkingConfig(model string, options ai.RequestOptions) *genai.ThinkingConfig {
	budget := options.ThinkingBudgetTokens
	effort := strings.ToLower(strings.TrimSpace(options.ReasoningEffort))
	if budget == 0 && effort == "" {
		return nil
	}
	if effort == "off" || effort == "none" {
		if usesThinkingLevel(model) {
			return &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelMinimal}
		}
		zero := int32(0)
		return &genai.ThinkingConfig{ThinkingBudget: &zero}
	}
	if usesThinkingLevel(model) {
		return &genai.ThinkingConfig{
			IncludeThoughts: true,
			ThinkingLevel:   thinkingLevel(effort),
		}
	}
	if budget < 0 {
		value := int32(budget)
		return &genai.ThinkingConfig{IncludeThoughts: true, ThinkingBudget: &value}
	}
	if budget <= 0 {
		budget = defaultThinkingBudget(model, effort)
	}
	value := int32(budget)
	return &genai.ThinkingConfig{IncludeThoughts: true, ThinkingBudget: &value}
}

func thinkingLevel(effort string) genai.ThinkingLevel {
	switch effort {
	case "minimal":
		return genai.ThinkingLevelMinimal
	case "low":
		return genai.ThinkingLevelLow
	case "medium":
		return genai.ThinkingLevelMedium
	default:
		return genai.ThinkingLevelHigh
	}
}

func defaultThinkingBudget(model, effort string) int {
	if strings.Contains(model, "2.5-pro") {
		switch effort {
		case "minimal":
			return 128
		case "low":
			return 2048
		case "medium":
			return 8192
		default:
			return 32768
		}
	}
	if strings.Contains(model, "2.5-flash-lite") {
		switch effort {
		case "minimal":
			return 512
		case "low":
			return 2048
		case "medium":
			return 8192
		default:
			return 24576
		}
	}
	if strings.Contains(model, "2.5-flash") {
		switch effort {
		case "minimal":
			return 128
		case "low":
			return 2048
		case "medium":
			return 8192
		default:
			return 24576
		}
	}
	return -1
}

var thinkingLevelModel = regexp.MustCompile(`(?i)^(gemini-3(?:\.\d+)?-(?:pro|flash)|gemini-(?:[4-9]|[1-9]\d)|gemini-flash-(?:lite-)?latest|gemma-?4)`)

func usesThinkingLevel(model string) bool {
	return thinkingLevelModel.MatchString(model)
}

var geminiMajorVersion = regexp.MustCompile(`(?i)^gemini(?:-live)?-(\d+)`)

func requiresToolCallID(model string) bool {
	if major, ok := googleMajorVersion(model); ok {
		return major >= 3
	}
	return strings.HasPrefix(model, "claude-") || strings.HasPrefix(model, "gpt-oss-")
}

func normalizeToolCallID(model, id string) string {
	if !requiresToolCallID(model) {
		return id
	}
	var normalized strings.Builder
	normalized.Grow(len(id))
	for index := 0; index < len(id) && normalized.Len() < 64; index++ {
		character := id[index]
		if (character >= 'a' && character <= 'z') ||
			(character >= 'A' && character <= 'Z') ||
			(character >= '0' && character <= '9') ||
			character == '_' || character == '-' {
			normalized.WriteByte(character)
			continue
		}
		normalized.WriteByte('_')
	}
	return normalized.String()
}

func supportsMultimodalFunctionResponse(model string) bool {
	if major, ok := googleMajorVersion(model); ok {
		return major >= 3
	}
	return true
}

func googleMajorVersion(model string) (int, bool) {
	match := geminiMajorVersion.FindStringSubmatch(model)
	if len(match) != 2 {
		return 0, false
	}
	major, err := strconv.Atoi(match[1])
	if err != nil {
		return 0, false
	}
	return major, true
}

func imagePart(image *ai.ImageContent) (*genai.Part, error) {
	data, mediaType, uri, err := imagePayload(image)
	if err != nil {
		return nil, err
	}
	if uri != "" {
		return &genai.Part{FileData: &genai.FileData{FileURI: uri, MIMEType: mediaType}}, nil
	}
	return &genai.Part{InlineData: &genai.Blob{Data: data, MIMEType: mediaType}}, nil
}

func functionResponseImagePart(image *ai.ImageContent) (*genai.FunctionResponsePart, error) {
	data, mediaType, uri, err := imagePayload(image)
	if err != nil {
		return nil, err
	}
	if uri != "" {
		return &genai.FunctionResponsePart{
			FileData: &genai.FunctionResponseFileData{FileURI: uri, MIMEType: mediaType},
		}, nil
	}
	return &genai.FunctionResponsePart{
		InlineData: &genai.FunctionResponseBlob{Data: data, MIMEType: mediaType},
	}, nil
}

func imagePayload(image *ai.ImageContent) ([]byte, string, string, error) {
	if image == nil {
		return nil, "", "", errors.New("google: image block is missing its payload")
	}
	data := strings.TrimSpace(image.Data)
	uri := strings.TrimSpace(image.URL)
	if (data == "") == (uri == "") {
		return nil, "", "", errors.New("google: image must set exactly one of data or url")
	}
	mediaType := strings.TrimSpace(image.MediaType)
	if data != "" {
		if mediaType == "" {
			return nil, "", "", errors.New("google: inline image requires a media type")
		}
		decoded, err := base64.StdEncoding.DecodeString(data)
		if err != nil {
			return nil, "", "", fmt.Errorf("google: decode inline image: %w", err)
		}
		return decoded, mediaType, "", nil
	}
	if strings.HasPrefix(uri, "data:") {
		decoded, parsedType, err := parseDataURL(uri)
		if err != nil {
			return nil, "", "", err
		}
		if mediaType == "" {
			mediaType = parsedType
		}
		return decoded, mediaType, "", nil
	}
	if mediaType == "" {
		return nil, "", "", errors.New("google: image URL requires a media type")
	}
	return nil, mediaType, uri, nil
}

func parseDataURL(value string) ([]byte, string, error) {
	rest := strings.TrimPrefix(value, "data:")
	comma := strings.IndexByte(rest, ',')
	if comma < 0 {
		return nil, "", errors.New("google: invalid data URL")
	}
	meta := rest[:comma]
	payload := rest[comma+1:]
	if !strings.HasSuffix(meta, ";base64") {
		return nil, "", errors.New("google: only base64 data URLs are supported")
	}
	mediaType := strings.TrimSuffix(meta, ";base64")
	if mediaType == "" {
		return nil, "", errors.New("google: data URL requires a media type")
	}
	decoded, err := base64.StdEncoding.DecodeString(payload)
	if err != nil {
		return nil, "", fmt.Errorf("google: decode data URL: %w", err)
	}
	return decoded, mediaType, nil
}

func argumentsMap(raw json.RawMessage) (map[string]any, error) {
	if len(raw) == 0 || strings.TrimSpace(string(raw)) == "" {
		return map[string]any{}, nil
	}
	var arguments map[string]any
	if err := json.Unmarshal(raw, &arguments); err != nil {
		return nil, err
	}
	if arguments == nil {
		arguments = map[string]any{}
	}
	return arguments, nil
}

func signatureBytes(signature string) []byte {
	signature = strings.TrimSpace(signature)
	if signature == "" {
		return nil
	}
	decoded, err := base64.StdEncoding.DecodeString(signature)
	if err != nil {
		return nil
	}
	return decoded
}

var jsonSchemaNameMaps = map[string]bool{
	"properties":        true,
	"patternProperties": true,
	"$defs":             true,
	"definitions":       true,
	"dependentSchemas":  true,
	"dependencies":      true,
}

var jsonSchemaSubschemas = map[string]bool{
	"additionalProperties":  true,
	"unevaluatedProperties": true,
	"propertyNames":         true,
	"items":                 true,
	"prefixItems":           true,
	"additionalItems":       true,
	"unevaluatedItems":      true,
	"contains":              true,
	"allOf":                 true,
	"anyOf":                 true,
	"oneOf":                 true,
	"not":                   true,
	"if":                    true,
	"then":                  true,
	"else":                  true,
	"contentSchema":         true,
}

func rewriteConstAsEnum(value any) any {
	return rewriteSchema(value, false)
}

func rewriteSchema(value any, isNameMap bool) any {
	switch typed := value.(type) {
	case []any:
		result := make([]any, len(typed))
		for index, item := range typed {
			result[index] = rewriteSchema(item, false)
		}
		return result
	case map[string]any:
		result := make(map[string]any, len(typed))
		for key, item := range typed {
			switch {
			case isNameMap || jsonSchemaSubschemas[key]:
				result[key] = rewriteSchema(item, false)
			case jsonSchemaNameMaps[key]:
				result[key] = rewriteSchema(item, true)
			default:
				result[key] = item
			}
		}
		if !isNameMap {
			if constant, ok := result["const"]; ok {
				if _, hasEnum := result["enum"]; !hasEnum {
					switch constant.(type) {
					case string, float64, int, int64, json.Number:
						delete(result, "const")
						result["enum"] = []any{constant}
					}
				}
			}
		}
		return result
	default:
		return value
	}
}
