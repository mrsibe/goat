package agent

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/mrsibe/goat/ai"
)

func validateToolArguments(spec ai.ToolSpec, raw json.RawMessage) error {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 {
		raw = json.RawMessage("{}")
	}

	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return fmt.Errorf("invalid JSON: trailing data")
	}
	if _, ok := value.(map[string]any); !ok {
		return fmt.Errorf("arguments must be a JSON object")
	}

	root := any(spec.Parameters)
	validator := schemaValidator{root: root}
	if err := validator.validate(root, value, "arguments"); err != nil {
		return err
	}
	return nil
}

type schemaValidator struct {
	root any
}

func (v schemaValidator) validate(schema, value any, path string) error {
	switch typed := schema.(type) {
	case nil, bool:
		if typed == false {
			return fmt.Errorf("%s is not allowed", path)
		}
		return nil
	case map[string]any:
		return v.validateObject(typed, value, path)
	default:
		return fmt.Errorf("%s schema must be an object or boolean", path)
	}
}

func (v schemaValidator) validateObject(schema map[string]any, value any, path string) error {
	if ref, ok := schema["$ref"].(string); ok {
		resolved, err := v.resolveRef(ref)
		if err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		return v.validate(resolved, value, path)
	}

	if nullable, _ := schema["nullable"].(bool); nullable && value == nil {
		return nil
	}
	if allowed, ok := schema["type"]; ok && !matchesTypes(value, allowed) {
		return fmt.Errorf("%s: expected %s, got %s", path, describeTypes(allowed), jsonType(value))
	}
	if expected, ok := schema["const"]; ok && !jsonEqual(value, expected) {
		return fmt.Errorf("%s: must equal %s", path, compactJSON(expected))
	}
	if values, ok := anyList(schema["enum"]); ok && !containsJSON(values, value) {
		return fmt.Errorf("%s: must be one of %s", path, compactJSON(values))
	}
	if values, ok := anyList(schema["allOf"]); ok {
		for _, item := range values {
			if err := v.validate(item, value, path); err != nil {
				return err
			}
		}
	}
	if values, ok := anyList(schema["anyOf"]); ok {
		var firstErr error
		for _, item := range values {
			if err := v.validate(item, value, path); err == nil {
				firstErr = nil
				break
			} else if firstErr == nil {
				firstErr = err
			}
		}
		if firstErr != nil {
			return fmt.Errorf("%s: does not match any allowed schema: %w", path, firstErr)
		}
	}
	if values, ok := anyList(schema["oneOf"]); ok {
		matches := 0
		for _, item := range values {
			if v.validate(item, value, path) == nil {
				matches++
			}
		}
		if matches != 1 {
			return fmt.Errorf("%s: must match exactly one schema, matched %d", path, matches)
		}
	}
	if negated, ok := schema["not"]; ok {
		if v.validate(negated, value, path) == nil {
			return fmt.Errorf("%s: matches a forbidden schema", path)
		}
	}

	switch typed := value.(type) {
	case map[string]any:
		if err := v.validateMap(schema, typed, path); err != nil {
			return err
		}
	case []any:
		if err := v.validateArray(schema, typed, path); err != nil {
			return err
		}
	case string:
		if err := validateString(schema, typed, path); err != nil {
			return err
		}
	case json.Number:
		if err := validateNumber(schema, typed, path); err != nil {
			return err
		}
	}
	return nil
}

func (v schemaValidator) validateMap(schema map[string]any, value map[string]any, path string) error {
	if minimum, ok := intKeyword(schema, "minProperties"); ok && len(value) < minimum {
		return fmt.Errorf("%s: must contain at least %d properties", path, minimum)
	}
	if maximum, ok := intKeyword(schema, "maxProperties"); ok && len(value) > maximum {
		return fmt.Errorf("%s: must contain at most %d properties", path, maximum)
	}

	properties, _ := schema["properties"].(map[string]any)
	for _, name := range stringList(schema["required"]) {
		if _, ok := value[name]; !ok {
			return fmt.Errorf("%s.%s: required property is missing", path, name)
		}
	}
	for _, key := range sortedKeys(value) {
		if propertySchema, ok := properties[key]; ok {
			if err := v.validate(propertySchema, value[key], path+"."+key); err != nil {
				return err
			}
			continue
		}
		switch additional := schema["additionalProperties"].(type) {
		case bool:
			if !additional {
				return fmt.Errorf("%s.%s: additional property is not allowed", path, key)
			}
		case map[string]any:
			if err := v.validate(additional, value[key], path+"."+key); err != nil {
				return err
			}
		}
	}
	return nil
}

func (v schemaValidator) validateArray(schema map[string]any, value []any, path string) error {
	if minimum, ok := intKeyword(schema, "minItems"); ok && len(value) < minimum {
		return fmt.Errorf("%s: must contain at least %d items", path, minimum)
	}
	if maximum, ok := intKeyword(schema, "maxItems"); ok && len(value) > maximum {
		return fmt.Errorf("%s: must contain at most %d items", path, maximum)
	}
	if unique, _ := schema["uniqueItems"].(bool); unique {
		for i := 0; i < len(value); i++ {
			for j := i + 1; j < len(value); j++ {
				if jsonEqual(value[i], value[j]) {
					return fmt.Errorf("%s[%d]: duplicate item", path, j)
				}
			}
		}
	}
	if itemSchema, ok := schema["items"]; ok {
		for index, item := range value {
			if err := v.validate(itemSchema, item, fmt.Sprintf("%s[%d]", path, index)); err != nil {
				return err
			}
		}
	}
	if prefix, ok := anyList(schema["prefixItems"]); ok {
		for index, item := range value {
			if index >= len(prefix) {
				break
			}
			if err := v.validate(prefix[index], item, fmt.Sprintf("%s[%d]", path, index)); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateString(schema map[string]any, value, path string) error {
	length := utf8.RuneCountInString(value)
	if minimum, ok := intKeyword(schema, "minLength"); ok && length < minimum {
		return fmt.Errorf("%s: length must be at least %d", path, minimum)
	}
	if maximum, ok := intKeyword(schema, "maxLength"); ok && length > maximum {
		return fmt.Errorf("%s: length must be at most %d", path, maximum)
	}
	if pattern, ok := schema["pattern"].(string); ok {
		expression, err := regexp.Compile(pattern)
		if err != nil {
			return fmt.Errorf("%s: invalid schema pattern: %w", path, err)
		}
		if !expression.MatchString(value) {
			return fmt.Errorf("%s: must match %q", path, pattern)
		}
	}
	return nil
}

func validateNumber(schema map[string]any, value json.Number, path string) error {
	number, err := value.Float64()
	if err != nil {
		return fmt.Errorf("%s: invalid number", path)
	}
	if minimum, ok := numberKeyword(schema, "minimum"); ok && number < minimum {
		return fmt.Errorf("%s: must be at least %s", path, strconv.FormatFloat(minimum, 'f', -1, 64))
	}
	if maximum, ok := numberKeyword(schema, "maximum"); ok && number > maximum {
		return fmt.Errorf("%s: must be at most %s", path, strconv.FormatFloat(maximum, 'f', -1, 64))
	}
	if minimum, ok := numberKeyword(schema, "exclusiveMinimum"); ok && number <= minimum {
		return fmt.Errorf("%s: must be greater than %s", path, strconv.FormatFloat(minimum, 'f', -1, 64))
	}
	if maximum, ok := numberKeyword(schema, "exclusiveMaximum"); ok && number >= maximum {
		return fmt.Errorf("%s: must be less than %s", path, strconv.FormatFloat(maximum, 'f', -1, 64))
	}
	if divisor, ok := numberKeyword(schema, "multipleOf"); ok {
		if divisor == 0 {
			return fmt.Errorf("%s: schema multipleOf must not be zero", path)
		}
		quotient := number / divisor
		if math.Abs(quotient-math.Round(quotient)) > 1e-9 {
			return fmt.Errorf("%s: must be a multiple of %s", path, strconv.FormatFloat(divisor, 'f', -1, 64))
		}
	}
	return nil
}

func (v schemaValidator) resolveRef(ref string) (any, error) {
	if ref == "#" {
		return v.root, nil
	}
	if !strings.HasPrefix(ref, "#/") {
		return nil, fmt.Errorf("unsupported schema reference %q", ref)
	}
	current := v.root
	for _, rawPart := range strings.Split(strings.TrimPrefix(ref, "#/"), "/") {
		part := strings.ReplaceAll(strings.ReplaceAll(rawPart, "~1", "/"), "~0", "~")
		switch typed := current.(type) {
		case map[string]any:
			next, ok := typed[part]
			if !ok {
				return nil, fmt.Errorf("schema reference %q does not exist", ref)
			}
			current = next
		case []any:
			index, err := strconv.Atoi(part)
			if err != nil || index < 0 || index >= len(typed) {
				return nil, fmt.Errorf("schema reference %q does not exist", ref)
			}
			current = typed[index]
		default:
			return nil, fmt.Errorf("schema reference %q does not exist", ref)
		}
	}
	return current, nil
}

func matchesTypes(value any, schema any) bool {
	switch typed := schema.(type) {
	case string:
		return matchesType(value, typed)
	default:
		values, ok := anyList(typed)
		if !ok {
			return false
		}
		for _, item := range values {
			if name, ok := item.(string); ok && matchesType(value, name) {
				return true
			}
		}
	}
	return false
}

func matchesType(value any, name string) bool {
	switch name {
	case "object":
		_, ok := value.(map[string]any)
		return ok
	case "array":
		_, ok := value.([]any)
		return ok
	case "string":
		_, ok := value.(string)
		return ok
	case "number":
		_, ok := value.(json.Number)
		return ok
	case "integer":
		number, ok := value.(json.Number)
		if !ok {
			return false
		}
		parsed, err := number.Float64()
		return err == nil && math.Trunc(parsed) == parsed
	case "boolean":
		_, ok := value.(bool)
		return ok
	case "null":
		return value == nil
	default:
		return false
	}
}

func describeTypes(value any) string {
	names := stringList(value)
	if len(names) == 0 {
		return "a valid type"
	}
	return strings.Join(names, " or ")
}

func jsonType(value any) string {
	switch value.(type) {
	case nil:
		return "null"
	case map[string]any:
		return "object"
	case []any:
		return "array"
	case string:
		return "string"
	case json.Number:
		return "number"
	case bool:
		return "boolean"
	default:
		return fmt.Sprintf("%T", value)
	}
}

func stringList(value any) []string {
	switch typed := value.(type) {
	case []string:
		return append([]string(nil), typed...)
	case []any:
		result := make([]string, 0, len(typed))
		for _, item := range typed {
			if text, ok := item.(string); ok {
				result = append(result, text)
			}
		}
		return result
	case string:
		return []string{typed}
	default:
		return nil
	}
}

func anyList(value any) ([]any, bool) {
	switch typed := value.(type) {
	case []any:
		return typed, true
	case []string:
		result := make([]any, len(typed))
		for index, item := range typed {
			result[index] = item
		}
		return result, true
	}
	reflected := reflect.ValueOf(value)
	if !reflected.IsValid() || (reflected.Kind() != reflect.Slice && reflected.Kind() != reflect.Array) {
		return nil, false
	}
	result := make([]any, reflected.Len())
	for index := range result {
		result[index] = reflected.Index(index).Interface()
	}
	return result, true
}

func intKeyword(schema map[string]any, key string) (int, bool) {
	number, ok := numberKeyword(schema, key)
	if !ok || math.Trunc(number) != number {
		return 0, false
	}
	return int(number), true
}

func numberKeyword(schema map[string]any, key string) (float64, bool) {
	switch typed := schema[key].(type) {
	case float64:
		return typed, true
	case float32:
		return float64(typed), true
	case int:
		return float64(typed), true
	case int64:
		return float64(typed), true
	case json.Number:
		value, err := typed.Float64()
		return value, err == nil
	default:
		return 0, false
	}
}

func containsJSON(values []any, target any) bool {
	for _, value := range values {
		if jsonEqual(value, target) {
			return true
		}
	}
	return false
}

func jsonEqual(left, right any) bool {
	if reflect.DeepEqual(left, right) {
		return true
	}
	leftNumber, leftOK := asFloat(left)
	rightNumber, rightOK := asFloat(right)
	return leftOK && rightOK && leftNumber == rightNumber
}

func asFloat(value any) (float64, bool) {
	switch typed := value.(type) {
	case json.Number:
		number, err := typed.Float64()
		return number, err == nil
	case float64:
		return typed, true
	case int:
		return float64(typed), true
	case int64:
		return float64(typed), true
	default:
		return 0, false
	}
}

func compactJSON(value any) string {
	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Sprintf("%v", value)
	}
	return string(data)
}

func sortedKeys(values map[string]any) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
