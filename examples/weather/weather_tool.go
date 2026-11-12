package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/mrsibe/goat/agent"
	"github.com/mrsibe/goat/ai"
)

var weatherTool = agent.NewTextTool(agent.ToolSpec{
	Name:        "get_weather",
	Description: "查询指定城市的实时天气",
	Parameters: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"city": map[string]any{
				"type":        "string",
				"description": "城市名，例如：北京",
			},
		},
		"required": []string{"city"},
	},
}, func(_ context.Context, call ai.ToolCall) (string, error) {
	var input struct {
		City string `json:"city"`
	}
	if err := json.Unmarshal(call.Arguments, &input); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}
	if input.City == "" {
		return "", errors.New("city is required")
	}

	weather := map[string]string{
		"北京": "晴，25°C，微风，紫外线较强",
		"上海": "多云，22°C，湿度较高",
	}
	if condition, ok := weather[input.City]; ok {
		return condition, nil
	}
	return "", fmt.Errorf("no weather data for %s", input.City)
})
