package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/mrsibe/goat/agent"
	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/ai/openai"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run() error {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return errors.New("OPENAI_API_KEY is required")
	}

	modelName := os.Getenv("OPENAI_MODEL")
	if modelName == "" {
		return errors.New("OPENAI_MODEL is required")
	}

	model, err := openai.New(openai.Config{
		APIKey:  apiKey,
		BaseURL: os.Getenv("OPENAI_BASE_URL"),
		Model:   modelName,
	})
	if err != nil {
		return err
	}

	assistant, err := agent.New(agent.Config{
		Model:        model,
		Tools:        []agent.Tool{weatherTool},
		SystemPrompt: "你是一个助手。需要外部信息时调用工具，拿到结果后用中文简洁回答。",
		MaxSteps:     8,
	})
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	lastStep := 0
	streamed := false
	result, err := assistant.Run(ctx, "北京今天适合户外活动吗？", agent.WithOnEvent(func(event agent.Event) {
		switch event.Type {
		case agent.EventMessageDelta:
			if event.Step != lastStep {
				fmt.Printf("\nAssistant[%d]: ", event.Step)
				lastStep = event.Step
			}
			fmt.Print(event.Text)
			streamed = true
		case agent.EventToolStart:
			fmt.Printf("\nAction: %s(%s)\n", event.ToolCall.Name, event.ToolCall.Arguments)
		case agent.EventToolEnd:
			fmt.Print("Observation: ")
			for _, block := range event.ToolResult.Content {
				if block.Type == ai.BlockText {
					fmt.Print(block.Text)
				}
			}
			fmt.Println()
		}
	}))
	if err != nil {
		return err
	}

	if !streamed {
		fmt.Printf("Assistant: %s\n", result.Answer)
	}
	fmt.Printf("\n[%d model turns, %d tokens]\n", result.Steps, result.Usage.TotalTokens)
	return nil
}
