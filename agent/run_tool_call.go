package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/mrsibe/goat/ai"
)

// ToolCallOutcome is the final result of one tool call after hooks ran.
type ToolCallOutcome struct {
	ToolCall   ai.ToolCall
	Result     ToolResult
	IsError    bool
	DurationMs int64
}

// RunToolCallOptions configures RunToolCall.
type RunToolCallOptions struct {
	Tools            []Tool
	AssistantMessage ai.Message
	Context          AgentContext
	BeforeToolCall   BeforeToolCall
	AfterToolCall    AfterToolCall
}

// RunToolCall runs one tool call through argument preparation, schema
// validation, before/after hooks, and execution, without emitting agent
// events or appending messages. Failures are reported as error outcomes.
func RunToolCall(ctx context.Context, call ai.ToolCall, options RunToolCallOptions) ToolCallOutcome {
	outcome := ToolCallOutcome{ToolCall: ai.CloneToolCall(call)}
	registry, err := newToolRegistry(options.Tools)
	if err != nil {
		return failToolCall(outcome, err.Error())
	}
	tool, spec, ok := registry.get(call.Name)
	if !ok {
		return failToolCall(outcome, fmt.Sprintf("unknown tool %q", call.Name))
	}

	arguments := ai.CloneRawMessage(call.Arguments)
	if len(strings.TrimSpace(string(arguments))) == 0 {
		arguments = json.RawMessage("{}")
	}
	if call.ArgumentsError != "" {
		return failToolCall(outcome, fmt.Sprintf("invalid arguments for tool %q: %s", call.Name, call.ArgumentsError))
	}
	if spec.PrepareArguments != nil {
		prepared, prepareErr := spec.PrepareArguments(ai.CloneRawMessage(arguments))
		if prepareErr != nil {
			return failToolCall(outcome, fmt.Sprintf("invalid arguments for tool %q: %s", call.Name, prepareErr))
		}
		arguments = ai.CloneRawMessage(prepared)
	}
	if validateErr := validateToolArguments(spec.aiToolSpec(), arguments); validateErr != nil {
		return failToolCall(outcome, fmt.Sprintf("invalid arguments for tool %q: %s", call.Name, validateErr))
	}
	parsed, parseErr := parseJSONArguments(arguments)
	if parseErr != nil {
		return failToolCall(outcome, fmt.Sprintf("invalid arguments for tool %q: %s", call.Name, parseErr))
	}

	prepared := preparedToolCall{
		call: ai.ToolCall{
			ID:        call.ID,
			Name:      call.Name,
			Arguments: arguments,
		},
		tool: tool,
		args: parsed,
	}
	if options.BeforeToolCall != nil {
		decision, hookErr := options.BeforeToolCall(ctx, BeforeToolCallContext{
			AssistantMessage: ai.CloneMessage(options.AssistantMessage),
			ToolCall:         ai.CloneToolCall(prepared.call),
			Args:             parsed,
			Context:          cloneAgentContext(options.Context),
		})
		if hookErr != nil {
			return failToolCall(outcome, hookErr.Error())
		}
		if ctx.Err() != nil {
			return failToolCall(outcome, "Operation aborted")
		}
		if decision != nil && decision.Block {
			reason := decision.Reason
			if reason == "" {
				reason = "tool execution was blocked"
			}
			result := ErrorResult(reason)
			result.Terminate = decision.Terminate
			outcome.Result = result
			outcome.IsError = true
			return outcome
		}
	}
	if ctx.Err() != nil {
		return failToolCall(outcome, "Operation aborted")
	}

	started := time.Now()
	result, executeErr := executeToolSafely(ctx, prepared)
	outcome.DurationMs = time.Since(started).Milliseconds()
	if executeErr != nil {
		result = ErrorResult(executeErr.Error())
	}
	outcome.Result, outcome.IsError = applyToolCallHooks(ctx, options, prepared, result)
	recordNestedOutcome(ctx, prepared.call, argsOf(prepared), outcome)
	return outcome
}

func executeToolSafely(ctx context.Context, prepared preparedToolCall) (result ToolResult, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			result = ErrorResult(fmt.Sprintf("tool %q panicked: %v", prepared.call.Name, recovered))
			err = nil
		}
	}()
	result, err = prepared.tool.Execute(ctx, prepared.call.ID, ai.CloneRawMessage(prepared.call.Arguments), nil)
	if err != nil {
		return ToolResult{}, err
	}
	if len(result.Content) == 0 {
		result.Content = []ai.ContentBlock{{Type: ai.BlockText, Text: ""}}
	}
	if validateErr := validateToolResult(result); validateErr != nil {
		return ToolResult{}, fmt.Errorf("tool %q returned invalid content: %s", prepared.call.Name, validateErr)
	}
	return result.clone(), nil
}

func applyToolCallHooks(
	ctx context.Context,
	options RunToolCallOptions,
	prepared preparedToolCall,
	result ToolResult,
) (ToolResult, bool) {
	result = result.clone()
	isError := result.IsError
	if options.AfterToolCall == nil {
		return result, isError
	}
	override, err := options.AfterToolCall(ctx, AfterToolCallContext{
		AssistantMessage: ai.CloneMessage(options.AssistantMessage),
		ToolCall:         ai.CloneToolCall(prepared.call),
		Args:             prepared.args,
		Result:           result.clone(),
		IsError:          isError,
		Context:          cloneAgentContext(options.Context),
	})
	if err != nil {
		return ErrorResult(err.Error()), true
	}
	if override == nil {
		return result, isError
	}
	if override.Content != nil {
		result.Content = ai.CloneContentBlocks(override.Content)
		if override.StructuredContent == nil {
			result.StructuredContent = nil
		}
	}
	if override.Details != nil {
		result.Details = ai.CloneRawMessage(override.Details)
	}
	if override.StructuredContent != nil {
		result.StructuredContent = ai.CloneRawMessage(override.StructuredContent)
	}
	if override.Usage != nil {
		usage := *override.Usage
		result.Usage = &usage
	}
	if override.IsError != nil {
		isError = *override.IsError
		result.IsError = isError
	}
	if override.Terminate != nil {
		result.Terminate = *override.Terminate
	}
	result.IsError = isError
	if validateErr := validateToolResult(result); validateErr != nil {
		return ErrorResult(fmt.Sprintf("tool %q returned invalid content: %s", prepared.call.Name, validateErr)), true
	}
	return result, isError
}

func recordNestedOutcome(ctx context.Context, call ai.ToolCall, args json.RawMessage, outcome ToolCallOutcome) {
	recorder := nestedRecorderFromContext(ctx)
	if recorder == nil {
		return
	}
	status := "ok"
	message := ""
	if outcome.IsError {
		status = "error"
		message = outcome.Result.Text()
	}
	recorder.record(call.ID, call.Name, args, status, outcome.DurationMs, message)
}

func argsOf(prepared preparedToolCall) json.RawMessage {
	return ai.CloneRawMessage(prepared.call.Arguments)
}

func failToolCall(outcome ToolCallOutcome, message string) ToolCallOutcome {
	outcome.Result = ErrorResult(message)
	outcome.IsError = true
	return outcome
}
