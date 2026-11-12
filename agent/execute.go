package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/mrsibe/goat/ai"
)

type preparedToolCall struct {
	call      ai.ToolCall
	tool      Tool
	args      any
	immediate *ToolResult
}

type executedToolCall struct {
	result     ToolResult
	durationMs int64
}

func (a *Agent) executeToolCalls(
	ctx context.Context,
	agentContext AgentContext,
	assistant ai.Message,
	calls []ai.ToolCall,
	step int,
) ([]ai.Message, bool) {
	registry, err := newToolRegistry(agentContext.Tools)
	if err != nil {
		results := make([]ai.Message, len(calls))
		for index, call := range calls {
			callCopy := ai.CloneToolCall(call)
			a.emit(Event{Type: EventToolStart, Step: step, ToolCall: &callCopy})
			result := ErrorResult(err.Error())
			a.emitToolEnd(callCopy, result, 0, step)
			results[index] = toolResultMessage(callCopy, result, 0)
		}
		return results, false
	}

	sequential := a.toolExecutionMode() == ToolExecutionSequential
	if !sequential {
		for _, call := range calls {
			_, spec, ok := registry.get(call.Name)
			if ok && spec.ExecutionMode == ToolExecutionSequential {
				sequential = true
				break
			}
		}
	}
	if sequential {
		return a.executeToolCallsSequential(ctx, agentContext, assistant, calls, step, registry)
	}
	return a.executeToolCallsParallel(ctx, agentContext, assistant, calls, step, registry)
}

func (a *Agent) executeToolCallsSequential(
	ctx context.Context,
	agentContext AgentContext,
	assistant ai.Message,
	calls []ai.ToolCall,
	step int,
	registry toolRegistry,
) ([]ai.Message, bool) {
	results := make([]ai.Message, 0, len(calls))
	finalized := make([]ToolResult, 0, len(calls))
	for _, call := range calls {
		prepared := a.prepareToolCall(ctx, agentContext, assistant, call, step, registry)
		var result ToolResult
		var durationMs int64
		if prepared.immediate != nil {
			result = prepared.immediate.clone()
		} else {
			executed := a.executePreparedTool(ctx, prepared, step)
			result, durationMs = a.finalizeTool(ctx, agentContext, assistant, prepared, executed, step)
		}
		a.emitToolEnd(prepared.call, result, durationMs, step)
		results = append(results, toolResultMessage(prepared.call, result, durationMs))
		finalized = append(finalized, result)
		if ctx.Err() != nil {
			break
		}
	}
	return results, allToolsTerminate(finalized)
}

func (a *Agent) executeToolCallsParallel(
	ctx context.Context,
	agentContext AgentContext,
	assistant ai.Message,
	calls []ai.ToolCall,
	step int,
	registry toolRegistry,
) ([]ai.Message, bool) {
	type completion struct {
		index      int
		result     ToolResult
		durationMs int64
	}
	completed := make(chan completion, len(calls))
	var wait sync.WaitGroup
	finalized := make([]ToolResult, len(calls))
	results := make([]ai.Message, len(calls))

	preparedCalls := make([]preparedToolCall, len(calls))
	for index, call := range calls {
		preparedCalls[index] = a.prepareToolCall(ctx, agentContext, assistant, call, step, registry)
	}

	for index, prepared := range preparedCalls {
		if prepared.immediate != nil {
			completed <- completion{index: index, result: prepared.immediate.clone()}
			continue
		}
		wait.Add(1)
		go func(index int, prepared preparedToolCall) {
			defer wait.Done()
			executed := a.executePreparedTool(ctx, prepared, step)
			result, durationMs := a.finalizeTool(ctx, agentContext, assistant, prepared, executed, step)
			completed <- completion{index: index, result: result, durationMs: durationMs}
		}(index, prepared)
	}
	wait.Wait()
	close(completed)

	for item := range completed {
		a.emitToolEnd(calls[item.index], item.result, item.durationMs, step)
		finalized[item.index] = item.result
		results[item.index] = toolResultMessage(calls[item.index], item.result, item.durationMs)
	}
	return results, allToolsTerminate(finalized)
}

func (a *Agent) prepareToolCall(
	ctx context.Context,
	agentContext AgentContext,
	assistant ai.Message,
	call ai.ToolCall,
	step int,
	registry toolRegistry,
) preparedToolCall {
	callCopy := ai.CloneToolCall(call)
	a.emit(Event{Type: EventToolStart, Step: step, ToolCall: &callCopy})

	tool, spec, ok := registry.get(call.Name)
	if !ok {
		result := ErrorResult(fmt.Sprintf("unknown tool %q", call.Name))
		return preparedToolCall{call: callCopy, immediate: &result}
	}
	if len(strings.TrimSpace(string(callCopy.Arguments))) == 0 {
		callCopy.Arguments = json.RawMessage("{}")
	}
	if callCopy.ArgumentsError != "" {
		result := ErrorResult(fmt.Sprintf(
			"invalid arguments for tool %q: %s",
			call.Name,
			callCopy.ArgumentsError,
		))
		return preparedToolCall{call: callCopy, immediate: &result}
	}
	if spec.PrepareArguments != nil {
		prepared, err := spec.PrepareArguments(ai.CloneRawMessage(callCopy.Arguments))
		if err != nil {
			result := ErrorResult(fmt.Sprintf("invalid arguments for tool %q: %s", call.Name, err))
			return preparedToolCall{call: callCopy, immediate: &result}
		}
		callCopy.Arguments = ai.CloneRawMessage(prepared)
	}
	if err := validateToolArguments(spec.aiToolSpec(), callCopy.Arguments); err != nil {
		result := ErrorResult(fmt.Sprintf(
			"invalid arguments for tool %q: %s",
			call.Name,
			err,
		))
		return preparedToolCall{call: callCopy, immediate: &result}
	}
	args, err := parseJSONArguments(callCopy.Arguments)
	if err != nil {
		result := ErrorResult(fmt.Sprintf("invalid arguments for tool %q: %s", call.Name, err))
		return preparedToolCall{call: callCopy, immediate: &result}
	}
	if a.beforeToolCall != nil {
		decision, err := a.beforeToolCall(ctx, BeforeToolCallContext{
			AssistantMessage: ai.CloneMessage(assistant),
			ToolCall:         ai.CloneToolCall(callCopy),
			Args:             args,
			Context:          cloneAgentContext(agentContext),
		})
		if err != nil {
			result := ErrorResult(err.Error())
			return preparedToolCall{call: callCopy, immediate: &result}
		}
		if decision != nil && decision.Block {
			reason := decision.Reason
			if reason == "" {
				reason = "tool execution was blocked"
			}
			result := ErrorResult(reason)
			result.Terminate = decision.Terminate
			return preparedToolCall{call: callCopy, immediate: &result}
		}
	}
	return preparedToolCall{call: callCopy, tool: tool, args: args}
}

func (a *Agent) executePreparedTool(ctx context.Context, prepared preparedToolCall, step int) (executed executedToolCall) {
	started := time.Now()
	var updateMu sync.Mutex
	var updates sync.WaitGroup
	accepting := true
	closeUpdates := func() {
		updateMu.Lock()
		accepting = false
		updateMu.Unlock()
		updates.Wait()
	}
	defer func() {
		closeUpdates()
		if recovered := recover(); recovered != nil {
			executed.result = ErrorResult(fmt.Sprintf(
				"tool %q panicked: %v",
				prepared.call.Name,
				recovered,
			))
			executed.durationMs = time.Since(started).Milliseconds()
		}
	}()

	update := func(partial ToolResult) {
		updateMu.Lock()
		if !accepting {
			updateMu.Unlock()
			return
		}
		updates.Add(1)
		updateMu.Unlock()
		defer updates.Done()
		partialCopy := partial.clone()
		callCopy := ai.CloneToolCall(prepared.call)
		a.emit(Event{
			Type:          EventToolUpdate,
			Step:          step,
			ToolCall:      &callCopy,
			PartialResult: &partialCopy,
		})
	}

	var result ToolResult
	var err error
	if streaming, ok := prepared.tool.(StreamingTool); ok {
		result, err = streaming.ExecuteStream(ctx, prepared.call, update)
	} else {
		result, err = prepared.tool.Execute(ctx, prepared.call)
	}
	executed.durationMs = time.Since(started).Milliseconds()
	closeUpdates()
	if err != nil {
		executed.result = ErrorResult(err.Error())
		return executed
	}
	if len(result.Content) == 0 {
		result.Content = []ai.ContentBlock{{Type: ai.BlockText, Text: ""}}
	}
	result = result.clone()
	if err := validateToolResult(result); err != nil {
		executed.result = ErrorResult(fmt.Sprintf(
			"tool %q returned invalid content: %s",
			prepared.call.Name,
			err,
		))
		return executed
	}
	executed.result = result
	return executed
}

func (a *Agent) finalizeTool(
	ctx context.Context,
	agentContext AgentContext,
	assistant ai.Message,
	prepared preparedToolCall,
	executed executedToolCall,
	step int,
) (ToolResult, int64) {
	result := executed.result.clone()
	if a.afterToolCall != nil {
		override, err := a.afterToolCall(ctx, AfterToolCallContext{
			AssistantMessage: ai.CloneMessage(assistant),
			ToolCall:         ai.CloneToolCall(prepared.call),
			Args:             prepared.args,
			Result:           result.clone(),
			IsError:          result.IsError,
			Context:          cloneAgentContext(agentContext),
		})
		if err != nil {
			result = ErrorResult(err.Error())
		} else if override != nil {
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
				result.IsError = *override.IsError
			}
			if override.Terminate != nil {
				result.Terminate = *override.Terminate
			}
		}
	}
	result = result.clone()
	if err := validateToolResult(result); err != nil {
		result = ErrorResult(fmt.Sprintf("tool %q returned invalid content: %s", prepared.call.Name, err))
	}
	return result, executed.durationMs
}

func (a *Agent) failTruncatedToolCalls(calls []ai.ToolCall, step int) []ai.Message {
	results := make([]ai.Message, 0, len(calls))
	for _, call := range calls {
		callCopy := ai.CloneToolCall(call)
		a.emit(Event{Type: EventToolStart, Step: step, ToolCall: &callCopy})
		result := ErrorResult(fmt.Sprintf(
			"tool call %q was not executed because the response hit the output token limit; its arguments may be truncated. Re-issue the call with complete arguments.",
			call.Name,
		))
		a.emitToolEnd(callCopy, result, 0, step)
		results = append(results, toolResultMessage(callCopy, result, 0))
	}
	return results
}

func (a *Agent) emitToolEnd(call ai.ToolCall, result ToolResult, durationMs int64, step int) {
	callCopy := ai.CloneToolCall(call)
	resultCopy := result.clone()
	a.emit(Event{
		Type:       EventToolEnd,
		Step:       step,
		ToolCall:   &callCopy,
		ToolResult: &resultCopy,
		DurationMs: durationMs,
	})
}

func (a *Agent) toolExecutionMode() ToolExecutionMode {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.toolExecution
}

func toolResultMessage(call ai.ToolCall, result ToolResult, durationMs int64) ai.Message {
	return ai.Message{
		Role:              ai.RoleTool,
		Content:           ai.CloneContentBlocks(result.Content),
		ToolCallID:        call.ID,
		ToolName:          call.Name,
		IsError:           result.IsError,
		Details:           ai.CloneRawMessage(result.Details),
		StructuredContent: ai.CloneRawMessage(result.StructuredContent),
		Usage:             cloneUsage(result.Usage),
		DurationMs:        durationMs,
		Timestamp:         time.Now().UnixMilli(),
	}
}

func cloneUsage(usage *ai.Usage) *ai.Usage {
	if usage == nil {
		return nil
	}
	cloned := *usage
	return &cloned
}

func parseJSONArguments(raw json.RawMessage) (any, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, err
	}
	return value, nil
}

func allToolsTerminate(results []ToolResult) bool {
	if len(results) == 0 {
		return false
	}
	for _, result := range results {
		if !result.Terminate {
			return false
		}
	}
	return true
}
