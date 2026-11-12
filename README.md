# GOAT

**GOAT = Go Agent Toolkit**，一个供应商无关的 Go Agent SDK。它把模型协议、
消息模型和状态化 Agent 循环拆成可组合的包，方便你在上面构建 coding agent、
工作流 agent 或其他工具型 Agent。

仓库：<https://github.com/mrsibe/goat>

> 当前目标版本是 v0.4.0。v0.x 阶段允许破坏性 API，升级前请阅读
> [CHANGELOG.md](CHANGELOG.md) 的迁移说明。v1.0 之后遵循
> [STABILITY.md](STABILITY.md) 的兼容承诺；与 pi 的有意差异见
> [COMPATIBILITY.md](COMPATIBILITY.md)。

## 定位

GOAT 只负责 Agent SDK 的核心边界：

- `ai`：供应商无关的消息、内容块、工具声明、请求参数、流事件、usage 和重试工具。
- `ai/openai`：OpenAI Chat Completions 与 Responses 适配器，使用官方
  `openai-go` SDK。
- `ai/anthropic`：Anthropic Messages 适配器，使用官方
  `anthropic-sdk-go`。
- `agent`：状态化模型循环、工具执行、生命周期事件、输入队列和扩展钩子。
- `ai/faux`：可脚本化的测试模型，不依赖网络。

编码 Agent、文件工具、shell 工具、会话持久化和 CLI 在独立仓库
[`github.com/mrsibe/goding`](https://github.com/mrsibe/goding)。GOAT 主仓
不内置具体 Agent 产品。

## 安装

```bash
go get github.com/mrsibe/goat@v0.4.0
```

## 快速开始

```go
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/mrsibe/goat/agent"
	"github.com/mrsibe/goat/ai/openai"
)

func main() {
	model, err := openai.NewResponses(openai.Config{
		APIKey: os.Getenv("OPENAI_API_KEY"),
		Model:  "gpt-5",
	})
	if err != nil {
		panic(err)
	}

	assistant, err := agent.New(agent.Config{
		Model:        model,
		SystemPrompt: "你是一个简洁可靠的助手。",
	})
	if err != nil {
		panic(err)
	}

	assistant.Subscribe(func(event agent.Event) {
		if event.Type == agent.EventMessageDelta {
			fmt.Print(event.Text)
		}
	})

	result, err := assistant.Prompt(context.Background(), "你好")
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Answer)
}
```

## 工具

工具使用 `agent.ToolSpec` 声明模型可见的参数，以及 Go 侧的运行时提示：

```go
weather := agent.NewTextTool(agent.ToolSpec{
	Name:        "get_weather",
	Description: "查询城市天气",
	Parameters: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"city": map[string]any{"type": "string"},
		},
		"required": []string{"city"},
	},
}, func(ctx context.Context, call ai.ToolCall) (string, error) {
	var input struct {
		City string `json:"city"`
	}
	if err := json.Unmarshal(call.Arguments, &input); err != nil {
		return "", err
	}
	return "晴，25°C", nil
})
```

需要结构化结果、图片、details、usage 或 terminate 提示时，使用 `agent.NewTool`：

```go
type ToolResult struct {
	Content           []ai.ContentBlock
	Details           json.RawMessage
	StructuredContent json.RawMessage
	Usage             *ai.Usage
	IsError           bool
	Terminate         bool
}
```

需要流式进度时实现 `agent.StreamingTool`，或用 `agent.NewStreamingTool`：

```go
type StreamingTool interface {
	Tool
	ExecuteStream(ctx context.Context, call ai.ToolCall, update func(ToolResult)) (ToolResult, error)
}
```

工具错误、panic、未知工具、参数校验失败、`BeforeToolCall` 拒绝和非法工具结果
都会统一转换为 `IsError=true` 的 observation 交回模型。

## 模型协议

三种适配器实现同一个 `ai.Model` 接口，可以按业务替换：

```go
completions, _ := openai.NewCompletions(openai.Config{
	APIKey:  "YOUR_API_KEY",
	BaseURL: "https://api.deepseek.com",
	Model:   "deepseek-chat",
})

responses, _ := openai.NewResponses(openai.Config{
	APIKey: "YOUR_API_KEY",
	Model:  "gpt-5",
})

messages, _ := anthropic.New(anthropic.Config{
	APIKey: "YOUR_API_KEY",
	Model:  "claude-sonnet-5-5",
})
```

能力矩阵：

| 能力 | Chat Completions | Responses | Anthropic Messages |
| --- | --- | --- | --- |
| 文本流 | 支持 | 支持 | 支持 |
| 工具调用与工具结果 | 支持 | 支持 | 支持 |
| 用户图片输入 | 支持 | 支持 | 支持 |
| 工具结果图片 | 不支持 | 支持 | 支持 |
| thinking / 推理内容 | 不支持 | summary + encrypted content | thinking + signature / redacted |
| `tool_choice` | 支持 | 支持 | 支持 |
| `top_p` | 支持 | 支持 | 支持 |
| `reasoning_effort` | 支持 | 支持 | 不适用 |
| `thinking_budget_tokens` | 不适用 | 不适用 | 支持 |
| 缓存命中 usage | 支持 | 支持 | 支持 |
| 请求级缓存键 | `prompt_cache_key` | `prompt_cache_key` | 不适用 |

`Provider`、`ContextWindow`、`MaxOutputTokens` 是元数据，不会改变协议路由。
Anthropic 的 `max_tokens` 未指定时默认 4096，可由
`Config.MaxOutputTokens` 或 `RequestOptions.MaxTokens` 覆盖。

## 消息与内容

`ai.Message` 由角色和内容块组成。角色包括 `system`、`user`、`assistant`、
`tool`；内容块包括：

- `BlockText`：普通文本，带可选 `TextSignature`。
- `BlockThinking`：推理内容，带 `ThinkingSignature`、`ID` 和 `Redacted` 元数据。
- `BlockImage`：`ImageContent`，支持 base64 `Data + MediaType` 或 `URL`。
- `BlockToolCall`：模型请求的工具调用，带可选 `ThoughtSignature` 和 `Namespace`。

消息还携带 `Timestamp`、`Usage`、`StopReason`、`Provider`、`Model`、`API`、
`ErrorMessage` 和 `DurationMs` 等运行时元数据。工具结果消息额外支持
`Details`、`StructuredContent` 和 `Usage`。

工具结果使用 `ai.ToolResultMessage` 或 `ai.ToolTextMessage`：

```go
ai.ToolResultMessage("call_1", "screenshot", []ai.ContentBlock{
	{Type: ai.BlockText, Text: "页面预览"},
	{Type: ai.BlockImage, Image: &ai.ImageContent{
		Data:      base64PNG,
		MediaType: "image/png",
	}},
}, false)
```

辅助方法：

- `ai.TextMessage(role, text)`
- `ai.ToolMessage(callID, toolName, text)`
- `ai.ToolTextMessage(callID, toolName, text, isError)`
- `ai.ToolResultMessage(callID, toolName, content, isError)`
- `Message.Text()` / `Message.Thinking()` / `Message.ToolCalls()`
- `CloneMessage` / `CloneMessages` / `CloneContentBlocks` / `CloneRawMessage`

消息、工具调用和结果在 Agent 会话中会被深拷贝，调用方修改返回值不会污染内部
历史。

### Thinking 与图片

Anthropic thinking 的 `ThinkingSignature`、redacted payload，以及 OpenAI
Responses reasoning item 的 encrypted content 都会保留在历史中，并在下一轮
请求时回传。不要把 thinking 块当作普通文本拼接，否则会丢失签名。

OpenAI Responses 的 assistant message id / phase 会编码进 `TextSignature`，
function-call item id 保存在 `ai.ToolCall.ItemID`，两者都会在历史回放时还原。
`ai.ToolCall.ID` 始终是 provider 的 `call_id`，不会被拼接成复合字符串。

图片块要么设置 base64 `Data`，要么设置 `URL`，不能同时设置；base64 图片必须
提供 `MediaType`。如果不确定目标模型是否接受图片工具结果，先检查
`model.Info().SupportsToolResultImages`。

## 状态化 Agent

```go
assistant, err := agent.New(agent.Config{
	Model:             model,
	Tools:             tools,
	SystemPrompt:      "You are a helpful assistant.",
	MaxSteps:          0, // 0 表示不限制
	RequestOptions:    ai.RequestOptions{MaxTokens: 2048},
	ToolExecution:     agent.ToolExecutionParallel,
	ThinkingLevel:     agent.ThinkingOff,
	TransformMessages: transform,
	BeforeToolCall:    beforeTool,
	AfterToolCall:     afterTool,
})
```

主要字段：

- `Messages`：可选初始 transcript；如果首条不是 system，且设置了
  `SystemPrompt`，会自动插入。
- `MaxSteps`：可选安全阀，默认 0 不限制。耗尽返回 `agent.ErrMaxSteps`。
- `RequestOptions`：每次模型请求的默认采样、路由和 thinking 参数。
- `ToolExecution`：默认 `ToolExecutionParallel`；单工具声明
  `ExecutionModeSequential` 会强制整批顺序执行。
- `SteeringMode` / `FollowUpMode`：`QueueOneAtATime`（默认）或 `QueueAll`。
- `ThinkingLevel`：`off`、`minimal`、`low`、`medium`、`high`、`xhigh`、`max`。
- `ThinkingBudgets`：按 thinking level 提供 Anthropic 等协议的预算。
- `TransformMessages`：每次请求前改写消息列表，仅影响本次请求。
- `BeforeToolCall`：参数校验后、执行前策略钩子，可阻塞或终止工具批次。
- `AfterToolCall`：执行后、`tool_end` 前改写结果。
- `PrepareRequest` / `PrepareNextTurn` / `FinishTurn`：请求准备、下一轮准备和
  回合结束决策钩子。

核心方法：

```go
result, err := assistant.Prompt(ctx, "解释这个错误")
result, err = assistant.PromptMessages(ctx, message)
result, err = assistant.Continue(ctx)

assistant.Steer(ai.TextMessage(ai.RoleUser, "改用另一种方案"))
assistant.FollowUp(ai.TextMessage(ai.RoleUser, "完成后总结"))

assistant.Abort()
_ = assistant.WaitForIdle(ctx)
_ = assistant.Reset()

state := assistant.State()
_ = assistant.SetModel(nextModel)
_ = assistant.SetThinkingLevel(agent.ThinkingHigh)
_ = assistant.SetTools(nextTools)
assistant.SetMessages(messages)
```

`Agent` 同一时间只允许一个 run。运行中再次 `Prompt`、`Continue` 或 `Reset`
返回 `agent.ErrBusy`。`State()`、`Messages()` 和 `Result.Messages` 都返回深拷贝。
`Result.Messages` 是完整 transcript；`Result.NewMessages` 只包含本次 run 新增的
消息，语义与 pi 的 `newMessages` 对齐。

### 输入队列

- `Steer`：在当前 assistant turn 及其工具执行结束后注入，适合中途改变方向。
- `FollowUp`：在 agent 本来要停止时注入，适合追加任务。
- `PeekQueuedMessages`：预览下一次将被注入的消息，不消费。
- `ClearSteeringQueue` / `ClearFollowUpQueue` / `ClearAllQueues`：清理队列。

run 启动时也会检查 steering 队列，因此在 `Prompt` 前排队（或等待期间排队）的
steering 消息会进入首次模型请求。

`Continue` 对齐 pi 语义：transcript 不能为空，也不能只有 system 消息；最后一条
是 assistant 时，会先尝试 steering 队列，再尝试 follow-up 队列，否则返回
`agent.ErrCannotContinue`。

### 并行工具

并行模式下，所有工具调用先按模型顺序完成参数校验和 `BeforeToolCall` 预检，
再并发执行。`tool_end` 按完成顺序发出，但 tool result 消息始终按模型调用顺序
写入历史。

工具实现必须支持并发调用并且线程安全。工具流式 update 只在该次执行结束前有效；
执行结束后到达的 update 会被忽略。

### 消息变换

`TransformMessages` 每次模型调用前调用一次，收到的是历史的深拷贝。它返回的
消息只用于本次请求，不会写回 transcript：

```go
assistant, _ := agent.New(agent.Config{
	Model: model,
	TransformMessages: func(ctx context.Context, messages []ai.Message) ([]ai.Message, error) {
		if len(messages) <= 20 {
			return messages, nil
		}
		return append([]ai.Message{messages[0]}, messages[len(messages)-18:]...), nil
	},
})
```

SDK 不内置历史压缩。压缩策略、摘要调用和 token 预算由使用方决定。

## 事件

`Subscribe` 注册的回调同步执行，按注册顺序调用，并计入当前 run 的完成时间：

```text
agent_start
  turn_start
    message_start / message_delta / message_end
    thinking_start / thinking_delta / thinking_end
    tool_start / tool_update / tool_end
  turn_end
  agent_error   // 仅失败时，位于 agent_end 之前
agent_end
```

`Event.ToolResults` 在 `turn_end` 中一次给出本轮全部工具结果。`Event.ToolUpdate`
携带 `PartialResult`。`Event.DurationMs` 表示工具执行耗时。`Event.Err` 是原始
错误，`Event.Error` 是可序列化文本。事件回调阻塞会阻塞当前 run。

```go
assistant.Subscribe(func(event agent.Event) {
	switch event.Type {
	case agent.EventMessageDelta:
		fmt.Print(event.Text)
	case agent.EventThinkingDelta:
		log.Print(event.Thinking)
	case agent.EventToolStart:
		fmt.Printf("\ncall %s(%s)\n", event.ToolCall.Name, event.ToolCall.Arguments)
	case agent.EventToolEnd:
		fmt.Printf("result: %s\n", event.ToolResult.Text())
	}
})
```

provider 或流错误会追加带 `StopReasonError` 的 assistant 消息，完整发出事件，
并返回 `*agent.ModelError`。若失败发生在 assistant 消息产生之前（例如
`PrepareRequest` 或 `TransformMessages` 报错），Agent 会补发一条失败 assistant
消息和 `turn_end`。上下文取消返回 `context.Canceled`；provider 主动 abort 但
没有取消 context 时返回 `agent.ErrAborted`。

## 重试

Agent 不自动重试 provider 请求。`ai` 提供可复用的重试工具：

```go
err := ai.Retry(ctx, ai.RetryPolicy{
	Enabled:    true,
	MaxRetries: 3,
	BaseDelay:  time.Second,
	MaxDelay:   30 * time.Second,
}, func(ctx context.Context) error {
	_, err := model.Stream(ctx, request)
	return err
})
```

`ai.IsRetryable` 会识别 `APIError.Retryable`、HTTP 429 和 5xx；context 取消不会
重试。

## 工具参数校验

工具参数会在执行前按 `ToolSpec.Parameters` 校验。校验器覆盖对象、
`required`、常见 JSON 类型、数组、枚举、数值和字符串限制、
`additionalProperties`、本地 `$ref` 以及 `allOf/anyOf/oneOf/not`；其他供应商
专属 Schema 关键字由工具自身负责。需要兼容旧参数时使用
`ToolSpec.PrepareArguments`。

## 测试

`ai/faux` 提供脚本化模型，适合在不访问网络的情况下测试 Agent、工具循环和事件：

```go
model := faux.New(
	faux.Turn{Events: []ai.Event{
		{Type: ai.EventToolCallDelta, ContentIndex: 0, ToolCallDelta: &ai.ToolCallDelta{
			ID: "call_1", Name: "echo", ArgumentsDelta: `{"text":"hi"}`,
		}},
		{Type: ai.EventDone, StopReason: ai.StopReasonToolCalls},
	}},
	faux.Turn{Events: []ai.Event{
		{Type: ai.EventTextDelta, TextDelta: "done"},
		{Type: ai.EventDone, StopReason: ai.StopReasonStop},
	}},
)
```

`faux.Model.Requests` 记录每次请求，包含消息、工具和 `RequestOptions`。

## 迁移

### v0.3.0 的主要破坏性变化

1. `agent.Agent` 从不可变 Agent + `Session` 改为状态化 Agent。删除
   `NewSession`、`NewSessionWithHistory`、`Session` 和 `WithOnEvent`。
2. `Run` 保留为 `Prompt` 的便捷别名；事件改用 `Agent.Subscribe`。
3. `agent.ToolSpec` 取代直接传入 `ai.ToolSpec`；新增 `Label`、
   `OutputSchema`、`ExecutionMode` 和 `PrepareArguments`。
4. `BeforeToolCall` / `AfterToolCall` 改为接收上下文结构体并返回决策/覆盖结果。
5. `MaxSteps` 默认 0 不限制；`ToolExecution` 默认并行。
6. 输出长度截断且存在工具调用时不再直接失败，而是写入错误 observation 后继续。
7. `ErrModelStop` 和 `ErrOutputLimit` 删除；provider 错误使用 `ModelError`，
   长度停止通过 `Result.StopReason` 表达。
8. `ContentBlock.Signature` 改名为 `ThinkingSignature`。
9. `ai.ToolCall` 增加 `ThoughtSignature`、`Namespace`；`ai.Message` 增加运行时
   元数据、details、structured content 和 usage。
10. `coding`、`codingagent` 和 `cmd/goding` 继续留在独立仓库 `goding`。

### v0.4.0 的变化

1. `Result` 新增 `NewMessages`；`Messages` 继续表示完整 transcript。
2. OpenAI Responses 适配器保留 assistant message id、phase、text signature 和
   function-call item id；`ai.ToolCall` 新增 `ItemID`。
3. 新增 `agent` 语义一致性矩阵、`ai/internal/modeltest` 适配器契约测试和导出
   API 快照检查。
4. 新增 `STABILITY.md` 和 `COMPATIBILITY.md`，明确 v1.0 冻结范围和 pi 差异。
5. v0.4 不新增 provider、TUI、durable、MCP 或 coding-agent 功能。

## 边界

GOAT SDK 不包含：

- 会话持久化和恢复。
- 自动历史压缩、摘要和 token 预算管理。
- MCP、skills、子 Agent、RPC 或 TUI。
- OAuth、模型目录、路由和模型能力数据库。
- 图片生成、结构化分类器和 deferred response。
- 编码工具、文件/命令沙箱和审批策略。

这些属于具体 Agent 产品能力，应在 SDK 之上实现。`TransformMessages`、
`BeforeToolCall`、`AfterToolCall`、`PrepareRequest`、`PrepareNextTurn`、
`FinishTurn` 和自定义 `ai.Model` 是预留的扩展点。

## 开发

```bash
gofmt -w .
GOCACHE=/tmp/goat-gocache GOFLAGS=-buildvcs=false go test ./...
GOCACHE=/tmp/goat-gocache GOFLAGS=-buildvcs=false go test -race ./...
GOCACHE=/tmp/goat-gocache GOFLAGS=-buildvcs=false go vet ./...
GOCACHE=/tmp/goat-gocache GOFLAGS=-buildvcs=false go build ./...
bash scripts/api-snapshot.sh
```
