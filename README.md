# GOAT

[![Go Reference](https://pkg.go.dev/badge/github.com/mrsibe/goat.svg)](https://pkg.go.dev/github.com/mrsibe/goat)
[![CI](https://github.com/mrsibe/goat/actions/workflows/ci.yml/badge.svg)](https://github.com/mrsibe/goat/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**GOAT（Go Agent Toolkit）是一个供应商无关的 Go Agent SDK。**

它把模型协议、消息模型、流式事件和状态化 Agent 循环拆成可组合的包，让你用同一套
Agent 代码接入 OpenAI、Anthropic 或自定义模型，并在上面构建自己的工作流 Agent、
编码 Agent 或其他工具型产品。

GOAT 只做核心运行时。编码工具、文件/命令沙箱、会话持久化、MCP、TUI 和 CLI 属于
产品层，放在独立仓库 [`goding`](https://github.com/mrsibe/goding)。

> 当前版本：**v0.4.1**。v0.x 阶段允许破坏性变更，升级前请阅读
> [CHANGELOG.md](CHANGELOG.md)。v1.0 之后的兼容承诺见
> [STABILITY.md](STABILITY.md)，与 pi 的有意差异见
> [COMPATIBILITY.md](COMPATIBILITY.md)。

## 目录

- [为什么用 GOAT](#为什么用-goat)
- [安装](#安装)
- [快速开始](#快速开始)
- [包结构](#包结构)
- [模型适配器](#模型适配器)
- [工具](#工具)
- [Agent 运行时](#agent-运行时)
- [事件](#事件)
- [消息与内容](#消息与内容)
- [重试与参数校验](#重试与参数校验)
- [测试](#测试)
- [与 pi 的关系](#与-pi-的关系)
- [版本与稳定性](#版本与稳定性)
- [开发与发布](#开发与发布)
- [License](#license)

## 为什么用 GOAT

- **换模型不改 Agent 代码。** `ai.Model` 统一了请求、流事件、工具调用、thinking
  和 usage；OpenAI Chat Completions、OpenAI Responses、Anthropic Messages 和
  自定义模型可以按需替换。
- **流式优先。** 文本、thinking 和工具参数都有增量事件。`message_update` 携带
  pi 兼容的 `AssistantMessageEvent`，同时保留 `message_delta`、`thinking_*`
  等扁平事件，方便不同复杂度的消费者。
- **完整的工具运行时。** 参数校验、并行/顺序执行、流式进度、执行前后钩子、
  取消和统一错误 observation 都在 SDK 内处理。
- **状态化 Agent。** transcript、steering/follow-up 队列、abort、reset、
  thinking level 和运行时状态切换都是内置能力。
- **为测试而生。** `ai/faux` 提供脚本化模型，不访问网络即可测试 Agent 循环、
  工具执行和事件序列。
- **边界清晰。** SDK 不绑定具体产品形态，也不会替你决定持久化、压缩、审批或
  UI 策略。

## 安装

```bash
go get github.com/mrsibe/goat@v0.4.1
```

需要 Go 1.27.1 或更高版本。

## 快速开始

下面的例子使用 OpenAI Responses API，并在终端实时打印文本增量：

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

	result, err := assistant.Prompt(
		context.Background(),
		"用三句话解释 Go 的 goroutine。",
	)
	if err != nil {
		panic(err)
	}

	fmt.Printf("\n\nstop=%s tokens=%d\n", result.StopReason, result.Usage.TotalTokens)
}
```

`Prompt` 返回的 `Result` 包含完整答案、transcript、本次新增消息、步数、停止原因
和累计 usage。`Result.Messages` 是完整历史，`Result.NewMessages` 只包含本次 run
产生的消息。

## 包结构

| 包 | 职责 |
| --- | --- |
| `ai` | 供应商无关的消息、内容块、工具声明、请求参数、流事件、usage 和重试工具。 |
| `ai/openai` | OpenAI Chat Completions 与 Responses 适配器，基于官方 `openai-go` SDK。 |
| `ai/anthropic` | Anthropic Messages 适配器，基于官方 `anthropic-sdk-go`。 |
| `agent` | 状态化模型循环、工具执行、生命周期事件、输入队列和扩展钩子。 |
| `ai/faux` | 可脚本化的测试模型，不依赖网络。 |
| `examples/weather` | 一个最小可运行的天气工具示例。 |

## 模型适配器

所有适配器都实现同一个 `ai.Model` 接口：

```go
type Model interface {
	Info() ModelInfo
	Stream(ctx context.Context, request Request) (Stream, error)
}
```

选择适配器：

```go
// OpenAI Chat Completions（也兼容 DeepSeek 等 OpenAI 兼容服务）
completions, err := openai.NewCompletions(openai.Config{
	APIKey:  os.Getenv("OPENAI_API_KEY"),
	BaseURL: "https://api.deepseek.com",
	Model:   "deepseek-chat",
})

// OpenAI Responses
responses, err := openai.NewResponses(openai.Config{
	APIKey: os.Getenv("OPENAI_API_KEY"),
	Model:  "gpt-5",
})

// Anthropic Messages
messages, err := anthropic.New(anthropic.Config{
	APIKey: os.Getenv("ANTHROPIC_API_KEY"),
	Model:  "claude-sonnet-5-5",
})
```

`openai.New` 是 `openai.NewCompletions` 的兼容别名。`Provider`、`ContextWindow`、
`MaxOutputTokens` 是元数据，不会改变协议路由。Anthropic 未指定 `max_tokens` 时
默认 4096，可由 `Config.MaxOutputTokens` 或 `RequestOptions.MaxTokens` 覆盖。

### 能力矩阵

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

模型能力仍受具体模型限制。需要判断是否接受图片工具结果时，检查
`model.Info().SupportsToolResultImages`。

## 工具

### 文本工具

工具用 `agent.ToolSpec` 声明模型可见的参数，用函数实现执行逻辑：

```go
weather := agent.NewTextTool(agent.ToolSpec{
	Name:        "get_weather",
	Description: "查询指定城市的天气",
	Parameters: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"city": map[string]any{
				"type":        "string",
				"description": "城市名",
			},
		},
		"required":             []string{"city"},
		"additionalProperties": false,
	},
}, func(ctx context.Context, call ai.ToolCall) (string, error) {
	var input struct {
		City string `json:"city"`
	}
	if err := json.Unmarshal(call.Arguments, &input); err != nil {
		return "", err
	}
	return fmt.Sprintf("%s：晴，25°C", input.City), nil
})
```

`ToolSpec` 的字段：

| 字段 | 用途 |
| --- | --- |
| `Name` / `Description` / `Parameters` | 暴露给模型的工具声明。 |
| `Label` | 给 UI 或日志使用的可读名称。 |
| `OutputSchema` | 结构化输出元数据，供上层产品使用，不会自动发送给模型。 |
| `ExecutionMode` | 单工具覆盖并行/顺序执行策略。 |
| `PrepareArguments` | 校验前改写参数，用于兼容旧参数格式。 |

### 结构化结果与流式进度

需要返回图片、details、结构化内容、usage 或 terminate 提示时，使用
`agent.NewTool` 和 `agent.ToolResult`：

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

需要流式进度时实现 `agent.StreamingTool`，或使用 `agent.NewStreamingTool`：

```go
type StreamingTool interface {
	Tool
	ExecuteStream(
		ctx context.Context,
		call ai.ToolCall,
		update func(ToolResult),
	) (ToolResult, error)
}
```

执行期间的 update 会以 `tool_update` 事件发出，并携带 `PartialResult`。

### 执行语义

- 参数在执行前按 `ToolSpec.Parameters` 校验。
- 并行模式会先按模型顺序完成全部参数校验和 `BeforeToolCall` 预检，再并发执行。
- `tool_end` 按完成顺序发出，但 tool result 消息始终按模型调用顺序写入历史。
- 未知工具、参数错误、工具返回 error、panic、非法结果、钩子拒绝和 context
  取消都会转换成 `IsError=true` 的 observation 交回模型。
- context 取消后不会再启动新的工具执行；已取消的调用会得到
  `Operation aborted` observation。
- 并行工具必须支持并发调用并且线程安全。

## Agent 运行时

```go
assistant, err := agent.New(agent.Config{
	Model:             model,
	Tools:             tools,
	SystemPrompt:      "You are a helpful assistant.",
	MaxSteps:          0,
	RequestOptions:    ai.RequestOptions{MaxTokens: 2048},
	ToolExecution:     agent.ToolExecutionParallel,
	ThinkingLevel:     agent.ThinkingOff,
	TransformMessages: transform,
	BeforeToolCall:    beforeTool,
	AfterToolCall:     afterTool,
})
```

主要配置：

| 字段 | 说明 |
| --- | --- |
| `Model` | 必填，实现 `ai.Model`。 |
| `SystemPrompt` | 初始 system prompt；如果 `Messages` 首条不是 system，会自动插入。 |
| `Messages` | 可选初始 transcript，会被深拷贝。 |
| `Tools` | 工具列表；名称必须唯一。 |
| `MaxSteps` | 最大模型轮数，`0` 表示不限制；耗尽返回 `agent.ErrMaxSteps`。 |
| `RequestOptions` | 每次模型请求的默认采样、路由和 thinking 参数。 |
| `ToolExecution` | 默认 `ToolExecutionParallel`；单工具可声明顺序执行。 |
| `SteeringMode` / `FollowUpMode` | `QueueOneAtATime`（默认）或 `QueueAll`。 |
| `SessionID` | 转发给 provider 的缓存提示。 |
| `ThinkingLevel` | `off`、`minimal`、`low`、`medium`、`high`、`xhigh`、`max`。 |
| `ThinkingBudgets` | 按 thinking level 提供 Anthropic 等协议的预算。 |
| `TransformMessages` | 每次请求前改写消息列表，仅影响本次请求。 |
| `BeforeToolCall` / `AfterToolCall` | 工具执行前后的策略与结果改写。 |
| `PrepareRequest` / `PrepareNextTurn` / `FinishTurn` | 请求、下一轮和回合结束钩子。 |

### 核心方法

| 方法 | 用途 |
| --- | --- |
| `Prompt(ctx, text, images...)` | 追加一条 user 消息并运行 Agent。 |
| `PromptMessages(ctx, messages...)` | 追加任意消息并运行 Agent。 |
| `Continue(ctx)` | 从当前 transcript 继续；最后一条是 assistant 时尝试使用队列消息。 |
| `Steer(message)` | 在当前 assistant turn 和工具执行结束后注入消息。 |
| `FollowUp(message)` | 在 Agent 本来要停止时注入消息。 |
| `Abort()` | 取消当前 run。 |
| `WaitForIdle(ctx)` | 等待当前 run 结束。 |
| `Reset()` | 清空 transcript 并恢复初始 system prompt。 |
| `State()` / `Messages()` | 返回当前状态和 transcript 的深拷贝。 |
| `SetModel` / `SetTools` / `SetThinkingLevel` | 运行时切换模型、工具和 thinking level。 |
| `SetMessages(messages)` | 替换 transcript。 |
| `PeekQueuedMessages` | 预览下一次会注入的队列消息，不消费。 |
| `ClearSteeringQueue` / `ClearFollowUpQueue` / `ClearAllQueues` | 清理输入队列。 |

同一个 `Agent` 同时只允许一个 run。运行中再次 `Prompt`、`Continue` 或 `Reset`
会返回 `agent.ErrBusy`。

### 输入队列

- `Steer` 适合中途改变方向，例如“改用另一种方案”。
- `FollowUp` 适合追加任务，例如“完成后顺便写测试”。
- `QueueOneAtATime` 每次只注入一条，`QueueAll` 一次注入全部。
- run 启动时也会检查 steering 队列，因此在 `Prompt` 前排队的消息会进入首次模型请求。
- `Continue` 在 transcript 为空或只有 system 消息时返回 `agent.ErrNoMessages`；
  最后一条是 assistant 且队列为空时返回 `agent.ErrCannotContinue`。

### 扩展钩子

| 钩子 | 时机 | 典型用途 |
| --- | --- | --- |
| `TransformMessages` | 每次模型请求前 | 裁剪、压缩或重排本次请求的消息。 |
| `PrepareRequest` | 每次 provider 调用前 | 切换模型、工具上下文或 thinking level。 |
| `BeforeToolCall` | 参数校验后、执行前 | 审批、拦截、终止工具批次。 |
| `AfterToolCall` | 工具执行后、`tool_end` 前 | 改写 content、details、usage 或错误状态。 |
| `PrepareNextTurn` | 一轮结束后 | 为下一轮更新上下文或模型状态。 |
| `FinishTurn` | `turn_end` 前 | 决定继续还是结束。 |

`TransformMessages` 收到的是历史深拷贝，返回的消息只用于本次请求，不会写回
transcript。SDK 不内置历史压缩；压缩策略、摘要调用和 token 预算由使用方决定。

## 事件

`Subscribe` 注册的回调同步执行，按注册顺序调用，并计入当前 run 的完成时间。
事件顺序大致如下：

```text
agent_start
  turn_start
    message_start
      message_update / message_delta
      thinking_start / thinking_delta / thinking_end
    message_end
    tool_start / tool_update / tool_end
  turn_end
  ...                 # 有工具结果时继续下一轮
agent_error           # 仅失败时，位于 agent_end 之前
agent_end
```

`Event.Step` 从 1 开始。主要事件负载：

| 事件 | 关键负载 |
| --- | --- |
| `EventMessageUpdate` | `Message` 和原始 `AssistantMessageEvent`。 |
| `EventMessageDelta` | `Text`，扁平文本增量。 |
| `EventThinkingStart` / `Delta` / `End` | `Thinking`，扁平 thinking 事件。 |
| `EventToolStart` | `ToolCall`。 |
| `EventToolUpdate` | `PartialResult`。 |
| `EventToolEnd` | `ToolResult` 和 `DurationMs`。 |
| `EventTurnEnd` | `ToolResults`，本轮全部工具结果。 |
| `EventAgentEnd` | `Result`。 |
| `EventAgentError` | `Err`（原始错误）和 `Error`（可序列化文本）。 |

`message_update` 是 pi 兼容入口，适合需要工具参数增量的消费者；`message_delta`
和 `thinking_*` 是更简单的扁平事件，两者会同时发出。

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

### 错误语义

- provider 或流错误会追加带 `StopReasonError` 的 assistant 消息，完整发出事件，
  并返回 `*agent.ModelError`。
- context 取消返回 `context.Canceled` 或 `context.DeadlineExceeded`。
- provider 主动 abort 但没有取消 context 时返回 `agent.ErrAborted`。
- 达到 `MaxSteps` 返回 `agent.ErrMaxSteps`。
- 并发 run 返回 `agent.ErrBusy`。

## 消息与内容

`ai.Message` 由角色和内容块组成：

| 角色 | 用途 |
| --- | --- |
| `ai.RoleSystem` | system prompt。 |
| `ai.RoleUser` | 用户输入。 |
| `ai.RoleAssistant` | 模型输出，可包含文本、thinking 和工具调用。 |
| `ai.RoleTool` | 工具结果，通过 `ToolCallID` 关联请求。 |

内容块：

| 类型 | 说明 |
| --- | --- |
| `ai.BlockText` | 普通文本，可带 `TextSignature`。 |
| `ai.BlockThinking` | 推理内容，可带 `ThinkingSignature`、`ID` 和 `Redacted`。 |
| `ai.BlockImage` | base64 图片或 URL 图片。 |
| `ai.BlockToolCall` | 模型请求的工具调用，可带 `ThoughtSignature` 和 `Namespace`。 |

常用辅助函数：

```go
ai.TextMessage(role, text)
ai.ToolMessage(callID, toolName, text)
ai.ToolTextMessage(callID, toolName, text, isError)
ai.ToolResultMessage(callID, toolName, content, isError)

message.Text()
message.Thinking()
message.ToolCalls()
```

消息还携带 `Timestamp`、`Usage`、`StopReason`、`Provider`、`Model`、`API`、
`ErrorMessage` 和 `DurationMs` 等运行时元数据。消息、工具调用和结果在 Agent
会话中会被深拷贝，调用方修改返回值不会污染内部历史。

### Thinking 与签名

Anthropic thinking 的 `ThinkingSignature`、redacted payload，以及 OpenAI
Responses reasoning item 的 encrypted content 都会保留在历史中，并在下一轮请求
时回传。不要把 thinking 块当作普通文本拼接，否则会丢失签名。

OpenAI Responses 的 assistant message id / phase 会编码进 `TextSignature`，
function-call item id 保存在 `ai.ToolCall.ItemID`，两者都会在历史回放时还原。
`ai.ToolCall.ID` 始终是 provider 的 `call_id`，不会被拼接成复合字符串。

### 图片

图片块要么设置 base64 `Data`，要么设置 `URL`，不能同时设置；base64 图片必须
提供 `MediaType`。

```go
ai.ToolResultMessage("call_1", "screenshot", []ai.ContentBlock{
	{Type: ai.BlockText, Text: "页面预览"},
	{Type: ai.BlockImage, Image: &ai.ImageContent{
		Data:      base64PNG,
		MediaType: "image/png",
	}},
}, false)
```

## 重试与参数校验

Agent 不会自动重试 provider 请求。`ai` 提供可复用的重试工具：

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

工具参数校验器覆盖对象、`required`、常见 JSON 类型、数组、枚举、数值和字符串
限制、`additionalProperties`、本地 `$ref` 以及 `allOf/anyOf/oneOf/not`。其他
供应商专属 Schema 关键字由工具自身负责。

## 测试

`ai/faux` 提供脚本化模型，适合在不访问网络的情况下测试 Agent、工具循环和事件：

```go
model := faux.New(
	faux.Turn{Events: []ai.Event{
		{Type: ai.EventTextDelta, TextDelta: "hello"},
		{Type: ai.EventDone, StopReason: ai.StopReasonStop},
	}},
)
```

`faux.Model.Requests` 记录每次请求，包含消息、工具和 `RequestOptions`，可以用来
断言 Agent 实际发送给模型的内容。

## 与 pi 的关系

GOAT 的目标是把 pi 的核心 Agent 能力带到 Go，而不是复刻 pi coding agent：

- 模型协议、消息模型、流事件、工具循环、队列和钩子与 pi 的核心语义对齐。
- `message_update` 携带原始 `AssistantMessageEvent`，同时提供扁平事件。
- coding 工具、CLI/TUI、durable sessions、MCP、codemode 和扩展打包留在
  [`goding`](https://github.com/mrsibe/goding)。
- 已知差异和有意取舍记录在 [COMPATIBILITY.md](COMPATIBILITY.md)，包括 reset
  基线、动态工具声明、订阅者 signal、参数 coercion 和 provider 覆盖。

## 版本与稳定性

当前版本是 **v0.4.1**。v0.x 阶段允许破坏性 API，完整迁移说明见
[CHANGELOG.md](CHANGELOG.md)。

v1.0 的冻结范围、兼容承诺和弃用流程见 [STABILITY.md](STABILITY.md)。

## 开发与发布

### 本地验证

```bash
gofmt -w .
go test ./...
go test -race ./...
go vet ./...
go build ./...
bash scripts/api-snapshot.sh
```

`scripts/api-snapshot.sh` 会对比导出的公开 API 和 `testdata/api` 快照。修改公开
API 后，运行 `bash scripts/api-snapshot.sh -update` 并审查差异。

### CI

`.github/workflows/ci.yml` 在 `main` 分支 push、pull request 和 release workflow
调用时运行：

- `gofmt` 检查。
- `go mod tidy` 后确认 `go.mod` / `go.sum` 无差异。
- `go build`、`go vet` 和 API 快照检查。
- `go test -count=1` 和 `go test -race -count=1`。

### 发布

`.github/workflows/release.yml` 在推送 `v*` tag 时触发。它会先复用 CI 完成全部
验证，再用 GitHub CLI 创建 Release：

```bash
git tag -a v0.4.2 -m "GOAT v0.4.2"
git push origin main
git push origin v0.4.2
```

验证发布状态：

```bash
gh run list --workflow release.yml
gh release view v0.4.2
```

## License

[MIT](LICENSE)
