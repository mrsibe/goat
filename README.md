# GOAT

**GOAT = Go Agent Toolkit**，一个供应商无关的 Go Agent SDK。它把模型协议、
消息模型和 Agent 循环拆成可组合的包，方便你在上面构建自己的 coding agent、
工作流 agent 或其他工具型 Agent。

仓库：<https://github.com/mrsibe/goat>

> 当前目标版本是 v0.2.0。v0.x 阶段允许破坏性 API，升级前请阅读
> [CHANGELOG.md](CHANGELOG.md) 的迁移说明。

## 定位

GOAT 只负责 Agent SDK 的核心边界：

- `ai`：供应商无关的消息、内容块、工具声明、请求参数、流事件和 usage。
- `ai/openai`：OpenAI Chat Completions 与 Responses 适配器，使用官方
  `openai-go` SDK。
- `ai/anthropic`：Anthropic Messages 适配器，使用官方
  `anthropic-sdk-go`。
- `agent`：模型循环、工具执行、生命周期事件、会话历史和请求变换。
- `ai/faux`：可脚本化的测试模型，不依赖网络。

编码 Agent、文件工具、shell 工具、会话持久化和 CLI 已经迁移到独立仓库
[`github.com/mrsibe/goding`](https://github.com/mrsibe/goding)。GOAT 主仓
不再内置任何具体 Agent 产品。

## 安装

```bash
go get github.com/mrsibe/goat@v0.2.0
```

## 快速开始

```go
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/mrsibe/goat/agent"
	"github.com/mrsibe/goat/ai"
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

	result, err := assistant.Run(context.Background(), "你好")
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Answer)
}
```

需要工具时，用 `agent.NewTextTool` 或 `agent.NewTool` 注册：

```go
weather := agent.NewTextTool(ai.ToolSpec{
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

assistant, err := agent.New(agent.Config{
	Model: model,
	Tools: []agent.Tool{weather},
})
```

`json.Unmarshal` 的错误不需要转换成特殊结果：Agent 会把工具返回的错误、
panic、未知工具、参数校验失败和非法工具结果统一转换为错误 observation 交回
模型。

## 模型协议

三种适配器实现同一个 `ai.Model` 接口，可以按 run 或按业务替换：

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
`Config.MaxOutputTokens` 或 `RequestOptions.MaxTokens` 覆盖。OpenAI Chat
Completions 适用于 OpenAI、DeepSeek、OpenRouter、Ollama 等兼容服务；
`APIKey` 可以为空以连接本地无鉴权服务。

`ModelInfo` 的 `SupportsImages`、`SupportsToolResultImages`、
`SupportsThinking` 表示适配器协议能力。具体模型是否支持对应能力仍由模型本身
决定。

## 消息与内容

`ai.Message` 由角色和内容块组成。角色包括 `system`、`user`、`assistant`、
`tool`；内容块包括：

- `BlockText`：普通文本。
- `BlockThinking`：推理内容，带 `Signature`、`ID` 和 `Redacted` 元数据。
- `BlockImage`：`ImageContent`，支持 base64 `Data + MediaType` 或 `URL`。
- `BlockToolCall`：模型请求的工具调用。

工具结果使用 `ai.ToolResultMessage` 或 `ai.ToolTextMessage`。工具结果可以包含
多个文本和图片块，并通过 `IsError` 保留结构化错误状态：

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
- `CloneMessage` / `CloneMessages` / `CloneContentBlocks`

消息、工具调用和结果在 Agent 会话中会被深拷贝，调用方修改返回值不会污染内部
历史。

### Thinking 与图片

Anthropic thinking 的 `signature`、redacted payload，以及 OpenAI Responses
reasoning item 的 encrypted content 都会保留在历史中，并在下一轮请求时回传。
不要把 thinking 块当作普通文本拼接，否则会丢失签名。

图片块要么设置 base64 `Data`，要么设置 `URL`，不能同时设置；base64 图片必须
提供 `MediaType`。如果不确定目标模型是否接受图片工具结果，先检查
`model.Info().SupportsToolResultImages`。

## Agent 运行时

```go
assistant, err := agent.New(agent.Config{
	Model:             model,
	Tools:             tools,
	SystemPrompt:      "You are a helpful assistant.",
	MaxSteps:          10,
	RequestOptions:    ai.RequestOptions{MaxTokens: 2048},
	ToolExecution:     agent.ToolExecutionParallel,
	TransformMessages: transform,
	BeforeToolCall:    beforeTool,
	AfterToolCall:     afterTool,
})
```

主要字段：

- `MaxSteps`：最多模型轮数，默认 10。耗尽返回 `agent.ErrMaxSteps`。
- `RequestOptions`：每次模型请求的默认采样、路由和 thinking 参数。
- `ToolExecution`：`ToolExecutionSequential`（默认）或
  `ToolExecutionParallel`。
- `TransformMessages`：每次请求前改写消息列表，适合裁剪历史、注入上下文或
  做自定义压缩。
- `BeforeToolCall`：工具执行前策略钩子。返回错误会阻止执行，并把错误文本作为
  错误 observation 交给模型。
- `AfterToolCall`：工具执行后改写观察结果。

`Agent` 是不可变的，可被多个 `Session` 并发使用。单个 `Session` 不应并发调用
`Run`。`Session.Messages()` 返回深拷贝；`NewSessionWithHistory` 会忽略历史中
的系统消息，并使用 Agent 当前配置的 system prompt。

### 并行工具

并行模式下，所有工具调用先按模型顺序完成参数校验和 `BeforeToolCall` 预检，
再并发执行；最终结果仍按模型给出的调用顺序写入历史。因此：

- 工具实现必须支持并发调用并且线程安全。
- 事件中的 `tool_start` 顺序是模型顺序，`tool_end` 顺序是完成顺序。
- 结果消息顺序始终与 assistant 消息中的 tool call 顺序一致。

### 请求级覆盖

`Run` 可以临时覆盖 Agent 默认请求参数，未设置字段保留默认值：

```go
temperature := 0.2
topP := 0.9
result, err := assistant.Run(ctx, "解释这个错误", agent.WithRequestOptions(
	ai.RequestOptions{
		Temperature:          &temperature,
		TopP:                 &topP,
		MaxTokens:            4096,
		ReasoningEffort:      "medium",
		ThinkingBudgetTokens: 2048,
		ToolChoice:           ai.ToolChoice{Mode: ai.ToolChoiceAuto},
	},
))
```

`ToolChoice.Mode` 为空表示使用供应商默认值；`ToolChoiceNamed` 需要同时设置
`Name`。不同协议会忽略不适用于自己的字段。

### 消息变换

`TransformMessages` 每次模型调用前调用一次，收到的是历史的深拷贝。它返回的
消息只用于本次请求，不会写回 Session：

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

`Run` 的 `agent.WithOnEvent` 回调同步执行。事件类型：

```text
agent_start
  turn_start
    message_start / message_delta / message_end
    thinking_start / thinking_delta / thinking_end
    tool_start / tool_end
  turn_end
agent_end
agent_error
```

`Event.ToolResults` 在 `turn_end` 中一次给出本轮全部工具结果。`Event.Err` 是
原始错误，`Event.Error` 是可序列化文本。事件回调阻塞会阻塞当前 run。

```go
result, err := assistant.Run(ctx, "查天气", agent.WithOnEvent(func(event agent.Event) {
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
}))
```

## 工具

工具实现 `agent.Tool`：

```go
type Tool interface {
	Spec() ai.ToolSpec
	Execute(ctx context.Context, call ai.ToolCall) (ToolResult, error)
}
```

函数适配器：

- `agent.NewTextTool(spec, fn)`：函数返回文本。
- `agent.NewTool(spec, fn)`：函数返回 `agent.ToolResult`，可包含文本和图片。
- `agent.TextResult(text)` / `agent.ErrorResult(text)`：常用结果构造器。
- `agent.ToolResult.Text()`：拼接结果中的文本块。

执行语义：

- 工具错误和 panic 会变成 `IsError=true` 的 observation。
- 未知工具、参数校验失败、`BeforeToolCall` 拒绝和非法结果也会变成错误
  observation。
- 模型输出因 token 上限被截断时，不会执行任何工具调用；Agent 会为每个调用
  写入一条错误 observation，并在下一轮要求模型重新发起完整调用。
- `stop_reason=error`、流错误和上下文取消会终止 run，但保留已经产生的
  assistant partial 消息和 usage。

工具参数会在执行前按 `ToolSpec.Parameters` 校验。校验器覆盖对象、
`required`、常见 JSON 类型、数组、枚举、数值和字符串限制、
`additionalProperties`、本地 `$ref` 以及 `allOf/anyOf/oneOf/not`；其他供应商
专属 Schema 关键字由工具自身负责。

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

## v0.1 迁移

v0.2.0 的主要破坏性变化：

1. `ai.Model` 新增 `Info() ai.ModelInfo`。自定义模型需要实现 `Info`。
2. `ai.Request` 新增 `Options ai.RequestOptions`，支持采样、tool choice、
   reasoning 和 thinking 参数。
3. 流事件分拆为 `EventTextStart/Delta/End`、
   `EventThinkingStart/Delta/End`、
   `EventToolCallStart/Delta/End`、`EventUsage`、`EventDone`、`EventError`。
   `ToolCallDelta` 不再放在 `Event.ToolCall`，而是
   `Event.ToolCallDelta`，块位置使用 `Event.ContentIndex`。
4. `ai.ToolMessage` 从 `(callID, text)` 变为
   `(callID, toolName, text)`；推荐使用 `ai.ToolTextMessage`。
5. `agent.ToolFunc` 返回 `agent.ToolResult`，不再是 `string`；旧的文本工具
   改用 `agent.NewTextTool`。
6. `agent.ToolResult` 的 `CallID` 字段删除，`Content` 从 `string` 变为
   `[]ai.ContentBlock`；使用 `agent.TextResult`、`agent.ErrorResult` 或
   `ToolResult.Text()` 迁移。
7. `internal/faux` 移到公开包 `ai/faux`。
8. `coding`、`codingagent` 和 `cmd/goding` 从 SDK 主仓删除，迁移到
   `github.com/mrsibe/goding`。

## 边界

GOAT SDK 不包含：

- 会话持久化和恢复。
- 自动历史压缩、摘要和 token 预算管理。
- MCP、skills、子 Agent、RPC 或 TUI。
- OAuth、模型目录、路由和模型能力数据库。
- 编码工具、文件/命令沙箱和审批策略。

这些属于具体 Agent 产品能力，应在 SDK 之上实现。`TransformMessages`、
`BeforeToolCall`、`AfterToolCall` 和自定义 `ai.Model` 是预留的扩展点。

## 开发

```bash
gofmt -w .
GOCACHE=/tmp/goat-gocache GOFLAGS=-buildvcs=false go test ./...
GOCACHE=/tmp/goat-gocache GOFLAGS=-buildvcs=false go test -race ./...
GOCACHE=/tmp/goat-gocache GOFLAGS=-buildvcs=false go vet ./...
GOCACHE=/tmp/goat-gocache GOFLAGS=-buildvcs=false go build ./...
```
