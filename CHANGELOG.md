# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 和
[语义化版本](https://semver.org/lang/zh-CN/)。v0.x 阶段允许破坏性变更。

## [Unreleased]

## [0.5.0] - 2026-11-12

### Added

- `ai/azure`：Azure OpenAI v1 适配器，提供 `NewCompletions` 和 `NewResponses`。
  自动解析资源端点、注入 `api-key` 头和 `api-version` 查询参数，并复用
  `ai/openai` 的消息与流式转换。
- `ai/google`：Google Gemini 适配器，基于官方 `google.golang.org/genai` SDK。
  支持文本、thinking、thought signature、函数调用、图片输入/工具结果图片、
  usage 和 stop reason 映射。
- `ai/openrouter`：OpenRouter Chat Completions 与 Anthropic Messages 适配器，
  支持 Bearer 鉴权、`OPENROUTER_API_KEY` 回退和 OpenRouter reasoning 格式。
- `ai/deepseek`：DeepSeek Chat Completions 适配器，支持 `DEEPSEEK_API_KEY`
  回退、thinking 请求参数和 `reasoning_content` 历史回放。
- `ai/xai`：xAI Responses 适配器，支持 `XAI_API_KEY` 回退和 encrypted
  reasoning 内容。
- `openai.Config.ReasoningFormat` 和 `openai.Config.SupportsThinking`：为
  OpenAI 兼容 Chat Completions 端点选择 reasoning 请求/回放格式。
- Chat Completions 流现在识别 `reasoning_content`、`reasoning`、
  `reasoning_text` 和 OpenRouter `reasoning_details`，并生成 thinking 事件。
- `anthropic.Config.AuthToken`：使用 Bearer 鉴权，并阻止环境 API Key 同时发送
  `x-api-key`。
- `openai.Config.Query`：为请求附加查询参数，供 Azure 等兼容端点使用。
- `openai.Config.DisableEnvAuth`：阻止 SDK 从 `OPENAI_API_KEY` 注入
  `Authorization`，供使用其他鉴权头的适配器复用 OpenAI 协议实现。
- `ai.APIAzureOpenAIResponses` 和 `ai.APIGoogleGenerativeAI` API 标识。
- `openai.Config.Headers` 和 `anthropic.Config.Headers`：为兼容端点传递自定义
  HTTP header，支持 OpenRouter、网关鉴权和供应商路由头。
- README 增加 DeepSeek、Groq、Together、OpenRouter、Cerebras、Hugging Face、
  NVIDIA、Moonshot、Z.AI、Kimi、xAI、MiniMax 和 Vercel AI Gateway 等兼容端点
  的接入示例。

### Fixed

- 流式工具调用现在保留 `ai.ToolCall.ThoughtSignature`，确保 Gemini 的
  thought signature 能随 assistant 历史回放。
- Responses 请求的 `max_output_tokens` 低于 16 时自动提升到 16，避免
  OpenAI/Azure 拒绝过小的输出上限。

## [0.4.1] - 2026-11-12

### Added

- `agent.EventMessageUpdate` 和 `Event.AssistantMessageEvent`：暴露文本、
  thinking 和工具调用参数增量；`message_delta` / `thinking_*` 扁平事件继续保留。
- tag 触发的 Release workflow：复用 CI 验证后自动创建 GitHub Release。

### Changed

- CI 增加模块整洁性检查，并使用 `-count=1` 运行测试。

### Fixed

- context 取消后，工具调用不再在预检完成后继续执行；取消会产生
  `Operation aborted` 工具 observation。
- Agent 累积流式工具调用时保留 `ai.ToolCall.ItemID`。

## [0.4.0] - 2026-11-12

### Added

- `Result.NewMessages`：本次 run 新增的消息；`Result.Messages` 继续表示完整
  transcript。
- `ai.ToolCall.ItemID`：保留 OpenAI Responses 的 function-call item id。
- OpenAI Responses 回放元数据：assistant message id、phase、text signature 和
  function-call item id 的往返。
- `agent/conformance_test.go` 语义一致性矩阵，覆盖生命周期事件配对、队列、
  工具、错误和 transcript 形状。
- `ai/internal/modeltest` 共享适配器流契约测试，并接入 OpenAI Completions、
  OpenAI Responses 和 Anthropic Messages 主流程测试。
- `scripts/api-snapshot.sh` 和 `testdata/api/*.txt` 导出 API 快照检查。
- `STABILITY.md` 和 `COMPATIBILITY.md`，明确 v1.0 冻结范围和与 pi 的有意差异。

### Changed

- v0.4 定位为 v1.0 前的稳定化版本；不新增 provider、TUI、durable、MCP 或
  coding-agent 功能。
- `ai`、`agent` 增加包级稳定性说明；`ai/faux` 标记为测试专用包。
- `goding` 仍使用 v0.2 的 `Session`/`ai.ToolSpec` API，需要单独迁移到
  v0.3+ 的状态化 Agent 与 `agent.ToolSpec`；该迁移不在本仓库内完成。

### Fixed

- OpenAI Responses assistant 文本和工具调用不再丢失可回放的 provider item 元数据。
- 失败合成 assistant 消息现在也会进入 `Result.NewMessages`。

## [0.3.0] - 2026-11-12

> v0.3.0 未单独发布；其变更包含在 v0.4.0 中。

### Added

- `agent.Agent` 改为状态化运行时，新增 `Prompt`、`PromptMessages`、
  `Continue`、`Steer`、`FollowUp`、`Abort`、`WaitForIdle`、`Reset`、
  `State`、`Subscribe` 和状态修改方法。
- 新增 steering / follow-up 输入队列，以及 `QueueAll` /
  `QueueOneAtATime` 两种消费模式。
- 新增 `PrepareRequest`、`PrepareNextTurn`、`FinishTurn` 生命周期钩子。
- 新增 `ToolSpec` 运行时提示：`Label`、`OutputSchema`、
  `ExecutionMode` 和 `PrepareArguments`。
- 新增 `StreamingTool`、`NewStreamingTool`、`EventToolUpdate` 和工具执行耗时。
- 新增工具结果 `Details`、`StructuredContent`、`Usage`、`Terminate`。
- 新增 `ai.RetryPolicy`、`ai.Retry`、`ai.RetryDelay` 和 `ai.IsRetryable`。
- 新增消息运行时元数据：`Timestamp`、`Provider`、`Model`、`API`、
  `StopReason`、`ErrorMessage`、`DurationMs`、`Usage`。
- 新增 `ThinkingSignature`、`TextSignature`、`ThoughtSignature` 和
  `Namespace`。

### Changed

- `MaxSteps` 默认值改为 0，表示不限制。
- `ToolExecution` 默认值改为并行。
- 输出长度截断且存在工具调用时，不再直接失败，而是为每个调用写入错误
  observation 后继续。
- provider 错误改为返回 `*agent.ModelError`，同时保留完整 transcript 和事件。
- `Subscribe` 回调按注册顺序同步执行，并计入 run 完成时间。

### Removed

- 删除不可变 `Agent` + `Session` 模型：`NewSession`、
  `NewSessionWithHistory`、`Session`、`WithOnEvent` 和 `WithRequestOptions`。
- 删除 `ErrModelStop` 和 `ErrOutputLimit`。

### Fixed

- 并行工具 `tool_end` 按完成顺序发出，tool result 消息仍按模型调用顺序持久化。
- 工具流式 update 在执行结束后不再进入事件流，并且 run 会等待已在途的
  update 完成后再发出 `tool_end` / `agent_end`。
- 并行工具现在先完成全部参数校验和 `BeforeToolCall` 预检，再统一并发执行。
- `Continue` 在 busy 时先返回 `ErrBusy`，不再提前消费 steering / follow-up
  队列。
- run 开始时也会检查 steering 队列，`Prompt` 前排队的 steering 会进入首次请求。
- `AfterToolCall` 替换 `Content` 且未提供 `StructuredContent` 时会清除旧值。
- 未产生 assistant 消息的 run 失败（如 `PrepareRequest`、`TransformMessages`、
  `MaxSteps`）会补发失败 assistant 消息和 `turn_end`，再发出 `agent_error` /
  `agent_end`。
- 流式 `message_delta` / thinking 事件会同步更新 `State.StreamingMessage`。
- 工具结果 details、structured content、usage 和 duration 不再丢失。

## [0.2.0] - 2026-11-09

### Added

- 新增 `ai.ModelInfo` 和 `ai.Model.Info()`，用于暴露供应商、协议、上下文窗口、
  输出上限和协议能力。
- 新增 OpenAI Responses 适配器 `openai.NewResponses`。
- 新增 Anthropic Messages 适配器 `anthropic.New`，使用官方
  `anthropic-sdk-go`。
- 新增 thinking 内容块和流事件，保留 Anthropic thinking signature /
  redacted payload 与 OpenAI Responses encrypted reasoning content。
- 新增用户图片和工具结果图片支持；Chat Completions 明确拒绝工具结果图片。
- 新增 `RequestOptions`：`Temperature`、`TopP`、`MaxTokens`、`ToolChoice`、
  `SessionID`、`ReasoningEffort`、`ThinkingBudgetTokens`。
- 新增 `Config.ToolExecution`，支持顺序和并行工具执行。
- 新增 `Config.TransformMessages`，支持每次请求前改写消息历史。
- 新增 `Agent.WithRequestOptions`，支持单次 run 覆盖请求参数。
- 新增公开测试模型 `ai/faux`。
- 新增工具参数 JSON Schema 校验。
- 新增 `agent.TextResult`、`agent.ErrorResult` 和 `ToolResult.Text()`。
- 新增 `ai.ToolResultMessage`、`ai.ToolTextMessage`、`Message.Thinking()` 和
  内容块深拷贝辅助函数。
- 新增 CI 配置。

### Changed

- `ai.Event` 拆分为 start/delta/end、thinking、tool call、usage、done 和
  error 事件；工具调用增量移到 `Event.ToolCallDelta`。
- `agent.NewTool` 的 `ToolFunc` 现在返回 `ToolResult`；文本工具使用
  `agent.NewTextTool`。
- `agent.ToolResult.Content` 从 `string` 变为 `[]ai.ContentBlock`，移除
  `CallID` 字段。
- `ai.ToolMessage` 增加 `toolName` 参数。
- 模型错误和流错误会保留 assistant partial 消息和 usage。
- Anthropic 连续工具结果会合并到同一条 user 消息。
- 输出长度截断时不再执行可能不完整的工具调用。

### Removed

- 从 SDK 主仓移除 `coding`、`codingagent` 和 `cmd/goding`，迁移到
  `github.com/mrsibe/goding`。
- 移除 `internal/faux`，改用公开包 `ai/faux`。

### Fixed

- 工具参数非法 JSON 不再丢弃工具调用；错误会作为 `ArgumentsError` 保留，并由
  Agent 转换为错误 observation。
- 工具结果保留 `is_error` 状态并在历史中回传。
- 流式错误不再丢失已产生的 assistant 内容。
- 工具参数在 `BeforeToolCall` 之前完成校验。
- 工具结果文本和图片块在进入历史前完成校验。

## [0.1.0] - 2026-11-09

首个公开版本：

- 供应商无关的 `ai` 消息、工具和流事件。
- OpenAI Chat Completions 流式适配器。
- `agent` 模型循环、工具执行、会话历史和生命周期事件。
- 编码工具、编码 Agent 与 headless `goding` CLI。
