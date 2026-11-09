# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 和
[语义化版本](https://semver.org/lang/zh-CN/)。v0.x 阶段允许破坏性变更。

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
