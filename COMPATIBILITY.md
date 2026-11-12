# Compatibility with pi

GOAT targets semantic compatibility with the `pi-ai` and `pi-agent-core`
packages, not a line-by-line port of the Pi monorepo. Go API shape follows Go
conventions.

## In scope

- Provider-neutral messages, content blocks, tool calls, usage, and stream events.
- Stateful agent lifecycle, steering/follow-up queues, abort, reset, and hooks.
- Sequential and parallel tool execution, including preflight and streaming updates.
- OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages adapters.
- OpenAI Responses replay metadata: assistant message id, phase, text signature,
  and function-call item id.

## Intentional differences

- **Core only**: coding tools, CLI/TUI, durable sessions, MCP, codemode, and
  extension packaging live in the separate `goding` product repository.
- **Event shape**: GOAT emits pi-compatible `message_update` events with the
  original `AssistantMessageEvent`. It additionally exposes flat
  `message_delta`, `thinking_*`, and tool execution events for simpler
  consumers.
- **Result shape**: `Result.Messages` is the full transcript; `Result.NewMessages`
  contains only messages produced by the current run.
- **Retry**: the Agent does not retry provider requests automatically. Use
  `ai.Retry` and `ai.RetryPolicy` at the call site.
- **Provider surface**: GOAT ships OpenAI Completions, OpenAI Responses, and
  Anthropic Messages. Pi additionally ships Google, Bedrock, Mistral, Cloudflare,
  image generation, classification, and other provider-specific capabilities.
- **Auth and catalog**: GOAT adapters are configured explicitly with API keys,
  base URLs, and model names. There is no built-in model catalog, OAuth flow, or
  provider identity resolver.
- **Tool call ids**: `ai.ToolCall.ID` stays the provider `call_id`;
  `ai.ToolCall.ItemID` carries the OpenAI Responses item id. Pi combines both
  into one `callId|itemId` string.
- **MaxSteps**: GOAT adds `Config.MaxSteps` as a safety valve; `0` means
  unlimited. Pi has no equivalent core option.
- **Empty tool output**: GOAT normalizes an empty tool result to one empty text
  block because the built-in providers require non-empty tool output. Pi may
  preserve an empty content array.
- **Session persistence**: `Config.SessionID` is forwarded to provider cache
  hints. Durable conversation storage is a `goding` concern, not a core feature.
- **Reset baseline**: GOAT restores the configured initial system prompt. Pi
  retains the current system message. Dynamic tool declarations are not part of
  GOAT's agent state model.

## Conformance

- `agent/conformance_test.go` checks lifecycle event pairing and transcript shape.
- `ai/internal/modeltest` provides the shared adapter stream contract.
- `scripts/api-snapshot.sh` guards the exported API surface.
