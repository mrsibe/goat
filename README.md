# GOAT

**GOAT = Go Agent Toolkit**，一个参考 [pi](https://github.com/earendil-works/pi)
分层思路实现的 Go Agent SDK，并在其上组装了名为 `goding` 的无 TUI 编码 Agent。

仓库：<https://github.com/mrsibe/goat>

这不是 pi 的等价移植。这里保留了可复用的边界，刻意去掉了大规模产品功能：

- `ai`：供应商无关的消息、工具声明、流事件和 usage。
- `agent`：通用模型循环、工具执行、生命周期事件和会话历史。
- `coding`：编码系统提示词、根 `AGENTS.md`、`read/write/edit/bash` 工具。
- `codingagent`：把模型、工具和提示词组装成 goding，并提供 JSONL 会话。
- `cmd/goding`：headless CLI。

项目仍然是一个 Go module，包之间通过公开接口组合，没有拆成多个 module。

## 安全模型

`goding` **不会询问写文件或执行命令的审批**。

- `read/write/edit` 被限制在当前工作区内，并检查路径穿越和 symlink 逃逸。
- `bash` **不受工作区限制**，可以访问工作区外路径，也会继承当前用户权限。
- 文件工具限制不是操作系统沙箱，也无法防御 TOCTOU 或恶意本地进程。

需要更强隔离时，请在容器、虚拟机或独立用户中运行。

## SDK 快速开始

```go
package main

import (
	"context"
	"fmt"

	"github.com/mrsibe/goat/agent"
	"github.com/mrsibe/goat/ai/openai"
)

func main() {
	model, err := openai.New(openai.Config{
		APIKey:  "YOUR_API_KEY",
		BaseURL: "https://api.deepseek.com",
		Model:   "deepseek-chat",
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

`ai/openai` 使用 Chat Completions 流式协议，可连接 OpenAI、DeepSeek、OpenRouter、
Ollama 等兼容服务。API key 可以为空，方便本地无鉴权服务。

## 自定义工具

```go
echo := agent.NewTool(ai.ToolSpec{
	Name:        "echo",
	Description: "返回输入文本",
	Parameters: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"text": map[string]any{"type": "string"},
		},
		"required": []string{"text"},
	},
}, func(_ context.Context, call ai.ToolCall) (string, error) {
	var input struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(call.Arguments, &input); err != nil {
		return "", err
	}
	return input.Text, nil
})
```

工具返回的错误、未知工具和 panic 会作为错误 observation 交回模型。模型调用失败、
上下文取消、输出长度上限或 `MaxSteps` 耗尽会终止当前 run。

`agent.Config` 还提供 `BeforeToolCall` 和 `AfterToolCall`，可在 SDK 使用方实现策略、
审计或结果改写；goding 默认策略是完全放开。

## 事件与流

模型层 `ai.Stream` 返回 `start`、`text_delta`、`tool_call_delta`、`usage`、`done`。
Agent 层把这些事件和工具生命周期组合成：

```text
agent_start
turn_start
message_start / message_delta / message_end
tool_start / tool_end
turn_end
agent_end
agent_error
```

事件回调同步执行；长时间阻塞的回调会阻塞当前 run。

```go
result, err := assistant.Run(ctx, "查一下天气", agent.WithOnEvent(func(event agent.Event) {
	switch event.Type {
	case agent.EventMessageDelta:
		fmt.Print(event.Text)
	case agent.EventToolStart:
		fmt.Printf("\n调用 %s\n", event.ToolCall.Name)
	case agent.EventToolEnd:
		fmt.Printf("结果：%s\n", event.ToolResult.Content)
	}
}))
```

## 使用 goding

```bash
go install github.com/mrsibe/goat/cmd/goding@latest

# 或在仓库内构建
go build -o ./bin/goding ./cmd/goding

export OPENAI_API_KEY=...
export OPENAI_MODEL=deepseek-chat
export OPENAI_BASE_URL=https://api.deepseek.com

./bin/goding "解释一下这个仓库的目录结构"
printf '检查并修复测试失败\n' | ./bin/goding
```

主要参数：

```text
-m, --model string
-b, --base-url string
-k, --api-key string
-c, --continue
-r, --resume string
    --session-dir string
    --no-session
    --json
    --max-steps int
    --version
    --help
```

环境变量为 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL` 和 `GODING_HOME`。
消息来自位置参数；没有位置参数且 stdin 是管道时，读取整个 stdin。没有消息且 stdin
是终端时打印帮助。

文本模式只在 stdout 输出最终 assistant 文本。`--json` 每行输出一个 JSON 对象，
第一个事件是 `session_start`，随后是 agent 事件；错误事件为 `agent_error`。

退出码：

- `0`：成功。
- `1`：模型、agent、输出长度或 max-steps 错误。
- `2`：参数、配置、模型初始化或会话加载错误。
- `130`：被 SIGINT/SIGTERM 中断。

## 会话持久化

默认位置：

```text
$GODING_HOME/sessions/<workspace-sha256-16>/<session-id>.jsonl
```

`GODING_HOME` 默认 `~/.goding`。`--session-dir` 替换 `sessions` 根目录，仍然按工作区
哈希隔离。`--continue` 恢复当前工作区最近修改的会话，`--resume <id>` 恢复指定会话；
两者不能和 `--no-session` 同时使用。

JSONL 是线性追加格式，版本为 1：

```json
{"type":"session","version":1,"id":"...","timestamp":"...","cwd":"..."}
{"type":"message","timestamp":"...","message":{"role":"user","content":[...]}}
{"type":"message","timestamp":"...","message":{"role":"assistant","content":[...]}}
{"type":"usage","timestamp":"...","usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}
```

系统提示词不持久化，恢复时根据当前工作区和根 `AGENTS.md` 重新生成。没有换行结尾的
残缺末行会被忽略；内部损坏会明确报错，避免静默丢失上下文。v1 不支持分支、压缩、
多进程锁或会话迁移。

## 编码工具

- `read`：`path/offset/limit`，最多返回 2000 行或 50KB，附带续读 offset。
- `write`：创建或覆盖文件，自动创建父目录。
- `edit`：对原始文件做唯一、非重叠的精确文本替换；保留 BOM 和 LF/CRLF，拒绝混合换行。
- `bash`：`bash -lc`，合并 stdout/stderr，保留最后 2000 行或 50KB；截断时保留完整输出临时文件。

编码 Agent 自动读取工作区根目录的 `AGENTS.md`，不扫描父目录或子目录。内置工具输出会
受上述限制，第三方自定义工具需要自行控制结果大小；v1 不做自动历史压缩。

## 当前边界

v1 不包含 TUI、MCP、扩展系统、skills、子代理、RPC、并行工具执行、审批机制、
图片/thinking 内容、原生 Anthropic 适配器或自动会话压缩。工具执行的顺序与模型给出的
工具调用顺序一致。

## 验证

```bash
gofmt -w .
GOCACHE=/tmp/goat-gocache GOFLAGS=-buildvcs=false go test ./...
GOCACHE=/tmp/goat-gocache GOFLAGS=-buildvcs=false go vet ./...
GOCACHE=/tmp/goat-gocache GOFLAGS=-buildvcs=false go build ./...
```
