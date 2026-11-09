// Command goding is a headless coding agent powered by the GOAT SDK.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/mrsibe/goat/agent"
	"github.com/mrsibe/goat/ai"
	"github.com/mrsibe/goat/ai/openai"
	"github.com/mrsibe/goat/codingagent"
)

const version = "0.1.0"

type options struct {
	model       string
	baseURL     string
	apiKey      string
	continueID  bool
	resumeID    string
	sessionDir  string
	noSession   bool
	json        bool
	maxSteps    int
	showVersion bool
}

type modelLoader func(model, baseURL, apiKey string) (ai.Model, error)

func main() {
	info, _ := os.Stdin.Stat()
	stdinIsTerminal := info != nil && info.Mode()&os.ModeCharDevice != 0
	os.Exit(run(os.Args[1:], os.Stdin, stdinIsTerminal, os.Stdout, os.Stderr, defaultModelLoader))
}

func defaultModelLoader(model, baseURL, apiKey string) (ai.Model, error) {
	return openai.New(openai.Config{
		APIKey:  apiKey,
		BaseURL: baseURL,
		Model:   model,
	})
}

func run(args []string, stdin io.Reader, stdinIsTerminal bool, stdout, stderr io.Writer, loadModel modelLoader) int {
	opts, promptArgs, code := parseOptions(args, stderr)
	if code != 0 {
		return code
	}
	if opts.showVersion {
		fmt.Fprintf(stdout, "goding %s\n", version)
		return 0
	}
	if opts.continueID && opts.resumeID != "" {
		fmt.Fprintln(stderr, "goding: --continue and --resume are mutually exclusive")
		return 2
	}
	if opts.noSession && (opts.continueID || opts.resumeID != "") {
		fmt.Fprintln(stderr, "goding: --no-session cannot be combined with --continue or --resume")
		return 2
	}

	if opts.model == "" {
		opts.model = os.Getenv("OPENAI_MODEL")
	}
	if opts.baseURL == "" {
		opts.baseURL = os.Getenv("OPENAI_BASE_URL")
	}
	if opts.apiKey == "" {
		opts.apiKey = os.Getenv("OPENAI_API_KEY")
	}
	if strings.TrimSpace(opts.model) == "" {
		fmt.Fprintln(stderr, "goding: a model is required via --model or OPENAI_MODEL")
		return 2
	}

	prompt := strings.TrimSpace(strings.Join(promptArgs, " "))
	if prompt == "" {
		if stdinIsTerminal {
			printUsage(stderr)
			return 2
		}
		data, err := io.ReadAll(stdin)
		if err != nil {
			fmt.Fprintf(stderr, "goding: read stdin: %v\n", err)
			return 2
		}
		prompt = strings.TrimSpace(string(data))
	}
	if prompt == "" {
		fmt.Fprintln(stderr, "goding: prompt must not be empty")
		return 2
	}

	model, err := loadModel(opts.model, opts.baseURL, opts.apiKey)
	if err != nil {
		fmt.Fprintf(stderr, "goding: create model: %v\n", err)
		return 2
	}
	workspace, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(stderr, "goding: resolve workspace: %v\n", err)
		return 2
	}
	assistant, err := codingagent.New(codingagent.Config{
		Model:              model,
		Workspace:          workspace,
		MaxSteps:           opts.maxSteps,
		SessionDir:         opts.sessionDir,
		DisablePersistence: opts.noSession,
	})
	if err != nil {
		fmt.Fprintf(stderr, "goding: initialize agent: %v\n", err)
		return 2
	}

	var session *codingagent.Session
	switch {
	case opts.resumeID != "":
		session, err = assistant.Resume(opts.resumeID)
	case opts.continueID:
		session, err = assistant.Continue()
	default:
		session, err = assistant.NewSession()
	}
	if err != nil {
		fmt.Fprintf(stderr, "goding: open session: %v\n", err)
		return 2
	}
	defer session.Close()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	var encodeErr error
	encoder := json.NewEncoder(stdout)
	if opts.json {
		if err := encoder.Encode(map[string]any{
			"type":       "session_start",
			"session_id": session.ID(),
		}); err != nil {
			fmt.Fprintf(stderr, "goding: write JSON event: %v\n", err)
			return 1
		}
	}
	onEvent := func(event agent.Event) {
		if !opts.json || encodeErr != nil {
			return
		}
		if err := encoder.Encode(event); err != nil {
			encodeErr = err
		}
	}
	result, runErr := session.Run(ctx, prompt, onEvent)
	if encodeErr != nil {
		fmt.Fprintf(stderr, "goding: write JSON event: %v\n", encodeErr)
		return 1
	}
	if closeErr := session.Close(); closeErr != nil {
		fmt.Fprintf(stderr, "goding: close session: %v\n", closeErr)
		return 1
	}
	if runErr != nil {
		if ctx.Err() != nil || errors.Is(runErr, context.Canceled) {
			return 130
		}
		fmt.Fprintf(stderr, "goding: %v\n", runErr)
		return 1
	}
	if !opts.json && result.Answer != "" {
		fmt.Fprintln(stdout, result.Answer)
	}
	return 0
}

func parseOptions(args []string, stderr io.Writer) (options, []string, int) {
	var opts options
	flags := flag.NewFlagSet("goding", flag.ContinueOnError)
	flags.SetOutput(stderr)
	flags.Usage = func() { printUsage(stderr) }
	flags.StringVar(&opts.model, "m", "", "model id")
	flags.StringVar(&opts.model, "model", "", "model id")
	flags.StringVar(&opts.baseURL, "b", "", "OpenAI-compatible base URL")
	flags.StringVar(&opts.baseURL, "base-url", "", "OpenAI-compatible base URL")
	flags.StringVar(&opts.apiKey, "k", "", "API key (optional for local servers)")
	flags.StringVar(&opts.apiKey, "api-key", "", "API key (optional for local servers)")
	flags.BoolVar(&opts.continueID, "c", false, "continue the most recent session")
	flags.BoolVar(&opts.continueID, "continue", false, "continue the most recent session")
	flags.StringVar(&opts.resumeID, "r", "", "resume a specific session id")
	flags.StringVar(&opts.resumeID, "resume", "", "resume a specific session id")
	flags.StringVar(&opts.sessionDir, "session-dir", "", "session storage root")
	flags.BoolVar(&opts.noSession, "no-session", false, "disable session persistence")
	flags.BoolVar(&opts.json, "json", false, "emit agent events as JSON lines")
	flags.IntVar(&opts.maxSteps, "max-steps", 0, "maximum model turns (default 10)")
	flags.BoolVar(&opts.showVersion, "version", false, "print version and exit")

	if err := flags.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return options{}, nil, 0
		}
		return options{}, nil, 2
	}
	return opts, flags.Args(), 0
}

func printUsage(writer io.Writer) {
	fmt.Fprintln(writer, `goding - headless coding agent

Usage:
  goding [options] [message...]

Options:
  -m, --model string       Model id (or OPENAI_MODEL)
  -b, --base-url string    OpenAI-compatible base URL (or OPENAI_BASE_URL)
  -k, --api-key string     API key (or OPENAI_API_KEY; optional for local servers)
  -c, --continue           Continue the most recent session for this workspace
  -r, --resume string      Resume a specific session id
      --session-dir string Session storage root (or $GODING_HOME/sessions)
      --no-session         Disable session persistence
      --json               Emit agent events as JSON lines on stdout
      --max-steps int      Maximum model turns (default 10)
      --version            Print version and exit
      --help               Print this help and exit

If no message is provided and stdin is piped, stdin is used as the prompt.

SECURITY: There are no approval prompts. File tools are confined to the
workspace, but bash is unrestricted and is not an OS sandbox. Run goding in a
container or isolated account when stronger isolation is required.`)
}
