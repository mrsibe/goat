# API Stability

GOAT v1.0 is the first stable release line. The exported API of the packages
below follows semantic versioning; breaking changes require a major version and
must be documented in `CHANGELOG.md`.

## Stable packages

- `github.com/mrsibe/goat/ai`
- `github.com/mrsibe/goat/agent`
- `github.com/mrsibe/goat/ai/openai`
- `github.com/mrsibe/goat/ai/anthropic`
- `github.com/mrsibe/goat/ai/azure`
- `github.com/mrsibe/goat/ai/google`
- `github.com/mrsibe/goat/ai/openrouter`
- `github.com/mrsibe/goat/ai/deepseek`
- `github.com/mrsibe/goat/ai/xai`

## Test-only packages

- `github.com/mrsibe/goat/ai/faux` is a deterministic test model. Its exported
  API is not covered by the v1.0 compatibility promise.

## Policy

- Exported identifiers in stable packages follow semantic versioning.
- Breaking changes require a major version after v1.0.
- Additive changes are allowed in minor versions.
- Deprecated identifiers remain for at least one minor release and carry a
  `Deprecated:` comment.
- Internal packages and unexported behavior are not part of the public contract.

## v1.0 freeze criteria (met)

- The `agent` conformance suite and the pi parity matrix pass.
- Provider contract tests pass for every built-in adapter.
- The exported API snapshot is reviewed and committed (`testdata/api`).
- `COMPATIBILITY.md` documents every intentional deviation from pi.
