# API Stability

GOAT is pre-1.0. The v0.4.x release line is the final planned stabilization pass
before v1.0; breaking changes remain possible until v1.0 and must be documented
in `CHANGELOG.md`.

## Stable packages after v1.0

- `github.com/mrsibe/goat/ai`
- `github.com/mrsibe/goat/agent`
- `github.com/mrsibe/goat/ai/openai`
- `github.com/mrsibe/goat/ai/anthropic`

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

## v1.0 freeze criteria

- The `agent` conformance suite passes.
- Provider contract tests pass for every built-in adapter.
- The exported API snapshot is reviewed and committed.
- `COMPATIBILITY.md` documents every intentional deviation from pi.
