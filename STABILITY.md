# API Stability

GOAT is on the 0.x line (currently 0.6). Exported APIs are stable enough to
build on, but a minor release may still make breaking changes. Strict semantic
versioning starts at 1.0. Every breaking change must be documented in
`CHANGELOG.md` and marked **破坏性**.

## Packages in the compatibility promise

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
  API is not covered by the compatibility promise.

## Policy

- 0.x: breaking changes are allowed in a minor release and must be marked
  **破坏性** in `CHANGELOG.md`.
- Every exported API change must update the snapshot under `testdata/api`
  (`bash scripts/api-snapshot.sh -update`).
- Deprecated identifiers remain for at least one minor release and carry a
  `Deprecated:` comment.
- Internal packages and unexported behavior are not part of the public contract.

## v1.0 freeze criteria (not yet met)

- ✅ The `agent` conformance suite passes and the pi parity matrix has no
  outstanding 🚧 entries.
- ✅ Provider contract tests pass for every built-in adapter.
- ✅ The exported API snapshot is reviewed and committed (`testdata/api`).
- ✅ `COMPATIBILITY.md` documents every intentional deviation from pi.
- ⬜ Simplification pass: the package layout and duplicated code paths are
  collapsed before the API is frozen.
