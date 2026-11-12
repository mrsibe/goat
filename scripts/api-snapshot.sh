#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

packages=(
  "ai"
  "agent"
  "ai/openai"
  "ai/anthropic"
  "ai/faux"
)

update=false
if [[ "${1:-}" == "-update" ]]; then
  update=true
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [-update]" >&2
  exit 2
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

failed=false
for package in "${packages[@]}"; do
  name="${package//\//-}"
  actual="$tmp/$name.txt"
  snapshot="$root/testdata/api/$name.txt"
  GOFLAGS="${GOFLAGS:-} -buildvcs=false" go doc -all "./$package" >"$actual"
  if $update; then
    mkdir -p "$(dirname "$snapshot")"
    cp "$actual" "$snapshot"
    echo "updated $snapshot"
    continue
  fi
  if [[ ! -f "$snapshot" ]]; then
    echo "missing API snapshot: $snapshot (run $0 -update)" >&2
    failed=true
    continue
  fi
  if ! diff -u "$snapshot" "$actual"; then
    echo "API snapshot changed for $package; review it and run $0 -update" >&2
    failed=true
  fi
done

if $failed; then
  exit 1
fi
