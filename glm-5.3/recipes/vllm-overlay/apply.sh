#!/usr/bin/env bash
set -euo pipefail

readonly BASE_COMMIT='c01b50e390e6d3d0019aa53f41ff1198c8105e5a'
readonly PATCH_SHA256='2b8b715c1021f1f1e3fad23b03c864970f124832178209b00eaae74ed9f2651f'
readonly MANIFEST_SHA256='5921028ebd11487c511625a0eb7623e66a42c05b72206741ba5bd2fc7a7f8d2a'
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly PATCH_FILE="$SCRIPT_DIR/vllm-c01b50e-to-895c5d5.patch"
readonly MANIFEST_FILE="$SCRIPT_DIR/SHA256SUMS"

usage() {
  echo "usage: $0 [--check-only] /path/to/vllm" >&2
}

check_only=0
if [[ "${1:-}" == '--check-only' ]]; then
  check_only=1
  shift
fi
if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

checkout="$(realpath -- "$1")"
git -C "$checkout" rev-parse --is-inside-work-tree >/dev/null
actual_commit="$(git -C "$checkout" rev-parse HEAD)"
if [[ "$actual_commit" != "$BASE_COMMIT" ]]; then
  echo "expected vLLM $BASE_COMMIT, found $actual_commit" >&2
  exit 1
fi

echo "$PATCH_SHA256  $PATCH_FILE" | sha256sum --check --strict
echo "$MANIFEST_SHA256  $MANIFEST_FILE" | sha256sum --check --strict

if (cd "$checkout" && sha256sum --check --status --strict "$MANIFEST_FILE"); then
  echo 'exact deployed source overlay is already present'
  exit 0
fi

if [[ -n "$(git -C "$checkout" status --porcelain --untracked-files=no -- vllm)" ]]; then
  echo 'refusing to overwrite existing vllm/ changes' >&2
  exit 1
fi

git -C "$checkout" apply --unidiff-zero --check "$PATCH_FILE"
if (( check_only )); then
  echo 'patch applies cleanly to the pinned base'
  exit 0
fi

git -C "$checkout" apply --unidiff-zero "$PATCH_FILE"
(cd "$checkout" && sha256sum --check --strict "$MANIFEST_FILE")
echo 'applied and verified the 21-file deployed source overlay'
