#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
readonly here
package="$(cd "$here/.." && pwd)"
readonly package
readonly results_root="${1:-${RESULTS_ROOT:-$package/results}}"
readonly chart_python="${CHART_PYTHON:-python3}"

python3 "$here/extract_results.py" \
  --results-root "$results_root" --output-root "$package"
if ! "$chart_python" -c 'import matplotlib' >/dev/null 2>&1; then
  echo "CHART_PYTHON=$chart_python cannot import matplotlib" >&2
  exit 2
fi
"$chart_python" "$here/render_charts.py" --package-root "$package"
python3 "$here/update_readme.py" --package-root "$package"

printf 'GLM-5.2 publication artifacts refreshed from %s\n' "$results_root"
