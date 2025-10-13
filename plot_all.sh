#!/usr/bin/env bash
set -u
shopt -s globstar 2>/dev/null || true

PY=python3

# find NUL-separated to match read -d ''
find . -type f -name '*_fig.py' \
  -not -path './.venv/*' -not -path './venv/*' -not -path './env/*' \
  -not -path '*/__pycache__/*' -not -path '*/.git/*' -print0 |
  while IFS= read -r -d '' file; do
    echo "==== Running: $file ===="
    "$PY" "$file"
    rc=$?
    echo "Exit code: $rc"
    if [ $rc -ne 0 ]; then
      echo "ERROR: $file exited with $rc" >&2
    fi
    echo
  done
