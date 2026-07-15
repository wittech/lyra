#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS="$ROOT/dependencies"

git -C "$ROOT" submodule update --init --recursive -- \
  dependencies/OpenXR-SDK \
  dependencies/args \
  dependencies/dlss \
  dependencies/fmt \
  dependencies/glfw \
  dependencies/imgui \
  dependencies/pybind11 \
  dependencies/tinylogger \
  dependencies/zlib

echo "GUI dependencies are ready in $DEPS"
