#!/usr/bin/env bash
set -euo pipefail

RELEASE_TAG="tqp-v0.3.0"
REPO="https://github.com/TheTom/llama-cpp-turboquant.git"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="${ROOT}/bin/turboquant-plus-${RELEASE_TAG}"
BUILD_DIR="/tmp/llama-cpp-turboquant-build"

echo "=== Cloning llama-cpp-turboquant @ ${RELEASE_TAG} ==="
rm -rf "${BUILD_DIR}"
git clone --branch "${RELEASE_TAG}" --depth 1 "${REPO}" "${BUILD_DIR}"

echo "=== Configuring cmake (CUDA) ==="
cmake -S "${BUILD_DIR}" -B "${BUILD_DIR}/build" \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release

echo "=== Building llama-server and llama-quantize ==="
cmake --build "${BUILD_DIR}/build" \
    --config Release \
    --target llama-server llama-quantize \
    -j"$(nproc)"

echo "=== Installing binaries and shared libraries to ${BIN_DIR} ==="
mkdir -p "${BIN_DIR}"
cp "${BUILD_DIR}/build/bin/"* "${BIN_DIR}/"

echo "=== Cleanup ==="
rm -rf "${BUILD_DIR}"

echo ""
echo "Done. Binaries and libraries installed to:"
echo "  ${BIN_DIR}/"
echo ""
echo "Add to your shell or run before using:"
echo "  export LD_LIBRARY_PATH=\"${BIN_DIR}:\${LD_LIBRARY_PATH:-}\""
echo ""
echo "Verify: ${BIN_DIR}/llama-server --version 2>&1 | head -1"
