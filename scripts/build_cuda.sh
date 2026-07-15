#!/usr/bin/env bash
set -euo pipefail

RELEASE_TAG="tqp-v0.3.0"
REPO="https://github.com/TheTom/llama-cpp-turboquant"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="${ROOT}/bin/turboquant-plus-tqp-v0.3.0"

echo "=== Building TurboQuant CUDA binaries (${RELEASE_TAG}) ==="

if ! command -v nvcc &>/dev/null; then
    echo "ERROR: CUDA toolkit not found (nvcc missing)." >&2
    echo "Install with: apt-get install -y cuda-toolkit-12-4" >&2
    exit 1
fi

echo "CUDA version: $(nvcc --version | grep release | awk '{print $5, $6}')"

BUILD_DIR="/tmp/llama-cpp-tq-build"
rm -rf "${BUILD_DIR}"
git clone --branch "${RELEASE_TAG}" --depth 1 "${REPO}" "${BUILD_DIR}"

echo "Configuring CMake..."
cmake -B "${BUILD_DIR}/build" -S "${BUILD_DIR}" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="native" \
    -DGGML_CURL=ON

echo "Compiling (using $(nproc) threads)..."
cmake --build "${BUILD_DIR}/build" --config Release -j"$(nproc)"

echo "Installing binaries to ${BIN_DIR}..."
mkdir -p "${BIN_DIR}"
for bin in llama-server llama-quantize llama-perplexity; do
    if [ -f "${BUILD_DIR}/build/bin/${bin}" ]; then
        cp "${BUILD_DIR}/build/bin/${bin}" "${BIN_DIR}/${bin}"
        echo "  ${bin} -> ${BIN_DIR}/${bin}"
    else
        echo "  WARNING: ${bin} not found in build output" >&2
    fi
done

rm -rf "${BUILD_DIR}"

echo ""
echo "=== Done. Verify with: ==="
echo "  ${BIN_DIR}/llama-server --version 2>&1 | head -1"
