#!/usr/bin/env bash
set -euo pipefail

RELEASE_TAG="tqp-v0.3.0"
REPO="TheTom/llama-cpp-turboquant"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="${ROOT}/bin"

OS="$(uname -s)"
ARCH="$(uname -m)"

download_and_extract() {
    local url="$1"
    local archive="$2"
    local extract_dir="$3"

    echo "Downloading ${url}"
    if command -v curl &>/dev/null; then
        curl -fL -o "${archive}" "${url}"
    elif command -v wget &>/dev/null; then
        wget -qO "${archive}" "${url}"
    else
        echo "Error: curl or wget required" >&2
        exit 1
    fi

    mkdir -p "${extract_dir}"
    case "${archive}" in
        *.tar.gz) tar -xzf "${archive}" -C "${extract_dir}" ;;
        *.zip)
            if command -v unzip &>/dev/null; then
                unzip -qo "${archive}" -d "${extract_dir}"
            else
                echo "Error: unzip required for .zip archives" >&2
                exit 1
            fi
            ;;
        *) echo "Error: unknown archive format ${archive}" >&2; exit 1 ;;
    esac
    rm -f "${archive}"
    echo "Extracted to ${extract_dir}"
}

pick_asset() {
    case "${OS}" in
        Darwin)
            case "${ARCH}" in
                arm64) echo "turboquant-plus-${RELEASE_TAG}-macos-arm64-metal.tar.gz" ;;
                *) echo "Error: Apple Silicon (arm64) required for Metal build" >&2; exit 1 ;;
            esac
            ;;
        Linux)
            case "${ARCH}" in
                x86_64)
                    if command -v nvidia-smi &>/dev/null; then
                        echo "WARNING: No Linux CUDA binary available from ${REPO}." >&2
                        echo "The CPU build will be used. For GPU acceleration, compile from source:" >&2
                        echo "  https://github.com/${REPO}" >&2
                        echo "  cmake -B build -DGGML_CUDA=ON && cmake --build build" >&2
                    fi
                    echo "turboquant-plus-${RELEASE_TAG}-linux-x64-cpu.tar.gz"
                    ;;
                *) echo "Error: unsupported Linux arch ${ARCH}" >&2; exit 1 ;;
            esac
            ;;
        MINGW*|MSYS*|CYGWIN*|Windows*)
            case "${ARCH}" in
                x86_64) echo "turboquant-plus-${RELEASE_TAG}-windows-x64-cuda12.4.zip" ;;
                *) echo "Error: unsupported Windows arch ${ARCH}" >&2; exit 1 ;;
            esac
            ;;
        *) echo "Error: unsupported OS ${OS}" >&2; exit 1 ;;
    esac
}

main() {
    local asset
    asset="$(pick_asset)"

    local base_url="https://github.com/${REPO}/releases/download/${RELEASE_TAG}"
    local url="${base_url}/${asset}"
    local archive="${BIN_DIR}/${asset}"

    mkdir -p "${BIN_DIR}"
    download_and_extract "${url}" "${archive}" "${BIN_DIR}"

    echo "Done. Binaries available in ${BIN_DIR}"
    echo "Verify with: ${BIN_DIR}/bin/llama-server --version"
}

main "$@"
