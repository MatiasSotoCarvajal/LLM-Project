#!/usr/bin/env bash
set -euo pipefail

RELEASE_TAG="tqp-v0.3.0"
REPO="TheTom/llama-cpp-turboquant"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="${ROOT}/bin"

OS="$(uname -s)"
ARCH="$(uname -m)"

HF_CUDA_REPO="yosoyalguien/llama-binaries-cuda"
HF_CUDA_FILE="llama-tq-binaries-universal.tar.gz"

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

main() {
    local asset=""
    local url=""
    local extract_dir="${BIN_DIR}"

    case "${OS}" in
        Darwin)
            case "${ARCH}" in
                arm64)
                    asset="turboquant-plus-${RELEASE_TAG}-macos-arm64-metal.tar.gz"
                    url="https://github.com/${REPO}/releases/download/${RELEASE_TAG}/${asset}"
                    ;;
                *) echo "Error: Apple Silicon (arm64) required for Metal build" >&2; exit 1 ;;
            esac
            ;;
        Linux)
            case "${ARCH}" in
                x86_64)
                    if command -v nvidia-smi &>/dev/null; then
                        echo "Detected NVIDIA GPU. Using prebuilt CUDA binary from ${HF_CUDA_REPO}"
                        asset="${HF_CUDA_FILE}"
                        url="https://huggingface.co/${HF_CUDA_REPO}/resolve/main/${HF_CUDA_FILE}"
                        extract_dir="${BIN_DIR}/turboquant-plus-${RELEASE_TAG}"
                    else
                        asset="turboquant-plus-${RELEASE_TAG}-linux-x64-cpu.tar.gz"
                        url="https://github.com/${REPO}/releases/download/${RELEASE_TAG}/${asset}"
                    fi
                    ;;
                *) echo "Error: unsupported Linux arch ${ARCH}" >&2; exit 1 ;;
            esac
            ;;
        MINGW*|MSYS*|CYGWIN*|Windows*)
            case "${ARCH}" in
                x86_64)
                    asset="turboquant-plus-${RELEASE_TAG}-windows-x64-cuda12.4.zip"
                    url="https://github.com/${REPO}/releases/download/${RELEASE_TAG}/${asset}"
                    ;;
                *) echo "Error: unsupported Windows arch ${ARCH}" >&2; exit 1 ;;
            esac
            ;;
        *) echo "Error: unsupported OS ${OS}" >&2; exit 1 ;;
    esac

    local archive="${BIN_DIR}/${asset}"

    mkdir -p "${BIN_DIR}"
    download_and_extract "${url}" "${archive}" "${extract_dir}"

    echo ""
    echo "Done. Binaries available in ${extract_dir}"
    echo "Verify with: ${extract_dir}/llama-server --version 2>&1 | head -1"
}

main "$@"
