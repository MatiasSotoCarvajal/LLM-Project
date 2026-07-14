import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from backend.config import local_folder
MODELS: list[dict] = [
    {
        "repo_id": "unsloth/gemma-4-E4B-it-GGUF",
        "filenames": ["gemma-4-E4B-it-Q8_0.gguf"],
    },
    {
        "repo_id": "unsloth/gemma-4-E2B-it-GGUF",
        "filenames": ["gemma-4-E2B-it-UD-Q8_K_XL.gguf"],
    },
    {
        "repo_id": "unsloth/Qwen3.5-9B-GGUF",
        "filenames": ["Qwen3.5-9B-Q8_0.gguf"],
    },
    {
        "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "filenames": ["Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"],
    },
]


def is_downloaded(repo_id: str, filename: str | None = None) -> bool:
    folder = local_folder(repo_id)
    if not folder.exists() or not any(folder.iterdir()):
        return False
    if filename is None:
        return True
    return (folder / filename).exists()


def download_file(repo_id: str, filename: str) -> Path:
    local_path = local_folder(repo_id)
    local_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id} / {filename} -> {local_path} ...")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_path),
    )
    print(f"Done: {downloaded}")
    return Path(downloaded)


def download_model(model: dict) -> list[Path]:
    repo_id = model["repo_id"]
    filenames = model["filenames"]
    downloaded: list[Path] = []
    for filename in filenames:
        if is_downloaded(repo_id, filename):
            print(f"Already downloaded, skipping: {repo_id} / {filename}")
            continue
        downloaded.append(download_file(repo_id, filename))
    return downloaded


def download_all(models: list[dict] | None = None) -> list[Path]:
    targets = models if models is not None else MODELS
    downloaded: list[Path] = []
    for model in targets:
        downloaded.extend(download_model(model))
    return downloaded


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download specific quantizations from the Hugging Face Hub into ./models."
    )
    parser.add_argument(
        "repo_id",
        nargs="?",
        default=None,
        help="Optional single repository id. Must be combined with -f/--filename.",
    )
    parser.add_argument(
        "-f",
        "--filename",
        default=None,
        help="Specific file (quantization) to download from the repo, e.g. 'model-Q4_K_M.gguf'.",
    )
    parser.add_argument(
        "-n",
        "--name",
        default=None,
        help="Optional local folder name under ./models. Defaults to the repo id with '/' replaced by '__'.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.repo_id is not None:
        if args.filename is None:
            print("Error: --filename is required when passing a repo_id.")
            raise SystemExit(1)
        model = {"repo_id": args.repo_id, "filenames": [args.filename]}
        download_model(model)
    else:
        download_all()
