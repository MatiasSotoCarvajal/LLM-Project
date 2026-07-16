import argparse
import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import hf_hub_download

from backend.config import local_folder


class ModelEntry(TypedDict, total=False):
    repo_id: str
    filenames: list[str]
    local_repo_id: str


MODELS: list[ModelEntry] = [
    {
        "repo_id": "unsloth/gemma-4-E4B-it-GGUF",
        "filenames": ["gemma-4-E4B-it-Q8_0.gguf"],
    },
    # {
    #     "repo_id": "yosoyalguien/gemma-4-E4B-it-GGUF-TQ4_1S",
    #     "filenames": ["gemma-4-E4B-it-tq4_1s.gguf"],
    #     "local_repo_id": "unsloth/gemma-4-E4B-it-GGUF",
    # },
    {
        "repo_id": "unsloth/gemma-4-E2B-it-GGUF",
        "filenames": ["gemma-4-E2B-it-Q8_0.gguf"],
    },
    # {
    #     "repo_id": "yosoyalguien/gemma-4-E2B-it-GGUF-TQ4_1S",
    #     "filenames": ["gemma-4-E2B-it-tq4_1s.gguf"],
    #     "local_repo_id": "unsloth/gemma-4-E2B-it-GGUF",
    # },
    {
        "repo_id": "unsloth/Qwen3.5-9B-GGUF",
        "filenames": ["Qwen3.5-9B-Q8_0.gguf"],
    },
    # {
    #     "repo_id": "yosoyalguien/Qwen3.5-9B-GGUF-TQ4_1S",
    #     "filenames": ["Qwen3.5-9B-tq4_1s.gguf"],
    #     "local_repo_id": "unsloth/Qwen3.5-9B-GGUF",
    # },
    {
        "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "filenames": ["Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"],
    },
    # {
    #     "repo_id": "yosoyalguien/Meta-Llama-3.1-8B-Instruct-GGUF-TQ4_1S",
    #     "filenames": ["Meta-Llama-3.1-8B-Instruct-tq4_1s.gguf"],
    #     "local_repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    # },
]


def _local_folder(entry: ModelEntry) -> Path:
    target = entry.get("local_repo_id", entry["repo_id"])
    return local_folder(target)


def is_downloaded(entry: ModelEntry, filename: str | None = None) -> bool:
    folder = _local_folder(entry)
    if not folder.exists() or not any(folder.iterdir()):
        return False
    if filename is None:
        return True
    return (folder / filename).exists()


def download_file(entry: ModelEntry, filename: str) -> Path:
    local_path = _local_folder(entry)
    local_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {entry['repo_id']} / {filename} -> {local_path} ...")
    downloaded = hf_hub_download(
        repo_id=entry["repo_id"],
        filename=filename,
        local_dir=str(local_path),
    )
    print(f"Done: {downloaded}")
    return Path(downloaded)


def download_model(entry: ModelEntry) -> list[Path]:
    downloaded: list[Path] = []
    for filename in entry["filenames"]:
        if is_downloaded(entry, filename):
            print(f"Already downloaded, skipping: {entry['repo_id']} / {filename}")
            continue
        downloaded.append(download_file(entry, filename))
    return downloaded


def download_all(models: list[ModelEntry] | None = None) -> list[Path]:
    targets = models if models is not None else MODELS
    downloaded: list[Path] = []
    for entry in targets:
        downloaded.extend(download_model(entry))
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
    parser.add_argument(
        "--q8-only",
        action="store_true",
        help="Download only Q8_0 (or Q8_K) quantizations, skipping BF16/F16/F32.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.repo_id is not None:
        if args.filename is None:
            print("Error: --filename is required when passing a repo_id.")
            raise SystemExit(1)
        entry: ModelEntry = {"repo_id": args.repo_id, "filenames": [args.filename]}
        if args.name is not None:
            entry["local_repo_id"] = args.name
        download_model(entry)
    else:
        targets = MODELS
        if args.q8_only:
            targets = []
            for entry in MODELS:
                q8_files = [f for f in entry["filenames"] if "Q8" in f or "q8" in f]
                if q8_files:
                    targets.append({**entry, "filenames": q8_files})
        download_all(targets)
