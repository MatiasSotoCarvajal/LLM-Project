import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"


def local_folder(repo_id: str) -> Path:
    return MODELS_DIR / repo_id.replace("/", "__")


def is_downloaded(repo_id: str) -> bool:
    folder = local_folder(repo_id)
    return folder.exists() and any(folder.iterdir())


def download_local_model(repo_id: str, destination_folder: str | None = None) -> Path:
    if destination_folder is None:
        destination_folder = repo_id.replace("/", "__")

    local_path = MODELS_DIR / destination_folder
    local_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id} -> {local_path} ...")
    snapshot_download(repo_id=repo_id, local_dir=str(local_path))
    print(f"Done: {local_path}")
    return local_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a model from the Hugging Face Hub into ./models."
    )
    parser.add_argument(
        "repo_id",
        help="Hugging Face repository id, e.g. 'NousResearch/Nous-Hermes-2-Pro-Llama-3-8B' or 'majentik/gemma-4-E4B-TurboQuant-MLX-8bit'.",
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
    download_local_model(args.repo_id, args.name)
