import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, whoami

from backend.config import MODELS_DIR, TQ_PATTERN, find_gguf_models, repo_id_from_folder

HF_USER_DEFAULT = "yosoyalguien"
TURBOQUANT_VERSION = "turboquant-plus-tqp-v0.2.0"
QUANTIZE_BIN_URL = "https://github.com/ggml-org/llama.cpp"


def model_base_name(repo_id: str) -> str:
    return repo_id.split("/")[-1]


def tq_suffix_from_filename(filename: str) -> str:
    match = TQ_PATTERN.search(filename)
    return f"tq{match.group(1)}" if match else "tq"


def build_repo_name(hf_user: str, parent_repo_id: str, tq_suffix: str) -> str:
    base = model_base_name(parent_repo_id)
    return f"{hf_user}/{base}-{tq_suffix.upper()}"


def build_model_card(
    repo_name: str,
    parent_repo_id: str,
    tq_suffix: str,
    filename: str,
) -> str:
    return f"""---
base_model:
- {parent_repo_id}
license: other
tags:
- turboquant
- gguf
- quantization
- {tq_suffix}
---

# {repo_name}

TurboQuant ({tq_suffix.upper()}) quantization of [{parent_repo_id}](https://huggingface.co/{parent_repo_id}).

## Details

| Field | Value |
|---|---|
| Parent model | [{parent_repo_id}](https://huggingface.co/{parent_repo_id}) |
| Quantization type | {tq_suffix.upper()} |
| Quantization tool | [{TURBOQUANT_VERSION}]({QUANTIZE_BIN_URL}) |
| File | `{filename}` |

## Usage

Use with [llama.cpp](https://github.com/ggml-org/llama.cpp) (TurboQuant fork) or any GGUF-compatible runtime that supports the {tq_suffix.upper()} type.

```bash
llama-server -m {filename} --port 8080
```

## Disclaimer

This model was quantized using TurboQuant, an experimental KV cache compression and quantization method. Quality may differ from the parent model. Refer to the parent model for licensing and usage terms.
"""


def upload_model(
    model_path: Path,
    hf_user: str,
    api: HfApi,
    dry_run: bool,
    private: bool,
) -> str | None:
    folder_name = model_path.parent.name
    parent_repo_id = repo_id_from_folder(folder_name)
    tq_suffix = tq_suffix_from_filename(model_path.name)
    repo_name = build_repo_name(hf_user, parent_repo_id, tq_suffix)
    card = build_model_card(repo_name, parent_repo_id, tq_suffix, model_path.name)

    print(f"  Repo:    {repo_name}")
    print(f"  Parent:  {parent_repo_id}")
    print(f"  File:    {model_path.name}")
    print(f"  Type:    {tq_suffix.upper()}")

    if dry_run:
        print(f"  [dry-run] Would create {repo_name} and upload {model_path.name}")
        print(f"  [dry-run] Model card:\n{card}")
        return repo_name

    try:
        create_repo(repo_name, repo_type="model", private=private, exist_ok=True)
    except Exception as exc:
        print(f"  FAILED creating repo: {exc}", file=sys.stderr)
        return None

    try:
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=model_path.name,
            repo_id=repo_name,
            repo_type="model",
        )
    except Exception as exc:
        print(f"  FAILED uploading model: {exc}", file=sys.stderr)
        return None

    try:
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(card)
            tmp_path = tmp.name

        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="model",
        )
        Path(tmp_path).unlink(missing_ok=True)
    except Exception as exc:
        print(f"  FAILED uploading model card: {exc}", file=sys.stderr)

    print(f"  Done: https://huggingface.co/{repo_name}")
    return repo_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload TurboQuant GGUF models to Hugging Face, linked to their parent models."
    )
    parser.add_argument(
        "-u",
        "--user",
        default=HF_USER_DEFAULT,
        help=f"Hugging Face username/namespace (default: {HF_USER_DEFAULT}).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repositories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without making API calls.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    models = find_gguf_models(fn_filter=lambda p: TQ_PATTERN.search(p.stem))
    if not models:
        print(f"No TurboQuant (.gguf with tq suffix) models found under {MODELS_DIR}")
        sys.exit(0)

    print(f"Found {len(models)} TurboQuant model(s) under {MODELS_DIR}\n")

    if args.dry_run:
        api = None
    else:
        api = HfApi()
        try:
            user = whoami()
            actual_user = user.get("name", args.user)
            if actual_user != args.user:
                print(f"Logged in as {actual_user}, using that instead of {args.user}")
                args.user = actual_user
        except Exception:
            print("Not logged in to Hugging Face. Run: hf auth login", file=sys.stderr)
            sys.exit(1)

    uploaded = []
    failures = 0
    for model_path in models:
        print(f"[{model_path.parent.name}/{model_path.name}]")
        result = upload_model(model_path, args.user, api, args.dry_run, args.private) # type: ignore
        if result is not None:
            uploaded.append(result)
        else:
            failures += 1
        print("-" * 60)

    print(f"\nUploaded: {len(uploaded)}  Failed: {failures}")
    if failures:
        sys.exit(1)
