import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import MODELS_DIR, QUANTIZE_BIN, find_gguf_models

DEFAULT_TYPE = "TQ4_1S"
SUFFIX_BY_TYPE = {
    "TQ1_0": "tq1_0",
    "TQ2_0": "tq2_0",
    "TQ3_1S": "tq3_1s",
    "TQ4_1S": "tq4_1s",
    "Q4_K_M": "Q4_K_M",
    "Q4_K_S": "Q4_K_S",
    "Q5_K_M": "Q5_K_M",
    "Q8_0": "Q8_0",
}

_SOURCE_PRIORITY = ["F32", "f32", "BF16", "bf16", "F16", "f16"]


def _precision_rank(stem: str) -> int:
    for i, tag in enumerate(_SOURCE_PRIORITY):
        if tag in stem:
            return i
    return len(_SOURCE_PRIORITY)


def _is_tq_suffixed(stem: str) -> bool:
    return any(stem.lower().endswith(f"-{s}") for s in SUFFIX_BY_TYPE.values())


def pick_sources(model_paths: list[Path]) -> list[Path]:
    by_folder: dict[Path, list[Path]] = defaultdict(list)
    for p in model_paths:
        by_folder[p.parent].append(p)

    sources: list[Path] = []
    for folder, files in sorted(by_folder.items()):
        candidates = [f for f in files if not _is_tq_suffixed(f.stem)]
        if not candidates:
            candidates = files
        candidates.sort(key=lambda p: _precision_rank(p.stem))
        sources.append(candidates[0])
    return sources


def output_path(model_path: Path, quant_type: str) -> Path:
    suffix = SUFFIX_BY_TYPE.get(quant_type, quant_type.lower())
    stem = model_path.stem
    for existing in SUFFIX_BY_TYPE.values():
        if stem.endswith((f"-{existing}", f"_{existing}")):
            base = stem[: -(len(existing) + 1)]
            return model_path.with_name(f"{base}-{suffix}.gguf")
    return model_path.with_name(f"{stem}-{suffix}.gguf")


def already_quantized(model_path: Path, quant_type: str) -> bool:
    stem = model_path.stem.lower()
    suffix = SUFFIX_BY_TYPE.get(quant_type, quant_type.lower()).lower()
    return stem.endswith((f"-{suffix}", f"_{suffix}"))


def quantize_model(model_path: Path, quant_type: str, threads: int | None, dry_run: bool, allow_requantize: bool = True, delete_source: bool = False) -> int:
    if already_quantized(model_path, quant_type):
        print(f"Already {quant_type}, skipping: {model_path.name}")
        return 0

    out_path = output_path(model_path, quant_type)
    if out_path.exists():
        print(f"Output exists, skipping: {out_path}")
        return 0

    cmd = [str(QUANTIZE_BIN)]
    if allow_requantize:
        cmd.append("--allow-requantize")
    cmd += [str(model_path), str(out_path), quant_type]
    if threads is not None:
        cmd.append(str(threads))

    print(f"Quantizing {model_path.name} -> {out_path.name} [{quant_type}]")
    if dry_run:
        print(" ".join(cmd))
        return 0

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout, end="")
    if proc.returncode != 0:
        print(f"FAILED ({proc.returncode}): {model_path.name}", file=sys.stderr)
        if out_path.exists():
            out_path.unlink()
            print(f"Removed incomplete output: {out_path.name}")
    elif delete_source:
        source_rank = _precision_rank(model_path.stem)
        if source_rank < len(_SOURCE_PRIORITY):
            print(f"Deleting source: {model_path.name}")
            model_path.unlink()
        else:
            print(f"Skipping delete -- source is already quantized ({model_path.name})")
    return proc.returncode


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize every .gguf model found under ./models using the TurboQuant llama-quantize binary."
    )
    parser.add_argument(
        "repo_id",
        nargs="?",
        default=None,
        help="Optional repo id (e.g. unsloth/Qwen3.5-9B-GGUF) to quantize a single model.",
    )
    parser.add_argument(
        "-t",
        "--type",
        default=DEFAULT_TYPE,
        help=f"Quantization type (default: {DEFAULT_TYPE}). E.g. TQ4_1S, TQ3_1S, Q4_K_M, Q8_0.",
    )
    parser.add_argument(
        "-j",
        "--threads",
        type=int,
        default=None,
        help="Number of threads. Defaults to llama-quantize auto.",
    )
    parser.add_argument(
        "--no-allow-requantize",
        action="store_true",
        help="Disable --allow-requantize (enabled by default).",
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete the source GGUF after a successful quantization (only for unquantized sources: F32/BF16/F16).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without executing them.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not QUANTIZE_BIN.exists():
        print(f"Quantize binary not found: {QUANTIZE_BIN}", file=sys.stderr)
        sys.exit(1)

    models = find_gguf_models()
    if not models:
        print(f"No .gguf models found under {MODELS_DIR}")
        sys.exit(0)

    if args.repo_id:
        folder = MODELS_DIR / args.repo_id.replace("/", "__")
        models = [p for p in models if p.parent == folder]
        if not models:
            print(f"No .gguf files found for {args.repo_id} in {folder}")
            sys.exit(1)

    sources = pick_sources(models)
    print(f"Found {len(sources)} source model(s) under {MODELS_DIR} (best precision per folder)")
    failures = 0
    for model_path in sources:
        allow_rq = not args.no_allow_requantize
        if _precision_rank(model_path.stem) >= len(_SOURCE_PRIORITY):
            print(f"Source is already quantized ({model_path.name}), allowing requantize")
        else:
            allow_rq = False
            print(f"Source is unquantized ({model_path.name}), using clean quantization")

        rc = quantize_model(
            model_path,
            args.type,
            args.threads,
            args.dry_run,
            allow_requantize=allow_rq,
            delete_source=args.delete_source,
        )
        failures += 1 if rc != 0 else 0
        print("-" * 60)

    if failures:
        print(f"{failures} model(s) failed.", file=sys.stderr)
        sys.exit(1)
    print("All done.")
