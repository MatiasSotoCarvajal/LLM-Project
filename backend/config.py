import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
BIN_NAME = "turboquant-plus-tqp-v0.3.0"

MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))

SERVER_BIN = ROOT / "bin" / BIN_NAME / "llama-server"
QUANTIZE_BIN = ROOT / "bin" / BIN_NAME / "llama-quantize"

RESULTS_DIR = ROOT / "results"

PERPLEXITY_BIN = ROOT / "bin" / BIN_NAME / "llama-perplexity"
DEFAULT_PPL_FILE = ROOT / "benchmarks" / "ppl_sample.txt"

TQ_PATTERN = re.compile(r"-tq(\d_\w+)", re.IGNORECASE)

QUANT_SUFFIXES: list[str] = [
    "tq1_0", "tq2_0", "tq3_1s", "tq4_1s",
    "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q8_0",
]


def local_folder(repo_id: str) -> Path:
    return MODELS_DIR / repo_id.replace("/", "__")


def repo_id_from_folder(folder_name: str) -> str:
    return folder_name.replace("__", "/")


def find_gguf_models(fn_filter=None) -> list[Path]:
    if not MODELS_DIR.exists():
        return []
    files = sorted(MODELS_DIR.rglob("*.gguf"))
    if fn_filter is None:
        return files
    return [f for f in files if fn_filter(f)]
