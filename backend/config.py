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

_N_GPU_LAYERS_ENV = os.environ.get("N_GPU_LAYERS", "")
N_GPU_LAYERS: int | None = int(_N_GPU_LAYERS_ENV) if _N_GPU_LAYERS_ENV else None

_N_BATCH_ENV = os.environ.get("N_BATCH", "")
N_BATCH: int | None = int(_N_BATCH_ENV) if _N_BATCH_ENV else None

_N_UBATCH_ENV = os.environ.get("N_UBATCH", "")
N_UBATCH: int | None = int(_N_UBATCH_ENV) if _N_UBATCH_ENV else None

_FLASH_ATTN_ENV = os.environ.get("FLASH_ATTN", "0")
FLASH_ATTN: bool = _FLASH_ATTN_ENV in ("1", "true", "True", "yes")


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
