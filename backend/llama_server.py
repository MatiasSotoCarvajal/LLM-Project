import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "Models"
SERVER_BIN = ROOT / "bin" / "turboquant-plus-tqp-v0.2.0" / "llama-server"

DEFAULT_KV_CACHE_TYPE = "q8_0"
DEFAULT_V_CACHE_TYPE = "turbo3"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
STARTUP_TIMEOUT = 600
REQUEST_TIMEOUT = 600


def find_model(model_id: str) -> Path:
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}")

    direct = MODELS_DIR / model_id
    if direct.is_file():
        return direct

    folder_name = model_id.replace("/", "__")
    folder = MODELS_DIR / folder_name
    if folder.is_dir():
        ggufs = sorted(folder.glob("*.gguf"))
        if ggufs:
            return ggufs[0]

    matches = sorted(MODELS_DIR.rglob(f"{model_id}*.gguf"))
    if matches:
        return matches[0]

    matches = sorted(MODELS_DIR.rglob(f"*{model_id}*.gguf"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Model '{model_id}' not found under {MODELS_DIR}")


def run(
    model_id: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    cache_type_k: str = DEFAULT_KV_CACHE_TYPE,
    cache_type_v: str = DEFAULT_V_CACHE_TYPE,
    extra_args: list[str] | None = None,
    capture_output: bool = False,
) -> subprocess.Popen:
    model_path = find_model(model_id)
    print(f"Using model: {model_path}")

    if not SERVER_BIN.exists():
        raise FileNotFoundError(f"Server binary not found: {SERVER_BIN}")

    cmd = [
        str(SERVER_BIN),
        "-m", str(model_path),
        "--host", host,
        "--port", str(port),
        "--cache-type-k", cache_type_k,
        "--cache-type-v", cache_type_v,
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"Starting server: {' '.join(cmd)}")
    stdout = subprocess.PIPE if capture_output else subprocess.DEVNULL
    proc = subprocess.Popen(cmd, stdout=stdout, stderr=subprocess.STDOUT, text=True)
    return proc


def wait_for_server(host: str, port: int, timeout: int = STARTUP_TIMEOUT) -> None:
    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"Server did not become healthy at {url} within {timeout}s")


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def run_single(
    model_id: str,
    prompt: str,
    system_prompt: str | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    cache_type_k: str = DEFAULT_KV_CACHE_TYPE,
    cache_type_v: str = DEFAULT_V_CACHE_TYPE,
    max_tokens: int = 512,
    temperature: float = 0.0,
    extra_args: list[str] | None = None,
) -> dict:
    proc = run(
        model_id,
        host=host,
        port=port,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        extra_args=extra_args,
    )
    try:
        wait_for_server(host, port)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        r = requests.post(
            f"http://{host}:{port}/v1/chat/completions",
            json={
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    finally:
        stop_server(proc)


if __name__ == "__main__":
    print("You cannot run this module alone.")
    raise SystemExit(1)
