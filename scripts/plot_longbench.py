"""Per-model LongBench charts: same model, different KV-cache quantization.

For each model, draw one chart comparing its KV-cache quantization configs
(f16/f16 full precision, q8_0/q8_0 standard 8-bit, and the experimental
TurboQuant variants) across accuracy, decode speed, memory, and TTFT --
each shown *relative to the f16/f16 full-precision baseline (= 100%)*.

Why KV config and not weight_quant: in results_cuda every row is weight_quant
Q8_0; the axis that actually varies is the KV cache type (cache_type_k /
cache_type_v). Grouping by weight_quant would collapse every model to a single
bar. So we compare KV configs instead.

Data sources / quirks:
  * accuracy, decode speed, TTFT  -> longbench_examples.csv (per example)
  * peak memory (rss_gb_peak)     -> longbench_summary.json (not in the CSV)
  * decode_tokens_per_second uses 1e6 as a div-by-zero sentinel; those rows are
    dropped before averaging, else one model's speed mean explodes to ~11000.
  * kv_cache_mib was never logged, so "memory" is whole-process RSS
    (weights + activations + KV cache), a proxy for the KV footprint.

Usage:
    python scripts/plot_longbench.py            # uses results_cuda/ -> results_graphs/
    python scripts/plot_longbench.py --input-folder results_cuda \\
        --csv longbench_examples.csv --summary longbench_summary.json \\
        --output-folder results_graphs
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TPS_SENTINEL = 1e6          # decode_tokens_per_second div-by-zero marker to drop
BASELINE_CONFIG = "f16/f16"  # full precision = 100% reference

# Preferred config order + colourblind-safe palette (baselines grey/blue,
# TurboQuant variants warm so they stand out).
CONFIG_ORDER = ["f16/f16", "q8_0/q8_0", "turbo4/q8_0", "turbo4/turbo4"]
CONFIG_PALETTE = {
    "f16/f16":       "#4d4d4d",
    "q8_0/q8_0":     "#56B4E9",
    "turbo4/q8_0":   "#E69F00",
    "turbo4/turbo4": "#D55E00",
}

# Metric -> (column in the aggregated frame, x-axis label with "better" direction).
METRICS = [
    ("accuracy",     "Accuracy\n(↑ better)"),
    ("decode_speed", "Decode Speed\n(↑ better)"),
    ("memory_gb",    "Memory\n(↓ better)"),
    ("ttft",         "TTFT\n(↓ better)"),
]


def short_model(name: str) -> str:
    base = name.split("/")[-1]
    for junk in ("-GGUF", "-Instruct", "-it", "-Meta"):
        base = base.replace(junk, "")
    return base.strip("-")


def load_memory(summary_path: str) -> pd.DataFrame:
    """Peak process RSS (GB) per (model, config) from the summary JSON, if present."""
    if not summary_path or not os.path.exists(summary_path):
        print(f"  [Info] No summary JSON at '{summary_path}' -- memory bars will be skipped.")
        return pd.DataFrame(columns=["model", "config", "memory_gb"])
    data = json.loads(open(summary_path).read())
    rows = [
        {"model": e["model"],
         "config": f'{e["cache_type_k"]}/{e["cache_type_v"]}',
         "memory_gb": e.get("rss_gb_peak")}
        for e in data.get("results", []) if e.get("rss_gb_peak") is not None
    ]
    return pd.DataFrame(rows)


def config_order(configs) -> list[str]:
    present = set(configs)
    return [c for c in CONFIG_ORDER if c in present] + sorted(present - set(CONFIG_ORDER))


def generate_multi_model_plots(input_folder, csv_filename, summary_filename=None,
                               output_folder="results_graphs"):
    csv_path = os.path.join(input_folder, csv_filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find the file at '{csv_path}'. Check your input folder name!")
    os.makedirs(output_folder, exist_ok=True)

    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    df["config"] = df["cache_type_k"].astype(str) + "/" + df["cache_type_v"].astype(str)

    summary_path = os.path.join(input_folder, summary_filename) if summary_filename else None
    mem_df = load_memory(summary_path)

    unique_models = df["model"].unique()
    print(f"Found {len(unique_models)} unique models: {list(unique_models)}\n")

    for model_name in unique_models:
        print(f"Processing model: {model_name}...")
        model_df = df[df["model"] == model_name]

        # accuracy + ttft over all rows; decode speed over sentinel-filtered rows.
        agg = model_df.groupby("config").agg(
            accuracy=("is_correct", "mean"),
            ttft=("ttft_s", "mean"),
        )
        clean = model_df[model_df["decode_tokens_per_second"] < TPS_SENTINEL]
        agg["decode_speed"] = clean.groupby("config")["decode_tokens_per_second"].mean()

        # memory for this model, keyed by config.
        mm = mem_df[mem_df["model"] == model_name].set_index("config")["memory_gb"]
        agg["memory_gb"] = mm
        agg = agg.reset_index()

        # pick baseline (prefer f16/f16, else first available config).
        baseline = BASELINE_CONFIG if BASELINE_CONFIG in agg["config"].values else agg["config"].iloc[0]
        base = agg[agg["config"] == baseline].iloc[0]
        if baseline != BASELINE_CONFIG:
            print(f"  [Info] '{BASELINE_CONFIG}' missing; using '{baseline}' as baseline.")

        # relative % vs baseline for each metric (NaN-safe: skip metrics with no data).
        records = []
        for col, label in METRICS:
            if pd.isna(base.get(col)) or base.get(col) in (0, None):
                continue
            for _, r in agg.iterrows():
                if pd.isna(r[col]):
                    continue
                records.append({"config": r["config"], "Metric": label,
                                "Percentage": r[col] / base[col] * 100})
        melted_df = pd.DataFrame(records)
        if melted_df.empty:
            print(f"  [Skip] No plottable metrics for {model_name}.")
            continue

        order = config_order(agg["config"].unique())
        palette = {c: CONFIG_PALETTE.get(c, "#999999") for c in order}

        plt.figure(figsize=(11, 6))
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(data=melted_df, x="Metric", y="Percentage",
                         hue="config", hue_order=order, palette=palette)

        plt.axhline(100, color="red", linestyle="--", linewidth=1.5)
        clean_title = short_model(model_name)
        plt.title(f"KV quantization vs f16/f16 baseline: {clean_title}",
                  fontsize=14, fontweight="bold", pad=15)
        plt.ylabel(f"% relative to {baseline} baseline (100%)", fontsize=11)
        plt.xlabel("")
        plt.ylim(0, max(melted_df["Percentage"].max() + 15, 120))

        for p in ax.patches:
            h = p.get_height()
            if h and h > 0:
                ax.annotate(f"{h:.0f}%", (p.get_x() + p.get_width() / 2., h),
                            ha="center", va="bottom", xytext=(0, 3),
                            textcoords="offset points", fontsize=8, fontweight="bold")

        n = int(model_df.groupby("config").size().max())
        plt.figtext(0.5, 0.01,
                    f"~{n} examples/config — small sample, small gaps may be noise. "
                    f"Memory = whole-process RSS (kv_cache_mib not logged).",
                    ha="center", fontsize=8, color="0.45")
        plt.legend(title="KV config (key/value)", loc="upper right", fontsize=9)
        plt.tight_layout(rect=(0, 0.03, 1, 1))

        safe = f"relative_{clean_title.replace('.', '_').replace('/', '_')}.png"
        out_path = os.path.join(output_folder, safe)
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"  [Saved] {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-folder", default="results_cuda")
    parser.add_argument("--csv", default="longbench_examples.csv")
    parser.add_argument("--summary", default="longbench_summary.json")
    parser.add_argument("--output-folder", default="results_graphs")
    args = parser.parse_args()

    generate_multi_model_plots(
        input_folder=args.input_folder,
        csv_filename=args.csv,
        summary_filename=args.summary,
        output_folder=args.output_folder,
    )
