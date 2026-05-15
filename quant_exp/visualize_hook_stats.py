import argparse
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from rich.console import Console
from rich.table import Table


def load_stats(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def build_rows(data: dict, kind_filter: str | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    module_infos = data.get("module_infos", {})
    hook_stats = data.get("hook_stats", {})
    for module_name, stats in hook_stats.items():
        kind = stats["kind"]
        if kind_filter and kind != kind_filter:
            continue
        module_info = module_infos.get(module_name, {})
        rows.append(
            {
                "name": module_name,
                "kind": kind,
                "calls": int(stats["calls"]),
                "input_rms_mean": float(stats["input"]["rms_mean"]),
                "output_rms_mean": float(stats["output"]["rms_mean"]),
                "input_absmax": float(stats["input"]["absmax"]),
                "output_absmax": float(stats["output"]["absmax"]),
                "weight_absmax": float(module_info.get("weight_absmax", 0.0)),
                "weight_rms": float(module_info.get("weight_rms", 0.0)),
                "in_features": int(module_info.get("in_features", 0)),
                "out_features": int(module_info.get("out_features", 0)),
            }
        )
    return rows


def print_meta_summary(data: dict) -> None:
    console = Console()
    meta = data.get("meta", {})
    table = Table(title="Calibration Stats Metadata")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="magenta")
    for key, value in meta.items():
        table.add_row(str(key), str(value))
    console.print(table)


def print_kind_summary(data: dict, kind_filter: str | None = None) -> None:
    aggregates = data.get("aggregates", {}).get("by_kind", {})
    console = Console()
    table = Table(title="Kind Aggregates")
    table.add_column("Kind", style="cyan")
    table.add_column("Modules", justify="right")
    table.add_column("Calls", justify="right")
    table.add_column("In RMS", justify="right")
    table.add_column("Out RMS", justify="right")
    table.add_column("In AbsMax", justify="right")
    table.add_column("Out AbsMax", justify="right")
    table.add_column("W AbsMax", justify="right")
    for kind, stats in aggregates.items():
        if kind_filter and kind != kind_filter:
            continue
        table.add_row(
            kind,
            str(stats["module_count"]),
            str(stats["calls_total"]),
            f"{float(stats['input_rms_mean']):.5f}",
            f"{float(stats['output_rms_mean']):.5f}",
            f"{float(stats['input_absmax_mean']):.5f}",
            f"{float(stats['output_absmax_mean']):.5f}",
            f"{float(stats['weight_absmax_mean']):.5f}",
        )
    console.print(table)


def print_top_modules(rows: list[dict[str, object]], metric: str, top_k: int) -> None:
    console = Console()
    table = Table(title=f"Top {top_k} Modules by {metric}")
    table.add_column("Kind", style="cyan")
    table.add_column("Module", style="white")
    table.add_column(metric, justify="right", style="magenta")
    table.add_column("Calls", justify="right")
    table.add_column("Shape", justify="right")
    sorted_rows = sorted(rows, key=lambda row: float(row[metric]), reverse=True)[:top_k]
    for row in sorted_rows:
        shape = f"{row['in_features']}->{row['out_features']}"
        table.add_row(row["kind"], row["name"], f"{float(row[metric]):.5f}", str(row["calls"]), shape)
    console.print(table)


def plot_kind_aggregates(data: dict, output_dir: pathlib.Path, kind_filter: str | None = None) -> None:
    aggregates = data.get("aggregates", {}).get("by_kind", {})
    items = [(kind, stats) for kind, stats in aggregates.items() if not kind_filter or kind == kind_filter]
    if not items:
        return
    kinds = [kind for kind, _ in items]
    input_rms = [float(stats["input_rms_mean"]) for _, stats in items]
    output_rms = [float(stats["output_rms_mean"]) for _, stats in items]
    x = range(len(kinds))
    plt.figure(figsize=(10, 4))
    plt.bar(x, input_rms, width=0.4, label="input_rms_mean")
    plt.bar([idx + 0.4 for idx in x], output_rms, width=0.4, label="output_rms_mean")
    plt.xticks([idx + 0.2 for idx in x], kinds, rotation=45, ha="right")
    plt.ylabel("RMS")
    plt.title("Activation RMS by Linear Kind")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "kind_rms.png", dpi=160)
    plt.close()


def plot_top_modules(rows: list[dict[str, object]], metric: str, output_dir: pathlib.Path, top_k: int) -> None:
    selected = sorted(rows, key=lambda row: float(row[metric]), reverse=True)[:top_k]
    if not selected:
        return
    names = [row["name"].split(".")[-3:] for row in selected]
    labels = [".".join(name) for name in names]
    values = [float(row[metric]) for row in selected]
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(f"Top {top_k} Modules by {metric}")
    plt.tight_layout()
    plt.savefig(output_dir / f"top_{metric}.png", dpi=160)
    plt.close()


def plot_channel_profile(data: dict, output_dir: pathlib.Path, module_name: str) -> None:
    stats = data["hook_stats"][module_name]
    module_info = data["module_infos"][module_name]
    input_absmax = stats["input"]["channel_absmax"]
    output_absmax = stats["output"]["channel_absmax"]
    output_rms = stats["output"]["channel_rms"]
    weight_absmax = module_info["weight_per_out_channel_absmax"]

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    if input_absmax is not None:
        plt.plot(input_absmax.numpy())
    plt.title(f"Input Channel AbsMax: {module_name}")
    plt.ylabel("absmax")

    plt.subplot(3, 1, 2)
    if output_absmax is not None:
        plt.plot(output_absmax.numpy())
    plt.title("Output Channel AbsMax")
    plt.ylabel("absmax")

    plt.subplot(3, 1, 3)
    plt.plot(weight_absmax.numpy(), label="weight_absmax")
    if output_rms is not None:
        plt.plot(output_rms.numpy(), label="output_rms")
    plt.title("Weight / Output Channel Profile")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"channel_profile_{sanitize_filename(module_name)}.png", dpi=160)
    plt.close()


def select_channel_module(rows: list[dict[str, object]], requested_module: str | None) -> str | None:
    if requested_module is not None:
        return requested_module
    if not rows:
        return None
    top_row = max(rows, key=lambda row: float(row["output_absmax"]))
    return str(top_row["name"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize linear hook calibration stats.")
    parser.add_argument("--stats", required=True, help="Path to the .pt stats file created by bf16_baseline.py.")
    parser.add_argument(
        "--output-dir",
        default="/root/autodl-tmp/openpi/quant_exp/artifacts/visualizations",
        help="Directory for generated plots.",
    )
    parser.add_argument("--top-k", type=int, default=12, help="Number of top modules to display and plot.")
    parser.add_argument("--kind", default=None, help="Optional kind filter, for example expert_mlp.")
    parser.add_argument("--module", default=None, help="Optional module name used for the channel profile plot.")
    args = parser.parse_args()

    data = load_stats(args.stats)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(data, kind_filter=args.kind)
    print_meta_summary(data)
    print_kind_summary(data, kind_filter=args.kind)
    print_top_modules(rows, "output_absmax", args.top_k)
    print_top_modules(rows, "weight_absmax", args.top_k)

    plot_kind_aggregates(data, output_dir, kind_filter=args.kind)
    plot_top_modules(rows, "output_absmax", output_dir, args.top_k)
    plot_top_modules(rows, "weight_absmax", output_dir, args.top_k)

    selected_module = select_channel_module(rows, args.module)
    if selected_module is not None:
        plot_channel_profile(data, output_dir, selected_module)
        print(f"channel_profile_module: {selected_module}")

    print(f"visualization_output_dir: {output_dir}")


if __name__ == "__main__":
    main()
