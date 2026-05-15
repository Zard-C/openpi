import argparse
import pathlib
import re

import torch
from rich.console import Console
from rich.table import Table


def load_stats(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def natural_key(value: str) -> list[object]:
    parts = re.split(r"(\d+)", value)
    key: list[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return key


def compute_beta(teacher_value: float, student_value: float, eps: float = 1e-8) -> float:
    return float(teacher_value / max(student_value, eps))


def compute_channel_beta(teacher: torch.Tensor | None, student: torch.Tensor | None, eps: float = 1e-8):
    if teacher is None or student is None:
        return None
    teacher = teacher.to(torch.float32)
    student = student.to(torch.float32)
    return teacher / student.clamp_min(eps)


def build_beta_entries(
    teacher_stats: dict,
    student_stats: dict,
    *,
    kind: str,
    module_regex: str,
) -> list[dict[str, object]]:
    teacher_hook_stats = teacher_stats["hook_stats"]
    student_hook_stats = student_stats["hook_stats"]
    common_modules = sorted(set(teacher_hook_stats) & set(student_hook_stats), key=natural_key)
    pattern = re.compile(module_regex)

    rows: list[dict[str, object]] = []
    for module_name in common_modules:
        teacher_entry = teacher_hook_stats[module_name]
        student_entry = student_hook_stats[module_name]

        if teacher_entry["kind"] != kind:
            continue
        if not pattern.search(module_name):
            continue

        teacher_output_rms = float(teacher_entry["output"]["rms_mean"])
        student_output_rms = float(student_entry["output"]["rms_mean"])
        beta = compute_beta(teacher_output_rms, student_output_rms)
        channel_beta = compute_channel_beta(
            teacher_entry["output"].get("channel_rms"),
            student_entry["output"].get("channel_rms"),
        )

        rows.append(
            {
                "name": module_name,
                "kind": teacher_entry["kind"],
                "teacher_output_rms": teacher_output_rms,
                "student_output_rms": student_output_rms,
                "teacher_output_absmax": float(teacher_entry["output"]["absmax"]),
                "student_output_absmax": float(student_entry["output"]["absmax"]),
                "beta": beta,
                "channel_beta": channel_beta,
                "channel_beta_mean": float(channel_beta.mean().item()) if channel_beta is not None else None,
                "channel_beta_max": float(channel_beta.max().item()) if channel_beta is not None else None,
            }
        )

    return rows


def print_beta_table(rows: list[dict[str, object]], top_k: int) -> None:
    console = Console()
    table = Table(title=f"Top {top_k} Beta Entries")
    table.add_column("Module", style="cyan")
    table.add_column("Teacher RMS", justify="right")
    table.add_column("Student RMS", justify="right")
    table.add_column("Beta", justify="right", style="magenta")
    table.add_column("Channel Beta Mean", justify="right")
    table.add_column("Channel Beta Max", justify="right")
    for row in rows[:top_k]:
        table.add_row(
            str(row["name"]),
            f"{float(row['teacher_output_rms']):.6f}",
            f"{float(row['student_output_rms']):.6f}",
            f"{float(row['beta']):.6f}",
            "-" if row["channel_beta_mean"] is None else f"{float(row['channel_beta_mean']):.6f}",
            "-" if row["channel_beta_max"] is None else f"{float(row['channel_beta_max']):.6f}",
        )
    console.print(table)


def save_beta_results(rows: list[dict[str, object]], teacher_stats: dict, student_stats: dict, output_path: str) -> None:
    payload = {
        "meta": {
            "teacher": teacher_stats.get("meta", {}),
            "student": student_stats.get("meta", {}),
        },
        "beta": {row["name"]: row for row in rows},
    }
    output_file = pathlib.Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract OHB beta values from teacher/student hook stats.")
    parser.add_argument("--teacher", required=True, help="Teacher stats .pt path.")
    parser.add_argument("--student", required=True, help="Student stats .pt path.")
    parser.add_argument("--kind", default="expert_mlp", help="Linear kind to compare.")
    parser.add_argument(
        "--module-regex",
        default=r"mlp\.down_proj$",
        help="Regex used to select modules for beta extraction.",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Number of beta rows to print.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output .pt path for extracted beta values.",
    )
    args = parser.parse_args()

    teacher_stats = load_stats(args.teacher)
    student_stats = load_stats(args.student)
    rows = build_beta_entries(
        teacher_stats,
        student_stats,
        kind=args.kind,
        module_regex=args.module_regex,
    )
    if not rows:
        raise ValueError("No matching modules found for beta extraction.")

    print_beta_table(rows, top_k=args.top_k)
    first_row = rows[0]
    print(f"first_beta_module: {first_row['name']}")
    print(f"first_beta_value: {float(first_row['beta']):.6f}")

    if args.output is not None:
        save_beta_results(rows, teacher_stats, student_stats, args.output)
        print(f"beta_output: {args.output}")


if __name__ == "__main__":
    main()