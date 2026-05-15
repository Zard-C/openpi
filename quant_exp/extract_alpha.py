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


def compute_alpha(teacher_std: float, student_std: float, eps: float = 1e-8) -> float:
    return float(teacher_std / max(student_std, eps))


def compute_per_head_alpha(teacher: torch.Tensor | None, student: torch.Tensor | None, eps: float = 1e-8):
    if teacher is None or student is None:
        return None
    teacher = teacher.to(torch.float32)
    student = student.to(torch.float32)
    return teacher / student.clamp_min(eps)


def build_alpha_entries(
    teacher_stats: dict,
    student_stats: dict,
    *,
    kind: str,
    module_regex: str,
) -> list[dict[str, object]]:
    teacher_attention_stats = teacher_stats.get("attention_stats", {})
    student_attention_stats = student_stats.get("attention_stats", {})
    pattern = re.compile(module_regex)
    common_modules = sorted(set(teacher_attention_stats) & set(student_attention_stats), key=natural_key)

    rows: list[dict[str, object]] = []
    for module_name in common_modules:
        teacher_entry = teacher_attention_stats[module_name]
        student_entry = student_attention_stats[module_name]
        if teacher_entry["kind"] != kind:
            continue
        if not pattern.search(module_name):
            continue

        teacher_std = float(teacher_entry["logits_std_mean"])
        student_std = float(student_entry["logits_std_mean"])
        alpha = compute_alpha(teacher_std, student_std)
        per_head_alpha = compute_per_head_alpha(
            teacher_entry.get("per_head_std_mean"),
            student_entry.get("per_head_std_mean"),
        )
        rows.append(
            {
                "name": module_name,
                "kind": teacher_entry["kind"],
                "teacher_logits_std": teacher_std,
                "student_logits_std": student_std,
                "teacher_logits_absmax": float(teacher_entry["logits_absmax"]),
                "student_logits_absmax": float(student_entry["logits_absmax"]),
                "alpha": alpha,
                "per_head_alpha": per_head_alpha,
                "per_head_alpha_mean": float(per_head_alpha.mean().item()) if per_head_alpha is not None else None,
                "per_head_alpha_max": float(per_head_alpha.max().item()) if per_head_alpha is not None else None,
                "query_scale_last": student_entry.get("query_scale_last"),
                "key_scale_last": student_entry.get("key_scale_last"),
            }
        )
    return rows


def print_alpha_table(rows: list[dict[str, object]], top_k: int) -> None:
    console = Console()
    table = Table(title=f"Top {top_k} Alpha Entries")
    table.add_column("Module", style="cyan")
    table.add_column("Teacher Std", justify="right")
    table.add_column("Student Std", justify="right")
    table.add_column("Alpha", justify="right", style="magenta")
    table.add_column("Per-Head Mean", justify="right")
    table.add_column("Per-Head Max", justify="right")
    table.add_column("Q Scale", justify="right")
    table.add_column("K Scale", justify="right")
    for row in rows[:top_k]:
        table.add_row(
            str(row["name"]),
            f"{float(row['teacher_logits_std']):.6f}",
            f"{float(row['student_logits_std']):.6f}",
            f"{float(row['alpha']):.6f}",
            "-" if row["per_head_alpha_mean"] is None else f"{float(row['per_head_alpha_mean']):.6f}",
            "-" if row["per_head_alpha_max"] is None else f"{float(row['per_head_alpha_max']):.6f}",
            "-" if row["query_scale_last"] is None else f"{float(row['query_scale_last']):.6f}",
            "-" if row["key_scale_last"] is None else f"{float(row['key_scale_last']):.6f}",
        )
    console.print(table)


def save_alpha_results(rows: list[dict[str, object]], teacher_stats: dict, student_stats: dict, output_path: str) -> None:
    payload = {
        "meta": {
            "teacher": teacher_stats.get("meta", {}),
            "student": student_stats.get("meta", {}),
        },
        "alpha": {row["name"]: row for row in rows},
    }
    output_file = pathlib.Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ATM alpha values from teacher/student attention stats.")
    parser.add_argument("--teacher", required=True, help="Teacher stats .pt path.")
    parser.add_argument("--student", required=True, help="Student stats .pt path.")
    parser.add_argument("--kind", default="expert_attention", help="Attention kind to compare.")
    parser.add_argument(
        "--module-regex",
        default=r"self_attn$",
        help="Regex used to select attention modules for alpha extraction.",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Number of alpha rows to print.")
    parser.add_argument("--output", default=None, help="Optional output .pt path for alpha values.")
    args = parser.parse_args()

    teacher_stats = load_stats(args.teacher)
    student_stats = load_stats(args.student)
    rows = build_alpha_entries(
        teacher_stats,
        student_stats,
        kind=args.kind,
        module_regex=args.module_regex,
    )
    if not rows:
        raise ValueError("No matching attention modules found for alpha extraction.")

    print_alpha_table(rows, top_k=args.top_k)
    first_row = rows[0]
    print(f"first_alpha_module: {first_row['name']}")
    print(f"first_alpha_value: {float(first_row['alpha']):.6f}")

    if args.output is not None:
        save_alpha_results(rows, teacher_stats, student_stats, args.output)
        print(f"alpha_output: {args.output}")


if __name__ == "__main__":
    main()