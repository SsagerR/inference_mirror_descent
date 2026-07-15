#!/usr/bin/env python3
"""Inspect toy exponential-energy diffusion vs MALA sweep CSVs for missing or suspicious cells."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


METHODS = ["exponential_energy", "orig_energy_mala"]
DEFAULT_METRICS = [
    "js",
    "kl_sample_target",
    "w1",
    "ks",
    "sliced_w1",
    "marginal_ks_x",
    "marginal_ks_y",
    "sample_mean_q",
    "target_mean_q",
    "acceptance_rate",
    "weight_max",
    "dataset_ess_over_N",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-file", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--stage", default="final_clean")
    p.add_argument("--expected-seeds", type=int, default=3)
    p.add_argument("--metrics", default="auto")
    return p.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def as_float(value, default=math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def beta_value(row: dict[str, str]) -> str:
    return row.get("beta_input") or row.get("beta") or ""


def numeric_sort_key(value: str) -> tuple[float, str]:
    f = as_float(value)
    return (f if math.isfinite(f) else math.inf, str(value))


def sorted_unique(values) -> list[str]:
    return sorted({str(v) for v in values if v not in {None, ""}}, key=numeric_sort_key)


def cell_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (str(row.get("sample_budget", "")), str(beta_value(row)), str(row.get("mala_steps", "")))


def method_cell_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (*cell_key(row), str(row.get("method", "")))


def present_metrics(rows: list[dict[str, str]], requested: str) -> list[str]:
    available = set().union(*(row.keys() for row in rows)) if rows else set()
    if requested.strip().lower() == "auto":
        return [m for m in DEFAULT_METRICS if m in available]
    return [m for m in requested.replace(",", " ").split() if m.strip() and m.strip() in available]


def metric_required_for_method(metric: str, method: str) -> bool:
    if metric == "acceptance_rate" and method == "exponential_energy":
        return False
    return True


def audit_method_cells(rows: list[dict[str, str]], metrics: list[str], expected_seeds: int) -> list[dict]:
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[method_cell_key(row)].append(row)

    out = []
    all_cells = sorted({cell_key(row) for row in rows}, key=lambda k: (numeric_sort_key(k[0]), numeric_sort_key(k[1]), numeric_sort_key(k[2])))
    for budget, beta, steps in all_cells:
        for method in METHODS:
            group = groups.get((budget, beta, steps, method), [])
            seeds = sorted_unique(row.get("dataset_seed") for row in group)
            metric_nan_counts = {}
            for metric in metrics:
                if metric_required_for_method(metric, method):
                    metric_nan_counts[f"{metric}_nan_count"] = sum(
                        1 for row in group if not math.isfinite(as_float(row.get(metric)))
                    )
                else:
                    metric_nan_counts[f"{metric}_nan_count"] = 0
            status = "ok"
            if len(seeds) != expected_seeds or len(group) != expected_seeds or any(v for v in metric_nan_counts.values()):
                status = "bad"
            row_out = {
                "sample_budget": budget,
                "beta": beta,
                "mala_steps": steps,
                "method": method,
                "n_rows": len(group),
                "n_seeds": len(seeds),
                "seeds": " ".join(seeds),
                "expected_seeds": expected_seeds,
                "status": status,
            }
            row_out.update(metric_nan_counts)
            out.append(row_out)
    return out


def paired_metric_rows(rows: list[dict[str, str]], metric: str) -> list[dict]:
    groups: dict[tuple[str, str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        budget, beta, steps = cell_key(row)
        seed = str(row.get("dataset_seed", ""))
        groups[(budget, beta, steps, seed)][str(row.get("method", ""))] = row

    out = []
    for (budget, beta, steps, seed), methods in sorted(
        groups.items(),
        key=lambda kv: (numeric_sort_key(kv[0][0]), numeric_sort_key(kv[0][1]), numeric_sort_key(kv[0][2]), numeric_sort_key(kv[0][3])),
    ):
        exp_row = methods.get("exponential_energy")
        mala_row = methods.get("orig_energy_mala")
        exp_value = as_float(exp_row.get(metric) if exp_row else None)
        mala_value = as_float(mala_row.get(metric) if mala_row else None)
        delta = exp_value - mala_value if math.isfinite(exp_value) and math.isfinite(mala_value) else math.nan
        out.append({
            "metric": metric,
            "sample_budget": budget,
            "beta": beta,
            "mala_steps": steps,
            "dataset_seed": seed,
            "exponential_energy": exp_value,
            "orig_energy_mala": mala_value,
            "delta_exp_minus_mala": delta,
            "winner_lower_is_better": "exponential_energy" if math.isfinite(delta) and delta < 0 else ("orig_energy_mala" if math.isfinite(delta) and delta > 0 else ""),
            "has_both_methods": int(exp_row is not None and mala_row is not None),
        })
    return out


def summarize_cells(paired_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in paired_rows:
        grouped[(row["metric"], row["sample_budget"], row["beta"], row["mala_steps"])].append(row)

    out = []
    for (metric, budget, beta, steps), group in sorted(
        grouped.items(),
        key=lambda kv: (kv[0][0], numeric_sort_key(kv[0][1]), numeric_sort_key(kv[0][2]), numeric_sort_key(kv[0][3])),
    ):
        exp_vals = [float(r["exponential_energy"]) for r in group if math.isfinite(float(r["exponential_energy"]))]
        mala_vals = [float(r["orig_energy_mala"]) for r in group if math.isfinite(float(r["orig_energy_mala"]))]
        deltas = [float(r["delta_exp_minus_mala"]) for r in group if math.isfinite(float(r["delta_exp_minus_mala"]))]
        out.append({
            "metric": metric,
            "sample_budget": budget,
            "beta": beta,
            "mala_steps": steps,
            "n_paired_seeds": len(deltas),
            "exp_mean": mean(exp_vals) if exp_vals else math.nan,
            "exp_min": min(exp_vals) if exp_vals else math.nan,
            "exp_max": max(exp_vals) if exp_vals else math.nan,
            "mala_mean": mean(mala_vals) if mala_vals else math.nan,
            "mala_min": min(mala_vals) if mala_vals else math.nan,
            "mala_max": max(mala_vals) if mala_vals else math.nan,
            "delta_mean_exp_minus_mala": mean(deltas) if deltas else math.nan,
            "delta_median_exp_minus_mala": median(deltas) if deltas else math.nan,
            "exp_seed_wins": sum(1 for d in deltas if d < 0),
            "mala_seed_wins": sum(1 for d in deltas if d > 0),
        })
    return out


def summarize_by_beta(cell_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in cell_rows:
        grouped[(row["metric"], row["beta"])].append(row)

    out = []
    for (metric, beta), group in sorted(grouped.items(), key=lambda kv: (kv[0][0], numeric_sort_key(kv[0][1]))):
        deltas = [as_float(r["delta_mean_exp_minus_mala"]) for r in group]
        deltas = [d for d in deltas if math.isfinite(d)]
        out.append({
            "metric": metric,
            "beta": beta,
            "n_cells": len(deltas),
            "exp_cell_wins": sum(1 for d in deltas if d < 0),
            "mala_cell_wins": sum(1 for d in deltas if d > 0),
            "mean_delta_exp_minus_mala": mean(deltas) if deltas else math.nan,
            "median_delta_exp_minus_mala": median(deltas) if deltas else math.nan,
        })
    return out


def main() -> None:
    args = parse_args()
    all_rows = read_rows(args.metrics_file)
    rows = [row for row in all_rows if (row.get("stage") or "") == args.stage]
    if not rows:
        raise SystemExit(f"no rows with stage={args.stage!r} in {args.metrics_file}")
    metrics = present_metrics(rows, args.metrics)
    if not metrics:
        raise SystemExit("no requested metrics are present in the CSV")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_method_cells(rows, metrics, args.expected_seeds)
    write_rows(args.out_dir / "audit_method_cells.csv", audit)

    all_cell_summaries = []
    for metric in metrics:
        paired = paired_metric_rows(rows, metric)
        cell_summary = summarize_cells(paired)
        by_beta = summarize_by_beta(cell_summary)
        write_rows(args.out_dir / "per_seed" / f"{metric}.csv", paired)
        write_rows(args.out_dir / "per_cell" / f"{metric}.csv", cell_summary)
        write_rows(args.out_dir / "by_beta" / f"{metric}.csv", by_beta)
        all_cell_summaries.extend(cell_summary)
    write_rows(args.out_dir / "all_cell_summaries.csv", all_cell_summaries)

    bad = [row for row in audit if row["status"] != "ok"]
    print(f"rows at stage={args.stage}: {len(rows)}")
    print(f"metrics inspected: {', '.join(metrics)}")
    print(f"method-cell audit rows: {len(audit)}")
    print(f"bad method cells: {len(bad)}")
    if bad:
        print("first bad cells:")
        for row in bad[:10]:
            print(row)
    print(f"wrote inspection tables to {args.out_dir}")


if __name__ == "__main__":
    main()
