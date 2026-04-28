#!/usr/bin/env python3
"""Generate paper statistics from benchmark result CSVs.

Outputs:
  <out-dir>/pass_rates.tex        — LaTeX table: pass rates per check × mode
  <out-dir>/distributions.tex     — LaTeX table: mean±std / median / P25–P75
  <out-dir>/physics_distributions.pdf
  <out-dir>/grasp_distribution.pdf
  <out-dir>/summary.txt           — plain-text summary of all numbers

Usage:
    python -m habitat_object_benchmark.scripts.generate_paper_stats \
        --results-dir data/datasets/amara-spatial-10k/results \
        --out-dir paper/stats
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(n, total):
    return f"{n} ({100 * n / total:.1f}\\%)" if total else "–"

def _dist_row(series, label, fmt=".4f"):
    s = series.dropna()
    if s.empty:
        return label, "–", "–", "–", "–"
    return (
        label,
        f"{s.mean():{fmt}} ± {s.std():{fmt}}",
        f"{s.median():{fmt}}",
        f"{s.quantile(0.25):{fmt}}",
        f"{s.quantile(0.75):{fmt}}",
    )

def _latex_table(headers, rows, caption, label):
    col_fmt = "l" + "r" * (len(headers) - 1)
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(c) for c in row) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{label}}}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ── Markdown helpers ─────────────────────────────────────────────────────────

# Arrow suffix for each metric column (appended to header label)
METRIC_ARROW = {
    # physics pass-rate columns (higher count = worse for failure metrics)
    "Settles":             " ↑",
    "Flies away":          " ↓",
    "Sinks permanently":   " ↓",
    "Sinks (trajectory)":  " ↓",
    # physics distribution metrics
    "XZ disp":             " ↓",
    "final Y offset":      " ↑",
    "min Y offset":        " ↑",
    "settle time":         " ↓",
    # graspability pass-rate columns
    "Any success":         " ↑",
    "Perfect (100%)":      " ↑",
    "None (0%)":           " ↓",
    "Mean rate":           " ↑",
    # graspability distribution metrics
    "success rate":        " ↑",
    "grasp width":         " ↓",
}

# For each metric, is a higher numeric value better?
METRIC_HIGHER_BETTER = {
    "Settles":             True,
    "Flies away":          False,
    "Sinks permanently":   False,
    "Sinks (trajectory)":  False,
    "XZ disp":             False,
    "final Y offset":      True,
    "min Y offset":        True,
    "settle time":         False,
    "Any success":         True,
    "Perfect (100%)":      True,
    "None (0%)":           False,
    "Mean rate":           True,
    "success rate":        True,
    "grasp width":         False,
}


def _arrow(col_header):
    for key, arrow in METRIC_ARROW.items():
        if key in col_header:
            return arrow
    return ""


def _higher_better(col_header):
    for key, hb in METRIC_HIGHER_BETTER.items():
        if key in col_header:
            return hb
    return None  # unknown — no bolding


def _extract_num(cell_str):
    """Extract the first float from a cell string like '443 (88.6%)' or '0.0709 ± 0.1891'."""
    import re
    m = re.search(r"-?\d+\.?\d*", str(cell_str))
    return float(m.group()) if m else None


def _bold_winners(headers, rows, group_size=None):
    """Return rows with the winner cell bolded.

    group_size: if set, compare rows in groups of that size (e.g. one row per
    mode per metric), bolding the winner within each group independently.
    If None, compare all rows globally (suited for pass-rate tables where
    each row is a different mode for the same set of metrics).

    Direction is determined by the column header first; if the header has no
    known direction, falls back to the row label (col 0) of the first row in
    the group — useful for distribution tables where each group shares a metric.
    """
    if len(rows) < 2:
        return rows
    rows = [list(r) for r in rows]
    n = len(rows)
    gs = group_size or n

    for col_idx, header in enumerate(headers[1:], start=1):
        for start in range(0, n, gs):
            group = rows[start:start + gs]
            # Determine direction: try column header, then row label of first row
            hb = _higher_better(header)
            if hb is None:
                hb = _higher_better(str(group[0][0]))
            if hb is None:
                continue
            nums = [_extract_num(r[col_idx]) for r in group]
            valid = [(i, v) for i, v in enumerate(nums) if v is not None]
            if len(valid) < 2:
                continue
            best_local = max(valid, key=lambda x: x[1] if hb else -x[1])[0]
            best_val = nums[best_local]
            if sum(1 for _, v in valid if v == best_val) == 1:
                rows[start + best_local][col_idx] = f"**{rows[start + best_local][col_idx]}**"
    return rows


def _md_table(headers, rows, bold_winners=False, group_size=None):
    if bold_winners:
        rows = _bold_winners(headers, rows, group_size=group_size)
    # Add arrows to headers
    annotated = [h + _arrow(h) for h in headers]
    sep = "| " + " | ".join("---" for _ in annotated) + " |"
    lines = ["| " + " | ".join(annotated) + " |", sep]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _pct_plain(n, total):
    return f"{n} ({100 * n / total:.1f}%)" if total else "–"


# ── Physics stats ─────────────────────────────────────────────────────────────

def physics_stats(csv_path: Path, out_dir: Path, summary_lines: list, md_sections: list):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    for col in ["physics_settles", "flies_away", "sinks_permanently", "sinks_below_floor"]:
        df[col] = df[col].map({"True": True, "False": False, True: True, False: False})

    modes = df["collision_mode"].unique().tolist()
    total = len(df[df["collision_mode"] == modes[0]])

    summary_lines.append("=== Physics ===")
    summary_lines.append(f"Assets: {total}")

    # ── Pass-rate table ───────────────────────────────────────────────────────
    rate_rows = []
    for mode in modes:
        m = df[df["collision_mode"] == mode]
        n = len(m)
        row = [
            mode.replace("_", "\\_"),
            _pct(int(m["physics_settles"].sum()),    n),
            _pct(int(m["flies_away"].sum()),         n),
            _pct(int(m["sinks_permanently"].sum()),  n),
            _pct(int(m["sinks_below_floor"].sum()),  n),
        ]
        rate_rows.append(row)
        summary_lines.append(
            f"  [{mode}] settles={int(m['physics_settles'].sum())}/{n} "
            f"flies={int(m['flies_away'].sum())} "
            f"sinks_perm={int(m['sinks_permanently'].sum())} "
            f"sinks_traj={int(m['sinks_below_floor'].sum())}"
        )

    rate_tex = _latex_table(
        ["Mode", "Settles", "Flies away", "Sinks permanently", "Sinks (trajectory)"],
        rate_rows,
        "Physics stability pass rates per collision mode.",
        "physics_pass_rates",
    )
    (out_dir / "pass_rates.tex").write_text(rate_tex)

    # ── Distribution table ────────────────────────────────────────────────────
    phys_metrics = [
        ("displacement_m",   "XZ disp (m)"),
        ("final_y_offset_m", "final Y offset (m)"),
        ("min_y_offset_m",   "min Y offset (m)"),
        ("settle_time_s",    "settle time (s)"),
    ]
    dist_rows = []
    dist_rows_plain = []  # interleaved by metric for per-group bolding
    for mode in modes:
        m = df[df["collision_mode"] == mode]
        for col, short in phys_metrics:
            label       = f"{mode.replace('_', chr(95))} {short}"
            label_plain = f"{mode} {short}"
            dist_rows.append(_dist_row(m[col], label))
            s = m[col].dropna()
            summary_lines.append(
                f"  [{mode}] {col}: mean={s.mean():.4f} std={s.std():.4f} "
                f"median={s.median():.4f} P25={s.quantile(0.25):.4f} P75={s.quantile(0.75):.4f}"
            )
    # Build plain rows interleaved by metric so group_size=len(modes) works correctly
    for col, short in phys_metrics:
        for mode in modes:
            m = df[df["collision_mode"] == mode]
            dist_rows_plain.append(_dist_row(m[col], f"{mode} {short}"))

    dist_tex = _latex_table(
        ["Metric", "Mean ± Std", "Median", "P25", "P75"],
        dist_rows,
        "Distribution of physics stability metrics.",
        "physics_distributions",
    )
    (out_dir / "distributions.tex").write_text(dist_tex)

    # ── Markdown ──────────────────────────────────────────────────────────────
    md_rate_rows = []
    for mode in modes:
        m = df[df["collision_mode"] == mode]
        n = len(m)
        md_rate_rows.append([
            mode,
            _pct_plain(int(m["physics_settles"].sum()), n),
            _pct_plain(int(m["flies_away"].sum()),      n),
            _pct_plain(int(m["sinks_permanently"].sum()), n),
            _pct_plain(int(m["sinks_below_floor"].sum()), n),
        ])

    md_sections.append("## Physics Stability\n")
    md_sections.append(f"**Assets evaluated:** {total}\n")
    md_sections.append("### Pass Rates\n")
    md_sections.append(_md_table(
        ["Mode", "Settles", "Flies away", "Sinks permanently", "Sinks (trajectory)"],
        md_rate_rows,
        bold_winners=True,
    ))
    md_sections.append("\n### Metric Distributions\n")
    md_sections.append(_md_table(
        ["Metric", "Mean ± Std", "Median", "P25", "P75"],
        dist_rows_plain,
        bold_winners=True,
        group_size=len(modes),
    ))
    md_sections.append("\n### Per-asset Results\n")
    for mode in modes:
        m = df[df["collision_mode"] == mode].copy()
        m = m.sort_values("asset_id")
        md_sections.append(f"#### {mode}\n")
        per_asset_rows = []
        for _, row in m.iterrows():
            per_asset_rows.append([
                row["asset_id"],
                "✅" if row["physics_settles"] else "❌",
                f"{row['displacement_m']:.4f}" if pd.notna(row["displacement_m"]) else "–",
                "⚠️" if row["flies_away"] else "ok",
                f"{row['final_y_offset_m']:.4f}" if pd.notna(row["final_y_offset_m"]) else "–",
                "⚠️" if row["sinks_permanently"] else "ok",
                f"{row['min_y_offset_m']:.4f}" if pd.notna(row["min_y_offset_m"]) else "–",
                "⚠️" if row["sinks_below_floor"] else "ok",
                f"{row['settle_time_s']:.3f}" if "settle_time_s" in row and pd.notna(row["settle_time_s"]) else "–",
                row["error"] if pd.notna(row["error"]) else "",
            ])
        md_sections.append(_md_table(
            ["Asset", "Settles", "XZ disp (m)", "Flies away", "Final Y offset (m)", "Sinks permanently", "Min Y offset (m)", "Sinks (trajectory)", "Settle time (s)", "Error"],
            per_asset_rows,
        ))
        md_sections.append("")

    # ── Figures ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metrics = [
        ("displacement_m",   "XZ Displacement (m)",  1.5),
        ("final_y_offset_m", "Final Y Offset (m)",   None),
        ("min_y_offset_m",   "Min Y Offset (m)",     None),
        ("settle_time_s",    "Settle Time (s)",       None),
    ]
    colors = {"convex_hull": "#3498db", "vhacd": "#e67e22"}
    for ax, (col, xlabel, vline) in zip(axes, metrics):
        for mode in modes:
            vals = df[df["collision_mode"] == mode][col].dropna()
            ax.hist(vals, bins=40, alpha=0.6, label=mode.replace("_", " "), color=colors.get(mode))
        if vline is not None:
            ax.axvline(vline, color="red", linestyle="--", linewidth=1, label="threshold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    fig.suptitle("Physics Stability Metric Distributions")
    fig.tight_layout()
    fig.savefig(out_dir / "physics_distributions.pdf", bbox_inches="tight")
    plt.close(fig)


# ── Graspability stats ────────────────────────────────────────────────────────

def grasp_stats(csv_path: Path, out_dir: Path, summary_lines: list, md_sections: list):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    modes = df["collision_mode"].unique().tolist()
    total = len(df[df["collision_mode"] == modes[0]])

    summary_lines.append("\n=== Graspability ===")
    summary_lines.append(f"Assets: {total}")

    rate_rows = []
    for mode in modes:
        m = df[df["collision_mode"] == mode]
        n = len(m)
        any_success  = int((m["grasp_success_rate"] > 0).sum())
        full_success = int((m["grasp_success_rate"] == 1.0).sum())
        zero         = int((m["grasp_success_rate"] == 0.0).sum())
        row = [
            mode.replace("_", "\\_"),
            _pct(any_success, n),
            _pct(full_success, n),
            _pct(zero, n),
            f"{m['grasp_success_rate'].mean():.3f}",
        ]
        rate_rows.append(row)
        summary_lines.append(
            f"  [{mode}] any_success={any_success}/{n} "
            f"perfect={full_success} zero={zero} "
            f"mean_rate={m['grasp_success_rate'].mean():.3f}"
        )

    rate_tex = _latex_table(
        ["Mode", "Any success", "Perfect (100\\%)", "None (0\\%)", "Mean rate"],
        rate_rows,
        "Graspability pass rates per collision mode.",
        "grasp_pass_rates",
    )
    # Append to pass_rates.tex
    existing = (out_dir / "pass_rates.tex").read_text()
    (out_dir / "pass_rates.tex").write_text(existing + "\n\n" + rate_tex)

    # Distribution rows
    grasp_metrics = [
        ("grasp_success_rate", "success rate"),
        ("mean_grasp_width_m", "grasp width (m)"),
    ]
    dist_rows = []
    dist_rows_plain = []
    for mode in modes:
        m = df[df["collision_mode"] == mode]
        for col, short in grasp_metrics:
            dist_rows.append(_dist_row(m[col], f"{mode.replace('_', chr(95))} {short}", fmt=".3f"))
            s = m[col].dropna()
            summary_lines.append(
                f"  [{mode}] {col}: mean={s.mean():.4f} std={s.std():.4f} "
                f"median={s.median():.4f} P25={s.quantile(0.25):.4f} P75={s.quantile(0.75):.4f}"
            )
    # Interleaved by metric for per-group bolding
    for col, short in grasp_metrics:
        for mode in modes:
            m = df[df["collision_mode"] == mode]
            dist_rows_plain.append(_dist_row(m[col], f"{mode} {short}", fmt=".3f"))

    existing = (out_dir / "distributions.tex").read_text()
    dist_tex = _latex_table(
        ["Metric", "Mean ± Std", "Median", "P25", "P75"],
        dist_rows,
        "Distribution of graspability metrics.",
        "grasp_distributions",
    )
    (out_dir / "distributions.tex").write_text(existing + "\n\n" + dist_tex)

    # Markdown
    md_rate_rows = []
    for mode in modes:
        m = df[df["collision_mode"] == mode]
        n = len(m)
        any_success  = int((m["grasp_success_rate"] > 0).sum())
        full_success = int((m["grasp_success_rate"] == 1.0).sum())
        zero         = int((m["grasp_success_rate"] == 0.0).sum())
        md_rate_rows.append([
            mode,
            _pct_plain(any_success, n),
            _pct_plain(full_success, n),
            _pct_plain(zero, n),
            f"{m['grasp_success_rate'].mean():.3f}",
        ])

    md_sections.append("\n## Graspability\n")
    md_sections.append(f"**Assets evaluated:** {total}\n")
    md_sections.append("### Pass Rates\n")
    md_sections.append(_md_table(
        ["Mode", "Any success", "Perfect (100%)", "None (0%)", "Mean rate"],
        md_rate_rows,
        bold_winners=True,
    ))
    md_sections.append("\n### Metric Distributions\n")
    md_sections.append(_md_table(
        ["Metric", "Mean ± Std", "Median", "P25", "P75"],
        dist_rows_plain,
        bold_winners=True,
        group_size=len(modes),
    ))
    md_sections.append("\n### Per-asset Results\n")
    for mode in modes:
        m = df[df["collision_mode"] == mode].copy()
        m = m.sort_values("asset_id")
        md_sections.append(f"#### {mode}\n")
        per_asset_rows = []
        for _, row in m.iterrows():
            pct = f"{row['grasp_success_rate'] * 100:.0f}%" if pd.notna(row["grasp_success_rate"]) else "–"
            per_asset_rows.append([
                row["asset_id"],
                pct,
                f"{int(row['grasp_successes'])}/{int(row['grasp_trials'])}" if pd.notna(row["grasp_successes"]) else "–",
                f"{row['mean_grasp_width_m']:.4f}" if pd.notna(row["mean_grasp_width_m"]) else "–",
                row["error"] if pd.notna(row["error"]) else "",
            ])
        md_sections.append(_md_table(
            ["Asset", "success rate", "Successes", "grasp width (m)", "Error"],
            per_asset_rows,
        ))
        md_sections.append("")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    colors = {"convex_hull": "#3498db", "vhacd": "#e67e22"}
    for ax, (col, xlabel) in zip(axes, [
        ("grasp_success_rate", "Grasp Success Rate"),
        ("mean_grasp_width_m", "Mean Grasp Width (m)"),
    ]):
        for mode in modes:
            vals = df[df["collision_mode"] == mode][col].dropna()
            ax.hist(vals, bins=30, alpha=0.6, label=mode.replace("_", " "), color=colors.get(mode))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    fig.suptitle("Graspability Metric Distributions")
    fig.tight_layout()
    fig.savefig(out_dir / "grasp_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


# ── Comparison table (multi-dataset) ─────────────────────────────────────────

def _comparison_section(datasets: dict, check: str, metrics: list, md_sections: list):
    """Build a cross-dataset comparison table for one check type.

    datasets: { name: results_dir }
    metrics:  [(csv_col, label, fmt, higher_better)]
    """
    # Collect per-dataset, per-mode stats
    rows_by_metric = {label: [] for _, label, _, _ in metrics}
    dataset_mode_cols = []  # column headers: "name (mode)"

    for ds_name, results_dir in datasets.items():
        csv_path = results_dir / check / f"{check}_results.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        for col in ["physics_settles", "flies_away", "sinks_permanently", "sinks_below_floor"]:
            if col in df.columns:
                df[col] = df[col].map({"True": True, "False": False, True: True, False: False})
        modes = df["collision_mode"].unique().tolist()
        for mode in modes:
            col_label = f"{ds_name} ({mode})"
            dataset_mode_cols.append(col_label)
            m = df[df["collision_mode"] == mode]
            for csv_col, label, fmt, _ in metrics:
                if csv_col in df.columns:
                    s = m[csv_col]
                    if s.dtype == bool or set(s.dropna().unique()).issubset({True, False}):
                        val = f"{s.sum()} ({100*s.mean():.1f}%)"
                    else:
                        val = f"{s.mean():{fmt}}"
                else:
                    val = "–"
                rows_by_metric[label].append(val)

    if not dataset_mode_cols:
        return

    headers = ["Metric"] + dataset_mode_cols
    rows = []
    for _, label, _, hb in metrics:
        row_vals = rows_by_metric[label]
        # Bold winner
        nums = [_extract_num(v) for v in row_vals]
        valid = [(i, v) for i, v in enumerate(nums) if v is not None]
        best_i = None
        if len(valid) >= 2:
            best_i = max(valid, key=lambda x: x[1] if hb else -x[1])[0]
            best_val = nums[best_i]
            if sum(1 for _, v in valid if v == best_val) > 1:
                best_i = None  # tie
        cells = []
        for i, v in enumerate(row_vals):
            cells.append(f"**{v}**" if i == best_i else v)
        rows.append([label + _arrow(label)] + cells)

    md_sections.append(f"\n## {check.title()} — Cross-dataset Comparison\n")
    md_sections.append(_md_table(headers, rows))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results-dir", type=Path,
                       help="Single dataset results dir")
    group.add_argument("--dataset", metavar="NAME=PATH", action="append", dest="datasets",
                       help="Multi-dataset: repeatable NAME=path/to/results pairs")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve dataset dict
    if args.results_dir:
        datasets = {args.results_dir.parent.name: args.results_dir}
    else:
        datasets = {}
        for entry in args.datasets:
            if "=" not in entry:
                parser.error(f"--dataset must be NAME=PATH, got: {entry!r}")
            name, path = entry.split("=", 1)
            datasets[name] = Path(path)

    summary_lines = []
    md_sections   = ["# Benchmark Results\n"]

    # Per-dataset sections
    for ds_name, results_dir in datasets.items():
        md_sections.append(f"\n---\n# Dataset: {ds_name}\n")
        summary_lines.append(f"\n{'='*40}\nDataset: {ds_name}\n{'='*40}")

        physics_csv = results_dir / "physics" / "physics_results.csv"
        if physics_csv.exists():
            print(f"[{ds_name}] Processing physics...")
            physics_stats(physics_csv, args.out_dir / ds_name, summary_lines, md_sections)
        else:
            print(f"[{ds_name}] no physics CSV")

        grasp_csv = results_dir / "graspability" / "graspability_results.csv"
        if grasp_csv.exists():
            print(f"[{ds_name}] Processing graspability...")
            grasp_stats(grasp_csv, args.out_dir / ds_name, summary_lines, md_sections)
        else:
            print(f"[{ds_name}] no graspability CSV")

    # Cross-dataset comparison (only meaningful with 2+ datasets)
    if len(datasets) > 1:
        _comparison_section(datasets, "physics", [
            ("physics_settles",  "Settles",            ".1%", True),
            ("displacement_m",   "XZ disp (m)",        ".4f", False),
            ("final_y_offset_m", "final Y offset (m)", ".4f", True),
            ("min_y_offset_m",   "min Y offset (m)",   ".4f", True),
            ("settle_time_s",    "settle time (s)",    ".3f", False),
            ("flies_away",       "Flies away",         ".1%", False),
            ("sinks_permanently","Sinks permanently",  ".1%", False),
            ("sinks_below_floor","Sinks (trajectory)", ".1%", False),
        ], md_sections)
        _comparison_section(datasets, "graspability", [
            ("grasp_success_rate", "success rate",  ".3f", True),
            ("mean_grasp_width_m", "grasp width (m)",".4f", False),
        ], md_sections)

    (args.out_dir / "summary.txt").write_text("\n".join(summary_lines))
    (args.out_dir / "results.md").write_text("\n".join(md_sections))

    print(f"\nOutputs written to {args.out_dir}/")
    print("  results.md, summary.txt, pass_rates.tex, distributions.tex, *.pdf")
    print("  summary.txt")
    print("  results.md")


if __name__ == "__main__":
    main()
