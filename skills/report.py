from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from schemas.experiment_ir import ExperimentIR


def build_summary_table(experiments: List[ExperimentIR]) -> str:
    header = ["run_id", "action_id", "runtime_s", "verdict"]
    rows = [header]
    for exp in experiments:
        action_id = exp.action.action_id if exp.action else "baseline"
        rows.append(
            [
                exp.run_id,
                action_id,
                f"{exp.results.runtime_seconds:.4f}",
                exp.verdict,
            ]
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = []
    for row in rows:
        line = " | ".join(col.ljust(widths[i]) for i, col in enumerate(row))
        lines.append(line)
    return "\n".join(lines)


def write_report(
    experiments: List[ExperimentIR],
    baseline: ExperimentIR,
    best: Optional[ExperimentIR],
    report_dir: Path,
) -> Dict[str, object]:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_table = build_summary_table(experiments)
    failures = [
        {
            "run_id": exp.run_id,
            "action_id": exp.action.action_id if exp.action else "baseline",
            "reasons": exp.reasons,
        }
        for exp in experiments
        if exp.verdict == "FAIL"
    ]

    best_action = best.action.action_id if best and best.action else "baseline"
    best_speedup = None
    if best:
        best_speedup = best.results.derived_metrics.get("speedup_vs_baseline")
    report = {
        "baseline_runtime_seconds": baseline.results.runtime_seconds,
        "best_action_id": best_action,
        "best_runtime_seconds": best.results.runtime_seconds if best else None,
        "best_speedup_vs_baseline": best_speedup,
        "experiments": [exp.model_dump() for exp in experiments],
        "failures": failures,
    }

    report_path = report_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Optimization Report",
        "",
        "## Summary",
        "",
        "```\n" + summary_table + "\n```",
        "",
        f"- Baseline runtime: {baseline.results.runtime_seconds:.4f}s",
        f"- Best action: {best_action}",
        f"- Best runtime: {best.results.runtime_seconds:.4f}s" if best else "- Best runtime: n/a",
        f"- Best speedup: {best_speedup:.3f}x" if best_speedup else "- Best speedup: n/a",
        "",
    ]
    if failures:
        md_lines.extend(
            [
                "## Failures",
                "",
                "```\n" + json.dumps(failures, indent=2) + "\n```",
                "",
            ]
        )
    report_md = report_dir / "report.md"
    report_md.write_text("\n".join(md_lines), encoding="utf-8")
    return {"report_json": str(report_path), "report_md": str(report_md), "summary_table": summary_table}
