from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from schemas.experiment_ir import ExperimentIR


def _is_profile_probe_exp(exp: ExperimentIR) -> bool:
    if not exp or not exp.action:
        return False
    action_id = str(exp.action.action_id or "")
    run_id = str(exp.run_id or "")
    return action_id.startswith("profile_probe.") or run_id.endswith("-profile")


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


def _markdown_table(rows: List[Dict[str, object]], columns: List[str]) -> str:
    if not rows:
        return "No data."
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body_lines = []
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            values.append(str(value))
        body_lines.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep] + body_lines)


def _build_attempts_table(experiments: List[ExperimentIR]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for exp in experiments:
        action_id = exp.action.action_id if exp.action else "baseline"
        speedup = (
            exp.results.derived_metrics.get("speedup_vs_base_run")
            or exp.results.derived_metrics.get("speedup_vs_baseline_check")
            or exp.results.derived_metrics.get("speedup_vs_baseline")
        )
        rows.append(
            {
                "run_id": exp.run_id,
                "action_id": action_id,
                "family": exp.action.family if exp.action else "baseline",
                "runtime_s": f"{exp.results.runtime_seconds:.4f}",
                "speedup": f"{speedup:.3f}x" if speedup is not None else "n/a",
                "verdict": exp.verdict,
            }
        )
    return rows


def _llm_detail_for_run(llm_summary: Optional[Dict[str, object]], run_id: str) -> Optional[str]:
    if not llm_summary:
        return None
    details = llm_summary.get("experiment_details")
    if isinstance(details, dict):
        detail = details.get(run_id)
        if detail:
            return str(detail)
    return None


def _collect_effective_actions(
    experiments: List[ExperimentIR],
    baseline_runtime: float,
    min_improvement_pct: float,
    llm_summary: Optional[Dict[str, object]],
) -> List[Dict[str, object]]:
    exp_map = {exp.run_id: exp for exp in experiments}
    rows: List[Dict[str, object]] = []
    for exp in experiments:
        if _is_profile_probe_exp(exp):
            continue
        if exp.action is None or exp.verdict != "PASS":
            continue
        base_exp = exp_map.get(exp.base_run_id) if exp.base_run_id else None
        base_runtime = (
            base_exp.results.runtime_seconds
            if base_exp and base_exp.results.runtime_seconds > 0
            else baseline_runtime
        )
        if base_runtime <= 0:
            continue
        improvement = (base_runtime - exp.results.runtime_seconds) / base_runtime
        if improvement < min_improvement_pct:
            continue
        speedup = (
            exp.results.derived_metrics.get("speedup_vs_base_run")
            or exp.results.derived_metrics.get("speedup_vs_baseline_check")
            or exp.results.derived_metrics.get("speedup_vs_baseline")
        )
        detail = _llm_detail_for_run(llm_summary, exp.run_id)
        rows.append(
            {
                "run_id": exp.run_id,
                "action_id": exp.action.action_id,
                "family": exp.action.family,
                "base_run_id": exp.base_run_id,
                "base_action_id": exp.base_action_id,
                "base_runtime_seconds": base_runtime,
                "runtime_seconds": exp.results.runtime_seconds,
                "speedup_vs_baseline": speedup or 0.0,
                "improvement_vs_base": improvement,
                "description": exp.action.description,
                "detail": detail,
            }
        )
    rows.sort(key=lambda item: item.get("runtime_seconds", 0.0))
    return rows


def _top_improvement_steps(
    effective_actions: List[Dict[str, object]],
    limit: int = 5,
) -> List[Dict[str, object]]:
    ordered = sorted(
        effective_actions,
        key=lambda item: float(item.get("improvement_vs_base", 0.0)),
        reverse=True,
    )
    return ordered[:limit]


def _build_failure_table(experiments: List[ExperimentIR]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for exp in experiments:
        if exp.verdict != "FAIL":
            continue
        action_id = exp.action.action_id if exp.action else "baseline"
        rows.append(
            {
                "run_id": exp.run_id,
                "action_id": action_id,
                "reasons": "; ".join(exp.reasons) if exp.reasons else "n/a",
            }
        )
    return rows


def _build_phase_table(phase_transitions: Optional[List[Dict[str, object]]]) -> List[Dict[str, object]]:
    if not phase_transitions:
        return []
    rows: List[Dict[str, object]] = []
    for item in phase_transitions:
        rows.append(
            {
                "iteration": item.get("iteration"),
                "from": item.get("from_phase"),
                "to": item.get("to_phase"),
                "frozen_run_id": item.get("frozen_run_id"),
                "frozen_build_id": item.get("frozen_build_id"),
                "reason": item.get("reason"),
            }
        )
    return rows


def _build_candidate_policy_table(candidate_policy: Optional[Dict[str, object]]) -> List[Dict[str, object]]:
    if not isinstance(candidate_policy, dict):
        return []
    rows: List[Dict[str, object]] = []
    order = candidate_policy.get("default_direction_order")
    if order:
        rows.append({"key": "default_direction_order", "value": ",".join(order)})
    max_candidates = candidate_policy.get("max_candidates_per_round")
    if max_candidates is not None:
        rows.append({"key": "max_candidates_per_round", "value": max_candidates})
    patch_budget = candidate_policy.get("patch_budgets")
    if isinstance(patch_budget, dict):
        for k, v in patch_budget.items():
            rows.append({"key": f"patch_budgets.{k}", "value": v})
    return rows


def _write_flow_figure(report_dir: Path) -> Optional[str]:
    path = _write_flow_png(report_dir)
    if path:
        return path
    return _write_flow_svg(report_dir)


def _write_flow_png(report_dir: Path) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    steps = _flow_steps()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    y_positions = list(range(len(steps)))[::-1]
    for idx, step in enumerate(steps):
        y = y_positions[idx]
        ax.text(
            0.1,
            y,
            step,
            fontsize=10,
            va="center",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8EEF3", edgecolor="#2F3A44"),
        )
        if idx < len(steps) - 1:
            ax.annotate(
                "",
                xy=(0.08, y - 0.9),
                xytext=(0.08, y - 0.1),
                arrowprops=dict(arrowstyle="->", color="#2F3A44", lw=1.2),
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, len(steps))
    fig.tight_layout()
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "optimization_flow.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _write_flow_svg(report_dir: Path) -> Optional[str]:
    steps = _flow_steps()
    width = 640
    height = 48 * len(steps) + 40
    box_w = 360
    box_h = 28
    x = 40
    y = 20
    svg_lines = [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\">",
        "<style>text{font-family:Arial, sans-serif;font-size:12px;fill:#2F3A44;}</style>",
    ]
    for idx, step in enumerate(steps):
        y_pos = y + idx * 48
        svg_lines.append(
            f"<rect x=\"{x}\" y=\"{y_pos}\" width=\"{box_w}\" height=\"{box_h}\" "
            "rx=\"6\" ry=\"6\" fill=\"#E8EEF3\" stroke=\"#2F3A44\"/>"
        )
        svg_lines.append(
            f"<text x=\"{x + 10}\" y=\"{y_pos + 18}\">{_escape_svg(step)}</text>"
        )
        if idx < len(steps) - 1:
            x_mid = x + 18
            y1 = y_pos + box_h
            y2 = y_pos + 48
            svg_lines.append(
                f"<line x1=\"{x_mid}\" y1=\"{y1}\" x2=\"{x_mid}\" y2=\"{y2}\" "
                "stroke=\"#2F3A44\" stroke-width=\"1.2\" marker-end=\"url(#arrow)\"/>"
            )
    svg_lines.append(
        "<defs><marker id=\"arrow\" markerWidth=\"8\" markerHeight=\"8\" refX=\"4\" "
        "refY=\"4\" orient=\"auto\" markerUnits=\"strokeWidth\">"
        "<path d=\"M0,0 L8,4 L0,8 z\" fill=\"#2F3A44\"/></marker></defs>"
    )
    svg_lines.append("</svg>")
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "optimization_flow.svg"
    path.write_text("\n".join(svg_lines), encoding="utf-8")
    return str(path)


def _flow_steps() -> List[str]:
    return [
        "Baseline",
        "Profile/Features",
        "Direction Selection",
        "Analyst",
        "Planner",
        "Optimizer",
        "Ranker",
        "Execute (Patch/Build/Run)",
        "Verify",
        "Reviewer",
        "Record/Report",
        "Iterate/Stop",
    ]


def _collect_iteration_trends(
    experiments: List[ExperimentIR],
    baseline_runtime: float,
) -> Dict[str, List[float]]:
    iterations: Dict[int, float] = {}
    for exp in experiments:
        if exp.action is None or exp.verdict != "PASS":
            continue
        exp_id = exp.exp_id or ""
        if exp_id.endswith("-validate"):
            continue
        match = re.match(r"^iter(\\d+)-", exp_id)
        if not match:
            continue
        iteration = int(match.group(1))
        runtime = exp.results.runtime_seconds
        if runtime <= 0:
            continue
        current = iterations.get(iteration)
        if current is None or runtime < current:
            iterations[iteration] = runtime
    labels = sorted(iterations.keys())
    runtime_series = [iterations[idx] for idx in labels]
    speedup_series = [
        (baseline_runtime / value) if value > 0 else 0.0 for value in runtime_series
    ]
    best_so_far: List[float] = []
    best_runtime = None
    for value in runtime_series:
        if best_runtime is None or value < best_runtime:
            best_runtime = value
        best_so_far.append((baseline_runtime / best_runtime) if best_runtime else 0.0)
    return {
        "labels": labels,
        "runtime_series": runtime_series,
        "speedup_series": speedup_series,
        "best_so_far_speedup": best_so_far,
    }


def _write_trend_figure(
    report_dir: Path,
    filename: str,
    title: str,
    labels: List[int],
    values: List[float],
    y_label: str,
) -> Optional[str]:
    if not labels or not values:
        return None
    path = _write_trend_png(report_dir, filename, title, labels, values, y_label)
    if path:
        return path
    svg_name = Path(filename).with_suffix(".svg").name
    return _write_trend_svg(report_dir, svg_name, title, labels, values, y_label)


def _write_trend_png(
    report_dir: Path,
    filename: str,
    title: str,
    labels: List[int],
    values: List[float],
    y_label: str,
) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(labels, values, marker="o", color="#1F6FEB", linewidth=1.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("iteration")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _write_trend_svg(
    report_dir: Path,
    filename: str,
    title: str,
    labels: List[int],
    values: List[float],
    y_label: str,
) -> Optional[str]:
    width = 640
    height = 360
    pad = 50
    max_val = max(values)
    min_val = min(values)
    span = max_val - min_val if max_val != min_val else 1.0
    x_step = (width - 2 * pad) / max(1, len(labels) - 1)
    points = []
    for idx, value in enumerate(values):
        x = pad + idx * x_step
        y = height - pad - ((value - min_val) / span) * (height - 2 * pad)
        points.append((x, y))
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / filename
    svg_lines = [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\">",
        "<style>text{font-family:Arial, sans-serif;font-size:12px;fill:#2F3A44;}</style>",
        f"<text x=\"{pad}\" y=\"{pad - 20}\">{_escape_svg(title)}</text>",
        f"<text x=\"{pad}\" y=\"{height - 12}\">iteration</text>",
        f"<text x=\"{10}\" y=\"{pad}\">{_escape_svg(y_label)}</text>",
        f"<line x1=\"{pad}\" y1=\"{height - pad}\" x2=\"{width - pad}\" y2=\"{height - pad}\" stroke=\"#9BA7B0\"/>",
        f"<line x1=\"{pad}\" y1=\"{pad}\" x2=\"{pad}\" y2=\"{height - pad}\" stroke=\"#9BA7B0\"/>",
        f"<polyline points=\"{polyline}\" fill=\"none\" stroke=\"#1F6FEB\" stroke-width=\"2\"/>",
    ]
    for x, y in points:
        svg_lines.append(
            f"<circle cx=\"{x:.1f}\" cy=\"{y:.1f}\" r=\"3\" fill=\"#1F6FEB\"/>"
        )
    svg_lines.append("</svg>")
    path.write_text("\n".join(svg_lines), encoding="utf-8")
    return str(path)


def _escape_svg(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\"", "&quot;")
        .replace("'", "&apos;")
    )


def write_report(
    experiments: List[ExperimentIR],
    baseline: ExperimentIR,
    best: Optional[ExperimentIR],
    report_dir: Path,
    success_info: Optional[Dict[str, object]] = None,
    agent_trace_path: Optional[str] = None,
    llm_summary_zh: Optional[Dict[str, object]] = None,
    candidate_policy: Optional[Dict[str, object]] = None,
    review_decision: Optional[Dict[str, object]] = None,
    phase_transitions: Optional[List[Dict[str, object]]] = None,
    composite_exp: Optional[ExperimentIR] = None,
    min_improvement_pct: float = 0.0,
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
    best_repro = _find_repro_script(report_dir, best.run_id) if best else None
    best_source_patch = _best_by_applies_to(experiments, "source_patch")
    best_source_patch_action = (
        best_source_patch.action.action_id if best_source_patch and best_source_patch.action else None
    )
    best_source_patch_speedup = None
    if best_source_patch:
        best_source_patch_speedup = best_source_patch.results.derived_metrics.get("speedup_vs_baseline")
    best_source_patch_repro = (
        _find_repro_script(report_dir, best_source_patch.run_id) if best_source_patch else None
    )
    best_action_chain = _best_action_chain(experiments, best)
    best_effective_combo = _effective_action_combo(experiments, best)
    effective_actions = _collect_effective_actions(
        experiments,
        baseline.results.runtime_seconds,
        min_improvement_pct,
        llm_summary_zh,
    )
    code_patch_status = "not_reached"
    if best_source_patch:
        code_patch_status = "pass"
    thread_policy = _extract_thread_policy(candidate_policy)
    thread_candidates = _collect_thread_candidates(experiments)
    trend = _collect_iteration_trends(experiments, baseline.results.runtime_seconds)
    runtime_trend_path = _write_trend_figure(
        report_dir,
        "runtime_trend.png",
        "Best Runtime per Iteration",
        trend.get("labels", []),
        trend.get("runtime_series", []),
        "runtime_s",
    )
    speedup_trend_path = _write_trend_figure(
        report_dir,
        "best_speedup.png",
        "Best-So-Far Speedup",
        trend.get("labels", []),
        trend.get("best_so_far_speedup", []),
        "speedup",
    )
    flow_path = _write_flow_figure(report_dir)
    report = {
        "baseline_runtime_seconds": baseline.results.runtime_seconds,
        "best_action_id": best_action,
        "best_action_chain": best_action_chain,
        "best_effective_combo": best_effective_combo,
        "best_runtime_seconds": best.results.runtime_seconds if best else None,
        "best_speedup_vs_baseline": best_speedup,
        "best_repro_script": best_repro,
        "best_source_patch_action_id": best_source_patch_action,
        "best_source_patch_runtime_seconds": best_source_patch.results.runtime_seconds if best_source_patch else None,
        "best_source_patch_speedup_vs_baseline": best_source_patch_speedup,
        "best_source_patch_repro_script": best_source_patch_repro,
        "code_patch_status": code_patch_status,
        "experiments": [exp.model_dump() for exp in experiments],
        "failures": failures,
    }
    if composite_exp:
        report["final_composite"] = {
            "run_id": composite_exp.run_id,
            "action_id": composite_exp.action.action_id if composite_exp.action else None,
            "runtime_seconds": composite_exp.results.runtime_seconds,
            "speedup_vs_baseline": composite_exp.results.derived_metrics.get("speedup_vs_baseline"),
            "verdict": composite_exp.verdict,
        }
    report["figures"] = [
        {
            "id": "optimization_flow",
            "title": "Optimization Flow",
            "type": "image",
            "path": flow_path,
        },
        {
            "id": "runtime_trend",
            "title": "Best Runtime per Iteration",
            "type": "image",
            "path": runtime_trend_path,
            "labels": trend.get("labels", []),
            "values": trend.get("runtime_series", []),
        },
        {
            "id": "best_speedup",
            "title": "Best-So-Far Speedup",
            "type": "image",
            "path": speedup_trend_path,
            "labels": trend.get("labels", []),
            "values": trend.get("best_so_far_speedup", []),
        },
    ]
    report["figures"] = [item for item in report["figures"] if item.get("path")]
    attempts_table = _build_attempts_table(experiments)
    failure_table = _build_failure_table(experiments)
    phase_table = _build_phase_table(phase_transitions)
    policy_table = _build_candidate_policy_table(candidate_policy)
    report["tables"] = {
        "attempts": attempts_table,
        "failures": failure_table,
        "phases": phase_table,
        "candidate_policy": policy_table,
    }
    if thread_policy:
        report["thread_candidate_policy"] = thread_policy
    if thread_candidates:
        report["thread_candidates_observed"] = thread_candidates
    if candidate_policy and "thread_candidate_policy" not in report:
        report["candidate_policy"] = candidate_policy
    if success_info:
        report["success"] = success_info
    if effective_actions:
        report["effective_actions"] = effective_actions
    if llm_summary_zh:
        report["llm_summary_zh"] = llm_summary_zh
    if agent_trace_path:
        report["agent_trace_path"] = agent_trace_path
    if review_decision:
        report["review_decision"] = review_decision
    if phase_transitions:
        report["phase_transitions"] = phase_transitions
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
                f"- Best effective combo: {' | '.join(best_effective_combo)}" if best_effective_combo else "- Best effective combo: n/a",
                f"- Action chain (audit): {' -> '.join(best_action_chain)}",
        f"- Best runtime: {best.results.runtime_seconds:.4f}s" if best else "- Best runtime: n/a",
        f"- Best speedup: {best_speedup:.3f}x" if best_speedup is not None else "- Best speedup: n/a",
        f"- Best repro script: {best_repro}" if best_repro else "- Best repro script: n/a",
        "",
    ]
    if composite_exp:
        composite_speedup = composite_exp.results.derived_metrics.get("speedup_vs_baseline")
        md_lines.extend(
            [
                "## Final Composite Validation",
                "",
                f"- Composite runtime: {composite_exp.results.runtime_seconds:.4f}s",
                f"- Composite speedup: {composite_speedup:.3f}x"
                if composite_speedup is not None
                else "- Composite speedup: n/a",
                f"- Composite verdict: {composite_exp.verdict}",
                "",
            ]
        )
    key_steps = _top_improvement_steps(effective_actions) if effective_actions else []
    if key_steps:
        md_lines.extend(
            [
                "## Key Improvement Steps",
                "",
                _markdown_table(
                    [
                        {
                            "action_id": item.get("action_id"),
                            "family": item.get("family"),
                            "base_action_id": item.get("base_action_id"),
                            "runtime_s": f"{item.get('runtime_seconds', 0.0):.4f}",
                            "delta_vs_base": f"{item.get('improvement_vs_base', 0.0):.2%}",
                        }
                        for item in key_steps
                    ],
                    ["action_id", "family", "base_action_id", "runtime_s", "delta_vs_base"],
                ),
                "",
            ]
        )
    if effective_actions:
        md_lines.extend(
            [
                "## Effective Actions (vs baseline)",
                "",
                _markdown_table(
                    [
                        {
                            "action_id": item.get("action_id"),
                            "family": item.get("family"),
                            "base_action_id": item.get("base_action_id"),
                            "runtime_s": f"{item.get('runtime_seconds', 0.0):.4f}",
                            "speedup": f"{item.get('speedup_vs_baseline', 0.0):.3f}x",
                            "delta_vs_base": f"{item.get('improvement_vs_base', 0.0):.2%}",
                        }
                        for item in effective_actions
                    ],
                    ["action_id", "family", "base_action_id", "runtime_s", "speedup", "delta_vs_base"],
                ),
                "",
            ]
        )
    if best_source_patch:
        md_lines.extend(
            [
                "## Best Source Patch",
                "",
                f"- Best source_patch action: {best_source_patch_action}",
                f"- Runtime: {best_source_patch.results.runtime_seconds:.4f}s",
                f"- Speedup vs baseline: {best_source_patch_speedup:.3f}x"
                if best_source_patch_speedup is not None
                else "- Speedup vs baseline: n/a",
                f"- Repro script: {best_source_patch_repro}"
                if best_source_patch_repro
                else "- Repro script: n/a",
                "",
            ]
        )
    md_lines.extend(
        [
            "## Code Optimization Milestone",
            "",
            f"- Status: {code_patch_status}",
            "",
        ]
    )
    md_lines.extend(["## Optimization Flow", ""])
    if flow_path:
        md_lines.extend([f"![Optimization Flow](figures/{Path(flow_path).name})", ""])
    else:
        md_lines.extend(["- Flow diagram unavailable (matplotlib missing).", ""])
    md_lines.extend(["## Performance Trends", ""])
    if runtime_trend_path:
        md_lines.append(f"![Best Runtime per Iteration](figures/{Path(runtime_trend_path).name})")
    else:
        md_lines.append("- Runtime trend unavailable (insufficient data or matplotlib missing).")
    md_lines.append("")
    if speedup_trend_path:
        md_lines.append(f"![Best-So-Far Speedup](figures/{Path(speedup_trend_path).name})")
    else:
        md_lines.append("- Speedup trend unavailable (insufficient data or matplotlib missing).")
    md_lines.append("")
    md_lines.extend(
        [
            "## Attempts Table",
            "",
            _markdown_table(
                attempts_table,
                ["run_id", "action_id", "family", "runtime_s", "speedup", "verdict"],
            ),
            "",
        ]
    )
    if failure_table:
        md_lines.extend(
            [
                "## Failure Summary",
                "",
                _markdown_table(failure_table, ["run_id", "action_id", "reasons"]),
                "",
            ]
        )
    if phase_table:
        md_lines.extend(
            [
                "## Phase Transitions",
                "",
                _markdown_table(
                    phase_table,
                    ["iteration", "from", "to", "frozen_run_id", "frozen_build_id", "reason"],
                ),
                "",
            ]
        )
    if policy_table:
        md_lines.extend(
            [
                "## Candidate Policy",
                "",
                _markdown_table(policy_table, ["key", "value"]),
                "",
            ]
        )
    if thread_policy or thread_candidates:
        md_lines.extend(
            [
                "## Candidate Thread Strategy",
                "",
                "```",
                json.dumps(
                    {
                        "policy": thread_policy,
                        "observed_candidates": thread_candidates,
                    },
                    indent=2,
                ),
                "```",
                "",
            ]
        )
    if success_info:
        achieved = success_info.get("achieved_improvement_pct", 0.0)
        target = success_info.get("target_improvement_pct", 0.0)
        md_lines.extend(
            [
                "## Success Criteria",
                "",
                f"- Success: {success_info.get('success')}",
                f"- Reason: {success_info.get('reason')}",
                f"- Target improvement: {target:.2%}",
                f"- Achieved improvement: {achieved:.2%}",
                "",
            ]
        )
    if agent_trace_path:
        md_lines.extend(
            [
                "## Agent Trace",
                "",
                f"- Trace path: {agent_trace_path}",
                "",
            ]
        )
    if review_decision:
        md_lines.extend(
            [
                "## Convergence Review",
                "",
                "```",
                json.dumps(review_decision, indent=2),
                "```",
                "",
            ]
        )
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
    report_zh = report_dir / "report_zh.md"
    report_zh.write_text(
        _build_report_zh(
            experiments,
            baseline,
            best,
            summary_table,
            success_info,
            agent_trace_path,
            llm_summary_zh,
            best_repro,
            thread_policy,
            thread_candidates,
            review_decision,
            phase_transitions,
            report_dir,
            candidate_policy,
            composite_exp,
            min_improvement_pct,
        ),
        encoding="utf-8",
    )
    return {
        "report_json": str(report_path),
        "report_md": str(report_md),
        "report_zh": str(report_zh),
        "summary_table": summary_table,
    }


def _build_report_zh(
    experiments: List[ExperimentIR],
    baseline: ExperimentIR,
    best: Optional[ExperimentIR],
    summary_table: str,
    success_info: Optional[Dict[str, object]],
    agent_trace_path: Optional[str],
    llm_summary_zh: Optional[Dict[str, object]],
    best_repro: Optional[str],
    thread_policy: Optional[Dict[str, object]],
    thread_candidates: List[int],
    review_decision: Optional[Dict[str, object]],
    phase_transitions: Optional[List[Dict[str, object]]],
    report_dir: Path,
    candidate_policy: Optional[Dict[str, object]],
    composite_exp: Optional[ExperimentIR],
    min_improvement_pct: float,
) -> str:
    best_action = best.action.action_id if best and best.action else "baseline"
    best_speedup = best.results.derived_metrics.get("speedup_vs_baseline") if best else None
    best_source_patch = _best_by_applies_to(experiments, "source_patch")
    best_source_patch_action = (
        best_source_patch.action.action_id if best_source_patch and best_source_patch.action else None
    )
    best_source_patch_speedup = None
    if best_source_patch:
        best_source_patch_speedup = best_source_patch.results.derived_metrics.get("speedup_vs_baseline")
    best_source_patch_repro = (
        _find_repro_script(report_dir, best_source_patch.run_id) if best_source_patch else None
    )
    code_patch_status = "not_reached"
    if best_source_patch:
        code_patch_status = "pass"
    effective_actions = _collect_effective_actions(
        experiments,
        baseline.results.runtime_seconds,
        min_improvement_pct,
        llm_summary_zh,
    )
    attempts = [exp.action.action_id for exp in experiments if exp.action]
    chosen = best_action if best_action != "baseline" else "baseline"
    rejected = [action_id for action_id in attempts if action_id != chosen]
    analysis_lines = (
        _extract_llm_overall_analysis(llm_summary_zh)
        if llm_summary_zh
        else _build_zh_analysis(baseline, best, experiments)
    )

    lines = [
        "# 优化报告（中文）",
        "",
        "## 总览",
        "",
        "```\n" + summary_table + "\n```",
        "",
        f"- Baseline 运行时间: {baseline.results.runtime_seconds:.4f}s",
        f"- 最优动作: {best_action}",
            f"- 最优组合(生效): {' | '.join(_effective_action_combo(experiments, best))}"
            if _effective_action_combo(experiments, best)
            else "- 最优组合(生效): n/a",
            f"- 动作链(审计): {' -> '.join(_best_action_chain(experiments, best))}",
        f"- 最优运行时间: {best.results.runtime_seconds:.4f}s" if best else "- 最优运行时间: n/a",
        f"- 相对加速: {best_speedup:.3f}x" if best_speedup is not None else "- 相对加速: n/a",
        f"- 复现脚本: {best_repro}" if best_repro else "- 复现脚本: n/a",
        "",
    ]
    if composite_exp:
        composite_speedup = composite_exp.results.derived_metrics.get("speedup_vs_baseline")
        composite_repro = _find_repro_script(report_dir, composite_exp.run_id)
        lines.extend(
            [
                "## 最终组合验证",
                "",
                f"- 组合运行时间: {composite_exp.results.runtime_seconds:.4f}s",
                f"- 相对加速: {composite_speedup:.3f}x"
                if composite_speedup is not None
                else "- 相对加速: n/a",
                f"- 判定: {composite_exp.verdict}",
                f"- 复现脚本: {composite_repro}" if composite_repro else "- 复现脚本: n/a",
                "",
            ]
        )
    if effective_actions:
        lines.extend(["## 有效改动汇总", ""])
        for item in effective_actions:
            detail = item.get("detail") or item.get("description") or ""
            lines.append(
                f"- {item.get('action_id')} ({item.get('family')}): "
                f"{item.get('runtime_seconds', 0.0):.4f}s, "
                f"相对基准动作 {item.get('base_action_id')}, "
                f"提升 {item.get('improvement_vs_base', 0.0):.2%}. {detail}"
            )
        lines.append("")
    key_steps = _top_improvement_steps(effective_actions) if effective_actions else []
    if key_steps:
        lines.extend(["## 关键提升步骤", ""])
        for item in key_steps:
            lines.append(
                f"- {item.get('action_id')} ({item.get('family')}): "
                f"{item.get('runtime_seconds', 0.0):.4f}s, "
                f"相对基准动作 {item.get('base_action_id')}, "
                f"提升 {item.get('improvement_vs_base', 0.0):.2%}"
            )
        lines.append("")
    if best_source_patch:
        lines.extend(
            [
                "## 代码优化最佳结果",
                "",
                f"- 最佳 source_patch 动作: {best_source_patch_action}",
                f"- 运行时间: {best_source_patch.results.runtime_seconds:.4f}s",
                f"- 相对加速: {best_source_patch_speedup:.3f}x"
                if best_source_patch_speedup is not None
                else "- 相对加速: n/a",
                f"- 复现脚本: {best_source_patch_repro}"
                if best_source_patch_repro
                else "- 复现脚本: n/a",
                "",
            ]
        )
    lines.extend(
        [
            "## 代码优化里程碑",
            "",
            f"- 状态: {code_patch_status}",
            "",
        ]
    )
    trend = _collect_iteration_trends(experiments, baseline.results.runtime_seconds)
    runtime_trend_path = _write_trend_figure(
        report_dir,
        "runtime_trend.png",
        "Best Runtime per Iteration",
        trend.get("labels", []),
        trend.get("runtime_series", []),
        "runtime_s",
    )
    speedup_trend_path = _write_trend_figure(
        report_dir,
        "best_speedup.png",
        "Best-So-Far Speedup",
        trend.get("labels", []),
        trend.get("best_so_far_speedup", []),
        "speedup",
    )
    flow_path = _write_flow_figure(report_dir)
    attempts_table = _build_attempts_table(experiments)
    failure_table = _build_failure_table(experiments)
    phase_table = _build_phase_table(phase_transitions)
    policy_table = _build_candidate_policy_table(candidate_policy)
    lines.extend(["## 优化流程", ""])
    if flow_path:
        lines.extend([f"![Optimization Flow](figures/{Path(flow_path).name})", ""])
    else:
        lines.extend(["- 优化流程图不可用（matplotlib 缺失）。", ""])
    lines.extend(["## 性能趋势", ""])
    if runtime_trend_path:
        lines.append(f"![Best Runtime per Iteration](figures/{Path(runtime_trend_path).name})")
    else:
        lines.append("- 运行时间曲线不可用（数据不足或 matplotlib 缺失）。")
    lines.append("")
    if speedup_trend_path:
        lines.append(f"![Best-So-Far Speedup](figures/{Path(speedup_trend_path).name})")
    else:
        lines.append("- 加速比曲线不可用（数据不足或 matplotlib 缺失）。")
    lines.append("")
    lines.extend(
        [
            "## 实验尝试表",
            "",
            _markdown_table(
                attempts_table,
                ["run_id", "action_id", "family", "runtime_s", "speedup", "verdict"],
            ),
            "",
        ]
    )
    if failure_table:
        lines.extend(
            [
                "## 失败概览表",
                "",
                _markdown_table(failure_table, ["run_id", "action_id", "reasons"]),
                "",
            ]
        )
    if phase_table:
        lines.extend(
            [
                "## 阶段切换与冻结点",
                "",
                _markdown_table(
                    phase_table,
                    ["iteration", "from", "to", "frozen_run_id", "frozen_build_id", "reason"],
                ),
                "",
            ]
        )
    if policy_table:
        lines.extend(
            [
                "## 候选策略",
                "",
                _markdown_table(policy_table, ["key", "value"]),
                "",
            ]
        )
    if thread_policy or thread_candidates:
        lines.extend(
            [
                "## 候选线程生成策略",
                "",
                "```",
                json.dumps(
                    {
                        "policy": thread_policy,
                        "observed_candidates": thread_candidates,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                "```",
                "",
            ]
        )
    if success_info:
        achieved = success_info.get("achieved_improvement_pct", 0.0)
        target = success_info.get("target_improvement_pct", 0.0)
        lines.extend(
            [
                "## 成功判据",
                "",
                f"- 是否满足: {success_info.get('success')}",
                f"- 结论原因: {success_info.get('reason')}",
                f"- 目标提升: {target:.2%}",
                f"- 实际提升: {achieved:.2%}",
                "",
            ]
        )
    if review_decision:
        lines.extend(
            [
                "## 收敛评审",
                "",
                "```",
                json.dumps(review_decision, indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
    lines.append("## 实验列表")
    for exp in experiments:
        action_id = exp.action.action_id if exp.action else "baseline"
        speedup = exp.results.derived_metrics.get("speedup_vs_baseline")
        speedup_str = f"{speedup:.3f}x" if speedup is not None else "n/a"
        reason_text = (
            _llm_reason_for_run(llm_summary_zh, exp.run_id)
            if llm_summary_zh
            else _format_zh_reason(exp, best_action)
        )
        lines.append(
            f"- {action_id}: {exp.verdict}, {exp.results.runtime_seconds:.4f}s, speedup {speedup_str}, 原因: {reason_text}"
        )
    failure_items = [
        exp for exp in experiments if exp.verdict == "FAIL"
    ]
    if failure_items:
        lines.extend(["", "## 失败概览", ""])
        for exp in failure_items:
            action_id = exp.action.action_id if exp.action else "baseline"
            reasons = "；".join(exp.reasons) if exp.reasons else "无"
            lines.append(f"- {action_id}: {reasons}")
    detail_items = _collect_llm_details(llm_summary_zh, experiments) if llm_summary_zh else []
    if detail_items:
        lines.extend(["", "## 实验分析（逐条）", ""])
        for action_id, detail in detail_items:
            lines.append(f"### {action_id}")
            lines.append(detail)
    if analysis_lines:
        lines.extend(["", "## 简要分析"])
        lines.extend([f"- {line}" for line in analysis_lines])
    selection_reason = _extract_llm_selection_reason(llm_summary_zh)
    if selection_reason:
        lines.extend(["", "## 选择理由", f"- {selection_reason}"])
    lines.extend(
        [
            "",
            "## 最终选择",
            f"- 采用的优化: {chosen}",
            f"- 未采用的优化: {', '.join(rejected) if rejected else '无'}",
            "",
        ]
    )
    if agent_trace_path:
        lines.extend(
            [
                "## Agent Trace",
                "",
                f"- Trace path: {agent_trace_path}",
                "",
            ]
        )
    return "\n".join(lines)


def _extract_thread_policy(candidate_policy: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not isinstance(candidate_policy, dict):
        return None
    families = candidate_policy.get("families")
    if not isinstance(families, dict):
        return None
    parallel_cfg = families.get("parallel_omp")
    if not isinstance(parallel_cfg, dict):
        return None
    threads_cfg = parallel_cfg.get("threads")
    if not isinstance(threads_cfg, dict):
        return None
    return threads_cfg


def _collect_thread_candidates(experiments: List[ExperimentIR]) -> List[int]:
    candidates: List[int] = []
    for exp in experiments:
        if not exp.action:
            continue
        env = exp.job.env or {}
        value = env.get("OMP_NUM_THREADS")
        if value is None:
            continue
        try:
            candidates.append(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(set(candidates))


def _find_repro_script(report_dir: Path, run_id: Optional[str]) -> Optional[str]:
    if not run_id:
        return None
    candidate = report_dir / "runs" / run_id / "repro.sh"
    if candidate.exists():
        return str(candidate)
    return None


def _best_by_applies_to(
    experiments: List[ExperimentIR], target: str
) -> Optional[ExperimentIR]:
    best = None
    for exp in experiments:
        if _is_profile_probe_exp(exp):
            continue
        if exp.verdict != "PASS" or not exp.action:
            continue
        if target not in (exp.action.applies_to or []):
            continue
        if best is None or exp.results.runtime_seconds < best.results.runtime_seconds:
            best = exp
    return best


def _best_action_chain(
    experiments: List[ExperimentIR],
    best_exp: Optional[ExperimentIR],
) -> List[str]:
    if not best_exp:
        return ["baseline"]
    by_run = {exp.run_id: exp for exp in experiments}
    chain: List[str] = []
    current = best_exp
    seen: set[str] = set()
    while current and current.run_id not in seen:
        seen.add(current.run_id)
        if current.action and current.action.action_id:
            chain.append(current.action.action_id)
        else:
            chain.append("baseline")
        if not current.base_run_id:
            break
        current = by_run.get(current.base_run_id)
    chain.reverse()
    if chain and chain[0] != "baseline":
        chain = ["baseline"] + [item for item in chain if item != "baseline"]
    return chain


def _effective_action_combo(
    experiments: List[ExperimentIR],
    best_exp: Optional[ExperimentIR],
) -> List[str]:
    if not best_exp:
        return []
    by_run = {exp.run_id: exp for exp in experiments}
    applied: Dict[str, str] = {}
    current = best_exp
    seen: set[str] = set()
    while current and current.run_id not in seen:
        seen.add(current.run_id)
        if current.action and current.action.applies_to:
            for target in current.action.applies_to:
                if target not in applied:
                    applied[target] = current.action.action_id
        if not current.base_run_id:
            break
        current = by_run.get(current.base_run_id)
    ordered = ["run_config", "input_script", "build_config", "source_patch"]
    combo: List[str] = []
    for key in ordered:
        if key in applied:
            combo.append(f"{key}={applied[key]}")
    for key in sorted(k for k in applied.keys() if k not in ordered):
        combo.append(f"{key}={applied[key]}")
    return combo


def _build_zh_analysis(
    baseline: ExperimentIR,
    best: Optional[ExperimentIR],
    experiments: List[ExperimentIR],
) -> List[str]:
    lines: List[str] = []
    if not best or not best.action:
        return lines
    best_speedup = best.results.derived_metrics.get("speedup_vs_baseline")
    best_family = best.action.family
    comm_ratio, output_ratio = _ratio_from_profile(baseline)
    if comm_ratio is not None and output_ratio is not None:
        if comm_ratio < 0.2 and output_ratio < 0.2:
            lines.append("基线以计算为主，优先尝试并行与亲和性方向。")
        elif comm_ratio >= 0.2:
            lines.append("基线通信占比较高，理论上应优先尝试通信优化方向。")
        elif output_ratio >= 0.2:
            lines.append("基线 I/O 占比较高，理论上应优先尝试 I/O 优化方向。")
    if best_speedup is not None:
        lines.append(f"最终选用 {best.action.action_id}，相对基线提升 {best_speedup:.3f}x。")
    if best_family:
        lines.append(f"选择方向: {best_family}，是本轮候选中表现最优的方向。")
    variance_cv = _extract_variance_cv(best)
    if variance_cv is not None:
        lines.append(f"稳定性（CV）≈ {variance_cv:.3f}，满足方差门禁。")
    lower_actions = _list_lower_actions(best, experiments)
    if lower_actions:
        lines.append(f"其余候选提升较小或无明显提升：{', '.join(lower_actions)}。")
    return lines


def _ratio_from_profile(exp: ExperimentIR) -> tuple[Optional[float], Optional[float]]:
    timing = exp.profile_report.timing_breakdown or {}
    total = timing.get("total", 0.0) or 0.0
    if total <= 0.0:
        return None, None
    comm_ratio = (timing.get("comm", 0.0) / total) if total else 0.0
    output_ratio = (timing.get("output", 0.0) / total) if total else 0.0
    return comm_ratio, output_ratio


def _extract_variance_cv(exp: ExperimentIR) -> Optional[float]:
    cv = exp.results.derived_metrics.get("variance_cv")
    if cv is not None:
        return float(cv)
    cv = exp.results.correctness_metrics.get("variance_cv")
    if cv is not None:
        return float(cv)
    return None


def _list_lower_actions(best: ExperimentIR, experiments: List[ExperimentIR]) -> List[str]:
    best_id = best.action.action_id if best.action else "baseline"
    items: List[str] = []
    for exp in experiments:
        if exp.action is None:
            continue
        if exp.action.action_id == best_id:
            continue
        speedup = exp.results.derived_metrics.get("speedup_vs_baseline")
        if speedup is None or speedup <= 1.0:
            items.append(exp.action.action_id)
    return items[:5]


def _format_zh_reason(exp: ExperimentIR, best_action_id: str) -> str:
    if exp.action is None:
        return "基线"
    if exp.reasons:
        return "门禁失败: " + ", ".join(exp.reasons)

    parts: List[str] = []
    if exp.run_id.endswith("-validate"):
        parts.append("最优复跑验证")
    elif exp.action.action_id == best_action_id:
        parts.append("最终选用")
    else:
        parts.append("未选用")

    speedup = exp.results.derived_metrics.get("speedup_vs_baseline")
    if speedup is not None:
        if speedup >= 1.2:
            parts.append("提升明显")
        elif speedup >= 1.05:
            parts.append("提升有限")
        elif speedup >= 1.0:
            parts.append("接近基线")
        else:
            parts.append("性能下降")

    correctness_skipped = exp.results.correctness_metrics.get("correctness_skipped_reason")
    if correctness_skipped:
        parts.append(f"正确性跳过: {correctness_skipped}")
    parts.append("通过门禁")
    return "，".join(parts)


def _llm_reason_for_run(llm_summary: Optional[Dict[str, object]], run_id: str) -> str:
    if not llm_summary:
        return "未提供原因"
    reasons = llm_summary.get("experiment_reasons", {})
    if isinstance(reasons, dict):
        return str(reasons.get(run_id, "未提供原因"))
    return "未提供原因"


def _collect_llm_details(
    llm_summary: Optional[Dict[str, object]],
    experiments: List[ExperimentIR],
) -> List[Tuple[str, str]]:
    if not llm_summary:
        return []
    details = llm_summary.get("experiment_details", {})
    if not isinstance(details, dict):
        return []
    items: List[Tuple[str, str]] = []
    for exp in experiments:
        action_id = exp.action.action_id if exp.action else "baseline"
        detail = details.get(exp.run_id)
        if detail:
            items.append((action_id, str(detail)))
    return items


def _extract_llm_overall_analysis(llm_summary: Optional[Dict[str, object]]) -> List[str]:
    if not llm_summary:
        return []
    analysis = llm_summary.get("overall_analysis")
    if isinstance(analysis, list):
        return [str(item) for item in analysis]
    return []


def _extract_llm_selection_reason(llm_summary: Optional[Dict[str, object]]) -> Optional[str]:
    if not llm_summary:
        return None
    selection_reason = llm_summary.get("selection_reason")
    if selection_reason:
        return str(selection_reason)
    return None
