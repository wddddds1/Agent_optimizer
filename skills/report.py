from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    success_info: Optional[Dict[str, object]] = None,
    agent_trace_path: Optional[str] = None,
    llm_summary_zh: Optional[Dict[str, object]] = None,
    candidate_policy: Optional[Dict[str, object]] = None,
    review_decision: Optional[Dict[str, object]] = None,
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
    thread_policy = _extract_thread_policy(candidate_policy)
    thread_candidates = _collect_thread_candidates(experiments)
    report = {
        "baseline_runtime_seconds": baseline.results.runtime_seconds,
        "best_action_id": best_action,
        "best_runtime_seconds": best.results.runtime_seconds if best else None,
        "best_speedup_vs_baseline": best_speedup,
        "best_repro_script": best_repro,
        "experiments": [exp.model_dump() for exp in experiments],
        "failures": failures,
    }
    if thread_policy:
        report["thread_candidate_policy"] = thread_policy
    if thread_candidates:
        report["thread_candidates_observed"] = thread_candidates
    if candidate_policy and "thread_candidate_policy" not in report:
        report["candidate_policy"] = candidate_policy
    if success_info:
        report["success"] = success_info
    if llm_summary_zh:
        report["llm_summary_zh"] = llm_summary_zh
    if agent_trace_path:
        report["agent_trace_path"] = agent_trace_path
    if review_decision:
        report["review_decision"] = review_decision

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
        f"- Best speedup: {best_speedup:.3f}x" if best_speedup is not None else "- Best speedup: n/a",
        f"- Best repro script: {best_repro}" if best_repro else "- Best repro script: n/a",
        "",
    ]
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
) -> str:
    best_action = best.action.action_id if best and best.action else "baseline"
    best_speedup = best.results.derived_metrics.get("speedup_vs_baseline") if best else None
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
        f"- 最优运行时间: {best.results.runtime_seconds:.4f}s" if best else "- 最优运行时间: n/a",
        f"- 相对加速: {best_speedup:.3f}x" if best_speedup is not None else "- 相对加速: n/a",
        f"- 复现脚本: {best_repro}" if best_repro else "- 复现脚本: n/a",
        "",
    ]
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
