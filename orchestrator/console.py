from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TextIO, Tuple

from schemas.action_ir import ActionIR
from schemas.plan_ir import PlanIR
from schemas.ranking_ir import RankedActions
from schemas.experiment_ir import ExperimentIR
from schemas.job_ir import JobIR
from schemas.profile_report import ProfileReport


@dataclass
class ConsoleUI:
    enabled: bool = True
    stream: TextIO = sys.stdout
    baseline_runtime: Optional[float] = None
    show_output_preview: bool = True
    preview_bytes: int = 2048

    def header(
        self,
        job: JobIR,
        selection_mode: str,
        top_k: int,
        direction_top_k: int,
        llm_enabled: bool,
        llm_model: str,
        artifacts_dir: str,
        baseline_repeats: int,
        baseline_stat: str,
        validate_top1_repeats: int,
        min_improvement_pct: float,
    ) -> None:
        if not self.enabled:
            return
        self._section("运行开始")
        self._kv("算例", job.case_id)
        self._kv("工作目录", job.workdir)
        self._kv("可执行文件", job.lammps_bin)
        self._kv(
            "预算",
            f"迭代={job.budgets.max_iters} 运行={job.budgets.max_runs} 限时={job.budgets.max_wall_seconds}s",
        )
        self._kv(
            "选择",
            f"模式={selection_mode} top_k={top_k} direction_top_k={direction_top_k}",
        )
        llm_status = "启用" if llm_enabled else "关闭"
        self._kv("LLM", f"{llm_status} model={llm_model}")
        self._kv("产物目录", artifacts_dir)
        self._kv(
            "基线",
            f"重复={baseline_repeats} 统计={baseline_stat} 复跑Top1={validate_top1_repeats} 最小提升={min_improvement_pct:.2%}",
        )
        self._print("")

    def analysis(
        self,
        bottlenecks: List[str],
        allowed_families: List[str],
        confidence: str,
        rationale: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return
        self._section("分析")
        rationale = rationale or "无"
        self._agent(
            "AnalystAgent",
            [
                f"瓶颈: {', '.join(bottlenecks) if bottlenecks else '无'}",
                f"允许动作族: {', '.join(allowed_families) if allowed_families else '无'}",
                f"画像置信度: {confidence}",
                f"理由: {rationale}",
            ],
        )

    def plan_summary(self, plan: PlanIR) -> None:
        if not self.enabled:
            return
        self._section("计划")
        self._agent(
            "PlannerAgent",
            [
                f"选择方向: {', '.join(plan.chosen_families) if plan.chosen_families else '无'}",
                f"候选上限: {plan.max_candidates}",
                f"Top1 复跑: {plan.evaluation.top1_validation_repeats}",
                f"理由: {plan.reason}",
            ],
        )

    def candidates(
        self,
        actions: List[ActionIR],
        ranking_mode: str,
        selection_mode: str,
        llm_explanation: Optional[str],
    ) -> None:
        if not self.enabled:
            return
        self._section("候选动作")
        self._agent(
            "OptimizerAgent",
            [
                f"排序模式: {ranking_mode}",
                f"选择模式: {selection_mode}",
                f"候选数: {len(actions)}",
            ],
        )
        if actions:
            for idx, action in enumerate(actions, start=1):
                effect = ",".join(action.expected_effect)
                self._print(
                    f"  {idx}. {action.action_id} | 方向={action.family} | 风险={action.risk_level} | 预期效果={effect}"
                )
                if action.description:
                    self._print(f"     描述: {action.description}")
        if llm_explanation:
            self._agent("OptimizerAgent", ["LLM 解释:"] + _split_lines(llm_explanation, "  - "))

    def rank_summary(self, ranked: RankedActions) -> None:
        if not self.enabled:
            return
        self._section("排序结果")
        self._agent(
            "RouterRankerAgent",
            [
                f"候选数: {len(ranked.ranked)}",
                f"被拒绝: {len(ranked.rejected)}",
                f"说明: {ranked.scoring_notes or 'heuristic'}",
            ],
        )

    def run_start(
        self,
        exp_id: str,
        action: Optional[ActionIR],
        env_overrides: Dict[str, str],
        run_args: List[str],
    ) -> None:
        if not self.enabled:
            return
        self._section(f"运行 {exp_id}")
        if action is None:
            self._agent("OptimizerAgent", ["动作: 基线（无修改）"])
        else:
            self._agent(
                "OptimizerAgent",
                [
                    f"动作ID: {action.action_id}",
                    f"方向: {action.family}",
                    f"风险: {action.risk_level}",
                    f"作用范围: {', '.join(action.applies_to)}",
                ],
            )
            if action.description:
                self._print(f"  描述: {action.description}")
        if env_overrides:
            self._print(f"  环境覆盖: {_fmt_env(env_overrides)}")
        if run_args:
            self._print(f"  运行参数: {' '.join(run_args)}")

    def profile_result(self, run_output, profile: ProfileReport) -> None:
        if not self.enabled:
            return
        lines = [
            f"运行耗时: {run_output.runtime_seconds:.4f}s",
            f"退出码: {run_output.exit_code}",
            f"样本数: {len(run_output.samples)}",
        ]
        timing = _format_timing(profile)
        if timing:
            lines.append(f"耗时分解: {timing}")
        self._agent("ProfilerAgent", lines)
        if self.show_output_preview:
            self.output_preview(run_output)

    def output_preview(self, run_output) -> None:
        if not self.enabled or not self.show_output_preview:
            return
        previews = [
            ("stdout", run_output.stdout_path),
            ("stderr", run_output.stderr_path),
            ("time", run_output.time_output_path),
            ("log", run_output.log_path),
        ]
        lines: List[str] = ["原始输出预览（尾部截断）:"]
        for label, path in previews:
            preview = _read_preview_text(path, self.preview_bytes)
            if not preview:
                continue
            text, truncated, size = preview
            suffix = "（截断）" if truncated else ""
            lines.append(f"{label}: {size} bytes{suffix}")
            for line in text.splitlines()[-12:]:
                lines.append(f"  {line}")
        if len(lines) > 1:
            self._agent("ProfilerAgent", lines)

    def verify_result(self, exp: ExperimentIR) -> None:
        if not self.enabled:
            return
        speedup = exp.results.derived_metrics.get("speedup_vs_baseline")
        speedup_str = f"{speedup:.3f}x" if speedup is not None else "n/a"
        variance_cv = _extract_variance_cv(exp)
        correctness = exp.results.correctness_metrics.get("correctness_skipped_reason")
        lines = [
            f"判定: {exp.verdict}",
            f"运行耗时: {exp.results.runtime_seconds:.4f}s",
            f"相对基线加速: {speedup_str}",
        ]
        if variance_cv is not None:
            lines.append(f"方差CV: {variance_cv:.3f}")
        if correctness:
            lines.append(f"正确性: 跳过（{correctness}）")
        if exp.reasons:
            lines.append(f"原因: {', '.join(exp.reasons)}")
        self._agent("VerifierAgent", lines)

    def iteration_summary(
        self,
        iteration: int,
        best_exp: Optional[ExperimentIR],
        best_overall: Optional[ExperimentIR],
    ) -> None:
        if not self.enabled:
            return
        self._section(f"第 {iteration:03d} 轮小结")
        if best_exp is None:
            self._print("  本轮最优: 无（无通过候选）")
        else:
            speedup = best_exp.results.derived_metrics.get("speedup_vs_baseline")
            speedup_str = f"{speedup:.3f}x" if speedup is not None else "n/a"
            action_id = best_exp.action.action_id if best_exp.action else "baseline"
            self._print(
                f"  本轮最优: {action_id} 耗时={best_exp.results.runtime_seconds:.4f}s 加速={speedup_str}"
            )
        if best_overall is None:
            self._print("  历史最优: 无")
            return
        speedup = best_overall.results.derived_metrics.get("speedup_vs_baseline")
        speedup_str = f"{speedup:.3f}x" if speedup is not None else "n/a"
        action_id = best_overall.action.action_id if best_overall.action else "baseline"
        self._print(
            f"  历史最优: {action_id} 耗时={best_overall.results.runtime_seconds:.4f}s 加速={speedup_str}"
        )

    def stop(self, reason: str) -> None:
        if not self.enabled:
            return
        self._section("停止")
        self._print(f"  原因: {reason}")

    def final(self, best: Optional[ExperimentIR], report_md: str, report_zh: Optional[str]) -> None:
        if not self.enabled:
            return
        self._section("最终结果")
        if best is None:
            self._print("  最优: 无")
        else:
            action_id = best.action.action_id if best.action else "baseline"
            speedup = best.results.derived_metrics.get("speedup_vs_baseline")
            speedup_str = f"{speedup:.3f}x" if speedup is not None else "n/a"
            self._print(
                f"  最优: {action_id} 耗时={best.results.runtime_seconds:.4f}s 加速={speedup_str}"
            )
        self._print(f"  报告: {report_md}")
        if report_zh:
            self._print(f"  中文报告: {report_zh}")

    def update_baseline(self, exp: ExperimentIR) -> None:
        self.baseline_runtime = exp.results.runtime_seconds

    def _section(self, title: str) -> None:
        self._print("")
        self._print(f"=== {title} ===")

    def _agent(self, name: str, lines: List[str]) -> None:
        self._print(f"[{name}]")
        for line in lines:
            self._print(f"  {line}")

    def _kv(self, key: str, value: str) -> None:
        self._print(f"- {key}: {value}")

    def _print(self, line: str) -> None:
        if not self.enabled:
            return
        self.stream.write(line + "\n")
        self.stream.flush()


def _fmt_env(env: Dict[str, str]) -> str:
    return " ".join(f"{key}={value}" for key, value in env.items())


def _format_timing(profile: ProfileReport) -> str:
    timing = profile.timing_breakdown or {}
    total = timing.get("total", 0.0) or 0.0
    if total <= 0.0:
        return ""
    parts: List[str] = [f"总计={total:.4f}s"]
    for key in ("pair", "kspace", "neigh", "comm", "modify", "output"):
        value = timing.get(key)
        if value is None:
            continue
        ratio = (value / total) if total else 0.0
        parts.append(f"{key}={value:.4f}s({ratio:.0%})")
    return " ".join(parts)


def _extract_variance_cv(exp: ExperimentIR) -> Optional[float]:
    cv = exp.results.derived_metrics.get("variance_cv")
    if cv is not None:
        return float(cv)
    cv = exp.results.correctness_metrics.get("variance_cv")
    if cv is not None:
        return float(cv)
    return None


def _split_lines(text: str, prefix: str) -> List[str]:
    items = [line.strip() for line in text.splitlines() if line.strip()]
    if not items:
        return []
    return [f"{prefix}{item}" for item in items]


def _read_preview_text(path: Optional[str], max_bytes: int) -> Optional[Tuple[str, bool, int]]:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    size = file_path.stat().st_size
    truncated = size > max_bytes
    if truncated:
        with file_path.open("rb") as handle:
            handle.seek(-max_bytes, 2)
            data = handle.read(max_bytes)
        text = data.decode("utf-8", errors="replace")
    else:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    return text, truncated, size
