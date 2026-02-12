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
from schemas.review_ir import ReviewDecision


def _format_diff_stats(stats: Optional[Dict[str, object]]) -> Optional[str]:
    """Format diff statistics into a compact summary string."""
    if not stats:
        return None
    files = stats.get("files")
    lines = stats.get("lines_changed")
    meaningful = stats.get("meaningful_lines")
    parts = []
    if isinstance(files, list) and files:
        parts.append(f"{len(files)}个文件")
    if isinstance(lines, int) and lines > 0:
        if isinstance(meaningful, int) and meaningful != lines:
            parts.append(f"{lines}行变更({meaningful}行有效)")
        else:
            parts.append(f"{lines}行变更")
    return f"[{', '.join(parts)}]" if parts else None


@dataclass
class ConsoleUI:
    enabled: bool = True
    stream: TextIO = sys.stdout
    baseline_runtime: Optional[float] = None
    verbose: bool = False
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
    ) -> None:
        if not self.enabled:
            return
        self._section("运行开始")
        self._kv("算例", job.case_id)
        self._kv("工作目录", job.workdir)
        self._kv("可执行文件", job.app_bin)
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
            f"重复={baseline_repeats} 统计={baseline_stat} 复跑Top1={validate_top1_repeats}",
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
            "PlannerAgent",
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
                f"排序依据: {ranking_mode}",
                f"选择模式: {selection_mode}",
                f"候选数: {len(actions)}",
            ],
        )
        if actions:
            self._print("  候选列表:")
            for idx, action in enumerate(actions, start=1):
                reason = action.notes or action.description or ""
                reason = _shorten(reason, 120) if reason else ""
                suffix = f" — {reason}" if reason else ""
                self._print(
                    f"    {idx}. {action.action_id} (方向={action.family}, 风险={action.risk_level}){suffix}"
                )
                if self.verbose:
                    effect = ",".join(action.expected_effect)
                    if effect:
                        self._print(f"       预期效果: {effect}")
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
        if ranked.ranked:
            self._print("  排序列表:")
            for idx, item in enumerate(ranked.ranked, start=1):
                action = item.action
                score = f"{item.score:.2f}"
                self._print(
                    f"    {idx}. {action.action_id} (方向={action.family}, 分数={score})"
                )

    def review_summary(self, decision: ReviewDecision) -> None:
        if not self.enabled:
            return
        self._section("收敛评审")
        self._agent(
            "ReviewerAgent",
            [
                f"是否停止: {decision.should_stop}",
                f"置信度: {decision.confidence:.2f}",
                f"建议动作: {decision.suggested_next_step}",
                f"理由: {decision.reason}",
            ],
        )

    def run_start(
        self,
        exp_id: str,
        action: Optional[ActionIR],
        env_overrides: Dict[str, str],
        run_args: List[str],
        base_run_id: Optional[str] = None,
        base_action_id: Optional[str] = None,
        run_cmd: Optional[str] = None,
        build_cmds: Optional[List[str]] = None,
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
            detail_lines = _action_change_summary(action, env_overrides)
            if detail_lines:
                self._print("  优化说明:")
                for line in detail_lines:
                    self._print(f"    - {line}")
            if action.description and self.verbose:
                self._print(f"  描述: {action.description}")
            if base_action_id or base_run_id:
                base_action_id = base_action_id or "baseline"
                base_run_id = base_run_id or "baseline"
                if self.verbose:
                    self._print(f"  基于: {base_action_id} ({base_run_id})")
        if env_overrides and self.verbose:
            self._print(f"  环境覆盖: {_fmt_env(env_overrides)}")
        if run_cmd:
            self._print(f"  运行指令: {run_cmd}")
        elif run_args and self.verbose:
            self._print(f"  运行参数: {' '.join(run_args)}")
        if build_cmds:
            self._print("  构建指令:")
            for cmd in build_cmds:
                self._print(f"    - {cmd}")

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
        best_combo: Optional[list[str]] = None,
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
        if best_combo:
            self._print(f"  当前最优组合(生效): {' | '.join(best_combo)}")

    def patch_debug(self, action_id: str, attempt: int, status: str, note: Optional[str] = None) -> None:
        if not self.enabled:
            return
        self._patch_line(
            action_id=action_id,
            step=f"debug#{attempt}",
            status=status,
            note=note,
        )

    def patch_proposal(
        self,
        action_id: str,
        status: str,
        note: Optional[str] = None,
        diff_stats: Optional[Dict[str, object]] = None,
    ) -> None:
        if not self.enabled:
            return
        stats_str = _format_diff_stats(diff_stats)
        self._patch_line(
            action_id=action_id,
            step="proposal(LLM生成补丁)",
            status=status,
            note=note,
            stats=stats_str,
        )

    def patch_review(
        self,
        action_id: str,
        stage: str,
        verdict: str,
        note: Optional[str] = None,
        diff_stats: Optional[Dict[str, object]] = None,
    ) -> None:
        if not self.enabled:
            return
        _STAGE_DESC = {
            "deterministic": "规则校验",
            "target_check": "目标文件校验",
            "llm": "LLM审查",
        }
        desc = _STAGE_DESC.get(stage, stage)
        stats_str = _format_diff_stats(diff_stats)
        self._patch_line(
            action_id=action_id,
            step=f"review.{stage}({desc})",
            status=verdict,
            note=note,
            stats=stats_str,
        )

    def patch_apply_check(self, action_id: str, ok: bool, reason: Optional[str] = None) -> None:
        if not self.enabled:
            return
        verdict = "OK" if ok else "FAIL"
        self._patch_line(
            action_id=action_id,
            step="apply_check(git apply试运行)",
            status=verdict,
            note=reason,
        )

    def patch_preflight(self, action_id: str, ok: bool, reason: Optional[str] = None) -> None:
        if not self.enabled:
            return
        verdict = "OK" if ok else "FAIL"
        self._patch_line(
            action_id=action_id,
            step="preflight(编译检查)",
            status=verdict,
            note=reason,
        )

    def _patch_line(
        self,
        action_id: str,
        step: str,
        status: str,
        note: Optional[str] = None,
        stats: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return
        line = f"[Patch] 动作={action_id} 步骤={step} 状态={status}"
        if stats:
            line += f" {stats}"
        if note:
            line += f" 说明={note}"
        self._print(line)

    def worktree_error(self, exp_id: str, attempts: int, last_error: str) -> None:
        if not self.enabled:
            return
        lines = [
            f"实验ID: {exp_id}",
            f"重试次数: {attempts}",
        ]
        if last_error:
            lines.append(f"错误: {last_error}")
        self._agent("GitWorktree", lines)

    def stop(self, reason: str) -> None:
        if not self.enabled:
            return
        self._section("停止")
        self._print(f"  原因: {reason}")

    def skip(self, reason: str) -> None:
        if not self.enabled:
            return
        self._section("跳过")
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


def _action_change_summary(action: ActionIR, env_overrides: Dict[str, str]) -> List[str]:
    lines: List[str] = []
    params = action.parameters or {}
    if action.notes:
        lines.append(f"理由: {action.notes}")
    expected = ",".join(action.expected_effect) if action.expected_effect else ""
    if expected:
        lines.append(f"预期效果: {expected}")
    if env_overrides:
        env_items = ", ".join(f"{key}={value}" for key, value in env_overrides.items())
        lines.append(f"环境变量: {env_items}")
    run_args_cfg = params.get("run_args")
    if isinstance(run_args_cfg, dict):
        flags = []
        for item in run_args_cfg.get("set_flags", []) or []:
            if not isinstance(item, dict):
                continue
            flag = item.get("flag")
            values = item.get("values") or []
            if flag:
                if values:
                    flags.append(f"{flag} {' '.join(str(v) for v in values)}")
                else:
                    flags.append(str(flag))
        if flags:
            lines.append(f"运行参数: {' | '.join(flags)}")
    input_edit = params.get("input_edit")
    if isinstance(input_edit, list) and input_edit:
        edits = []
        for item in input_edit:
            if not isinstance(item, dict):
                continue
            line = item.get("line") or item.get("value")
            if line:
                edits.append(str(line))
        if edits:
            if len(edits) > 3:
                edits = edits[:3] + ["..."]
            lines.append(f"输入修改: {', '.join(edits)}")
    build_pack = params.get("build_pack") or params.get("build_pack_id")
    if build_pack:
        lines.append(f"构建配置: {build_pack}")
    backend = params.get("backend_enable")
    if backend:
        lines.append(f"后端选择: {backend}")
    patch_family = params.get("patch_family")
    if patch_family:
        lines.append(f"补丁类型: {patch_family}")
    return lines


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


def _shorten(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


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
