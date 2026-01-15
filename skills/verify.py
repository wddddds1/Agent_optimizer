from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from schemas.action_ir import ActionIR
from schemas.profile_report import ProfileReport
from schemas.result_ir import ResultIR
from skills.metrics_parse import extract_error_lines, parse_thermo_table


@dataclass
class VerifyResult:
    verdict: str
    reasons: List[str]
    correctness_metrics: Dict[str, object]


def verify_run(
    action: Optional[ActionIR],
    result: ResultIR,
    profile: ProfileReport,
    gates: Dict[str, object],
    baseline_profile: Optional[ProfileReport],
) -> VerifyResult:
    reasons: List[str] = []
    correctness_metrics: Dict[str, object] = {}

    runtime_cfg = gates.get("runtime", {})
    if runtime_cfg.get("require_exit_code_zero", True) and result.exit_code != 0:
        reasons.append(f"nonzero exit code: {result.exit_code}")

    error_regex = runtime_cfg.get("error_regex")
    if error_regex and profile.log_path:
        try:
            log_text = Path(profile.log_path).read_text(encoding="utf-8")
            errors = extract_error_lines(log_text, error_regex)
            if errors:
                reasons.append("error pattern found in log")
        except FileNotFoundError:
            reasons.append("log file missing for runtime check")

    correctness_cfg = gates.get("correctness", {})
    require_correctness = False
    if action:
        if "input_script" in action.applies_to:
            require_correctness = True
        elif action.risk_level != "low":
            require_correctness = True

    if require_correctness:
        log_text = ""
        if profile.log_path:
            try:
                log_text = Path(profile.log_path).read_text(encoding="utf-8")
            except FileNotFoundError:
                reasons.append("log file missing for correctness check")
        metrics = parse_thermo_table(log_text)
        if not metrics:
            reasons.append("no thermo metrics available for correctness check")
        else:
            correctness_metrics["key_scalar_diffs"] = {}
            if baseline_profile and baseline_profile.log_path:
                try:
                    baseline_text = Path(baseline_profile.log_path).read_text(encoding="utf-8")
                    baseline_metrics = parse_thermo_table(baseline_text)
                except FileNotFoundError:
                    baseline_metrics = {}
                for key, value in metrics.items():
                    base = baseline_metrics.get(key)
                    if base is None:
                        continue
                    diff = abs(value - base)
                    rel = diff / max(abs(base), 1.0e-12)
                    correctness_metrics["key_scalar_diffs"][key] = {"abs": diff, "rel": rel}
                    abs_thresh = correctness_cfg.get("scalar_thresholds", {}).get("abs", 0.0)
                    rel_thresh = correctness_cfg.get("scalar_thresholds", {}).get("rel", 0.0)
                    if diff > abs_thresh and rel > rel_thresh:
                        reasons.append(f"correctness drift: {key}")
            else:
                reasons.append("baseline metrics missing for correctness check")
    else:
        if correctness_cfg.get("allow_skip_for_low_risk_run_config", False):
            correctness_metrics["correctness_skipped_reason"] = "low-risk run_config action"

    variance_cfg = gates.get("variance", {})
    samples = result.samples or []
    if len(samples) >= 2:
        mean = sum(samples) / len(samples)
        if mean > 0:
            var = sum((x - mean) ** 2 for x in samples) / len(samples)
            cv = (var ** 0.5) / mean
            correctness_metrics["variance_cv"] = cv
            if cv > variance_cfg.get("cv_max", 1.0):
                reasons.append("variance gate failed")

    verdict = "PASS" if not reasons else "FAIL"
    return VerifyResult(verdict=verdict, reasons=reasons, correctness_metrics=correctness_metrics)
