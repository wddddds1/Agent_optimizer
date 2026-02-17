from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional

from schemas.action_ir import ActionIR
from schemas.experiment_ir import ExperimentIR
from schemas.job_ir import JobIR
from schemas.profile_report import ProfileReport
from schemas.result_ir import ResultIR
from skills.applications import (
    requires_structured_correctness as app_requires_structured_correctness,
    supports_agentic_correctness as app_supports_agentic_correctness,
)
from skills.metrics_parse import extract_error_lines, parse_thermo_series, parse_thermo_table


@dataclass
class VerifyResult:
    verdict: str
    reasons: List[str]
    correctness_metrics: Dict[str, object]


def verify_run(
    job: JobIR,
    action: Optional[ActionIR],
    result: ResultIR,
    profile: ProfileReport,
    gates: Dict[str, object],
    baseline_exp: Optional[ExperimentIR],
    is_final_validation: bool = False,
    agentic_decider=None,
    agentic_cfg: Optional[Dict[str, object]] = None,
    contract_getter: Optional[Callable[[JobIR], Optional[Dict[str, object]]]] = None,
    contract_putter: Optional[Callable[[JobIR, Dict[str, object]], None]] = None,
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
    correctness_cfg, final_cfg = _apply_final_validation(correctness_cfg, is_final_validation)
    apply_relaxations = True
    if is_final_validation:
        apply_relaxations = bool(final_cfg.get("allow_relaxations", False))
    is_baseline = action is None
    # Always require correctness checks for every non-baseline run.
    # Baseline still respects baseline_require_thermo.
    require_correctness = False
    if is_baseline:
        require_correctness = bool(correctness_cfg.get("baseline_require_thermo", True))
    else:
        require_correctness = True
    if is_final_validation:
        require_correctness = True

    effective_cfg = _effective_correctness_cfg(
        correctness_cfg,
        action,
        apply_relaxations=apply_relaxations,
    )
    agentic_cfg = agentic_cfg or {}
    agentic_mode = str(agentic_cfg.get("mode", "")).lower()
    strict_structured_correctness = app_requires_structured_correctness(job.app)
    use_agent = (agentic_decider is not None and agentic_mode == "agent_only"
                 and not is_baseline and app_supports_agentic_correctness(job.app))

    if require_correctness:
        log_text = ""
        if profile.log_path:
            try:
                log_text = Path(profile.log_path).read_text(encoding="utf-8")
            except FileNotFoundError:
                reasons.append("log file missing for correctness check")
        series_cfg = effective_cfg.get("series_compare", {})
        series_window = int(series_cfg.get("window", 0)) if series_cfg else 0
        metrics = parse_thermo_table(log_text)
        series = parse_thermo_series(log_text, max_rows=series_window)
        baseline_text = ""
        if not is_baseline and baseline_exp and baseline_exp.profile_report.log_path:
            try:
                baseline_text = Path(baseline_exp.profile_report.log_path).read_text(encoding="utf-8")
            except FileNotFoundError:
                baseline_text = ""
        if not metrics and not series:
            correctness_metrics["no_thermo_metrics"] = True
            generic_eval = _evaluate_generic_contract(
                job=job,
                run_text=log_text,
                baseline_text=baseline_text if not is_baseline else "",
                is_baseline=is_baseline,
                contract_getter=contract_getter,
                contract_putter=contract_putter,
            )
            correctness_metrics["generic_contract"] = generic_eval
            status = str(generic_eval.get("status", "UNSURE")).upper()
            if status == "FAIL":
                reasons.append(f"generic correctness mismatch: {generic_eval.get('reason', 'mismatch')}")
            elif status == "UNSURE":
                if not use_agent and strict_structured_correctness:
                    reasons.append("no thermo metrics available for correctness check")
                elif not use_agent:
                    reasons.append(
                        f"correctness uncertain: {generic_eval.get('reason', 'generic contract unavailable')}"
                    )
        else:
            correctness_metrics["key_scalar_diffs"] = {}
            correctness_metrics["series_diffs"] = {}
            correctness_metrics["scalar_values"] = {}
            correctness_metrics["series_stats"] = {}
            adaptive_cfg = {}
            baseline_series = {}
            baseline_metrics = {}
            if not is_baseline:
                if baseline_text:
                    baseline_metrics = parse_thermo_table(baseline_text)
                    baseline_series = parse_thermo_series(baseline_text, max_rows=series_window)
                if not baseline_metrics and not baseline_series:
                    if not use_agent:
                        reasons.append("baseline metrics missing for correctness check")
                    correctness_metrics["baseline_metrics_missing"] = True
            if effective_cfg.get("adaptive_from_baseline") and baseline_series:
                adaptive_cfg = _adaptive_thresholds(
                    baseline_series,
                    effective_cfg,
                )
                if adaptive_cfg:
                    correctness_metrics["adaptive_thresholds"] = adaptive_cfg

            if baseline_metrics and metrics:
                for key, value in metrics.items():
                    base = baseline_metrics.get(key)
                    if base is None:
                        continue
                    diff = abs(value - base)
                    rel = diff / max(abs(base), 1.0e-12)
                    correctness_metrics["key_scalar_diffs"][key] = {"abs": diff, "rel": rel}
                    correctness_metrics["scalar_values"][key] = {"baseline": base, "run": value}
                    if not use_agent:
                        abs_thresh = effective_cfg.get("scalar_thresholds", {}).get("abs", 0.0)
                        rel_thresh = effective_cfg.get("scalar_thresholds", {}).get("rel", 0.0)
                        if adaptive_cfg:
                            key_cfg = adaptive_cfg.get("scalar_thresholds", {}).get(key, {})
                            abs_thresh = max(abs_thresh, float(key_cfg.get("abs", 0.0)))
                            rel_thresh = max(rel_thresh, float(key_cfg.get("rel", 0.0)))
                        if diff > abs_thresh and rel > rel_thresh:
                            reasons.append(f"correctness drift: {key}")

            if baseline_series and series:
                # Build timestep-aligned index for accurate comparison
                steps = series.get("Step", [])
                base_steps = baseline_series.get("Step", [])
                if steps and base_steps:
                    # Align by timestep intersection
                    base_step_idx = {int(s): i for i, s in enumerate(base_steps)}
                    aligned_indices = []  # (series_idx, baseline_idx)
                    for si, s in enumerate(steps):
                        bi = base_step_idx.get(int(s))
                        if bi is not None:
                            aligned_indices.append((si, bi))
                else:
                    # Fallback: align from end (legacy behaviour)
                    aligned_indices = None

                for key, values in series.items():
                    if key == "Step":
                        continue
                    base_values = baseline_series.get(key)
                    if not base_values:
                        continue
                    if aligned_indices is not None:
                        if not aligned_indices:
                            continue
                        diffs = [abs(values[si] - base_values[bi]) for si, bi in aligned_indices]
                        base_slice = [base_values[bi] for _, bi in aligned_indices]
                        run_slice = [values[si] for si, _ in aligned_indices]
                    else:
                        count = min(len(values), len(base_values))
                        if count == 0:
                            continue
                        diffs = [abs(values[-count + i] - base_values[-count + i]) for i in range(count)]
                        base_slice = base_values[-count:]
                        run_slice = values[-count:]
                    max_abs = max(diffs)
                    base_mag = max(max(abs(v) for v in base_slice), 1.0e-12)
                    max_rel = max_abs / base_mag
                    correctness_metrics["series_diffs"][key] = {"max_abs": max_abs, "max_rel": max_rel}
                    stats = _series_stats(base_slice, run_slice)
                    if stats:
                        correctness_metrics["series_stats"][key] = stats
                    if not use_agent:
                        abs_thresh = series_cfg.get("abs_max", 0.0) if series_cfg else 0.0
                        rel_thresh = series_cfg.get("rel_max", 0.0) if series_cfg else 0.0
                        if adaptive_cfg:
                            key_cfg = adaptive_cfg.get("series_compare", {}).get(key, {})
                            abs_thresh = max(abs_thresh, float(key_cfg.get("abs_max", 0.0)))
                            rel_thresh = max(rel_thresh, float(key_cfg.get("rel_max", 0.0)))
                        if max_abs > abs_thresh and max_rel > rel_thresh:
                            reasons.append(f"series drift: {key}")

            drift_cfg = effective_cfg.get("energy_drift", {})
            drift_metric = drift_cfg.get("metric", "TotEng")
            drift_limit = float(drift_cfg.get("rel_max", 0.0))
            drift_value = _energy_drift(series, drift_metric)
            baseline_drift = _energy_drift(baseline_series, drift_metric) if baseline_series else None
            if drift_value is not None:
                correctness_metrics["energy_drift"] = drift_value
            if baseline_drift is not None:
                correctness_metrics["energy_drift_baseline"] = baseline_drift
            if not use_agent and not is_baseline and drift_value is not None and drift_limit:
                if adaptive_cfg:
                    drift_cfg = adaptive_cfg.get("energy_drift", {})
                    drift_limit = max(drift_limit, float(drift_cfg.get("rel_max", 0.0)))
                if baseline_drift is not None:
                    # Only fail if drift got WORSE (larger magnitude), not if
                    # it improved (smaller magnitude than baseline).
                    if abs(drift_value) > abs(baseline_drift) and abs(drift_value - baseline_drift) > drift_limit:
                        reasons.append("energy drift delta exceeded")
                elif abs(drift_value) > drift_limit:
                    reasons.append("energy drift threshold exceeded")

        should_call_agent = use_agent and (
            "generic_contract" not in correctness_metrics
            or str(correctness_metrics["generic_contract"].get("status", "")).upper() == "UNSURE"
        )
        if should_call_agent:
            hard_failures = [
                r
                for r in reasons
                if not r.startswith("correctness drift")
                and not r.startswith("series drift")
                and "energy drift" not in r
                and not r.startswith("correctness uncertain:")
            ]
            payload = _build_agent_payload(
                job=job,
                action=action,
                profile=profile,
                result=result,
                correctness_metrics=correctness_metrics,
                hard_failures=hard_failures,
                run_log_text=log_text,
                baseline_log_text=baseline_text if not is_baseline else "",
                agentic_cfg=agentic_cfg,
            )
            decision = agentic_decider(payload) if agentic_decider else None
            if isinstance(decision, dict):
                correctness_metrics["agent_decision"] = decision
                verdict = str(decision.get("verdict", "")).upper()
                if verdict == "FAIL":
                    reasons.append(f"agent correctness: {decision.get('rationale', 'FAIL')}")
                elif verdict == "NEED_MORE_CONTEXT":
                    reasons.append("agent correctness: NEED_MORE_CONTEXT")

    else:
        if effective_cfg.get("allow_skip_for_low_risk_run_config", False):
            correctness_metrics["correctness_skipped_reason"] = "low-risk run_config action"

    variance_cfg = gates.get("variance", {})
    samples = result.samples or []
    if len(samples) >= 2:
        mean = sum(samples) / len(samples)
        if mean > 0:
            var = sum((x - mean) ** 2 for x in samples) / len(samples)
            cv = (var ** 0.5) / mean
            correctness_metrics["variance_cv"] = cv
            if (not is_baseline or variance_cfg.get("baseline_enforce", False)) and cv > variance_cfg.get(
                "cv_max", 1.0
            ):
                reasons.append("variance gate failed")

    output_files = correctness_cfg.get("output_files", [])
    if require_correctness and output_files and not is_baseline:
        baseline_workdir = (
            Path(baseline_exp.job.workdir)
            if baseline_exp and baseline_exp.job and baseline_exp.job.workdir
            else None
        )
        run_workdir = Path(job.workdir) if job.workdir else None
        if baseline_workdir and run_workdir:
            baseline_hashes = _hash_output_files(baseline_workdir, output_files)
            run_hashes = _hash_output_files(run_workdir, output_files)
            correctness_metrics["output_hashes"] = {"baseline": baseline_hashes, "run": run_hashes}
            for name, base_hash in baseline_hashes.items():
                run_hash = run_hashes.get(name)
                if base_hash is None or run_hash is None:
                    reasons.append(f"missing output file: {name}")
                elif base_hash != run_hash:
                    reasons.append(f"output hash mismatch: {name}")
        else:
            reasons.append("output file comparison unavailable")

    if action and action.family == "runtime_backend_select" and not use_agent:
        if effective_cfg.get("backend_select_allow_drift", False):
            drift_prefixes = ("correctness drift:", "series drift:", "energy drift")
            drift_reasons = [r for r in reasons if r.startswith(drift_prefixes)]
            if drift_reasons:
                reasons = [r for r in reasons if not r.startswith(drift_prefixes)]
                correctness_metrics["backend_select_drift_ignored"] = True

    # Runtime regression guard: fail if >2x slower than baseline.
    regression_limit = float(runtime_cfg.get("max_regression_factor", 2.0))
    if (
        not is_baseline
        and baseline_exp
        and baseline_exp.results
        and baseline_exp.results.runtime_seconds > 0
        and result.runtime_seconds > 0
    ):
        factor = result.runtime_seconds / baseline_exp.results.runtime_seconds
        if factor > regression_limit:
            reasons.append(
                f"runtime regression: {factor:.1f}x slower than baseline"
            )

    verdict = "PASS" if not reasons else "FAIL"
    return VerifyResult(verdict=verdict, reasons=reasons, correctness_metrics=correctness_metrics)


def _evaluate_generic_contract(
    job: JobIR,
    run_text: str,
    baseline_text: str,
    is_baseline: bool,
    contract_getter: Optional[Callable[[JobIR], Optional[Dict[str, object]]]] = None,
    contract_putter: Optional[Callable[[JobIR, Dict[str, object]], None]] = None,
) -> Dict[str, object]:
    run_sig = _generic_signature(run_text)
    if run_sig is None:
        if _signature_intentionally_suppressed(job):
            return {
                "status": "PASS",
                "reason": "signature intentionally suppressed by run args",
                "signature": None,
                "suppressed_signature": True,
            }
        return {"status": "UNSURE", "reason": "run signature unavailable"}

    if is_baseline:
        contract = {
            "version": 1,
            "kind": "generic_signature_v1",
            "app": job.app,
            "case_id": job.case_id,
            "signature": run_sig,
        }
        if contract_putter:
            try:
                contract_putter(job, contract)
            except Exception:
                pass
        return {
            "status": "PASS",
            "reason": "baseline contract recorded",
            "signature": run_sig,
        }

    baseline_sig = _generic_signature(baseline_text) if baseline_text else None
    if baseline_sig is None and contract_getter:
        try:
            cached = contract_getter(job)
        except Exception:
            cached = None
        if isinstance(cached, dict):
            cand = cached.get("signature")
            if isinstance(cand, dict):
                baseline_sig = cand
    if baseline_sig is None:
        return {"status": "UNSURE", "reason": "baseline signature unavailable", "signature": run_sig}

    verdict, mismatch = _compare_generic_signatures(baseline_sig, run_sig)
    if verdict == "PASS":
        return {
            "status": "PASS",
            "reason": "generic signature matched",
            "signature": run_sig,
            "baseline_signature": baseline_sig,
        }
    return {
        "status": "FAIL",
        "reason": mismatch or "signature mismatch",
        "signature": run_sig,
        "baseline_signature": baseline_sig,
    }


def _signature_intentionally_suppressed(job: JobIR) -> bool:
    app = str(getattr(job, "app", "") or "").strip().lower()
    run_args = [str(arg) for arg in (getattr(job, "run_args", []) or [])]
    for idx, token in enumerate(run_args):
        if token == "-o" and idx + 1 < len(run_args) and run_args[idx + 1] == "/dev/null":
            return app == "bwa"
    return False


def _generic_signature(text: str) -> Optional[Dict[str, object]]:
    if not text:
        return None
    lines = [line.rstrip("\n") for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    normalized = [_normalize_line(line) for line in lines if not _is_volatile_line(line)]
    payload = "\n".join(normalized).encode("utf-8", errors="replace")
    sig: Dict[str, object] = {
        "line_count": len(normalized),
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    sam = _sam_summary(lines)
    if sam:
        sig["sam"] = sam
    return sig


def _is_volatile_line(line: str) -> bool:
    lowered = line.lower()
    volatile_tokens = (
        "real time:",
        "cpu:",
        " user ",
        " sys",
        "maximum resident set size",
        "context switches",
        "instructions retired",
        "cycles elapsed",
        "peak memory footprint",
        "page faults",
        "messages sent",
        "messages received",
    )
    return any(token in lowered for token in volatile_tokens)


def _normalize_line(line: str) -> str:
    return " ".join(line.strip().split())


def _sam_summary(lines: List[str]) -> Optional[Dict[str, object]]:
    total = 0
    mapped = 0
    unmapped = 0
    primary = 0
    secondary = 0
    supplementary = 0
    nm_sum = 0
    mapq_sum = 0
    hash_xor = 0
    hash_sum = 0
    mask64 = (1 << 64) - 1

    for line in lines:
        if line.startswith("@"):
            continue
        parts = line.split("\t")
        if len(parts) < 11:
            continue
        try:
            flag = int(parts[1])
            pos = int(parts[3])
            mapq = int(parts[4])
        except (TypeError, ValueError):
            continue
        total += 1
        mapq_sum += mapq
        if flag & 0x4:
            unmapped += 1
        else:
            mapped += 1
        if flag & 0x100:
            secondary += 1
        if flag & 0x800:
            supplementary += 1
        if not (flag & 0x100) and not (flag & 0x800):
            primary += 1
        for field in parts[11:]:
            if field.startswith("NM:i:"):
                try:
                    nm_sum += int(field[5:])
                except ValueError:
                    pass
                break
        key = f"{parts[0]}\t{flag}\t{parts[2]}\t{pos}\t{parts[5]}\t{parts[9]}"
        digest = hashlib.sha1(key.encode("utf-8", errors="replace")).digest()
        value = int.from_bytes(digest[:8], "big", signed=False)
        hash_xor ^= value
        hash_sum = (hash_sum + value) & mask64

    if total < 10:
        return None
    return {
        "total": total,
        "mapped": mapped,
        "unmapped": unmapped,
        "primary": primary,
        "secondary": secondary,
        "supplementary": supplementary,
        "mapq_sum": mapq_sum,
        "nm_sum": nm_sum,
        "hash_xor": hash_xor,
        "hash_sum": hash_sum,
    }


def _compare_generic_signatures(
    baseline: Dict[str, object],
    run: Dict[str, object],
) -> tuple[str, str]:
    base_sam = baseline.get("sam")
    run_sam = run.get("sam")
    if isinstance(base_sam, dict) and isinstance(run_sam, dict):
        keys = [
            "total",
            "mapped",
            "unmapped",
            "primary",
            "secondary",
            "supplementary",
            "nm_sum",
            "hash_xor",
            "hash_sum",
        ]
        for key in keys:
            if int(run_sam.get(key, -1)) != int(base_sam.get(key, -1)):
                return "FAIL", f"sam_{key}_mismatch"
        return "PASS", ""

    if (
        str(run.get("sha256", "")) == str(baseline.get("sha256", ""))
        and int(run.get("line_count", -1)) == int(baseline.get("line_count", -1))
    ):
        return "PASS", ""
    return "FAIL", "text_signature_mismatch"


def _energy_drift(series: Dict[str, List[float]], key: str) -> Optional[float]:
    values = series.get(key) if series else None
    if not values or len(values) < 2:
        return None
    start = values[0]
    end = values[-1]
    denom = abs(start) if abs(start) > 1.0e-12 else 1.0
    return (end - start) / denom


def _series_stats(baseline_vals: List[float], run_vals: List[float]) -> Optional[Dict[str, float]]:
    if not baseline_vals or not run_vals:
        return None
    n = min(len(baseline_vals), len(run_vals))
    if n == 0:
        return None
    b = baseline_vals[:n]
    r = run_vals[:n]
    b_mean = sum(b) / n
    r_mean = sum(r) / n
    b_var = sum((x - b_mean) ** 2 for x in b) / n
    r_var = sum((x - r_mean) ** 2 for x in r) / n
    b_std = math.sqrt(b_var)
    r_std = math.sqrt(r_var)
    diff_mean = r_mean - b_mean
    rel_mean = diff_mean / max(abs(b_mean), 1.0e-12)
    return {
        "n": n,
        "baseline_mean": b_mean,
        "baseline_std": b_std,
        "run_mean": r_mean,
        "run_std": r_std,
        "mean_diff": diff_mean,
        "mean_rel": rel_mean,
    }


def _build_agent_payload(
    job: JobIR,
    action: Optional[ActionIR],
    profile: ProfileReport,
    result: ResultIR,
    correctness_metrics: Dict[str, object],
    hard_failures: List[str],
    run_log_text: str,
    baseline_log_text: str,
    agentic_cfg: Dict[str, object],
) -> Dict[str, object]:
    max_lines = int(agentic_cfg.get("max_log_lines", 80) or 80)
    payload = {
        "action": action.model_dump() if action else None,
        "job": {
            "app": job.app,
            "case_id": job.case_id,
            "run_args": job.run_args,
            "env": job.env,
            "input_script": job.input_script,
        },
        "profile": {
            "timing_breakdown": profile.timing_breakdown,
            "system_metrics": profile.system_metrics,
            "notes": profile.notes,
        },
        "runtime": {
            "runtime_seconds": result.runtime_seconds,
            "exit_code": result.exit_code,
        },
        "correctness_metrics": correctness_metrics,
        "hard_failures": hard_failures,
        "run_thermo_excerpt": _thermo_excerpt(run_log_text, max_lines=max_lines),
        "baseline_thermo_excerpt": _thermo_excerpt(baseline_log_text, max_lines=max_lines),
    }
    return payload


def _thermo_excerpt(text: str, max_lines: int = 80) -> str:
    if not text:
        return ""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Step") and len(line.split()) > 2:
            header_idx = i
    if header_idx is None:
        return "\n".join(lines[-max_lines:])
    excerpt = lines[header_idx : header_idx + max_lines]
    return "\n".join(excerpt)


def _adaptive_thresholds(
    baseline_series: Dict[str, List[float]],
    effective_cfg: Dict[str, object],
) -> Dict[str, object]:
    multiplier = float(effective_cfg.get("adaptive_multiplier", 3.0))
    floors = effective_cfg.get("adaptive_floor", {})
    if not isinstance(floors, dict):
        floors = {}
    scalar_abs_floor = float(floors.get("scalar_abs", 0.0))
    scalar_rel_floor = float(floors.get("scalar_rel", 0.0))
    series_abs_floor = float(floors.get("series_abs", 0.0))
    series_rel_floor = float(floors.get("series_rel", 0.0))
    drift_rel_floor = float(floors.get("drift_rel", 0.0))
    scalar_thresholds: Dict[str, Dict[str, float]] = {}
    series_thresholds: Dict[str, Dict[str, float]] = {}

    for key, values in baseline_series.items():
        if key == "Step" or not values:
            continue
        mean = sum(values) / len(values)
        diffs = [abs(v - mean) for v in values]
        max_abs = max(diffs)
        base_mag = max(abs(mean), 1.0e-12)
        max_rel = max_abs / base_mag
        scalar_thresholds[key] = {
            "abs": max(max_abs * multiplier, scalar_abs_floor),
            "rel": max(max_rel * multiplier, scalar_rel_floor),
        }
        series_thresholds[key] = {
            "abs_max": max(max_abs * multiplier, series_abs_floor),
            "rel_max": max(max_rel * multiplier, series_rel_floor),
        }

    drift_metric = effective_cfg.get("energy_drift", {}).get("metric", "TotEng")
    drift_value = _energy_drift(baseline_series, str(drift_metric))
    drift_rel = abs(drift_value) if drift_value is not None else 0.0
    energy_drift = {"rel_max": max(drift_rel * multiplier, drift_rel_floor)}
    return {
        "scalar_thresholds": scalar_thresholds,
        "series_compare": series_thresholds,
        "energy_drift": energy_drift,
    }


def _effective_correctness_cfg(
    correctness_cfg: Dict[str, object],
    action: Optional[ActionIR],
    apply_relaxations: bool = True,
) -> Dict[str, object]:
    if not apply_relaxations or not action:
        return correctness_cfg
    if action.family == "runtime_backend_select":
        relaxed = correctness_cfg.get("backend_select_relaxed", {})
        if isinstance(relaxed, dict) and relaxed:
            effective = dict(correctness_cfg)
            for key in ("scalar_thresholds", "series_compare", "energy_drift"):
                if key in relaxed:
                    effective[key] = relaxed[key]
            return effective
    if action.family == "neighbor_tune":
        relaxed = correctness_cfg.get("neighbor_tune_relaxed", {})
        if isinstance(relaxed, dict) and relaxed:
            effective = dict(correctness_cfg)
            for key in ("scalar_thresholds", "series_compare", "energy_drift"):
                if key in relaxed:
                    effective[key] = relaxed[key]
            return effective
    if not any(target in action.applies_to for target in ["build_config"]):
        return correctness_cfg
    params = action.parameters or {}
    numeric_risk = params.get("numeric_risk")
    if numeric_risk != "low":
        return correctness_cfg
    relaxed = correctness_cfg.get("build_config_relaxed", {})
    if not isinstance(relaxed, dict):
        return correctness_cfg
    effective = dict(correctness_cfg)
    for key in ("scalar_thresholds", "series_compare", "energy_drift"):
        if key in relaxed:
            effective[key] = relaxed[key]
    return effective


def _apply_final_validation(
    correctness_cfg: Dict[str, object],
    is_final_validation: bool,
) -> tuple[Dict[str, object], Dict[str, object]]:
    if not is_final_validation:
        return correctness_cfg, {}
    final_cfg = correctness_cfg.get("final_validation", {})
    if not isinstance(final_cfg, dict) or not final_cfg:
        merged = dict(correctness_cfg)
        merged["allow_skip_for_low_risk_run_config"] = False
        merged["backend_select_allow_drift"] = False
        return merged, {}
    merged = dict(correctness_cfg)
    for key, value in final_cfg.items():
        if key == "allow_relaxations":
            continue
        merged[key] = value
    return merged, final_cfg


def _hash_output_files(workdir: Path, files: List[str]) -> Dict[str, Optional[str]]:
    hashes: Dict[str, Optional[str]] = {}
    for name in files:
        path = Path(name)
        if not path.is_absolute():
            path = workdir / path
        if not path.exists():
            hashes[name] = None
            continue
        hashes[name] = _sha256_path(path)
    return hashes


def _sha256_path(path: Path) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
