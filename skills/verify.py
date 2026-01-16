from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from schemas.action_ir import ActionIR
from schemas.experiment_ir import ExperimentIR
from schemas.job_ir import JobIR
from schemas.profile_report import ProfileReport
from schemas.result_ir import ResultIR
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
    is_baseline = action is None
    require_correctness = False
    if is_baseline:
        require_correctness = bool(correctness_cfg.get("baseline_require_thermo", True))
    elif action:
        if any(target in action.applies_to for target in ["input_script", "source_patch", "build_config"]):
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
        series_cfg = correctness_cfg.get("series_compare", {})
        series_window = int(series_cfg.get("window", 0)) if series_cfg else 0
        metrics = parse_thermo_table(log_text)
        series = parse_thermo_series(log_text, max_rows=series_window)
        if not metrics and not series:
            reasons.append("no thermo metrics available for correctness check")
        else:
            correctness_metrics["key_scalar_diffs"] = {}
            correctness_metrics["series_diffs"] = {}
            baseline_series = {}
            baseline_metrics = {}
            if not is_baseline:
                if baseline_exp and baseline_exp.profile_report.log_path:
                    try:
                        baseline_text = Path(baseline_exp.profile_report.log_path).read_text(encoding="utf-8")
                        baseline_metrics = parse_thermo_table(baseline_text)
                        baseline_series = parse_thermo_series(baseline_text, max_rows=series_window)
                    except FileNotFoundError:
                        baseline_metrics = {}
                        baseline_series = {}
                if not baseline_metrics and not baseline_series:
                    reasons.append("baseline metrics missing for correctness check")

            if baseline_metrics and metrics:
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

            if baseline_series and series:
                abs_thresh = series_cfg.get("abs_max", 0.0) if series_cfg else 0.0
                rel_thresh = series_cfg.get("rel_max", 0.0) if series_cfg else 0.0
                for key, values in series.items():
                    base_values = baseline_series.get(key)
                    if not base_values:
                        continue
                    count = min(len(values), len(base_values))
                    if count == 0:
                        continue
                    diffs = [abs(values[-count + i] - base_values[-count + i]) for i in range(count)]
                    max_abs = max(diffs)
                    base_mag = max(max(abs(v) for v in base_values[-count:]), 1.0e-12)
                    max_rel = max_abs / base_mag
                    correctness_metrics["series_diffs"][key] = {"max_abs": max_abs, "max_rel": max_rel}
                    if max_abs > abs_thresh and max_rel > rel_thresh:
                        reasons.append(f"series drift: {key}")

            drift_cfg = correctness_cfg.get("energy_drift", {})
            drift_metric = drift_cfg.get("metric", "TotEng")
            drift_limit = float(drift_cfg.get("rel_max", 0.0))
            drift_value = _energy_drift(series, drift_metric)
            baseline_drift = _energy_drift(baseline_series, drift_metric) if baseline_series else None
            if drift_value is not None:
                correctness_metrics["energy_drift"] = drift_value
            if baseline_drift is not None:
                correctness_metrics["energy_drift_baseline"] = baseline_drift
            if not is_baseline and drift_value is not None and drift_limit:
                if baseline_drift is not None:
                    if abs(drift_value - baseline_drift) > drift_limit:
                        reasons.append("energy drift delta exceeded")
                elif abs(drift_value) > drift_limit:
                    reasons.append("energy drift threshold exceeded")
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

    verdict = "PASS" if not reasons else "FAIL"
    return VerifyResult(verdict=verdict, reasons=reasons, correctness_metrics=correctness_metrics)


def _energy_drift(series: Dict[str, List[float]], key: str) -> Optional[float]:
    values = series.get(key) if series else None
    if not values or len(values) < 2:
        return None
    start = values[0]
    end = values[-1]
    denom = abs(start) if abs(start) > 1.0e-12 else 1.0
    return (end - start) / denom


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
