from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from schemas.action_ir import ActionIR
from schemas.job_ir import JobIR
from skills.metrics_parse import parse_lammps_timing

_DEFAULT_MPI_LAUNCHER = "mpirun"


def apply_adapter(action: ActionIR, job: JobIR, adapter_cfg: Dict[str, object] | None = None) -> ActionIR:
    if job.app != "lammps":
        return action

    global _DEFAULT_MPI_LAUNCHER
    _DEFAULT_MPI_LAUNCHER = (adapter_cfg or {}).get("mpi_launcher", "mpirun")

    params = dict(action.parameters or {})
    _merge_env_bucket(params, "mpi_env")
    _merge_env_bucket(params, "io_env")
    _merge_env_bucket(params, "runtime_env")
    _merge_env_bucket(params, "lib_env")
    backend = params.get("backend_enable")
    if backend == "openmp":
        params["run_args"] = _merge_run_args(
            params.get("run_args"),
            {
                "set_flags": [
                    {
                        "flag": "-sf",
                        "values": ["omp"],
                        "arg_count": 1,
                    }
                ]
            },
        )
        backend_threads = params.get("backend_threads")
        if backend_threads is not None:
            params["run_args"] = _merge_run_args(
                params.get("run_args"),
                {
                    "set_flags": [
                        {
                            "flag": "-pk",
                            "values": ["omp", str(backend_threads)],
                            "arg_count": 2,
                        }
                    ]
                },
            )
    elif backend == "opt":
        params["run_args"] = _merge_run_args(
            params.get("run_args"),
            {
                "set_flags": [
                    {
                        "flag": "-sf",
                        "values": ["opt"],
                        "arg_count": 1,
                    }
                ]
            },
        )
    elif backend == "mpi":
        # Pure MPI — no LAMMPS suffix needed
        pass
    elif backend == "mpi_omp":
        # MPI+OMP hybrid: needs -sf omp + thread count
        params["run_args"] = _merge_run_args(
            params.get("run_args"),
            {
                "set_flags": [
                    {
                        "flag": "-sf",
                        "values": ["omp"],
                        "arg_count": 1,
                    }
                ]
            },
        )
        backend_threads = params.get("backend_threads")
        if backend_threads is not None:
            params["run_args"] = _merge_run_args(
                params.get("run_args"),
                {
                    "set_flags": [
                        {
                            "flag": "-pk",
                            "values": ["omp", str(backend_threads)],
                            "arg_count": 2,
                        }
                    ]
                },
            )

    if "input_edit" not in params:
        input_edit = _build_input_edit(params)
        if input_edit:
            params["input_edit"] = input_edit

    # Auto-generate launcher config from action parameters
    launcher = _build_launcher_config(params)
    if launcher:
        params["launcher"] = launcher

    action.parameters = params
    return action


def _build_launcher_config(params: Dict[str, object]) -> Optional[Dict[str, object]]:
    """Auto-generate launcher config from action parameters.

    If an explicit ``launcher`` dict exists, resolve ``type: "auto"`` to the
    system default.  Otherwise, derive launcher config from the legacy
    ``mpi_env.MPI_RANKS`` hint.
    """
    launcher = params.get("launcher")
    if isinstance(launcher, dict):
        # Resolve "auto" type → system default
        if launcher.get("type") == "auto":
            launcher["type"] = _DEFAULT_MPI_LAUNCHER
        return launcher

    # Derive from mpi_env.MPI_RANKS if present
    mpi_env = params.get("mpi_env", {})
    if not isinstance(mpi_env, dict):
        return None
    ranks_str = mpi_env.get("MPI_RANKS")
    if ranks_str and int(ranks_str) > 1:
        # Remove MPI_RANKS from mpi_env (it's not a real env var)
        mpi_env_clean = {k: v for k, v in mpi_env.items() if k != "MPI_RANKS"}
        if mpi_env_clean:
            params["mpi_env"] = mpi_env_clean
        else:
            params.pop("mpi_env", None)
        return {
            "type": _DEFAULT_MPI_LAUNCHER,
            "np": int(ranks_str),
            "extra_args": [],
        }
    return None


def input_edit_allowlist() -> list[str]:
    return [
        "neighbor",
        "neigh_modify",
        "thermo",
        "dump",
        "kspace_style",
        "kspace_modify",
        "comm_modify",
        "newton",
    ]


def parse_timing_breakdown(log_text: str) -> Dict[str, float]:
    return parse_lammps_timing(log_text)


def requires_structured_correctness() -> bool:
    return True


def supports_agentic_correctness() -> bool:
    return True


def ensure_output_capture(
    run_args: List[str],
    run_dir: Path,
) -> Tuple[List[str], List[str]]:
    """LAMMPS output is already captured in stdout.log — no args change needed."""
    log_path = run_dir / "log.lammps"
    stdout_path = run_dir / "stdout.log"
    capture_paths = []
    # Prefer log.lammps (thermo data); fall back to stdout.log
    if "-log" in run_args:
        idx = run_args.index("-log")
        if idx + 1 < len(run_args):
            capture_paths.append(run_args[idx + 1])
    if not capture_paths:
        capture_paths.append(str(log_path))
    capture_paths.append(str(stdout_path))
    return list(run_args), capture_paths


def compute_drift(
    baseline_path: str,
    candidate_path: str,
    thresholds: Dict[str, object],
) -> "DriftReport":
    """Compute LAMMPS output drift via thermo series comparison."""
    from skills.verify import DriftReport
    from skills.metrics_parse import parse_thermo_series

    bp = Path(baseline_path)
    cp = Path(candidate_path)
    if not bp.exists() or not cp.exists():
        missing = []
        if not bp.exists():
            missing.append(f"baseline: {baseline_path}")
        if not cp.exists():
            missing.append(f"candidate: {candidate_path}")
        return DriftReport(
            status="WARN",
            drift_metrics={},
            summary=f"Log file(s) missing: {', '.join(missing)}",
            details={"missing": missing},
            thresholds_used=thresholds,
        )

    base_text = bp.read_text(encoding="utf-8", errors="replace")
    cand_text = cp.read_text(encoding="utf-8", errors="replace")
    base_series = parse_thermo_series(base_text, max_rows=0)
    cand_series = parse_thermo_series(cand_text, max_rows=0)

    if not base_series or not cand_series:
        return DriftReport(
            status="WARN",
            drift_metrics={},
            summary="Thermo series unavailable for drift comparison",
            details={},
            thresholds_used=thresholds,
        )

    metrics: Dict[str, object] = {}
    reasons: List[str] = []

    # Energy drift: compare TotEng relative drift
    energy_key = "TotEng"
    energy_limit = float(thresholds.get("energy_drift_rel_max", 1.0e-4))
    for key in (energy_key, "PotEng", "KinEng"):
        base_vals = base_series.get(key, [])
        cand_vals = cand_series.get(key, [])
        if len(base_vals) < 2 or len(cand_vals) < 2:
            continue
        base_drift = _rel_drift(base_vals)
        cand_drift = _rel_drift(cand_vals)
        delta = abs(cand_drift - base_drift)
        metrics[f"{key}_drift_delta"] = delta
        metrics[f"{key}_baseline_drift"] = base_drift
        metrics[f"{key}_candidate_drift"] = cand_drift
        if key == energy_key and delta > energy_limit:
            reasons.append(f"{key}_drift_delta={delta:.2e} > {energy_limit:.2e}")

    # Temperature stability: coefficient of variation
    temp_cv_max = float(thresholds.get("temperature_cv_max", 0.05))
    temp_vals = cand_series.get("Temp", [])
    if len(temp_vals) >= 2:
        t_mean = sum(temp_vals) / len(temp_vals)
        if t_mean > 0:
            t_var = sum((v - t_mean) ** 2 for v in temp_vals) / len(temp_vals)
            t_cv = math.sqrt(t_var) / t_mean
            metrics["temperature_cv"] = t_cv
            if t_cv > temp_cv_max:
                reasons.append(f"temperature_cv={t_cv:.4f} > {temp_cv_max}")

    # Force drift (if Press data available as proxy)
    force_limit = float(thresholds.get("force_drift_rel_max", 1.0e-3))
    base_press = base_series.get("Press", [])
    cand_press = cand_series.get("Press", [])
    if len(base_press) >= 2 and len(cand_press) >= 2:
        count = min(len(base_press), len(cand_press))
        diffs = [abs(cand_press[-count + i] - base_press[-count + i]) for i in range(count)]
        base_mag = max(max(abs(v) for v in base_press[-count:]), 1.0e-12)
        max_rel = max(diffs) / base_mag
        metrics["press_drift_rel"] = max_rel
        if max_rel > force_limit:
            reasons.append(f"press_drift_rel={max_rel:.2e} > {force_limit:.2e}")

    if reasons:
        status = "FAIL"
        summary = "LAMMPS drift: " + "; ".join(reasons)
    else:
        status = "PASS"
        summary = "LAMMPS output within drift thresholds"

    return DriftReport(
        status=status,
        drift_metrics=metrics,
        summary=summary,
        details={
            "baseline_keys": list(base_series.keys()),
            "candidate_keys": list(cand_series.keys()),
        },
        thresholds_used=thresholds,
    )


def _rel_drift(values: List[float]) -> float:
    """Relative drift: (last - first) / |first|."""
    if len(values) < 2:
        return 0.0
    denom = abs(values[0]) if abs(values[0]) > 1.0e-12 else 1.0
    return (values[-1] - values[0]) / denom


def ensure_log_path(run_args: List[str], run_dir: Path) -> List[str]:
    args = list(run_args)
    log_path = run_dir / "log.lammps"
    if "-log" in args:
        idx = args.index("-log")
        if idx + 1 < len(args):
            args[idx + 1] = str(log_path)
        else:
            args.append(str(log_path))
        return args
    return args + ["-log", str(log_path)]


def _merge_env_bucket(params: Dict[str, object], key: str) -> None:
    bucket = params.get(key)
    if not isinstance(bucket, dict):
        return
    env = dict(params.get("env", {}) or {})
    for k, v in bucket.items():
        if k and v is not None:
            env[str(k)] = str(v)
    if env:
        params["env"] = env


def _build_input_edit(params: Dict[str, object]) -> Dict[str, object] | None:
    if "neighbor_skin" in params:
        return {
            "directive": "neighbor",
            "mode": "replace_line",
            "match": "^neighbor\\s+.*$",
            "replace": f"neighbor {params['neighbor_skin']} bin",
        }
    if "neighbor_every" in params:
        return {
            "directive": "neigh_modify",
            "mode": "replace_line",
            "match": "^neigh_modify\\s+.*$",
            "replace": f"neigh_modify every {params['neighbor_every']} delay 0 check yes",
        }
    if "output_thermo_every" in params:
        return {
            "directive": "thermo",
            "mode": "replace_line",
            "match": "^thermo\\s+\\d+.*$",
            "replace": f"thermo {params['output_thermo_every']}",
        }
    if "output_dump_every" in params:
        return {
            "directive": "dump",
            "mode": "replace_line",
            "match": "^dump\\s+\\S+\\s+\\S+\\s+\\S+\\s+\\d+.*$",
            "replace": f"dump 1 all custom {params['output_dump_every']} dump.lammpstrj id type x y z",
        }
    if "kspace_accuracy" in params:
        return {
            "directive": "kspace_modify",
            "mode": "replace_line",
            "match": "^kspace_modify\\s+.*$",
            "replace": f"kspace_modify accuracy {params['kspace_accuracy']}",
        }
    if "kspace_style" in params:
        return {
            "directive": "kspace_style",
            "mode": "replace_line",
            "match": "^kspace_style\\s+.*$",
            "replace": f"kspace_style {params['kspace_style']}",
        }
    if "comm_cutoff" in params:
        return {
            "directive": "comm_modify",
            "mode": "replace_line",
            "match": "^comm_modify\\s+.*$",
            "replace": f"comm_modify cutoff {params['comm_cutoff']}",
        }
    if "comm_mode" in params:
        return {
            "directive": "comm_modify",
            "mode": "replace_line",
            "match": "^comm_modify\\s+.*$",
            "replace": f"comm_modify mode {params['comm_mode']}",
        }
    if "newton_setting" in params:
        return {
            "directive": "newton",
            "mode": "replace_line",
            "match": "^newton\\s+.*$",
            "replace": f"newton {params['newton_setting']}",
        }
    return None


def _merge_run_args(
    base: Dict[str, object] | None,
    extra: Dict[str, object] | None,
) -> Dict[str, object]:
    merged: Dict[str, object] = {}
    if isinstance(base, dict):
        merged.update(base)
    if not isinstance(extra, dict):
        return merged
    set_flags = list(merged.get("set_flags", []))
    seen = {item.get("flag") for item in set_flags if isinstance(item, dict)}
    for item in extra.get("set_flags", []):
        if not isinstance(item, dict):
            continue
        flag = item.get("flag")
        if flag in seen:
            continue
        set_flags.append(item)
        seen.add(flag)
    if set_flags:
        merged["set_flags"] = set_flags
    return merged
