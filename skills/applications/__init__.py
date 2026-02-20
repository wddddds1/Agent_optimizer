from __future__ import annotations

import hashlib
import importlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from schemas.action_ir import ActionIR
    from schemas.job_ir import JobIR


def _app_module_name(app: str) -> str:
    # Convert arbitrary app names to python module-friendly suffixes.
    normalized = re.sub(r"[^a-zA-Z0-9_]", "_", str(app or "").strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return f"{normalized}_app" if normalized else ""


def _load_app_module(app: str):
    module_name = _app_module_name(app)
    if not module_name:
        return None
    try:
        return importlib.import_module(f"skills.applications.{module_name}")
    except ModuleNotFoundError:
        return None


def apply_adapter(
    action: "ActionIR",
    job: "JobIR",
    adapter_cfg: Optional[Dict[str, object]] = None,
) -> "ActionIR":
    module = _load_app_module(job.app)
    if module is not None and hasattr(module, "apply_adapter"):
        return module.apply_adapter(action, job, adapter_cfg)
    return action


def input_edit_allowlist(app: str) -> List[str]:
    module = _load_app_module(app)
    if module is not None and hasattr(module, "input_edit_allowlist"):
        result = module.input_edit_allowlist()
        if isinstance(result, list):
            return result
    return []


def parse_timing_breakdown(app: str, log_text: str) -> Dict[str, float]:
    module = _load_app_module(app)
    if module is not None and hasattr(module, "parse_timing_breakdown"):
        result = module.parse_timing_breakdown(log_text)
        if isinstance(result, dict):
            return result
    return {}


def requires_structured_correctness(app: str) -> bool:
    module = _load_app_module(app)
    if module is not None and hasattr(module, "requires_structured_correctness"):
        return bool(module.requires_structured_correctness())
    return False


def supports_agentic_correctness(app: str) -> bool:
    module = _load_app_module(app)
    if module is not None and hasattr(module, "supports_agentic_correctness"):
        return bool(module.supports_agentic_correctness())
    return True


def ensure_log_path(app: str, run_args: List[str], run_dir: Path) -> List[str]:
    module = _load_app_module(app)
    if module is not None and hasattr(module, "ensure_log_path"):
        result = module.ensure_log_path(run_args, run_dir)
        if isinstance(result, list):
            return [str(arg) for arg in result]
    return list(run_args)


def get_domain_knowledge(adapter_cfg: Optional[Dict] = None) -> Dict:
    """Extract domain knowledge from adapter config.

    Returns a dict with keys: application_type, kernel_semantics,
    common_pitfalls, effective_strategies.  Returns empty dict if
    no domain_knowledge section is present.
    """
    if not isinstance(adapter_cfg, dict):
        return {}
    dk = adapter_cfg.get("domain_knowledge")
    if not isinstance(dk, dict):
        return {}
    return {
        "application_type": dk.get("application_type", ""),
        "kernel_semantics": dk.get("kernel_semantics", []),
        "common_pitfalls": dk.get("common_pitfalls", []),
        "effective_strategies": dk.get("effective_strategies", []),
    }


def ensure_output_capture(
    app: str,
    run_args: List[str],
    run_dir: Path,
) -> Tuple[List[str], List[str]]:
    """Return (modified_run_args, capture_paths) with output routed to files.

    For apps that discard output (e.g. BWA with ``-o /dev/null``), rewrites
    run_args so output goes to a capture file instead.  For apps whose output
    is already captured (e.g. LAMMPS thermo in stdout.log), returns the
    original args unchanged and points to the existing capture path.

    Returns:
        (run_args, capture_paths) where *capture_paths* lists the files the
        drift checker should compare.
    """
    module = _load_app_module(app)
    if module is not None and hasattr(module, "ensure_output_capture"):
        return module.ensure_output_capture(run_args, run_dir)
    # Default fallback: capture stdout to a file.
    capture_path = str(run_dir / "captured_stdout.txt")
    return list(run_args), [capture_path]


def get_drift_checker(app: str) -> Callable:
    """Return the app-specific drift checker, or a generic hash-based fallback."""
    module = _load_app_module(app)
    if module is not None and hasattr(module, "compute_drift"):
        return module.compute_drift
    return _default_drift_checker


def _default_drift_checker(
    baseline_path: str,
    candidate_path: str,
    thresholds: Dict[str, object],
) -> "DriftReport":
    """Binary hash comparison fallback for unknown applications."""
    from skills.verify import DriftReport

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
            drift_metrics={"hash_match": False},
            summary=f"File(s) missing: {', '.join(missing)}",
            details={"missing": missing},
            thresholds_used=thresholds,
        )
    b_hash = _file_sha256(bp)
    c_hash = _file_sha256(cp)
    match = b_hash == c_hash
    warn_on_mismatch = bool(thresholds.get("warn_on_mismatch", True))
    fail_on_mismatch = bool(thresholds.get("fail_on_mismatch", False))
    if match:
        status = "PASS"
        summary = "Output hash matches baseline"
    elif fail_on_mismatch:
        status = "FAIL"
        summary = "Output hash differs from baseline"
    elif warn_on_mismatch:
        status = "WARN"
        summary = "Output hash differs from baseline (warning)"
    else:
        status = "PASS"
        summary = "Output hash differs from baseline (ignored)"
    return DriftReport(
        status=status,
        drift_metrics={"hash_match": match},
        summary=summary,
        details={"baseline_hash": b_hash, "candidate_hash": c_hash},
        thresholds_used=thresholds,
    )


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = [
    "apply_adapter",
    "ensure_log_path",
    "ensure_output_capture",
    "get_domain_knowledge",
    "get_drift_checker",
    "input_edit_allowlist",
    "parse_timing_breakdown",
    "requires_structured_correctness",
    "supports_agentic_correctness",
]
