from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from schemas.action_ir import ActionIR
from schemas.job_ir import JobIR


def apply_adapter(
    action: ActionIR,
    job: JobIR,
    adapter_cfg: Optional[Dict[str, object]] = None,
) -> ActionIR:
    """BWA adapter — minimal.

    BWA thread count is handled via set_flags (-t N) in action_space.yaml,
    so no special injection is needed here (unlike LAMMPS's -sf omp).
    """
    return action


# ---------------------------------------------------------------------------
# Output capture
# ---------------------------------------------------------------------------

def ensure_output_capture(
    run_args: List[str],
    run_dir: Path,
) -> Tuple[List[str], List[str]]:
    """Rewrite ``-o /dev/null`` to ``-o {run_dir}/output.sam`` for drift detection."""
    args = list(run_args)
    capture_path = str(run_dir / "output.sam")
    for idx, token in enumerate(args):
        if token == "-o" and idx + 1 < len(args) and args[idx + 1] == "/dev/null":
            args[idx + 1] = capture_path
            return args, [capture_path]
    # No -o /dev/null found — output may already go to a file or stdout.
    # Add explicit -o so we always have a capture file.
    if "-o" not in args:
        args.extend(["-o", capture_path])
        return args, [capture_path]
    # -o points somewhere else; use that path as the capture path.
    for idx, token in enumerate(args):
        if token == "-o" and idx + 1 < len(args):
            return args, [args[idx + 1]]
    return args, []


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def compute_drift(
    baseline_path: str,
    candidate_path: str,
    thresholds: Dict[str, object],
) -> "DriftReport":
    """Compute BWA output drift by comparing SAM summary statistics."""
    from skills.verify import DriftReport, _sam_summary

    bp = Path(baseline_path)
    cp = Path(candidate_path)
    if not bp.exists() or not cp.exists():
        missing = []
        if not bp.exists():
            missing.append(f"baseline: {baseline_path}")
        if not cp.exists():
            missing.append(f"candidate: {candidate_path}")
        return DriftReport(
            status="FAIL",
            drift_metrics={},
            summary=f"SAM file(s) missing: {', '.join(missing)}",
            details={"missing": missing},
            thresholds_used=thresholds,
        )

    base_lines = bp.read_text(encoding="utf-8", errors="replace").splitlines()
    cand_lines = cp.read_text(encoding="utf-8", errors="replace").splitlines()
    base_sam = _sam_summary(base_lines)
    cand_sam = _sam_summary(cand_lines)

    if not base_sam or not cand_sam:
        return DriftReport(
            status="WARN",
            drift_metrics={},
            summary="SAM summary unavailable (too few alignments)",
            details={"baseline_sam": base_sam, "candidate_sam": cand_sam},
            thresholds_used=thresholds,
        )

    metrics: Dict[str, object] = {}
    reasons: List[str] = []
    warnings: List[str] = []

    # Mapped rate delta
    base_mapped_rate = base_sam["mapped"] / max(base_sam["total"], 1)
    cand_mapped_rate = cand_sam["mapped"] / max(cand_sam["total"], 1)
    mapped_rate_delta = abs(cand_mapped_rate - base_mapped_rate)
    metrics["mapped_rate_delta"] = mapped_rate_delta
    metrics["baseline_mapped_rate"] = base_mapped_rate
    metrics["candidate_mapped_rate"] = cand_mapped_rate
    threshold = float(thresholds.get("mapped_rate_delta_max", 0.001))
    if mapped_rate_delta > threshold:
        reasons.append(
            f"mapped_rate_delta={mapped_rate_delta:.6f} > {threshold}"
        )

    # Unmapped rate delta
    base_unmapped_rate = base_sam["unmapped"] / max(base_sam["total"], 1)
    cand_unmapped_rate = cand_sam["unmapped"] / max(cand_sam["total"], 1)
    unmapped_rate_delta = abs(cand_unmapped_rate - base_unmapped_rate)
    metrics["unmapped_rate_delta"] = unmapped_rate_delta
    threshold = float(thresholds.get("unmapped_rate_delta_max", 0.002))
    if unmapped_rate_delta > threshold:
        reasons.append(
            f"unmapped_rate_delta={unmapped_rate_delta:.6f} > {threshold}"
        )

    # Mean NM (edit distance) delta
    base_mean_nm = base_sam["nm_sum"] / max(base_sam["mapped"], 1)
    cand_mean_nm = cand_sam["nm_sum"] / max(cand_sam["mapped"], 1)
    mean_nm_delta = abs(cand_mean_nm - base_mean_nm)
    metrics["mean_nm_delta"] = mean_nm_delta
    threshold = float(thresholds.get("mean_nm_delta_max", 0.5))
    if mean_nm_delta > threshold:
        reasons.append(f"mean_nm_delta={mean_nm_delta:.4f} > {threshold}")

    # Mean MAPQ delta
    base_mean_mapq = base_sam["mapq_sum"] / max(base_sam["total"], 1)
    cand_mean_mapq = cand_sam["mapq_sum"] / max(cand_sam["total"], 1)
    mean_mapq_delta = abs(cand_mean_mapq - base_mean_mapq)
    metrics["mean_mapq_delta"] = mean_mapq_delta
    threshold = float(thresholds.get("mean_mapq_delta_max", 1.0))
    if mean_mapq_delta > threshold:
        reasons.append(
            f"mean_mapq_delta={mean_mapq_delta:.4f} > {threshold}"
        )

    # Hash match (exact output)
    hash_match = (
        base_sam["hash_xor"] == cand_sam["hash_xor"]
        and base_sam["hash_sum"] == cand_sam["hash_sum"]
    )
    metrics["hash_match"] = hash_match
    if not hash_match and not reasons:
        warnings.append("alignment hash differs but metrics within thresholds")

    if reasons:
        status = "FAIL"
        summary = "BWA drift: " + "; ".join(reasons)
    elif warnings:
        status = "WARN"
        summary = "BWA drift warning: " + "; ".join(warnings)
    else:
        status = "PASS"
        summary = "BWA output within drift thresholds"

    return DriftReport(
        status=status,
        drift_metrics=metrics,
        summary=summary,
        details={"baseline_sam": base_sam, "candidate_sam": cand_sam},
        thresholds_used=thresholds,
    )
