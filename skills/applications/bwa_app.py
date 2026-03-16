from __future__ import annotations

import hashlib
import json
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
    """Route BWA output to ``{run_dir}/output.sam`` for drift detection.

    Always retarget ``-o`` to the current run directory to avoid inheriting
    stale output paths from baseline/base runs.
    """
    args = list(run_args)
    capture_path = str(run_dir / "output.sam")
    for idx, token in enumerate(args):
        if token == "-o" and idx + 1 < len(args):
            args[idx + 1] = capture_path
            return args, [capture_path]
    # No explicit -o found — add one so we always have a capture file.
    if "-o" not in args:
        args.extend(["-o", capture_path])
        return args, [capture_path]
    return args, [capture_path]


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def _load_sam_stats(path: Path) -> Optional[Dict[str, object]]:
    """Load SAM alignment statistics from a pre-extracted summary JSON or a SAM file.

    Accepts either:
    - ``*.sam_summary.json``: a JSON file produced by ``_bwa_shrink_capture_to_summary``
      in graph.py, containing the already-computed stats dict.  This path is preferred
      and avoids loading hundreds of gigabytes into memory.
    - any other file: treated as a SAM file and stream-parsed line by line so the
      full file is never loaded into RAM at once.
    """
    if not path.exists():
        return None

    if path.name.endswith(".sam_summary.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return None
            # Return with original types; all values are ints.
            return {k: int(v) for k, v in raw.items() if isinstance(v, (int, float))}
        except Exception:
            return None

    # Stream-parse the SAM file without loading it entirely into RAM.
    total = mapped = unmapped = primary = secondary = supplementary = 0
    nm_sum = mapq_sum = hash_xor = hash_sum = 0
    mask64 = (1 << 64) - 1
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                line = raw_line.rstrip("\n")
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
    except Exception:
        return None

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


def compute_drift(
    baseline_path: str,
    candidate_path: str,
    thresholds: Dict[str, object],
) -> "DriftReport":
    """Compute BWA output drift by comparing SAM summary statistics."""
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
            status="FAIL",
            drift_metrics={},
            summary=f"SAM file(s) missing: {', '.join(missing)}",
            details={"missing": missing},
            thresholds_used=thresholds,
        )

    base_sam = _load_sam_stats(bp)
    cand_sam = _load_sam_stats(cp)

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
