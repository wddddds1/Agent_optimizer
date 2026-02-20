from __future__ import annotations

from typing import Any, Dict, List

from schemas.profile_report import BottleneckClassification, ProfileReport


def build_profile_payload(
    profile: ProfileReport,
    *,
    hotspot_limit: int = 30,
) -> Dict[str, Any]:
    """Build a stable, LLM-friendly profile payload."""
    hotspots_raw = profile.tau_hotspots or []
    hotspots: List[Dict[str, Any]] = []
    for entry in hotspots_raw[: max(0, int(hotspot_limit))]:
        if not isinstance(entry, dict):
            continue
        hotspots.append(
            {
                "name": str(entry.get("name", "")),
                "file": str(entry.get("file", "")),
                "line": entry.get("line"),
                "exclusive_us": entry.get("exclusive_us"),
                "inclusive_us": entry.get("inclusive_us"),
                "calls": entry.get("calls"),
                "source": str(entry.get("source", "tau")),
            }
        )

    portrait = profile.bottleneck_portrait or build_bottleneck_portrait(profile)
    return {
        "timing_breakdown": profile.timing_breakdown,
        "system_metrics": profile.system_metrics,
        "notes": profile.notes,
        "tau_hotspots": hotspots,
        "function_hotspots": hotspots,
        "bottleneck_portrait": portrait,
        "profile_health": {
            "has_timing_breakdown": bool(profile.timing_breakdown),
            "hotspot_count": len(hotspots),
        },
    }


def build_bottleneck_portrait(profile: ProfileReport) -> Dict[str, Any]:
    """Build per-round staged bottleneck portrait from available counters.

    Values are best-effort and may contain ``unknown`` when counters are absent.
    """
    metrics = profile.system_metrics or {}
    timing = profile.timing_breakdown or {}

    def _f(key: str, default: float = 0.0) -> float:
        value = metrics.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    real = _f("time_real_sec")
    user = _f("time_user_sec")
    sys = _f("time_sys_sec")
    cpu_parallelism = (user + sys) / real if real > 0 else 0.0
    cpu_busy_ratio = min(1.0, max(0.0, cpu_parallelism)) if real > 0 else 0.0
    cpu_idle_ratio = max(0.0, 1.0 - cpu_busy_ratio) if real > 0 else 0.0
    if real <= 0:
        cpu_state = "unknown"
    elif cpu_busy_ratio >= 0.9:
        cpu_state = "busy"
    elif cpu_busy_ratio >= 0.65:
        cpu_state = "mixed"
    else:
        cpu_state = "idle_heavy"

    cache_miss_rate = _f("cache_miss_rate")
    llc_miss_rate = _f("llc_miss_rate")
    memory_bw_gbps = _f("memory_bw_gbps") or _f("memory_bandwidth_gbps") or _f("dram_bw_gbps")
    if memory_bw_gbps > 0:
        mem_state = "high" if memory_bw_gbps >= 50.0 else ("medium" if memory_bw_gbps >= 20.0 else "low")
    elif cache_miss_rate > 0.06 or llc_miss_rate > 0.02:
        mem_state = "high"
    elif cache_miss_rate > 0.0 or llc_miss_rate > 0.0:
        mem_state = "medium"
    else:
        mem_state = "unknown"

    stall_front = _f("stall_frontend_pct")
    stall_back = _f("stall_backend_pct")
    stall_mem = _f("stall_memory_pct")
    stall_branch = _f("stall_branch_pct")
    stall_candidates = {
        "frontend": stall_front,
        "backend": stall_back,
        "memory": stall_mem,
        "branch": stall_branch,
    }
    stall_known = [key for key, value in stall_candidates.items() if value > 0]
    if stall_known:
        dominant_stall = max(stall_known, key=lambda key: stall_candidates[key])
    elif cache_miss_rate > 0.06 or llc_miss_rate > 0.02:
        dominant_stall = "memory"
    elif _f("branch_miss_rate") > 0.05:
        dominant_stall = "branch"
    else:
        dominant_stall = "unknown"

    imbalance_cv = _f("thread_imbalance_cv")
    imbalance_pct = _f("load_imbalance_pct")
    if imbalance_cv > 0:
        imbalance_state = "high" if imbalance_cv >= 0.2 else ("medium" if imbalance_cv >= 0.1 else "low")
    elif imbalance_pct > 0:
        imbalance_state = "high" if imbalance_pct >= 20.0 else ("medium" if imbalance_pct >= 10.0 else "low")
    else:
        imbalance_state = "unknown"

    total = float(timing.get("total", 0.0) or 0.0)
    output_ratio = (float(timing.get("output", 0.0) or 0.0) / total) if total > 0 else 0.0
    io_blocks = _f("io_blocks")
    io_bytes = _f("io_bytes")
    if output_ratio >= 0.2 or io_blocks > 0:
        io_state = "high" if output_ratio >= 0.2 else "medium"
    elif output_ratio > 0:
        io_state = "low"
    else:
        io_state = "unknown"

    focus: List[str] = []
    if mem_state in {"high", "medium"} and dominant_stall in {"memory", "backend"}:
        focus.append("reshape memory access path and data layout")
    if cpu_state in {"idle_heavy", "mixed"} and imbalance_state in {"high", "medium"}:
        focus.append("rebalance thread workloads and reduce sync skew")
    if io_state in {"high", "medium"}:
        focus.append("batch or overlap I/O with compute")
    if not focus:
        focus.append("prioritize highest-share hotspot structural path")

    return {
        "cpu": {
            "state": cpu_state,
            "busy_ratio_single_core_norm": round(cpu_busy_ratio, 4),
            "idle_ratio_single_core_norm": round(cpu_idle_ratio, 4),
            "parallelism_estimate": round(cpu_parallelism, 4) if cpu_parallelism > 0 else 0.0,
        },
        "memory_bw": {
            "state": mem_state,
            "gbps": round(memory_bw_gbps, 3) if memory_bw_gbps > 0 else None,
            "cache_miss_rate": round(cache_miss_rate, 4) if cache_miss_rate > 0 else None,
            "llc_miss_rate": round(llc_miss_rate, 4) if llc_miss_rate > 0 else None,
        },
        "stall": {
            "dominant": dominant_stall,
            "frontend_pct": round(stall_front, 4) if stall_front > 0 else None,
            "backend_pct": round(stall_back, 4) if stall_back > 0 else None,
            "memory_pct": round(stall_mem, 4) if stall_mem > 0 else None,
            "branch_pct": round(stall_branch, 4) if stall_branch > 0 else None,
        },
        "thread_balance": {
            "state": imbalance_state,
            "imbalance_cv": round(imbalance_cv, 4) if imbalance_cv > 0 else None,
            "load_imbalance_pct": round(imbalance_pct, 4) if imbalance_pct > 0 else None,
        },
        "io_wait": {
            "state": io_state,
            "timing_output_ratio": round(output_ratio, 4) if output_ratio > 0 else None,
            "io_blocks": round(io_blocks, 4) if io_blocks > 0 else None,
            "io_bytes": round(io_bytes, 2) if io_bytes > 0 else None,
        },
        "structural_focus": focus,
    }


def classify_bottleneck(profile: ProfileReport) -> BottleneckClassification:
    """Classify the application's bottleneck type from profiling data.

    Uses IPC, cache miss rates, branch miss rates, and timing breakdown
    ratios to determine whether the application is compute-bound,
    memory-bound, branch-bound, or mixed.
    """
    metrics = profile.system_metrics or {}
    timing = profile.timing_breakdown or {}
    portrait = profile.bottleneck_portrait or build_bottleneck_portrait(profile)

    # Extract key metrics — try platform-specific key names
    instructions = float(
        metrics.get("instructions_retired", 0)
        or metrics.get("instructions", 0)
        or 0
    )
    cycles = float(
        metrics.get("cycles_elapsed", 0)
        or metrics.get("cycles", 0)
        or 0
    )
    ipc = instructions / cycles if cycles > 0 else 0.0

    cache_misses = float(metrics.get("cache_misses", 0) or 0)
    cache_refs = float(metrics.get("cache_references", 0) or 0)
    cache_miss_rate = cache_misses / cache_refs if cache_refs > 0 else 0.0

    # Pre-computed cache miss rate (from perf stat)
    if cache_miss_rate == 0.0:
        cache_miss_rate = float(metrics.get("cache_miss_rate", 0) or 0)

    branch_misses = float(metrics.get("branch_misses", 0) or 0)
    branches = float(metrics.get("branches", 0) or 0)
    branch_miss_rate = branch_misses / branches if branches > 0 else 0.0

    # Pre-computed branch miss rate (from perf stat)
    if branch_miss_rate == 0.0:
        branch_miss_rate = float(metrics.get("branch_miss_rate", 0) or 0)

    # L1/LLC miss rates from xctrace or perf
    l1_miss_rate = float(metrics.get("l1_dcache_miss_rate", 0) or 0)
    llc_miss_rate = float(metrics.get("llc_miss_rate", 0) or 0)

    # Classify bottleneck
    bottleneck = "mixed"
    rationale_parts: List[str] = []

    is_memory = False
    is_compute = False
    is_branch = False

    # IPC-based classification
    if ipc > 0:
        if ipc < 0.5:
            is_memory = True
            rationale_parts.append(f"IPC={ipc:.2f} (<0.5) suggests memory-bound")
        elif ipc > 2.0:
            is_compute = True
            rationale_parts.append(f"IPC={ipc:.2f} (>2.0) suggests compute-bound")
        else:
            rationale_parts.append(f"IPC={ipc:.2f} (moderate)")

    # Cache miss rate
    mem_state = str(((portrait.get("memory_bw") or {}).get("state", "")) if isinstance(portrait, dict) else "")
    stall_dom = str(((portrait.get("stall") or {}).get("dominant", "")) if isinstance(portrait, dict) else "")
    if cache_miss_rate > 0.05:
        is_memory = True
        rationale_parts.append(f"cache miss rate={cache_miss_rate:.1%}")
    if mem_state == "high" or stall_dom == "memory":
        is_memory = True
    if l1_miss_rate > 0.1 or llc_miss_rate > 0.02:
        is_memory = True
        rationale_parts.append(f"L1 miss={l1_miss_rate:.1%}, LLC miss={llc_miss_rate:.1%}")

    # Branch miss rate
    if branch_miss_rate > 0.05:
        is_branch = True
        rationale_parts.append(f"branch miss rate={branch_miss_rate:.1%}")

    # Timing breakdown heuristics (for LAMMPS-like apps)
    total = timing.get("total", 0.0) or 0.0
    if total > 0:
        pair_ratio = timing.get("pair", 0.0) / total
        comm_ratio = timing.get("comm", 0.0) / total
        neigh_ratio = timing.get("neigh", 0.0) / total
        if pair_ratio > 0.6:
            rationale_parts.append(f"pair dominates ({pair_ratio:.0%})")
            if not is_memory and not is_compute:
                is_compute = True
        if neigh_ratio > 0.15:
            is_memory = True
            rationale_parts.append(f"neigh build significant ({neigh_ratio:.0%})")

    # Determine final classification
    if is_memory and is_branch:
        bottleneck = "mixed"
    elif is_memory:
        bottleneck = "memory"
    elif is_compute:
        bottleneck = "compute"
    elif is_branch:
        bottleneck = "branch"

    # Direction filtering
    effective: List[str] = []
    ineffective: List[str] = []

    if bottleneck == "memory":
        effective = ["data_layout", "memory_path", "cache_blocking", "algorithmic"]
        ineffective = ["vectorization", "loop_fusion", "strength_reduction"]
    elif bottleneck == "compute":
        effective = ["vectorization", "algorithmic", "loop_fusion", "loop_interchange"]
        ineffective = ["memory_path", "cache_blocking"]
    elif bottleneck == "branch":
        effective = ["branch_opt", "algorithmic", "data_layout"]
        ineffective = ["vectorization", "loop_fusion"]
    else:  # mixed
        effective = [
            "data_layout", "memory_path", "vectorization",
            "algorithmic", "cache_blocking",
        ]
        ineffective = []

    # Estimate arithmetic intensity
    ai_estimate = "unknown"
    if ipc > 0:
        if ipc < 0.5:
            ai_estimate = "low (memory-bound)"
        elif ipc < 1.5:
            ai_estimate = "moderate"
        else:
            ai_estimate = "high (compute-bound)"

    mem_bw = "unknown"
    if cache_miss_rate > 0:
        mem_bw = f"cache miss rate {cache_miss_rate:.1%}"

    return BottleneckClassification(
        bottleneck_type=bottleneck,
        arithmetic_intensity_estimate=ai_estimate,
        memory_bandwidth_utilization=mem_bw,
        ipc=round(ipc, 3),
        branch_miss_rate=round(branch_miss_rate, 4),
        effective_directions=effective,
        ineffective_directions=ineffective,
        rationale="; ".join(rationale_parts) if rationale_parts else "insufficient metrics",
    )
