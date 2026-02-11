from __future__ import annotations

from typing import Dict, List

from schemas.profile_report import ProfileReport


def build_profile_features(profile: ProfileReport) -> Dict[str, object]:
    timing = profile.timing_breakdown or {}
    total = float(timing.get("total", 0.0) or 0.0)
    comm = float(timing.get("comm", 0.0) or 0.0)
    output = float(timing.get("output", 0.0) or 0.0)
    compute = 0.0
    for key in ("pair", "kspace", "neigh", "modify"):
        compute += float(timing.get(key, 0.0) or 0.0)
    neigh = float(timing.get("neigh", 0.0) or 0.0)

    comm_ratio = comm / total if total > 0.0 else 0.0
    io_ratio = output / total if total > 0.0 else 0.0
    compute_ratio = compute / total if total > 0.0 else 0.0
    neigh_ratio = neigh / total if total > 0.0 else 0.0

    # Hardware counters (from /usr/bin/time -l on macOS or perf stat on Linux)
    sys_metrics = profile.system_metrics or {}
    insns = float(sys_metrics.get("instructions_retired", 0) or 0)
    cycles = float(sys_metrics.get("cycles_elapsed", 0) or 0)
    ipc = insns / cycles if cycles > 0 else None

    time_user = float(sys_metrics.get("time_user_sec", 0) or 0)
    time_real = float(sys_metrics.get("time_real_sec", 0) or 0)
    user_sys_ratio = time_user / time_real if time_real > 0 else None

    # Cache/branch metrics (from perf stat, if available)
    cache_miss_rate = float(sys_metrics.get("cache_miss_rate", 0) or 0) or None
    branch_miss_rate = float(sys_metrics.get("branch_miss_rate", 0) or 0) or None

    tags: List[str] = []
    if io_ratio >= 0.2:
        tags.append("io_bound")
    if comm_ratio >= 0.2:
        tags.append("comm_bound")
    if compute_ratio >= 0.5 or not tags:
        tags.append("compute_bound")

    # Hardware-counter-driven tags
    if ipc is not None:
        if ipc < 1.0:
            tags.append("memory_bound")
        elif ipc > 2.5:
            tags.append("well_optimized")
    if cache_miss_rate is not None and cache_miss_rate > 0.05:
        tags.append("cache_thrashing")
    if branch_miss_rate is not None and branch_miss_rate > 0.03:
        tags.append("branch_heavy")

    features = {
        "metrics": {
            "io_ratio": io_ratio,
            "comm_ratio": comm_ratio,
            "compute_ratio": compute_ratio,
            "neigh_ratio": neigh_ratio,
            "imbalance_ratio": None,
            "ipc": ipc,
            "user_sys_ratio": user_sys_ratio,
            "cache_miss_rate": cache_miss_rate,
            "branch_miss_rate": branch_miss_rate,
        },
        "bottleneck_tags": tags,
        "structural_hazards": {
            "mpi_in_inner_loop": None,
        },
    }
    return features
