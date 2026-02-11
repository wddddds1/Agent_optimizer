from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

from skills.hardware_probe import get_system_topology


_DEFAULT_CAPS: Set[str] = {
    "runtime_parallel_knobs",
    "runtime_affinity_knobs",
    "build_toolchain",
    "source_patch",
}


def collect_system_caps(repo_root: Path | None = None) -> Dict[str, object]:
    topo = get_system_topology() or {}
    caps = set(_DEFAULT_CAPS)
    if not topo.get("physical_cores"):
        caps.discard("runtime_parallel_knobs")
    if repo_root and not (repo_root / ".git").exists():
        caps.discard("source_patch")
    if topo.get("mpi_launcher"):
        caps.add("mpi_runtime")
    topo["capabilities"] = sorted(caps)
    return topo
