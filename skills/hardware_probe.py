from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import psutil


def get_system_topology() -> Dict[str, object]:
    physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    logical = psutil.cpu_count(logical=True) or os.cpu_count() or physical
    topology: Dict[str, object] = {
        "physical_cores": int(physical),
        "logical_cores": int(logical),
    }
    _merge_dict(topology, _probe_lscpu())
    core_groups = _probe_core_groups(topology)
    if core_groups:
        topology["core_groups"] = core_groups
    _merge_dict(topology, _probe_mpi_runtime())
    return topology


def _probe_lscpu() -> Dict[str, object]:
    if platform.system() == "Darwin":
        return {}
    lscpu_path = _which("lscpu")
    if not lscpu_path:
        return {}
    try:
        result = subprocess.run(
            [lscpu_path, "-J"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {}
    try:
        import json

        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    items = payload.get("lscpu")
    if not isinstance(items, list):
        return {}
    fields = {}
    for item in items:
        field = str(item.get("field", "")).strip().rstrip(":")
        data = str(item.get("data", "")).strip()
        if field:
            fields[field] = data
    sockets = _to_int(fields.get("Socket(s)"))
    cores_per_socket = _to_int(fields.get("Core(s) per socket"))
    threads_per_core = _to_int(fields.get("Thread(s) per core"))
    numa_nodes = _to_int(fields.get("NUMA node(s)"))
    model_name = fields.get("Model name")
    info: Dict[str, object] = {}
    if sockets:
        info["sockets"] = sockets
    if cores_per_socket:
        info["cores_per_socket"] = cores_per_socket
    if threads_per_core:
        info["threads_per_core"] = threads_per_core
    if numa_nodes:
        info["numa_nodes"] = numa_nodes
    if model_name:
        info["model_name"] = model_name
    return info


def _probe_core_groups(topology: Dict[str, object]) -> List[Dict[str, object]]:
    groups: List[Dict[str, object]] = []
    if platform.system() == "Darwin":
        perf = _sysctl_int("hw.perflevel0.physicalcpu")
        eff = _sysctl_int("hw.perflevel1.physicalcpu")
        if perf:
            groups.append({"label": "performance", "count": perf})
        if eff:
            groups.append({"label": "efficiency", "count": eff})
        return groups
    sysfs_groups = _probe_linux_core_type(topology)
    if sysfs_groups:
        groups.extend(sysfs_groups)
    return groups


def _probe_linux_core_type(topology: Dict[str, object]) -> List[Dict[str, object]]:
    base = Path("/sys/devices/system/cpu")
    if not base.exists():
        return []
    counts: Dict[str, int] = {}
    for entry in base.glob("cpu[0-9]*"):
        core_type = entry / "topology" / "core_type"
        if not core_type.exists():
            continue
        try:
            value = core_type.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            counts[value] = counts.get(value, 0) + 1
    if not counts:
        return []
    threads_per_core = int(topology.get("threads_per_core") or 1)
    groups: List[Dict[str, object]] = []
    for key, value in counts.items():
        count = value // threads_per_core if threads_per_core > 0 else value
        groups.append({"label": f"type_{key}", "count": count})
    return groups


def _sysctl_int(name: str) -> Optional[int]:
    try:
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", name],
            check=True,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
        if not value:
            return None
        return int(value)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def _which(name: str) -> Optional[str]:
    paths = os.environ.get("PATH", "").split(os.pathsep)
    for path in paths:
        candidate = Path(path) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _to_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _probe_mpi_runtime() -> Dict[str, object]:
    """Detect available MPI runtime (mpirun/mpiexec/srun).

    Returns the first available launcher as the default.
    """
    detected: List[Dict[str, str]] = []
    for launcher in ["mpirun", "mpiexec", "srun"]:
        path = _which(launcher)
        if path:
            version_text = ""
            try:
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                version_text = (result.stdout + result.stderr).strip().split("\n")[0]
            except Exception:
                pass
            detected.append({"name": launcher, "path": path, "version": version_text})
    if not detected:
        return {}
    default = detected[0]
    return {
        "mpi_launcher": default["name"],
        "mpi_launcher_path": default["path"],
        "mpi_version": default["version"],
        "mpi_launchers_available": detected,
    }


def check_binary_mpi_support(binary_path: str) -> bool:
    """Check whether a binary is linked against a real MPI library.

    Uses ``otool -L`` on macOS or ``ldd`` on Linux to inspect shared
    library dependencies.  Returns *True* if a real MPI library (libmpi,
    libpmpi, libmpich, etc.) is found, *False* otherwise.
    """
    binary = Path(binary_path)
    if not binary.exists():
        return False
    system = platform.system()
    try:
        if system == "Darwin":
            result = subprocess.run(
                ["otool", "-L", str(binary)],
                capture_output=True, text=True, timeout=5,
            )
        else:
            result = subprocess.run(
                ["ldd", str(binary)],
                capture_output=True, text=True, timeout=5,
            )
        output = result.stdout.lower()
        mpi_markers = ["libmpi", "libpmpi", "libmpich", "libmsmpi"]
        return any(marker in output for marker in mpi_markers)
    except Exception:
        return False


def _merge_dict(target: Dict[str, object], source: Dict[str, object]) -> None:
    for key, value in source.items():
        if value is None:
            continue
        target[key] = value
