from __future__ import annotations

import os
import platform
import re
import socket
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _run(cmd: List[str]) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


def _parse_size_to_bytes(text: str) -> Optional[int]:
    if not text:
        return None
    value = text.strip().upper()
    match = re.match(r"^([0-9]+)([KMG])?B?$", value)
    if not match:
        return None
    num = int(match.group(1))
    unit = match.group(2)
    if unit == "K":
        return num * 1024
    if unit == "M":
        return num * 1024 * 1024
    if unit == "G":
        return num * 1024 * 1024 * 1024
    return num


def _read_first_line(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[0].strip()
    except Exception:
        return ""


def _linux_cpu_topology() -> Tuple[int, int, int]:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return 0, 0, 0
    text = cpuinfo.read_text(encoding="utf-8", errors="replace")
    physical_ids: List[str] = []
    core_ids: List[Tuple[str, str]] = []
    logical = 0
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        logical += 1
        pid = None
        cid = None
        for line in block.splitlines():
            if line.startswith("physical id"):
                pid = line.split(":", 1)[-1].strip()
            elif line.startswith("core id"):
                cid = line.split(":", 1)[-1].strip()
        if pid is not None:
            physical_ids.append(pid)
        if pid is not None and cid is not None:
            core_ids.append((pid, cid))
    sockets = len(set(physical_ids)) if physical_ids else 0
    physical = len(set(core_ids)) if core_ids else 0
    return physical, logical, sockets


def _linux_cache_info() -> Dict[str, Optional[int]]:
    cache_root = Path("/sys/devices/system/cpu/cpu0/cache")
    result: Dict[str, Optional[int]] = {
        "l1d_bytes": None,
        "l2_bytes": None,
        "l3_bytes": None,
        "line_size": None,
    }
    if not cache_root.exists():
        return result
    for index in cache_root.glob("index*"):
        level = _read_first_line(index / "level")
        ctype = _read_first_line(index / "type").lower()
        size = _parse_size_to_bytes(_read_first_line(index / "size"))
        if not level or size is None:
            continue
        if level == "1" and ctype == "data":
            result["l1d_bytes"] = size
        elif level == "2":
            result["l2_bytes"] = size
        elif level == "3":
            result["l3_bytes"] = size
    line_size = _read_first_line(cache_root / "index0" / "coherency_line_size")
    if line_size.isdigit():
        result["line_size"] = int(line_size)
    return result


def _linux_memory_info() -> Dict[str, Optional[int]]:
    meminfo = Path("/proc/meminfo")
    total_bytes = None
    if meminfo.exists():
        text = meminfo.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"^MemTotal:\s+(\d+)\s+kB", text, re.MULTILINE)
        if match:
            total_bytes = int(match.group(1)) * 1024
    page_size = None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
    except Exception:
        page_size = None
    return {"total_bytes": total_bytes, "page_size": page_size}


def _linux_numa_nodes() -> Optional[int]:
    node_root = Path("/sys/devices/system/node")
    if not node_root.exists():
        return None
    nodes = [p for p in node_root.glob("node*") if p.is_dir()]
    return len(nodes) if nodes else None


def _linux_network_interfaces() -> List[str]:
    net_root = Path("/sys/class/net")
    if not net_root.exists():
        return []
    names = sorted(p.name for p in net_root.iterdir() if p.is_dir())
    return names


def _mac_cache_info() -> Dict[str, Optional[int]]:
    def _sysctl_int(name: str) -> Optional[int]:
        value = _run(["sysctl", "-n", name])
        return int(value) if value.isdigit() else None

    return {
        "l1d_bytes": _sysctl_int("hw.l1dcachesize"),
        "l2_bytes": _sysctl_int("hw.l2cachesize"),
        "l3_bytes": _sysctl_int("hw.l3cachesize"),
        "line_size": _sysctl_int("hw.cachelinesize"),
    }


def _mac_perf_levels() -> List[Dict[str, Optional[object]]]:
    levels: List[Dict[str, Optional[object]]] = []
    for idx in range(4):
        name = _run(["sysctl", "-n", f"hw.perflevel{idx}.name"])
        physical = _run(["sysctl", "-n", f"hw.perflevel{idx}.physicalcpu"])
        logical = _run(["sysctl", "-n", f"hw.perflevel{idx}.logicalcpu"])
        if not name and not physical and not logical:
            continue
        entry: Dict[str, Optional[object]] = {
            "level": idx,
            "name": name or None,
            "physical_cores": int(physical) if physical.isdigit() else None,
            "logical_cores": int(logical) if logical.isdigit() else None,
        }
        levels.append(entry)
    return levels


def _mac_memory_info() -> Dict[str, Optional[int]]:
    total = _run(["sysctl", "-n", "hw.memsize"])
    page = _run(["sysctl", "-n", "hw.pagesize"])
    total_bytes = int(total) if total.isdigit() else None
    page_size = int(page) if page.isdigit() else None
    return {"total_bytes": total_bytes, "page_size": page_size}


def _mac_network_interfaces() -> List[str]:
    try:
        return [name for _, name in socket.if_nameindex()]
    except Exception:
        return []


def probe_system_info() -> Dict[str, object]:
    sysname = platform.system()
    os_info = {
        "system": sysname,
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    }

    cpu_info: Dict[str, object] = {
        "model": platform.processor() or "",
        "physical_cores": None,
        "logical_cores": os.cpu_count() or None,
        "sockets": None,
        "threads_per_core": None,
        "cache": {},
        "numa_nodes": None,
    }
    memory_info: Dict[str, Optional[int]] = {
        "total_bytes": None,
        "page_size": None,
    }
    net_info: Dict[str, object] = {"interfaces": []}

    if sysname == "Darwin":
        cpu_info["model"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        physical = _run(["sysctl", "-n", "hw.physicalcpu"])
        logical = _run(["sysctl", "-n", "hw.logicalcpu"])
        sockets = _run(["sysctl", "-n", "hw.packages"])
        cpu_info["physical_cores"] = int(physical) if physical.isdigit() else None
        cpu_info["logical_cores"] = int(logical) if logical.isdigit() else None
        cpu_info["sockets"] = int(sockets) if sockets.isdigit() else None
        if cpu_info["physical_cores"] and cpu_info["logical_cores"]:
            cpu_info["threads_per_core"] = (
                cpu_info["logical_cores"] // cpu_info["physical_cores"]
            )
        cpu_info["cache"] = _mac_cache_info()
        cpu_info["perf_levels"] = _mac_perf_levels()
        memory_info = _mac_memory_info()
        net_info["interfaces"] = _mac_network_interfaces()
    else:
        physical, logical, sockets = _linux_cpu_topology()
        if physical:
            cpu_info["physical_cores"] = physical
        if logical:
            cpu_info["logical_cores"] = logical
        if sockets:
            cpu_info["sockets"] = sockets
        if physical and logical:
            cpu_info["threads_per_core"] = logical // physical
        cpu_model = ""
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            text = cpuinfo.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"model name\s*:\s*(.+)", text)
            if match:
                cpu_model = match.group(1).strip()
        cpu_info["model"] = cpu_model or cpu_info["model"]
        cpu_info["cache"] = _linux_cache_info()
        cpu_info["numa_nodes"] = _linux_numa_nodes()
        memory_info = _linux_memory_info()
        net_info["interfaces"] = _linux_network_interfaces()

    return {
        "os": os_info,
        "cpu": cpu_info,
        "memory": memory_info,
        "network": net_info,
    }
