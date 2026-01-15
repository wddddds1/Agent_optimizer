from __future__ import annotations

import re
from typing import Dict, List


def parse_lammps_timing(log_text: str) -> Dict[str, float]:
    breakdown: Dict[str, float] = {}
    loop_match = re.search(r"Loop time of\\s+([0-9.]+)\\s+on", log_text)
    if loop_match:
        breakdown["total"] = float(loop_match.group(1))

    lines = log_text.splitlines()
    in_table = False
    for line in lines:
        if "MPI task timing breakdown" in line:
            in_table = True
            continue
        if in_table:
            if not line.strip():
                break
            if line.strip().startswith("Section"):
                continue
            if "|" not in line:
                continue
            cols = [c.strip() for c in line.split("|")]
            if len(cols) < 3:
                continue
            section = cols[0].lower()
            try:
                avg_time = float(cols[2])
            except ValueError:
                continue
            breakdown[section] = avg_time
    return breakdown


def parse_time_output(time_text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for line in time_text.splitlines():
        line = line.strip()
        if not line:
            continue
        multi = re.match(r"([0-9.]+)\\s+real\\s+([0-9.]+)\\s+user\\s+([0-9.]+)\\s+sys", line)
        if multi:
            metrics["time_real_sec"] = float(multi.group(1))
            metrics["time_user_sec"] = float(multi.group(2))
            metrics["time_sys_sec"] = float(multi.group(3))
            continue
        match = re.match(r"([0-9.]+)\\s+(.+)$", line)
        if not match:
            continue
        value = float(match.group(1))
        label = match.group(2)
        if label.startswith("maximum resident set size"):
            metrics["rss_mb"] = value / (1024.0 * 1024.0)
        elif label.startswith("page faults"):
            metrics["page_faults"] = value
        elif label.startswith("block input operations"):
            metrics["block_in"] = value
        elif label.startswith("block output operations"):
            metrics["block_out"] = value
        elif label.startswith("real"):
            metrics["time_real_sec"] = value
        elif label.startswith("user"):
            metrics["time_user_sec"] = value
        elif label.startswith("sys"):
            metrics["time_sys_sec"] = value
    if "block_in" in metrics or "block_out" in metrics:
        metrics["io_blocks"] = metrics.get("block_in", 0.0) + metrics.get("block_out", 0.0)
        metrics["io_bytes"] = metrics["io_blocks"] * 512.0
    return metrics


def parse_thermo_table(log_text: str) -> Dict[str, float]:
    lines = [ln.strip() for ln in log_text.splitlines() if ln.strip()]
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Step") and len(line.split()) > 2:
            header_idx = i
    if header_idx is None or header_idx + 1 >= len(lines):
        return {}
    header = lines[header_idx].split()
    last_row = None
    for line in lines[header_idx + 1 :]:
        cols = line.split()
        if len(cols) != len(header):
            continue
        try:
            float(cols[0])
        except ValueError:
            continue
        last_row = cols
    if not last_row:
        return {}
    metrics: Dict[str, float] = {}
    for key, value in zip(header, last_row):
        if key == "Step":
            continue
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def extract_error_lines(log_text: str, error_regex: str) -> List[str]:
    pattern = re.compile(error_regex)
    return [line for line in log_text.splitlines() if pattern.search(line)]
