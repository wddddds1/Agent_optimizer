from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Dict, List


def parse_lammps_timing(log_text: str) -> Dict[str, float]:
    breakdown: Dict[str, float] = {}
    loop_total = None
    loop_matches = re.findall(r"Loop time of\s+([0-9.]+)\s+on", log_text)
    if loop_matches:
        loop_total = float(loop_matches[-1])
        breakdown["total"] = loop_total
    wall_matches = re.findall(r"Total wall time:\s+([0-9]+):([0-9]+):([0-9]+)", log_text)
    if wall_matches:
        hours, minutes, seconds = wall_matches[-1]
        breakdown["wall_total"] = float(hours) * 3600.0 + float(minutes) * 60.0 + float(seconds)

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
            if section == "total" and loop_total is not None:
                breakdown["total_mpi"] = avg_time
            else:
                breakdown[section] = avg_time
    return breakdown


def parse_time_output(time_text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for line in time_text.splitlines():
        line = line.strip()
        if not line:
            continue
        multi = re.match(r"([0-9.]+)\s+real\s+([0-9.]+)\s+user\s+([0-9.]+)\s+sys", line)
        if multi:
            metrics["time_real_sec"] = float(multi.group(1))
            metrics["time_user_sec"] = float(multi.group(2))
            metrics["time_sys_sec"] = float(multi.group(3))
            continue
        match = re.match(r"([0-9.]+)\s+(.+)$", line)
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
        elif label.startswith("instructions retired"):
            metrics["instructions_retired"] = value
        elif label.startswith("cycles elapsed"):
            metrics["cycles_elapsed"] = value
        elif label.startswith("peak memory footprint"):
            metrics["peak_memory_bytes"] = value
        elif label.startswith("voluntary context switches"):
            metrics["voluntary_ctx_switches"] = value
        elif label.startswith("involuntary context switches"):
            metrics["involuntary_ctx_switches"] = value
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


def parse_thermo_series(log_text: str, max_rows: int = 0) -> Dict[str, List[float]]:
    lines = [ln.strip() for ln in log_text.splitlines() if ln.strip()]
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Step") and len(line.split()) > 2:
            header_idx = i
    if header_idx is None or header_idx + 1 >= len(lines):
        return {}
    header = lines[header_idx].split()
    rows: List[List[float]] = []
    for line in lines[header_idx + 1 :]:
        cols = line.split()
        if len(cols) != len(header):
            continue
        try:
            float(cols[0])
        except ValueError:
            continue
        try:
            rows.append([float(value) for value in cols])
        except ValueError:
            continue
    if not rows:
        return {}
    if max_rows and len(rows) > max_rows:
        rows = rows[-max_rows:]
    series: Dict[str, List[float]] = {}
    # Include Step column so callers can align by timestep
    for idx, key in enumerate(header):
        series[key] = [row[idx] for row in rows]
    return series


def parse_tau_profile(profile_dir: str, top_n: int = 30) -> List[Dict]:
    """Parse TAU profile files and return top hotspot functions.

    Reads ``profile.*.*.0`` files produced by TAU profiling.  Each file
    contains per-function timing data when function-level profiling or
    sampling is enabled.

    Returns a list of dicts sorted by exclusive time descending::

        {name, file, line, exclusive_us, inclusive_us, calls}

    Returns an empty list when no data is found (e.g. "0 aggregates").
    """
    from pathlib import Path
    import glob as _glob

    pdir = Path(profile_dir)
    if not pdir.is_dir():
        return []

    pattern = str(pdir / "profile.*.*.*")
    files = sorted(_glob.glob(pattern))
    if not files:
        return []

    merged: Dict[str, Dict] = {}

    for fpath in files:
        try:
            text = Path(fpath).read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        lines = text.splitlines()
        in_data = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Header line indicates start of function data
            if line.startswith("# Name") and "Calls" in line:
                in_data = True
                continue
            # Detect "0 templated_functions" → no data
            if re.match(r"^0\s+", line) and "aggregates" in line.lower():
                return []
            if not in_data:
                continue
            # Parse function data line:
            # "func_name [{file} {line}]" calls subrs excl incl profile_calls GROUP="..."
            m = re.match(
                r'"(.+?)"\s+'
                r'(\d+)\s+'      # calls
                r'(\d+)\s+'      # subrs
                r'([0-9.eE+\-]+)\s+'  # excl
                r'([0-9.eE+\-]+)\s+'  # incl
                r'(\d+)',        # profile_calls
                line,
            )
            if not m:
                continue
            raw_name = m.group(1)
            calls = int(m.group(2))
            excl = float(m.group(4))
            incl = float(m.group(5))

            # Extract file and line from name: "func [{file.c} {42}]"
            loc = re.search(r'\[\{(.+?)\}\s+\{(\d+)\}\]', raw_name)
            src_file = loc.group(1) if loc else ""
            src_line = int(loc.group(2)) if loc else 0
            func_name = re.sub(r'\s*\[\{.*', '', raw_name).strip()

            key = f"{func_name}:{src_file}:{src_line}"
            if key in merged:
                merged[key]["exclusive_us"] += excl
                merged[key]["inclusive_us"] += incl
                merged[key]["calls"] += calls
            else:
                merged[key] = {
                    "name": func_name,
                    "file": src_file,
                    "line": src_line,
                    "exclusive_us": excl,
                    "inclusive_us": incl,
                    "calls": calls,
                }

    results = sorted(merged.values(), key=lambda x: x["exclusive_us"], reverse=True)
    return results[:top_n]


def parse_xctrace_report(text: str, top_n: int = 30) -> List[Dict]:
    """Parse xctrace Time Profiler export text into hotspot entries.

    Best-effort parser: looks for lines starting with a percent value.
    """
    entries: List[Dict] = []
    if not text:
        return entries
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([0-9]+(?:\.[0-9]+)?)%\s+(.+)$", line)
        if not m:
            continue
        pct = float(m.group(1))
        rest = m.group(2).strip()
        name = rest
        file_path = ""
        line_no = None
        m2 = re.search(r"\(([^()]+\.(?:c|cc|cpp|cxx|m|mm|f|f90|F90)):(\d+)\)", rest)
        if m2:
            file_path = m2.group(1)
            try:
                line_no = int(m2.group(2))
            except ValueError:
                line_no = None
            name = rest.split(" (", 1)[0].strip()
        entries.append(
            {
                "name": name,
                "file": file_path,
                "line": line_no,
                "exclusive_us": pct,
                "inclusive_us": pct,
                "calls": 0,
                "source": "xctrace",
            }
        )
    entries.sort(key=lambda e: float(e.get("exclusive_us", 0.0)), reverse=True)
    if top_n and len(entries) > top_n:
        entries = entries[:top_n]
    return entries


def parse_xctrace_time_profile_xml(text: str, top_n: int = 30) -> List[Dict]:
    """Parse xctrace XML export from schema=time-profile into hotspot entries."""
    if not text or "<trace-query-result" not in text:
        return []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []
    if root.tag != "trace-query-result":
        return []

    id_map: Dict[str, ET.Element] = {}
    for elem in root.iter():
        elem_id = elem.attrib.get("id")
        if elem_id:
            id_map[elem_id] = elem

    def resolve(elem):
        if elem is None:
            return None
        ref = elem.attrib.get("ref")
        if ref:
            return id_map.get(ref, elem)
        return elem

    def frame_source(frame: ET.Element) -> tuple[str, int]:
        src = resolve(frame.find("source"))
        if src is None:
            return "", 0
        path_elem = resolve(src.find("path"))
        file_path = ""
        if path_elem is not None and path_elem.text:
            file_path = path_elem.text.strip()
        line_raw = src.attrib.get("line", "")
        try:
            line_no = int(line_raw)
        except (TypeError, ValueError):
            line_no = 0
        return file_path, line_no

    def frame_binary(frame: ET.Element) -> str:
        b = resolve(frame.find("binary"))
        if b is None:
            return ""
        return str(b.attrib.get("name", "") or "")

    def is_user_frame(name: str, binary: str, file_path: str) -> bool:
        if file_path and not file_path.startswith("/System/") and not file_path.startswith("/usr/lib/"):
            return True
        if not binary:
            return False
        if binary in {"dyld", "kernel.release.t6031"}:
            return False
        if binary.startswith("libsystem") or binary.startswith("libobjc") or binary.startswith("libc++"):
            return False
        return True

    merged: Dict[str, Dict] = {}
    for node in root.findall(".//node"):
        schema = node.find("schema")
        if schema is None or schema.attrib.get("name") != "time-profile":
            continue
        for row in node.findall("row"):
            weight = resolve(row.find("weight"))
            if weight is None or not (weight.text or "").strip():
                continue
            try:
                weight_ns = float((weight.text or "0").strip())
            except ValueError:
                continue
            weight_us = weight_ns / 1000.0
            bt = resolve(row.find("backtrace"))
            if bt is None:
                continue
            frames = bt.findall("frame")
            if not frames:
                continue

            chosen_name = ""
            chosen_file = ""
            chosen_line = 0

            for raw_frame in frames:
                frame = resolve(raw_frame)
                if frame is None:
                    continue
                name = str(frame.attrib.get("name", "") or "")
                binary = frame_binary(frame)
                file_path, line_no = frame_source(frame)
                if is_user_frame(name, binary, file_path):
                    chosen_name = name
                    chosen_file = file_path
                    chosen_line = line_no
                    break

            if not chosen_name:
                first = resolve(frames[0])
                if first is None:
                    continue
                chosen_name = str(first.attrib.get("name", "") or "")
                chosen_file, chosen_line = frame_source(first)
            if not chosen_name:
                continue

            key = f"{chosen_name}:{chosen_file}:{chosen_line}"
            if key in merged:
                merged[key]["exclusive_us"] += weight_us
                merged[key]["inclusive_us"] += weight_us
                merged[key]["calls"] += 1
            else:
                merged[key] = {
                    "name": chosen_name,
                    "file": chosen_file,
                    "line": chosen_line,
                    "exclusive_us": weight_us,
                    "inclusive_us": weight_us,
                    "calls": 1,
                    "source": "xctrace",
                }

    entries = sorted(merged.values(), key=lambda e: float(e.get("exclusive_us", 0.0)), reverse=True)
    if top_n and len(entries) > top_n:
        entries = entries[:top_n]
    return entries


def extract_error_lines(log_text: str, error_regex: str) -> List[str]:
    pattern = re.compile(error_regex)
    return [line for line in log_text.splitlines() if pattern.search(line)]
