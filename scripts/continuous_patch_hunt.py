#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _latest_session(sessions_dir: Path) -> Optional[Path]:
    candidates = [p for p in sessions_dir.iterdir() if p.is_dir() and p.name[:2] == "20"]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)


def _collect_experiments(session_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    runs_dir = session_dir / "runs"
    if not runs_dir.exists():
        return rows
    for run_dir in runs_dir.iterdir():
        exp_path = run_dir / "experiment.json"
        if not exp_path.exists():
            continue
        try:
            data = json.loads(exp_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(data)
    return rows


def _runtime(exp: Dict[str, object]) -> Optional[float]:
    try:
        return float(((exp.get("results") or {}).get("runtime_seconds")))  # type: ignore[arg-type]
    except Exception:
        return None


def _verdict(exp: Dict[str, object]) -> str:
    return str(exp.get("verdict", "")).upper()


def _action_family(exp: Dict[str, object]) -> str:
    action = exp.get("action") or {}
    if isinstance(action, dict):
        return str(action.get("family", ""))
    return ""


def _action_id(exp: Dict[str, object]) -> str:
    action = exp.get("action") or {}
    if isinstance(action, dict):
        return str(action.get("action_id", ""))
    return ""


def _run_id(exp: Dict[str, object]) -> str:
    return str(exp.get("run_id", ""))


def _evaluate_session(
    session_dir: Path,
    min_patches: int,
    gain_target: float,
) -> Tuple[bool, Dict[str, object]]:
    exps = _collect_experiments(session_dir)
    baseline = next((e for e in exps if _run_id(e) == "baseline"), None)
    baseline_rt = _runtime(baseline) if baseline else None

    phase1_rts = []
    for exp in exps:
        rid = _run_id(exp)
        if not rid.startswith("phase1-"):
            continue
        if _verdict(exp) != "PASS":
            continue
        rt = _runtime(exp)
        if rt and rt > 0:
            phase1_rts.append(rt)
    phase1_best = min(phase1_rts) if phase1_rts else None

    source_pass = []
    for exp in exps:
        if _action_family(exp) != "source_patch":
            continue
        if _verdict(exp) != "PASS":
            continue
        rt = _runtime(exp)
        if rt and rt > 0:
            source_pass.append((rt, _action_id(exp), _run_id(exp)))

    effective = []
    if phase1_best is not None:
        effective = [item for item in source_pass if item[0] < phase1_best]

    gain_vs_phase1 = None
    best_source_rt = min((item[0] for item in source_pass), default=None)
    if phase1_best is not None and best_source_rt is not None:
        gain_vs_phase1 = (phase1_best - best_source_rt) / phase1_best

    success = bool(
        phase1_best is not None
        and gain_vs_phase1 is not None
        and len(effective) >= min_patches
        and gain_vs_phase1 >= gain_target
    )
    summary: Dict[str, object] = {
        "session": str(session_dir),
        "baseline_runtime": baseline_rt,
        "phase1_best_runtime": phase1_best,
        "source_patch_pass_count": len(source_pass),
        "source_patch_effective_count": len(effective),
        "best_source_runtime": best_source_rt,
        "gain_vs_phase1": gain_vs_phase1,
        "effective_source_runs": [
            {"run_id": rid, "action_id": aid, "runtime_s": rt}
            for rt, aid, rid in sorted(effective, key=lambda item: item[0])
        ],
    }
    return success, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuous source patch hunt loop.")
    parser.add_argument("--case", default="bwa_ecoli_4m")
    parser.add_argument("--max-iters", type=int, default=20)
    parser.add_argument("--max-runs", type=int, default=260)
    parser.add_argument("--ui", default="console")
    parser.add_argument("--sessions-dir", default="artifacts/sessions")
    parser.add_argument("--min-patches", type=int, default=5)
    parser.add_argument("--gain-target", type=float, default=0.30)
    parser.add_argument("--sleep-seconds", type=float, default=3.0)
    args = parser.parse_args()

    sessions_dir = Path(args.sessions_dir)
    sessions_dir.mkdir(parents=True, exist_ok=True)

    round_idx = 0
    while True:
        round_idx += 1
        cmd = [
            "python3",
            "-m",
            "orchestrator.main",
            "--case",
            args.case,
            "--max-iters",
            str(args.max_iters),
            "--max-runs",
            str(args.max_runs),
            "--ui",
            str(args.ui),
        ]
        print(f"[round {round_idx}] run: {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd)
        print(f"[round {round_idx}] exit={proc.returncode}", flush=True)
        time.sleep(args.sleep_seconds)
        latest = _latest_session(sessions_dir)
        if latest is None:
            print(f"[round {round_idx}] no session found", flush=True)
            continue
        ok, summary = _evaluate_session(latest, args.min_patches, args.gain_target)
        print(json.dumps({"round": round_idx, "ok": ok, "summary": summary}, ensure_ascii=False), flush=True)
        if ok:
            print(f"[round {round_idx}] target reached, stop.", flush=True)
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
