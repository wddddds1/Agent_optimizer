from __future__ import annotations

from typing import Any, Dict, List

from schemas.profile_report import ProfileReport


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

    return {
        "timing_breakdown": profile.timing_breakdown,
        "system_metrics": profile.system_metrics,
        "notes": profile.notes,
        "tau_hotspots": hotspots,
        "function_hotspots": hotspots,
        "profile_health": {
            "has_timing_breakdown": bool(profile.timing_breakdown),
            "hotspot_count": len(hotspots),
        },
    }
