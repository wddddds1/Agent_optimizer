from __future__ import annotations

import hashlib
from typing import Dict

from schemas.profile_report import ProfileReport


def build_hotspot_signature(profile: ProfileReport) -> str:
    timing = profile.timing_breakdown or {}
    parts = [f"{key}:{timing.get(key)}" for key in sorted(timing.keys())]
    raw = "|".join(parts) + "|" + "|".join(profile.notes or [])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
