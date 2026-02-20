from schemas.profile_report import ProfileReport
from skills.profile_payload import build_bottleneck_portrait, build_profile_payload


def test_bottleneck_portrait_populates_expected_stages() -> None:
    profile = ProfileReport(
        timing_breakdown={"total": 10.0, "output": 2.5},
        system_metrics={
            "time_real_sec": 10.0,
            "time_user_sec": 5.0,
            "time_sys_sec": 1.0,
            "cache_miss_rate": 0.08,
            "branch_miss_rate": 0.02,
            "thread_imbalance_cv": 0.22,
        },
        notes=[],
    )
    portrait = build_bottleneck_portrait(profile)
    assert portrait["cpu"]["state"] in {"busy", "mixed", "idle_heavy", "unknown"}
    assert portrait["memory_bw"]["state"] in {"high", "medium", "low", "unknown"}
    assert portrait["stall"]["dominant"] in {"frontend", "backend", "memory", "branch", "unknown"}
    assert portrait["thread_balance"]["state"] in {"high", "medium", "low", "unknown"}
    assert portrait["io_wait"]["state"] in {"high", "medium", "low", "unknown"}
    assert isinstance(portrait["structural_focus"], list)


def test_build_profile_payload_includes_bottleneck_portrait() -> None:
    profile = ProfileReport(
        timing_breakdown={},
        system_metrics={"time_real_sec": 1.0, "time_user_sec": 0.9, "time_sys_sec": 0.05},
        notes=[],
        tau_hotspots=[],
    )
    payload = build_profile_payload(profile)
    assert "bottleneck_portrait" in payload
    assert isinstance(payload["bottleneck_portrait"], dict)
