from __future__ import annotations

from typing import Dict, List, Optional

from schemas.action_ir import ActionIR
from schemas.job_ir import JobIR

_DEFAULT_MPI_LAUNCHER = "mpirun"


def apply_adapter(action: ActionIR, job: JobIR, adapter_cfg: Dict[str, object] | None = None) -> ActionIR:
    if job.app != "lammps":
        return action

    global _DEFAULT_MPI_LAUNCHER
    _DEFAULT_MPI_LAUNCHER = (adapter_cfg or {}).get("mpi_launcher", "mpirun")

    params = dict(action.parameters or {})
    _merge_env_bucket(params, "mpi_env")
    _merge_env_bucket(params, "io_env")
    _merge_env_bucket(params, "runtime_env")
    _merge_env_bucket(params, "lib_env")
    backend = params.get("backend_enable")
    if backend == "openmp":
        params["run_args"] = _merge_run_args(
            params.get("run_args"),
            {
                "set_flags": [
                    {
                        "flag": "-sf",
                        "values": ["omp"],
                        "arg_count": 1,
                    }
                ]
            },
        )
        backend_threads = params.get("backend_threads")
        if backend_threads is not None:
            params["run_args"] = _merge_run_args(
                params.get("run_args"),
                {
                    "set_flags": [
                        {
                            "flag": "-pk",
                            "values": ["omp", str(backend_threads)],
                            "arg_count": 2,
                        }
                    ]
                },
            )
    elif backend == "opt":
        params["run_args"] = _merge_run_args(
            params.get("run_args"),
            {
                "set_flags": [
                    {
                        "flag": "-sf",
                        "values": ["opt"],
                        "arg_count": 1,
                    }
                ]
            },
        )
    elif backend == "mpi":
        # Pure MPI — no LAMMPS suffix needed
        pass
    elif backend == "mpi_omp":
        # MPI+OMP hybrid: needs -sf omp + thread count
        params["run_args"] = _merge_run_args(
            params.get("run_args"),
            {
                "set_flags": [
                    {
                        "flag": "-sf",
                        "values": ["omp"],
                        "arg_count": 1,
                    }
                ]
            },
        )
        backend_threads = params.get("backend_threads")
        if backend_threads is not None:
            params["run_args"] = _merge_run_args(
                params.get("run_args"),
                {
                    "set_flags": [
                        {
                            "flag": "-pk",
                            "values": ["omp", str(backend_threads)],
                            "arg_count": 2,
                        }
                    ]
                },
            )

    if "input_edit" not in params:
        input_edit = _build_input_edit(params)
        if input_edit:
            params["input_edit"] = input_edit

    # Auto-generate launcher config from action parameters
    launcher = _build_launcher_config(params)
    if launcher:
        params["launcher"] = launcher

    action.parameters = params
    return action


def _build_launcher_config(params: Dict[str, object]) -> Optional[Dict[str, object]]:
    """Auto-generate launcher config from action parameters.

    If an explicit ``launcher`` dict exists, resolve ``type: "auto"`` to the
    system default.  Otherwise, derive launcher config from the legacy
    ``mpi_env.MPI_RANKS`` hint.
    """
    launcher = params.get("launcher")
    if isinstance(launcher, dict):
        # Resolve "auto" type → system default
        if launcher.get("type") == "auto":
            launcher["type"] = _DEFAULT_MPI_LAUNCHER
        return launcher

    # Derive from mpi_env.MPI_RANKS if present
    mpi_env = params.get("mpi_env", {})
    if not isinstance(mpi_env, dict):
        return None
    ranks_str = mpi_env.get("MPI_RANKS")
    if ranks_str and int(ranks_str) > 1:
        # Remove MPI_RANKS from mpi_env (it's not a real env var)
        mpi_env_clean = {k: v for k, v in mpi_env.items() if k != "MPI_RANKS"}
        if mpi_env_clean:
            params["mpi_env"] = mpi_env_clean
        else:
            params.pop("mpi_env", None)
        return {
            "type": _DEFAULT_MPI_LAUNCHER,
            "np": int(ranks_str),
            "extra_args": [],
        }
    return None


def input_edit_allowlist() -> list[str]:
    return [
        "neighbor",
        "neigh_modify",
        "thermo",
        "dump",
        "kspace_style",
        "kspace_modify",
        "comm_modify",
        "newton",
    ]


def _merge_env_bucket(params: Dict[str, object], key: str) -> None:
    bucket = params.get(key)
    if not isinstance(bucket, dict):
        return
    env = dict(params.get("env", {}) or {})
    for k, v in bucket.items():
        if k and v is not None:
            env[str(k)] = str(v)
    if env:
        params["env"] = env


def _build_input_edit(params: Dict[str, object]) -> Dict[str, object] | None:
    if "neighbor_skin" in params:
        return {
            "directive": "neighbor",
            "mode": "replace_line",
            "match": "^neighbor\\s+.*$",
            "replace": f"neighbor {params['neighbor_skin']} bin",
        }
    if "neighbor_every" in params:
        return {
            "directive": "neigh_modify",
            "mode": "replace_line",
            "match": "^neigh_modify\\s+.*$",
            "replace": f"neigh_modify every {params['neighbor_every']} delay 0 check yes",
        }
    if "output_thermo_every" in params:
        return {
            "directive": "thermo",
            "mode": "replace_line",
            "match": "^thermo\\s+\\d+.*$",
            "replace": f"thermo {params['output_thermo_every']}",
        }
    if "output_dump_every" in params:
        return {
            "directive": "dump",
            "mode": "replace_line",
            "match": "^dump\\s+\\S+\\s+\\S+\\s+\\S+\\s+\\d+.*$",
            "replace": f"dump 1 all custom {params['output_dump_every']} dump.lammpstrj id type x y z",
        }
    if "kspace_accuracy" in params:
        return {
            "directive": "kspace_modify",
            "mode": "replace_line",
            "match": "^kspace_modify\\s+.*$",
            "replace": f"kspace_modify accuracy {params['kspace_accuracy']}",
        }
    if "kspace_style" in params:
        return {
            "directive": "kspace_style",
            "mode": "replace_line",
            "match": "^kspace_style\\s+.*$",
            "replace": f"kspace_style {params['kspace_style']}",
        }
    if "comm_cutoff" in params:
        return {
            "directive": "comm_modify",
            "mode": "replace_line",
            "match": "^comm_modify\\s+.*$",
            "replace": f"comm_modify cutoff {params['comm_cutoff']}",
        }
    if "comm_mode" in params:
        return {
            "directive": "comm_modify",
            "mode": "replace_line",
            "match": "^comm_modify\\s+.*$",
            "replace": f"comm_modify mode {params['comm_mode']}",
        }
    if "newton_setting" in params:
        return {
            "directive": "newton",
            "mode": "replace_line",
            "match": "^newton\\s+.*$",
            "replace": f"newton {params['newton_setting']}",
        }
    return None


def _merge_run_args(
    base: Dict[str, object] | None,
    extra: Dict[str, object] | None,
) -> Dict[str, object]:
    merged: Dict[str, object] = {}
    if isinstance(base, dict):
        merged.update(base)
    if not isinstance(extra, dict):
        return merged
    set_flags = list(merged.get("set_flags", []))
    seen = {item.get("flag") for item in set_flags if isinstance(item, dict)}
    for item in extra.get("set_flags", []):
        if not isinstance(item, dict):
            continue
        flag = item.get("flag")
        if flag in seen:
            continue
        set_flags.append(item)
        seen.add(flag)
    if set_flags:
        merged["set_flags"] = set_flags
    return merged
