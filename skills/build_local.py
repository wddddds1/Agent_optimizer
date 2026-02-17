from __future__ import annotations

import json
import platform
import time
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class BuildOutput:
    build_dir: str
    build_log_path: str
    cmake_cache_path: Optional[str]
    compile_commands_path: Optional[str]
    build_files: List[str]
    lammps_bin_path: Optional[str]
    provenance_path: Optional[str]
    provenance: Dict[str, object]
    build_seconds: Optional[float] = None

    @property
    def app_bin_path(self) -> Optional[str]:
        """Alias for lammps_bin_path (generic name)."""
        return self.lammps_bin_path


def build_job(
    build_cfg: Dict[str, object],
    source_root: Path,
    run_dir: Path,
) -> BuildOutput:
    build_system = str(build_cfg.get("build_system", "cmake"))
    if build_system == "make":
        return _build_make(build_cfg, source_root, run_dir)
    return _build_cmake(build_cfg, source_root, run_dir)


def _build_cmake(
    build_cfg: Dict[str, object],
    source_root: Path,
    run_dir: Path,
) -> BuildOutput:
    start_time = time.monotonic()
    build_dir = run_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    build_log_path = run_dir / "build.log"
    build_log_path.write_text("", encoding="utf-8")

    cmake_args = _normalize_args(build_cfg.get("cmake_args", []))
    if not _has_cmake_flag(cmake_args, "CMAKE_EXPORT_COMPILE_COMMANDS"):
        cmake_args.append("-D CMAKE_EXPORT_COMPILE_COMMANDS=ON")

    generator = str(build_cfg.get("generator") or "").strip()
    source_dir_raw = str(build_cfg.get("source_dir") or ".")
    source_dir = _resolve_path(source_dir_raw, source_root)
    cmake_cmd = ["cmake", "-S", str(source_dir), "-B", str(build_dir)]
    if generator:
        cmake_cmd.extend(["-G", generator])
    cmake_cmd.extend(cmake_args)

    env = os.environ.copy()
    env.update(_normalize_env(build_cfg.get("env", {})))

    _run_cmd(cmake_cmd, build_log_path, env=env)

    target = str(build_cfg.get("target") or "").strip()
    build_cmd = ["cmake", "--build", str(build_dir)]
    if target:
        build_cmd.extend(["--target", target])
    build_args = _normalize_args(build_cfg.get("build_args", []))
    if build_args:
        build_cmd.append("--")
        build_cmd.extend(build_args)
    _run_cmd(build_cmd, build_log_path, env=env)

    cmake_cache_path = _maybe_path(build_dir / "CMakeCache.txt")
    compile_commands_path = _maybe_path(build_dir / "compile_commands.json")
    build_files = [path for path in _maybe_paths([build_dir / "build.ninja", build_dir / "Makefile"])]
    lammps_bin = str(build_cfg.get("lammps_bin") or "lmp")
    lammps_bin_path = _maybe_path(build_dir / lammps_bin)

    provenance = _collect_build_provenance(
        cmake_cache_path, generator, cmake_args, build_args, env
    )
    provenance_path = run_dir / "build_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    build_seconds = time.monotonic() - start_time
    return BuildOutput(
        build_dir=str(build_dir),
        build_log_path=str(build_log_path),
        cmake_cache_path=cmake_cache_path,
        compile_commands_path=compile_commands_path,
        build_files=build_files,
        lammps_bin_path=lammps_bin_path,
        provenance_path=str(provenance_path),
        provenance=provenance,
        build_seconds=build_seconds,
    )


def _build_make(
    build_cfg: Dict[str, object],
    source_root: Path,
    run_dir: Path,
) -> BuildOutput:
    start_time = time.monotonic()
    build_dir = run_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    build_log_path = run_dir / "build.log"
    build_log_path.write_text("", encoding="utf-8")

    source_dir_raw = str(build_cfg.get("source_dir") or ".")
    source_dir = _resolve_path(source_dir_raw, source_root)

    env = os.environ.copy()
    env.update(_normalize_env(build_cfg.get("env", {})))

    # Clean (optional, on by default)
    make_clean = build_cfg.get("make_clean", True)
    if make_clean:
        _run_cmd(["make", "clean"], build_log_path, env=env, check=False, cwd=source_dir)

    # Build
    make_cmd = ["make"]
    build_args = _normalize_args(build_cfg.get("build_args", []))
    make_cmd.extend(build_args)
    target = str(build_cfg.get("target") or "").strip()
    if target:
        make_cmd.append(target)
    _run_cmd(make_cmd, build_log_path, env=env, cwd=source_dir)

    # Copy binary to build_dir for consistency with cmake path
    bin_name = str(build_cfg.get("app_bin") or build_cfg.get("lammps_bin") or "a.out")
    src_bin = source_dir / bin_name
    dst_bin = build_dir / bin_name
    if src_bin.exists():
        shutil.copy2(str(src_bin), str(dst_bin))

    bin_path = _maybe_path(dst_bin)

    provenance = {
        "build_system": "make",
        "source_dir": str(source_dir),
        "build_args": build_args,
    }
    provenance_path = run_dir / "build_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    return BuildOutput(
        build_dir=str(build_dir),
        build_log_path=str(build_log_path),
        cmake_cache_path=None,
        compile_commands_path=None,
        build_files=[],
        lammps_bin_path=bin_path,
        provenance_path=str(provenance_path),
        provenance=provenance,
        build_seconds=time.monotonic() - start_time,
    )


def collect_binary_provenance(lammps_bin: str, run_dir: Path) -> Dict[str, object]:
    binary_path = Path(lammps_bin)
    if not binary_path.exists():
        return {"lammps_bin": lammps_bin, "error": "binary missing"}

    lmp_help_path = run_dir / "lmp_help.txt"
    lmp_help_exit = _run_cmd(
        [str(binary_path), "-h"], lmp_help_path, env=os.environ.copy(), check=False
    )

    deps_path = run_dir / "lmp_deps.txt"
    if platform.system() == "Darwin":
        deps_cmd = ["otool", "-L", str(binary_path)]
    else:
        deps_cmd = ["ldd", str(binary_path)]
    deps_exit = _run_cmd(
        deps_cmd, deps_path, env=os.environ.copy(), check=False
    )

    return {
        "lammps_bin": str(binary_path),
        "lmp_help_path": str(lmp_help_path),
        "lmp_help_exit_code": lmp_help_exit,
        "otool_path": str(deps_path),
        "otool_exit_code": deps_exit,
    }


def _collect_build_provenance(
    cmake_cache_path: Optional[str],
    generator: str,
    cmake_args: List[str],
    build_args: List[str],
    env: Dict[str, str],
) -> Dict[str, object]:
    cmake_version = _capture_version(["cmake", "--version"], env)
    cache = _read_cmake_cache(cmake_cache_path) if cmake_cache_path else {}
    c_compiler = cache.get("CMAKE_C_COMPILER")
    cxx_compiler = cache.get("CMAKE_CXX_COMPILER")
    c_compiler_version = _capture_version([c_compiler, "--version"], env) if c_compiler else ""
    cxx_compiler_version = _capture_version([cxx_compiler, "--version"], env) if cxx_compiler else ""

    return {
        "cmake_version": cmake_version,
        "generator": generator,
        "cmake_args": cmake_args,
        "build_args": build_args,
        "c_compiler": c_compiler,
        "c_compiler_version": c_compiler_version,
        "cxx_compiler": cxx_compiler,
        "cxx_compiler_version": cxx_compiler_version,
    }


def _capture_version(cmd: List[str], env: Dict[str, str]) -> str:
    if not cmd or not cmd[0]:
        return ""
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        return (result.stdout or result.stderr or "").strip()
    except Exception:
        return ""


def _read_cmake_cache(path_str: Optional[str]) -> Dict[str, str]:
    if not path_str:
        return {}
    path = Path(path_str)
    if not path.exists():
        return {}
    cache: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("//") or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key_type, value = line.split("=", 1)
        if ":" in key_type:
            key, _ = key_type.split(":", 1)
        else:
            key = key_type
        cache[key.strip()] = value.strip()
    return cache


def _normalize_args(args: object) -> List[str]:
    if args is None:
        return []
    if isinstance(args, list):
        return [str(item) for item in args if str(item).strip()]
    if isinstance(args, str):
        return [item for item in shlex.split(args) if item.strip()]
    return [str(args)]


def _normalize_env(env: object) -> Dict[str, str]:
    if not isinstance(env, dict):
        return {}
    return {str(key): str(value) for key, value in env.items()}


def _resolve_path(path_str: str, root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _run_cmd(
    cmd: List[str],
    log_path: Path,
    env: Dict[str, str],
    check: bool = True,
    cwd: Path | str | None = None,
) -> int:
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd) if cwd else None,
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(cmd)}\n")
        if result.stdout:
            handle.write(result.stdout)
        if result.stderr:
            handle.write(result.stderr)
        handle.write("\n")
    if check and result.returncode != 0:
        raise RuntimeError(f"Build command failed: {' '.join(cmd)}")
    return result.returncode


def _maybe_path(path: Path) -> Optional[str]:
    return str(path) if path.exists() else None


def _maybe_paths(paths: List[Path]) -> List[str]:
    return [str(path) for path in paths if path.exists()]


def _has_cmake_flag(cmake_args: List[str], key: str) -> bool:
    key_upper = key.upper()
    for arg in cmake_args:
        if arg.upper().startswith(f"-D{key_upper}="):
            return True
    return False
