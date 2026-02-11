from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import exp
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from schemas.action_ir import ActionIR
from schemas.experience_ir import ExperienceRecord
from schemas.experiment_ir import ExperimentIR


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _detect_backend(run_args: Iterable[str]) -> Optional[str]:
    args = list(run_args or [])
    if "-sf" in args:
        try:
            idx = args.index("-sf")
            return args[idx + 1]
        except (ValueError, IndexError):
            return None
    return None


@dataclass
class ExperienceConfig:
    enabled: bool = True
    path: Optional[Path] = None
    min_gain_pct: float = 1.0
    min_gain_sigma_mult: float = 2.0
    max_cv: float = 0.03
    strong_gain_pct: float = 2.0
    decay_half_life_days: float = 14.0
    record_negative: bool = True
    app_match_boost: float = 1.2
    app_mismatch_penalty: float = 0.5
    case_match_boost: float = 1.2
    backend_mismatch_penalty: float = 0.7


@dataclass
class ExperienceMemory:
    config: ExperienceConfig
    records: List[ExperienceRecord] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: Dict[str, object], default_path: Path) -> "ExperienceMemory":
        conf = ExperienceConfig(
            enabled=bool(cfg.get("enabled", True)),
            min_gain_pct=_safe_float(cfg.get("min_gain_pct", 1.0)),
            min_gain_sigma_mult=_safe_float(cfg.get("min_gain_sigma_mult", 2.0)),
            max_cv=_safe_float(cfg.get("max_cv", 0.03)),
            strong_gain_pct=_safe_float(cfg.get("strong_gain_pct", 2.0)),
            decay_half_life_days=_safe_float(cfg.get("decay_half_life_days", 14.0)),
            record_negative=bool(cfg.get("record_negative", True)),
            app_match_boost=_safe_float(cfg.get("app_match_boost", 1.2)),
            app_mismatch_penalty=_safe_float(cfg.get("app_mismatch_penalty", 0.5)),
            case_match_boost=_safe_float(cfg.get("case_match_boost", 1.2)),
            backend_mismatch_penalty=_safe_float(cfg.get("backend_mismatch_penalty", 0.7)),
        )
        path_value = cfg.get("path")
        if isinstance(path_value, str) and path_value:
            conf.path = Path(path_value)
        else:
            conf.path = default_path
        memory = cls(conf)
        memory._load()
        return memory

    def _load(self) -> None:
        if not self.config.path:
            return
        path = self.config.path
        if not path.exists():
            return
        records: List[ExperienceRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = ExperienceRecord.model_validate_json(line)
            except Exception:
                continue
            records.append(record)
        self.records = records

    def _append(self, record: ExperienceRecord) -> None:
        if not self.config.path:
            return
        path = self.config.path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json())
            handle.write("\n")
        self.records.append(record)

    def record_experiment(self, exp: ExperimentIR, baseline_exp: ExperimentIR) -> None:
        if not self.config.enabled or exp.action is None:
            return
        baseline_runtime = _safe_float(baseline_exp.results.runtime_seconds)
        runtime = _safe_float(exp.results.runtime_seconds)
        if baseline_runtime <= 0 or runtime <= 0:
            return
        improvement_pct = (baseline_runtime - runtime) / baseline_runtime * 100.0
        speedup = baseline_runtime / runtime
        variance_cv = exp.results.derived_metrics.get("variance_cv")
        variance_cv = _safe_float(variance_cv, 0.0) if variance_cv is not None else None
        strength = "weak"
        weight = 0.0

        if exp.verdict == "PASS":
            sigma_pct = self._baseline_sigma_pct(baseline_exp)
            gain_ok = improvement_pct >= self.config.min_gain_pct
            sigma_ok = improvement_pct >= self.config.min_gain_sigma_mult * sigma_pct
            cv_ok = variance_cv is None or variance_cv <= self.config.max_cv
            if gain_ok and sigma_ok and cv_ok:
                strength = "strong" if improvement_pct >= self.config.strong_gain_pct else "weak"
                weight = 1.0 if strength == "strong" else 0.5
            else:
                return
        else:
            if not self.config.record_negative:
                return
            strength = "negative"
            weight = -0.5

        action = exp.action
        params = action.parameters or {}
        record = ExperienceRecord(
            action_id=action.action_id,
            family=action.family,
            outcome=exp.verdict,
            improvement_pct=improvement_pct,
            speedup_vs_baseline=speedup,
            variance_cv=variance_cv,
            case_id=exp.job.case_id,
            app=exp.job.app,
            backend=_detect_backend(exp.job.run_args),
            target_file=params.get("target_file"),
            patch_family=params.get("patch_family"),
            run_id=exp.run_id,
            timestamp=_iso_now(),
            strength=strength,
            weight=weight,
            evidence={
                "run_id": exp.run_id,
                "exp_path": str(exp.patch_path) if exp.patch_path else "",
            },
            # Persist deep analysis context when available
            origin=params.get("origin"),
            category=params.get("category"),
            diagnosis=params.get("diagnosis"),
            mechanism=params.get("mechanism"),
            compiler_gap=params.get("compiler_gap"),
            target_functions=params.get("target_functions"),
        )
        self._append(record)

    def score_actions(self, actions: Iterable[ActionIR], context: Dict[str, str]) -> Dict[str, float]:
        if not self.config.enabled or not self.records:
            return {action.action_id: 0.0 for action in actions}
        backend = context.get("backend")
        case_id = context.get("case_id")
        app = context.get("app")
        now = datetime.now(timezone.utc)
        decay_half = max(self.config.decay_half_life_days, 1e-6)
        scored: Dict[str, float] = {action.action_id: 0.0 for action in actions}
        for record in self.records:
            sim_base = 0.0
            for action in actions:
                sim = 0.0
                if action.action_id == record.action_id:
                    sim = 1.0
                elif action.family == record.family:
                    sim = 0.6
                elif record.patch_family and record.patch_family == (action.parameters or {}).get(
                    "patch_family"
                ):
                    sim = 0.5
                elif record.target_file and record.target_file == (action.parameters or {}).get(
                    "target_file"
                ):
                    sim = 0.5
                else:
                    record_family = record.patch_family
                    action_family = (action.parameters or {}).get("patch_family")
                    if record_family in ("loop_fusion", "loop_fission") and action_family in (
                        "loop_fusion",
                        "loop_fission",
                    ):
                        sim = 0.4
                if sim <= 0.0:
                    continue
                if backend and record.backend and backend != record.backend:
                    sim *= self.config.backend_mismatch_penalty
                if case_id and record.case_id and case_id == record.case_id:
                    sim *= self.config.case_match_boost
                if app and record.app:
                    if app == record.app:
                        sim *= self.config.app_match_boost
                    else:
                        sim *= self.config.app_mismatch_penalty
                sim_base = sim
                age_days = 0.0
                try:
                    ts = datetime.fromisoformat(record.timestamp)
                    age_days = abs((now - ts).total_seconds()) / 86400.0
                except Exception:
                    age_days = 0.0
                decay = exp(-age_days / decay_half) if decay_half > 0 else 1.0
                scored[action.action_id] += record.weight * sim_base * decay
        return scored

    def family_success_rates(
        self,
        app: str = "",
        backend: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Return per-patch-family success statistics.

        Returns ``{patch_family: {attempts, successes, avg_gain_pct,
        best_gain_pct, failure_rate}}``
        """
        stats: Dict[str, Dict[str, float]] = {}
        for rec in self.records:
            pf = rec.patch_family
            if not pf:
                continue
            if app and rec.app and rec.app != app:
                continue
            if backend and rec.backend and rec.backend != backend:
                continue
            if pf not in stats:
                stats[pf] = {
                    "attempts": 0,
                    "successes": 0,
                    "failures": 0,
                    "total_gain_pct": 0.0,
                    "best_gain_pct": 0.0,
                }
            s = stats[pf]
            s["attempts"] += 1
            if rec.outcome == "PASS" and rec.improvement_pct > 0:
                s["successes"] += 1
                s["total_gain_pct"] += rec.improvement_pct
                if rec.improvement_pct > s["best_gain_pct"]:
                    s["best_gain_pct"] = rec.improvement_pct
            else:
                s["failures"] += 1
        result: Dict[str, Dict[str, float]] = {}
        for pf, s in stats.items():
            att = s["attempts"]
            suc = s["successes"]
            result[pf] = {
                "attempts": att,
                "successes": suc,
                "avg_gain_pct": s["total_gain_pct"] / suc if suc else 0.0,
                "best_gain_pct": s["best_gain_pct"],
                "failure_rate": s["failures"] / att if att else 0.0,
            }
        return result

    def format_hints_for_prompt(
        self,
        app: str = "",
        backend: Optional[str] = None,
        top_k: int = 8,
    ) -> List[Dict[str, object]]:
        """Return formatted experience data for LLM prompt injection.

        Includes per-family statistics and, when available, analysis insights
        (diagnosis, mechanism, compiler_gap) from deep code analysis records.
        """
        rates = self.family_success_rates(app, backend)
        items = sorted(
            rates.items(),
            key=lambda kv: (-kv[1]["successes"], kv[1]["failure_rate"]),
        )
        hints: List[Dict[str, object]] = []
        for pf, s in items[:top_k]:
            hint: Dict[str, object] = {
                "patch_family": pf,
                "attempts": int(s["attempts"]),
                "successes": int(s["successes"]),
                "avg_gain_pct": round(s["avg_gain_pct"], 2),
                "best_gain_pct": round(s["best_gain_pct"], 2),
                "failure_rate": round(s["failure_rate"], 2),
            }
            # Attach the best analysis insight for this family (if any)
            best_insight = self._best_insight_for_family(pf, app, backend)
            if best_insight:
                hint["insight"] = best_insight
            hints.append(hint)
        return hints

    def _best_insight_for_family(
        self,
        patch_family: str,
        app: str = "",
        backend: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        """Find the most informative analysis insight for a patch family."""
        best = None
        best_improvement = -999.0
        for rec in self.records:
            if rec.patch_family != patch_family:
                continue
            if app and rec.app and rec.app != app:
                continue
            if backend and rec.backend and rec.backend != backend:
                continue
            diag = getattr(rec, "diagnosis", None)
            if not diag:
                continue
            if rec.improvement_pct > best_improvement:
                best_improvement = rec.improvement_pct
                insight: Dict[str, str] = {"diagnosis": diag}
                mech = getattr(rec, "mechanism", None)
                if mech:
                    insight["mechanism"] = mech
                gap = getattr(rec, "compiler_gap", None)
                if gap:
                    insight["compiler_gap"] = gap
                cat = getattr(rec, "category", None)
                if cat:
                    insight["category"] = cat
                insight["outcome"] = rec.outcome
                insight["improvement_pct"] = str(round(rec.improvement_pct, 2))
                best = insight
        return best

    def _baseline_sigma_pct(self, baseline_exp: ExperimentIR) -> float:
        samples = baseline_exp.results.samples or []
        if len(samples) < 2:
            return 0.0
        mean = sum(samples) / len(samples)
        if mean <= 0:
            return 0.0
        var = sum((x - mean) ** 2 for x in samples) / len(samples)
        sigma = var ** 0.5
        return (sigma / mean) * 100.0
