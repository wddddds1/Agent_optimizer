from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import Field, model_validator

from schemas.strict_base import StrictBaseModel


class OpportunityStatus(str, Enum):
    OK = "OK"
    NEED_MORE_CONTEXT = "NEED_MORE_CONTEXT"
    NEED_MORE_PROFILE = "NEED_MORE_PROFILE"
    NO_ACTIONABLE = "NO_ACTIONABLE"


class OpportunityMechanism(str, Enum):
    DATA_LAYOUT = "data_layout"
    MEMORY_PATH = "memory_path"
    VECTORIZATION = "vectorization"
    ALGORITHMIC = "algorithmic"
    SYNC = "sync"
    IO = "io"
    ALLOCATION = "allocation"
    MICRO_OPT = "micro_opt"


class HotspotEvidence(StrictBaseModel):
    file: str
    function: str
    line_range: str
    share: float


class ExpectedGain(StrictBaseModel):
    p50: float
    p90: float

    @model_validator(mode="after")
    def _validate_order(self) -> "ExpectedGain":
        if self.p50 < 0:
            raise ValueError("expected_gain.p50 must be >= 0")
        if self.p90 < self.p50:
            raise ValueError("expected_gain.p90 must be >= p50")
        return self


class Composability(StrictBaseModel):
    score: float = 0.5
    depends_on: List[str] = Field(default_factory=list)
    conflicts_with: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_score(self) -> "Composability":
        if self.score < 0.0 or self.score > 1.0:
            raise ValueError("composability.score must be in [0, 1]")
        return self


class ValidationPlan(StrictBaseModel):
    benchmark: str = ""
    metrics: List[str] = Field(default_factory=list)
    acceptance: str = ""


class OpportunityNode(StrictBaseModel):
    opportunity_id: str
    title: str
    hotspot: HotspotEvidence
    mechanism: OpportunityMechanism
    evidence_ids: List[str] = Field(default_factory=list)
    hypothesis: str
    expected_gain: ExpectedGain
    success_prob: float
    implementation_cost: float
    composability: Composability = Field(default_factory=Composability)
    validation_plan: ValidationPlan = Field(default_factory=ValidationPlan)
    target_files: List[str] = Field(default_factory=list)
    target_functions: List[str] = Field(default_factory=list)
    family_hint: str = ""
    notes: str = ""
    invalid: bool = False
    invalid_reasons: List[str] = Field(default_factory=list)
    meta: Dict[str, object] = Field(default_factory=dict)
    score: float = 0.0
    value_density: float = 0.0

    @model_validator(mode="after")
    def _validate_core(self) -> "OpportunityNode":
        if self.success_prob < 0.0 or self.success_prob > 1.0:
            raise ValueError("success_prob must be in [0, 1]")
        if self.implementation_cost <= 0:
            raise ValueError("implementation_cost must be > 0")
        if self.hotspot.share < 0.0:
            raise ValueError("hotspot.share must be >= 0")
        return self


class OpportunityGraph(StrictBaseModel):
    graph_id: str
    opportunities: List[OpportunityNode] = Field(default_factory=list)
    evidence_catalog: Dict[str, Dict[str, object]] = Field(default_factory=dict)
    invalid_nodes: List[Dict[str, object]] = Field(default_factory=list)
    ranking_notes: List[str] = Field(default_factory=list)


class OpportunityGraphResult(StrictBaseModel):
    status: OpportunityStatus
    graph: Optional[OpportunityGraph] = None
    missing: List[str] = Field(default_factory=list)
    needs_profile: List[str] = Field(default_factory=list)
    rationale: str = ""
    suggestions: List[str] = Field(default_factory=list)


class SelectedOpportunity(StrictBaseModel):
    opportunity: OpportunityNode
    expected_speedup: float
    success_prob: float
    composability: float
    implementation_cost: float
    objective_score: float
    value_density: float


class SelectedOpportunities(StrictBaseModel):
    selected: List[SelectedOpportunity] = Field(default_factory=list)
    dropped: List[Dict[str, object]] = Field(default_factory=list)
    macro_rule_applied: bool = False
    ranking_rationale: str = ""


MACRO_MECHANISMS = {
    OpportunityMechanism.DATA_LAYOUT,
    OpportunityMechanism.MEMORY_PATH,
    OpportunityMechanism.VECTORIZATION,
    OpportunityMechanism.ALGORITHMIC,
}


def validate_graph(graph: OpportunityGraph) -> OpportunityGraph:
    validated: List[OpportunityNode] = []
    invalid_nodes: List[Dict[str, object]] = list(graph.invalid_nodes)
    for node in graph.opportunities:
        reasons: List[str] = []
        if not node.hotspot.file or not node.hotspot.function or not node.hotspot.line_range:
            reasons.append("missing_hotspot_fields")
        if node.hotspot.share <= 0:
            reasons.append("hotspot.share must be > 0")
        if not node.evidence_ids:
            reasons.append("missing_evidence_ids")
        if not node.hypothesis.strip():
            reasons.append("missing_hypothesis")
        if node.expected_gain.p50 <= 0:
            reasons.append("expected_gain.p50 must be > 0")
        if node.expected_gain.p90 < node.expected_gain.p50:
            reasons.append("expected_gain.p90 < p50")
        if node.success_prob < 0 or node.success_prob > 1:
            reasons.append("success_prob out of range")
        if node.implementation_cost <= 0:
            reasons.append("implementation_cost <= 0")
        if not node.validation_plan.benchmark.strip() or not node.validation_plan.metrics:
            reasons.append("validation_plan incomplete")
        if reasons:
            node.invalid = True
            node.invalid_reasons = reasons
            invalid_nodes.append(
                {
                    "opportunity_id": node.opportunity_id,
                    "reasons": reasons,
                }
            )
        validated.append(node)
    return graph.model_copy(update={"opportunities": validated, "invalid_nodes": invalid_nodes})
