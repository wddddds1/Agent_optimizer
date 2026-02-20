from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import Field

from schemas.strict_base import LLMStatus, StrictBaseModel


class DataStructureInsight(StrictBaseModel):
    """A discovered data structure and its performance implications."""

    type_name: str
    defined_in: str = ""
    access_pattern: str = ""  # "AoS", "SoA", "indirect_2D", etc.
    hotspot_usage: str = ""
    cache_behavior: str = ""
    optimization_relevance: str = ""


class CompilerInsight(StrictBaseModel):
    """What the compiler already does and what it fails to do."""

    file_path: str
    vectorized_loops: List[str] = Field(default_factory=list)
    missed_optimizations: List[str] = Field(default_factory=list)
    simd_width_used: Optional[str] = None
    aliasing_issues: List[str] = Field(default_factory=list)
    inlining_decisions: List[str] = Field(default_factory=list)


class CallChainNode(StrictBaseModel):
    """A node in a critical call chain."""

    function_name: str
    file_path: str
    line_number: int = 0
    time_share_pct: float = 0.0
    description: str = ""


class OptimizationOpportunity(StrictBaseModel):
    """A single optimization opportunity discovered through deep analysis.

    NOT constrained to predefined patch_families.  The agent may reference
    a known family_hint OR describe an entirely novel optimization.
    """

    opportunity_id: str
    title: str
    category: str  # data_layout / algorithmic / branch_elimination / memory_access / parallelism / compiler_assist / novel
    family_hint: Optional[str] = None  # maps to known patch_family if applicable

    target_files: List[str] = Field(default_factory=list)
    target_functions: List[str] = Field(default_factory=list)

    # Core analysis
    diagnosis: str = ""
    mechanism: str = ""
    compiler_gap: str = ""  # WHY the compiler cannot do this

    # Evidence from the exploration
    evidence: List[str] = Field(default_factory=list)
    code_context: str = ""
    reference_code: str = ""
    assembly_evidence: str = ""
    compiler_report_evidence: str = ""

    # Ranking signals
    estimated_impact: str = ""  # "high", "medium", "low"
    confidence: float = 0.5
    risk_level: str = "medium"
    expected_effect: List[str] = Field(default_factory=list)

    # Dependencies and ordering
    depends_on: List[str] = Field(default_factory=list)
    conflicts_with: List[str] = Field(default_factory=list)
    composable_with: List[str] = Field(default_factory=list)

    # Difficulty estimate
    implementation_complexity: str = "medium"  # trivial / simple / medium / complex
    lines_of_change_estimate: int = 0

    priority_rank: int = 0  # 1 = highest priority


class DeepCodeAnalysisResult(StrictBaseModel):
    """Complete output of the Deep Code Analysis agent."""

    status: LLMStatus = "OK"
    missing_fields: List[str] = Field(default_factory=list)

    # Architecture understanding
    architecture_summary: str = ""
    call_chain: List[CallChainNode] = Field(default_factory=list)
    data_structures: List[DataStructureInsight] = Field(default_factory=list)

    # Compiler baseline
    compiler_insights: List[CompilerInsight] = Field(default_factory=list)
    compiler_baseline_summary: str = ""

    # Performance diagnosis
    bottleneck_diagnosis: str = ""
    hotspot_files: List[str] = Field(default_factory=list)

    # The ranked opportunities
    opportunities: List[OptimizationOpportunity] = Field(default_factory=list)

    # Strategy
    recommended_sequence: List[str] = Field(default_factory=list)
    strategy_rationale: str = ""

    # Algorithm pre-analysis (populated before deep analysis)
    algorithm_preanalysis: Optional[dict] = None

    # Meta
    total_files_explored: int = 0
    total_functions_analyzed: int = 0
    exploration_notes: List[str] = Field(default_factory=list)
