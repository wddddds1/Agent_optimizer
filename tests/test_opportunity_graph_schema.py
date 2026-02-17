from schemas.opportunity_graph import (
    Composability,
    ExpectedGain,
    HotspotEvidence,
    OpportunityGraph,
    OpportunityMechanism,
    OpportunityNode,
    ValidationPlan,
    validate_graph,
)


def _valid_node(opportunity_id: str = "opp1") -> OpportunityNode:
    return OpportunityNode(
        opportunity_id=opportunity_id,
        title="vectorize hot loop",
        hotspot=HotspotEvidence(
            file="third_party/bwa/bwa.c",
            function="bwa_mem",
            line_range="100-140",
            share=0.42,
        ),
        mechanism=OpportunityMechanism.VECTORIZATION,
        evidence_ids=["ev1"],
        hypothesis="increase SIMD utilization in inner loop",
        expected_gain=ExpectedGain(p50=0.08, p90=0.16),
        success_prob=0.6,
        implementation_cost=3.0,
        composability=Composability(score=0.7, depends_on=[], conflicts_with=[]),
        validation_plan=ValidationPlan(
            benchmark="run bwa chr1",
            metrics=["runtime_seconds", "speedup_vs_baseline"],
            acceptance="speedup >= 3%",
        ),
    )


def test_validate_graph_marks_missing_required_fields_invalid() -> None:
    invalid = _valid_node("opp_invalid")
    invalid.evidence_ids = []
    invalid.hotspot.share = 0.0
    invalid.validation_plan.metrics = []
    graph = OpportunityGraph(graph_id="g1", opportunities=[invalid], evidence_catalog={})

    checked = validate_graph(graph)

    assert checked.opportunities[0].invalid is True
    assert checked.invalid_nodes
    reasons = checked.invalid_nodes[0]["reasons"]
    assert "missing_evidence_ids" in reasons
    assert "hotspot.share must be > 0" in reasons
    assert "validation_plan incomplete" in reasons


def test_validate_graph_keeps_valid_node() -> None:
    graph = OpportunityGraph(graph_id="g2", opportunities=[_valid_node()], evidence_catalog={})

    checked = validate_graph(graph)

    assert checked.opportunities[0].invalid is False
    assert checked.invalid_nodes == []
