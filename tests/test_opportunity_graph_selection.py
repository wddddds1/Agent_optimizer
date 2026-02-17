from orchestrator.agent_llm import AgentConfig
from orchestrator.agents.code_analysis_agent import DeepCodeAnalysisAgent
from schemas.opportunity_graph import (
    Composability,
    ExpectedGain,
    HotspotEvidence,
    OpportunityGraph,
    OpportunityMechanism,
    OpportunityNode,
    ValidationPlan,
)


def _node(opportunity_id: str, mechanism: OpportunityMechanism, p50: float, cost: float) -> OpportunityNode:
    return OpportunityNode(
        opportunity_id=opportunity_id,
        title=opportunity_id,
        hotspot=HotspotEvidence(
            file="x.c",
            function="hot",
            line_range="1-10",
            share=0.4,
        ),
        mechanism=mechanism,
        evidence_ids=[f"ev-{opportunity_id}"],
        hypothesis="test",
        expected_gain=ExpectedGain(p50=p50, p90=max(p50, p50 * 1.5)),
        success_prob=0.6,
        implementation_cost=cost,
        composability=Composability(score=0.8),
        validation_plan=ValidationPlan(
            benchmark="bench",
            metrics=["runtime_seconds"],
            acceptance="improve",
        ),
    )


def test_select_topk_from_graph_prefers_macro_when_available(tmp_path) -> None:
    agent = DeepCodeAnalysisAgent(
        config=AgentConfig(enabled=False),
        repo_root=tmp_path,
        build_dir=tmp_path,
        experience_db=None,
    )
    graph = OpportunityGraph(
        graph_id="g",
        opportunities=[
            _node("micro1", OpportunityMechanism.MICRO_OPT, p50=0.20, cost=2.0),
            _node("vec1", OpportunityMechanism.VECTORIZATION, p50=0.10, cost=2.0),
            _node("algo1", OpportunityMechanism.ALGORITHMIC, p50=0.08, cost=2.0),
        ],
        evidence_catalog={},
    )

    selected = agent.select_topk_from_graph(graph, k=2, budget={}, experience_hints=[])
    mechanisms = [item.opportunity.mechanism.value for item in selected.selected]

    assert mechanisms[0] in {"vectorization", "algorithmic", "data_layout", "memory_path"}
    assert mechanisms[1] in {"vectorization", "algorithmic", "data_layout", "memory_path"}
