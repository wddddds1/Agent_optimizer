from orchestrator.router import RuleContext, filter_actions
from schemas.action_ir import ActionIR
from schemas.job_ir import Budgets, JobIR


def _job():
    return JobIR(
        case_id="test",
        workdir=".",
        lammps_bin="/bin/true",
        input_script="in.lj",
        env={},
        run_args=["-in", "in.lj"],
        budgets=Budgets(max_iters=1, max_runs=1, max_wall_seconds=1),
        tags=[],
    )


def test_preconditions_and_policy():
    actions = [
        ActionIR(
            action_id="neigh_every_2",
            family="neigh_modify",
            description="",
            applies_to=["input_script"],
            parameters={},
            preconditions=["input_contains:^\\s*neigh_modify\\b"],
            constraints=[],
            expected_effect=["mem_locality"],
            risk_level="medium",
        ),
        ActionIR(
            action_id="omp_threads_2",
            family="omp_threads",
            description="",
            applies_to=["run_config"],
            parameters={},
            preconditions=["app_is:lammps"],
            constraints=[],
            expected_effect=["compute_opt"],
            risk_level="low",
        ),
    ]
    policy = {
        "conditional_rules": [
            {
                "id": "no_neigh_if_long_range",
                "when": "input_contains:kspace_style|pppm|ewald|coul/long",
                "forbid_action_families": ["neigh_modify"],
            }
        ]
    }
    input_text = "kspace_style pppm 1.0\nneigh_modify every 1 delay 0 check yes\n"
    ctx = RuleContext(job=_job(), input_text=input_text, run_args=["-in", "in.lj"], env={})
    filtered = filter_actions(actions, ctx, allowed_families=None, policy=policy)
    ids = [a.action_id for a in filtered]
    assert "neigh_every_2" not in ids
    assert "omp_threads_2" in ids
