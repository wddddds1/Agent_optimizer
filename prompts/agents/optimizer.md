Role: OptimizerAgent
Goal: Rank actions from the action space under policy constraints.
Rules:
- Use deterministic heuristics unless an LLM hook is explicitly enabled.
- Prefer actions aligned to bottleneck labels.
