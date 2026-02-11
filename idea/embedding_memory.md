# Embedding-Based Experience Memory (Idea Proposal)

## Why This Matters
We already have a rule-based experience memory that boosts actions proven effective in similar scenarios. This is stable and auditable, but coarse. As we expand to different apps/cases, we want **finer similarity** without losing reproducibility. A hybrid approach (structured features + embeddings) gives us:

- Better generalization across cases without hand-crafted rules.
- Still auditable and deterministic for papers.
- A clear “learning” component suitable as a contribution.

## Key Insight
Use **structured features** as the hard filter and **embeddings** as a soft similarity bonus. Embeddings never override policy; they just bias ranking within allowed actions.

---

## Proposed Design

### 1. Experience Record Extension
Each verified action adds a record:

- `context_text`: normalized summary of case + hotspot + action
- `embedding`: vector of context_text (float list)
- `embedding_model_id`: name/version of embedding model
- `structured_keys`: case_id, app, backend, target_file, patch_family, family, etc.

Example context_text:
```
app=lammps; case=melt_xxlarge; backend=omp; hotspot=pair_lj_cut_omp.cpp;
ratio_pair=0.73; ratio_neigh=0.12; action=source_patch.loop_fusion;
expected=mem_locality,compute_opt
```

### 2. Similarity Scoring

Final score uses a mix:

```
score = w_struct * structured_similarity
      + w_embed  * cosine(embedding, context_embedding)
      + w_mem    * experience_weight
      + w_llm    * llm_score
      - w_risk   * risk_penalty
      + novelty
```

- Structured similarity is deterministic and auditable.
- Embedding similarity is soft; used only when embedding_model_id matches.
- We log all terms for reproducibility.

### 3. Deterministic Modes
- **Structured-only (default)**: no embedding, maximum reproducibility.
- **Hybrid**: use embedding if model is available and fixed.
- **Embedding-only** not recommended (too opaque).

### 4. Embedding Model Options
- Local model preferred for reproducibility (e.g., sentence-transformers).
- Store model ID + checksum in config.
- Cache embeddings in `artifacts/knowledge/embeddings.jsonl`.

### 5. Policy Safety
Embedding can **only affect ranking**, never bypass constraints:

- If policy forbids an action, embedding doesn’t matter.
- If evidence is missing, LLM score goes to 0.

---

## Minimal Implementation Plan

1. Extend `ExperienceRecord` with:
   - `context_text`
   - `embedding` (optional)
   - `embedding_model_id`
2. Add `skills/embedding_index.py`:
   - `encode(text) -> vector`
   - `cosine(vec1, vec2)`
3. Update ranker:
   - compute current context embedding once per iteration
   - compute embedding similarity for each action
   - include in scoring
4. Add config in `planner.yaml`:
   - `embedding.enabled: true/false`
   - `embedding.model_id`
   - `embedding.weight`

---

## Risks
- Non-determinism if model changes.
- Extra compute cost.
- Might bias toward irrelevant matches if context_text is too coarse.

Mitigation:
- Pin model version and store checksum.
- Keep embedding as a small weight only.
- Always log similarity breakdown in report.

---

## Value for Paper
This becomes a clear “learning-based prior” contribution:

- Structured priors = expert knowledge
- Embedding priors = learned similarity
- Demonstrates improved convergence vs. rule-only baseline

---

## Next Steps (optional)
- Implement hybrid mode and compare convergence vs. structured-only on xxlarge + at least 1 additional case.
- Add ablation table: structured-only vs hybrid vs embedding-only.

