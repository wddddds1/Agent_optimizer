"""LAMMPS-specific patch templates for algorithm-level optimisation patterns.

These provide concrete before/after code from the LAMMPS OPT backend
(pair_lj_cut_opt.cpp) as reference for the LLM.  The LLM adapts the
template to the target code rather than inventing optimisations from scratch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Algorithm-level templates (reference: pair_lj_cut_opt.cpp)
# ---------------------------------------------------------------------------

LAMMPS_PATCH_TEMPLATES: Dict[str, Dict[str, Dict[str, str]]] = {
    # ── param_table_pack ──────────────────────────────────────────────
    "param_table_pack": {
        "omp_dbl3": {
            "description": (
                "Pack LJ coefficient arrays (cutsq, lj1-lj4, offset) into "
                "a 64-byte cache-aligned struct so that all coefficients for "
                "one type pair are loaded in a single cache line.  "
                "Reference: pair_lj_cut_opt.cpp fast_alpha_t.  "
                "IMPORTANT: Add #include <cstdlib> and use std::malloc/free.  "
                "The std::free(tabsix) MUST be INSIDE the eval() function, "
                "after the outer for-loop but BEFORE the closing } of eval()."
            ),
            "before": """\
// OMP eval() hotspot anchor block (pair_lj_cut_omp.cpp):
for (int ii = iifrom; ii < iito; ++ii) {
  const int i = ilist[ii];
  const int itype = type[i];
  const int    * _noalias const jlist = firstneigh[i];
  const double * _noalias const cutsqi = cutsq[itype];
  const double * _noalias const offseti = offset[itype];
  const double * _noalias const lj1i = lj1[itype];
  const double * _noalias const lj2i = lj2[itype];
  const double * _noalias const lj3i = lj3[itype];
  const double * _noalias const lj4i = lj4[itype];
""",
            "after": """\
// STEP 1: Add #include <cstdlib> at top of file (for std::malloc/free)

// STEP 2: Inside eval(), BEFORE the outer for-loop (after variable decls),
//         insert struct definition + table allocation:
typedef struct {
    double cutsq, lj1, lj2, lj3, lj4, offset;
    double _pad[2];  // pad to 64 bytes
} fast_alpha_t;
int ntypes = atom->ntypes;
int ntypes2 = ntypes * ntypes;
auto *tabsix = (fast_alpha_t *)std::malloc(ntypes2 * sizeof(fast_alpha_t));
for (int it = 0; it < ntypes; it++)
    for (int jt = 0; jt < ntypes; jt++) {
        fast_alpha_t &a = tabsix[it * ntypes + jt];
        a.cutsq  = cutsq[it+1][jt+1];
        a.lj1    = lj1[it+1][jt+1];
        a.lj2    = lj2[it+1][jt+1];
        a.lj3    = lj3[it+1][jt+1];
        a.lj4    = lj4[it+1][jt+1];
        a.offset = offset[it+1][jt+1];
    }

// STEP 3: Inside outer loop, REPLACE the 6 row-pointer lines with:
const int itype0 = type[i] - 1;  // 0-based for flat table
auto *tabsixi = &tabsix[itype0 * ntypes];

// STEP 4: Inside inner loop, REPLACE coefficient access:
const int jtype0 = type[j] - 1;
fast_alpha_t &a = tabsixi[jtype0];
if (rsq < a.cutsq) {
    r2inv = 1.0/rsq;
    r6inv = r2inv*r2inv*r2inv;
    forcelj = r6inv * (a.lj1*r6inv - a.lj2);
    // energy: evdwl = r6inv*(a.lj3*r6inv - a.lj4) - a.offset;

// STEP 5: INSIDE eval(), AFTER the outer for-loop closes (after f[i].z += fztmp),
//         but BEFORE the function's closing }, insert:
std::free(tabsix);
// The function } comes AFTER this line.
""",
        },
        "serial": {
            "description": (
                "Pack LJ coefficient arrays into a flat cache-aligned struct. "
                "Same as omp_dbl3 but for serial pair_lj_cut.cpp."
            ),
            "before": """\
// Serial version: 2D array access with pointer indirection
if (rsq < cutsq[itype][jtype]) {
    forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
""",
            "after": """\
// Optimised: flat table lookup – same as omp_dbl3 template above
fast_alpha_t &a = tabsixi[jtype];
if (rsq < a.cutsq) {
    forcelj = r6inv * (a.lj1*r6inv - a.lj2);
""",
        },
    },

    # ── special_pair_split ────────────────────────────────────────────
    "special_pair_split": {
        "omp_dbl3": {
            "description": (
                "Split the inner loop body into a fast path for sbindex==0 "
                "(normal pairs, ~99% of all pairs) and a slow path for "
                "special bonds.  The fast path skips factor_lj multiplication "
                "and NEIGHMASK bit masking.  "
                "IMPORTANT: OpenMP uses dbl3_t; access x[j].x/y/z (NOT x[j][0]).  "
                "Keep j = jlist[jj] and sbmask(j) logic intact.  "
                "Reference: pair_lj_cut_opt.cpp lines 119-192.  "
                "Replace the original block starting at "
                "`factor_lj = special_lj[sbmask(j)];`."
            ),
            "before": """\
// OMP eval() structure (pair_lj_cut_omp.cpp):
// x uses dbl3_t struct: access as x[j].x, x[j].y, x[j].z (NOT x[j][0])
template <int EVFLAG, int EFLAG, int NEWTON_PAIR>
void PairLJCutOMP::eval(int iifrom, int iito, ThrData * const thr)
{
  const auto * _noalias const x = (dbl3_t *) atom->x[0];
  // ... variable declarations ...
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j].x;
      dely = ytmp - x[j].y;
      delz = ztmp - x[j].z;
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsqi[jtype]) {
        r2inv = 1.0/rsq;
        r6inv = r2inv*r2inv*r2inv;
        forcelj = r6inv * (lj1i[jtype]*r6inv - lj2i[jtype]);
        fpair = factor_lj*forcelj*r2inv;
        // ... force accumulation, energy, virial ...
      }
    }
// ... function closing } comes after inner loops ...
""",
            "after": """\
// STEP: Inside inner loop, split fast/slow path by sbindex.
// Anchor should include: `j = jlist[jj];`
// DO NOT use x[j][0]; must use x[j].x/.y/.z (dbl3_t).
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      int sbindex = sbmask(j);

      if (sbindex == 0) {
        // FAST PATH: no special bonds (~99% of pairs)
        // j index is clean – no NEIGHMASK needed
        delx = xtmp - x[j].x;
        dely = ytmp - x[j].y;
        delz = ztmp - x[j].z;
        rsq = delx*delx + dely*dely + delz*delz;
        jtype = type[j];

        if (rsq < cutsqi[jtype]) {
          r2inv = 1.0/rsq;
          r6inv = r2inv*r2inv*r2inv;
          forcelj = r6inv * (lj1i[jtype]*r6inv - lj2i[jtype]);
          fpair = forcelj*r2inv;    // no factor_lj multiply!
          // ... force accumulation ...
          if (EFLAG) {
            evdwl = r6inv*(lj3i[jtype]*r6inv-lj4i[jtype]) - offseti[jtype];
            // no factor_lj multiply on energy!
          }
          if (EVFLAG) ev_tally_thr(this,i,j,nlocal,NEWTON_PAIR,
                                   evdwl,0.0,fpair,delx,dely,delz,thr);
        }

      } else {
        // SLOW PATH: special bonds (rare)
        factor_lj = special_lj[sbindex];
        j &= NEIGHMASK;

        delx = xtmp - x[j].x;
        dely = ytmp - x[j].y;
        delz = ztmp - x[j].z;
        rsq = delx*delx + dely*dely + delz*delz;
        jtype = type[j];

        if (rsq < cutsqi[jtype]) {
          r2inv = 1.0/rsq;
          r6inv = r2inv*r2inv*r2inv;
          forcelj = r6inv * (lj1i[jtype]*r6inv - lj2i[jtype]);
          fpair = factor_lj*forcelj*r2inv;
          // ... force accumulation ...
          if (EFLAG) {
            evdwl = r6inv*(lj3i[jtype]*r6inv-lj4i[jtype]) - offseti[jtype];
            evdwl *= factor_lj;
          }
          if (EVFLAG) ev_tally_thr(this,i,j,nlocal,NEWTON_PAIR,
                                   evdwl,0.0,fpair,delx,dely,delz,thr);
        }
      }
    }
""",
        },
    },

    # ── flat_coeff_lookup ─────────────────────────────────────────────
    "flat_coeff_lookup": {
        "omp_dbl3": {
            "description": (
                "Flatten 2D coefficient arrays cutsq[i][j] into a 1D table "
                "cutsq_flat[i*ntypes+j] to eliminate one pointer indirection. "
                "Usually combined with param_table_pack."
            ),
            "before": """\
// 2D array with pointer indirection per row:
const double * cutsqi = cutsq[itype];   // load row pointer
if (rsq < cutsqi[jtype]) {              // then index into row
""",
            "after": """\
// 1D flat array – single index computation:
// (allocated before outer loop)
double *cutsq_flat = new double[ntypes2];
for (int it = 0; it < ntypes; it++)
    for (int jt = 0; jt < ntypes; jt++)
        cutsq_flat[it*ntypes+jt] = cutsq[it+1][jt+1];
// ...
if (rsq < cutsq_flat[(itype-1)*ntypes + (jtype-1)]) {
""",
        },
    },

    # ── neighbor_prefetch ─────────────────────────────────────────────
    "neighbor_prefetch": {
        "omp_dbl3": {
            "description": (
                "Insert software prefetch hints for the next neighbor's "
                "coordinates at the top of the inner loop."
            ),
            "before": """\
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;
""",
            "after": """\
    for (jj = 0; jj < jnum; jj++) {
      // prefetch next neighbor's position data
      if (jj + 4 < jnum) {
        int jpre = jlist[jj + 4] & NEIGHMASK;
        __builtin_prefetch(&x[jpre], 0, 1);
      }
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;
""",
        },
        "serial": {
            "description": "Software prefetch for serial pair_lj_cut.cpp.",
            "before": """\
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
""",
            "after": """\
    for (jj = 0; jj < jnum; jj++) {
      if (jj + 4 < jnum) {
        int jpre = jlist[jj + 4] & NEIGHMASK;
        __builtin_prefetch(&x[jpre], 0, 1);
      }
      j = jlist[jj];
""",
        },
    },

    # ── loop_fission (kept from original, refined) ────────────────────
    "loop_fission": {
        "omp_dbl3": {
            "description": (
                "Split EFLAG/VFLAG diagnostics from the force-only path.  "
                "Only valid when template parameter EFLAG or EVFLAG is 0.  "
                "NOTE: do NOT duplicate the entire loop body – only separate "
                "the ev_tally call."
            ),
            "before": """\
      if (rsq < cutsqi[jtype]) {
        // force computation
        if (EFLAG) { /* energy */ }
        if (EVFLAG) ev_tally_thr(...);
      }
""",
            "after": """\
      if (rsq < cutsqi[jtype]) {
        // force computation (always)
        if (EVFLAG) {
          if (EFLAG) { /* energy */ }
          ev_tally_thr(...);
        }
      }
""",
        },
    },
}

# ---------------------------------------------------------------------------
# Full OPT reference code for injection into CodePatch prompt context
# ---------------------------------------------------------------------------

_OPT_REFERENCE_PATH = "third_party/lammps/src/OPT/pair_lj_cut_opt.cpp"


def _read_opt_reference(repo_root: Path) -> str:
    """Read the full OPT reference implementation for prompt context."""
    opt_path = repo_root / _OPT_REFERENCE_PATH
    if opt_path.exists():
        return opt_path.read_text()
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_template(
    patch_family: str, backend: str = "serial"
) -> Optional[Dict[str, str]]:
    """Return the template dict for *patch_family* and *backend*, or ``None``."""
    family_templates = LAMMPS_PATCH_TEMPLATES.get(patch_family)
    if not family_templates:
        return None
    return family_templates.get(backend) or family_templates.get("serial")


def get_template_context(
    patch_family: str,
    backend: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> Optional[Dict[str, str]]:
    """Return template + full OPT reference for injection into CodePatch context.

    Returns a dict with keys: reference_file, description, before, after,
    full_reference (complete OPT source code).
    """
    bk = "omp_dbl3" if backend == "openmp_backend" else "serial"
    tmpl = get_template(patch_family, bk)
    if not tmpl:
        return None
    result: Dict[str, str] = {
        "reference_file": _OPT_REFERENCE_PATH,
        "description": tmpl.get("description", ""),
        "before": tmpl.get("before", ""),
        "after": tmpl.get("after", ""),
        "full_reference": "",
    }
    if repo_root:
        result["full_reference"] = _read_opt_reference(repo_root)
    return result


def format_templates_for_prompt(
    patch_families: List[str], backend: str = "serial"
) -> str:
    """Format relevant templates as a prompt section string."""
    sections: List[str] = []
    for family in patch_families:
        tmpl = get_template(family, backend)
        if not tmpl:
            continue
        sections.append(
            f"### {family}\n"
            f"{tmpl['description']}\n\n"
            f"**Before:**\n```cpp\n{tmpl['before']}```\n\n"
            f"**After:**\n```cpp\n{tmpl['after']}```"
        )
    return "\n\n".join(sections) if sections else ""
