# C5X finite-block reproducibility package

**Date:** 2026-07-12  
**Status:** corrected asserted freeze complete  
**Scope:** exact C5X certificates, C5AX--C5AZ regressions, and optional rooted downstream correction chain

## Claim boundary

This package reproduces finite statements and numerical regressions for one `n=2`, `4D`, `SU(2)` block. It does not reproduce an exact one-shell theorem, exceptional-sector control, a continuum construction, confinement, clustering, or a mass gap. C5BJ is excluded.

## Runtime

Run from:

```text
C:\Users\gratc\Documents\Codex\2026-06-20\gpd-help
```

Requirements are Python and NumPy. The wrapper resolves Python in this order:

1. a valid `PYTHON` environment variable;
2. `python` on `PATH`;
3. the bundled Codex Python runtime.

The wrapper freezes `PYTHONHASHSEED=0`, disables bytecode writes, and fixes common BLAS/OpenMP thread variables at one. The manifest records the actual Python executable/version, NumPy version, platform, and these environment values.

Seeded probes that use numerically constructed orthonormal bases can select different physical stress vectors when a different LAPACK runtime rotates a degenerate subspace. The package therefore freezes the runtime and treats probe-specific magnitudes as numerical diagnostics, while exact path certificates, invariant matrix constants, and semantic pass/fail decisions carry the portable gates.

## One-command runs

Paper-core reproduction:

```powershell
powershell -ExecutionPolicy Bypass -File calculations/run_c5x_paper_reproducibility.ps1
```

Core plus the corrected C5BA--C5BI dependency chain:

```powershell
powershell -ExecutionPolicy Bypass -File calculations/run_c5x_paper_reproducibility.ps1 -IncludeDownstream
```

Both commands write to `outputs/c5x-reproducibility-rerun/`. The second is slower because it reruns every rooted chart/density diagnostic through C5BI. Neither command executes C5BJ or advances the route.

## Frozen run record

The full command with `-IncludeDownstream` passed on 2026-07-12.

| item | frozen value |
|---|---|
| completion time UTC | `2026-07-12T18:37:00.7680033Z` |
| Python | `3.12.13`, CPython |
| NumPy | `2.3.5` |
| platform | `Windows-11-10.0.26200-SP0` |
| transitive local Python dependencies | 30 |
| exact certificate pass | `true` |
| downstream chain included | `true` |
| previous shifted-core constants preserved | `false` |
| C5BJ executed | `false` |
| project advanced | `false` |

## Exact certificates

`calculations/c5x_exact_finite_claim_certificates.py` uses integer/rational operations and exhaustive path-word comparison.

| gate | expected result |
|---|---:|
| exact face-row rank over the rationals | 17 |
| exact relation count | 7 |
| original same/inverse-cyclic path counts | 12 / 12 |
| original inverse cyclic shift | 6 |
| valid exact uniform shifts | 1, 5 |
| exact BCH witness defect | row 8, vector `(0,0,-2)` in doubled `Q2` |
| rooted path count | 24 |
| rooted min/max/mean segments | 8 / 16 / 13 |
| rooted unique edges | 84 |
| rooted path hash | `1EB6E829D66DCDFA6E12E18CACA3C09A3EDFC3EA56E7EBE0D7ED1E05CA01616C` |

The script fails if any exact gate changes.

## Core numerical regressions

### C5AX

`calculations/c5ax_reflection_retained_density.py`

Expected decision: linear artifact `False`, nonlinear basepoint defect `True`, retained density reflection-compatible `False`. The finite-g witness is an implementation illustration; the exact obstruction is supplied separately above.

### C5AY

`calculations/c5ay_reflection_covariant_retained_coordinates.py`

Expected decision: shifted-core repair `True`, Wilson-trace replacement needed now `False`, full common-stem theorem `False`. The eight-edge shifted-core constants are:

| quantity | value |
|---|---:|
| operator norm | `0.496353904572` |
| covariance L2 squared | `0.050689424924` |
| covariance L2 | `0.225143120979` |

These values are comparison data, not rooted constants.

### C5AZ

`calculations/c5az_shifted_coordinate_package_regression.py`

Expected decision:

| gate | value |
|---|---|
| common-stem finite package | `True` |
| rooted density recomputed | `True` |
| previous shifted-core constants preserved | `False` |
| exact shell theorem proved | `False` |

Corrected rooted constants:

| quantity | value |
|---|---:|
| operator norm | `1.0189627218075927` |
| covariance L2 squared | `0.21480467679414722` |
| covariance L2 | `0.463470254486895` |

## Downstream assertions

With `-IncludeDownstream`, the wrapper reruns C5BA--C5BI and checks:

- C5BA's budget uses the rooted covariance constant;
- C5BD rebuilds analytic source matrices on the frozen rooted path hash;
- no hardcoded shifted-core density scalar remains in C5BE/C5BF;
- C5BH reports that the current-`J0` truncated chart margin does not close;
- C5BI reports no compatibility with the zero current-`J0` chart room;
- C5BI does not claim a uniform compact-remainder theorem.

The corrected chart summary is:

| quantity | value |
|---|---:|
| current `J0` | 13082 |
| rooted adapted Q2 constant | `4.2787249510134995` |
| conservative Q3 constant | `28.00452547806159` |
| current truncated margin | `1.3008674431985208` |
| target | `0.5` |
| first truncated closing start | 58443 |

Therefore the old C5BI “partial pass at current J0” is invalidated. The corrected current-schedule C5BI status is fail/redesign, while the exact finite paper claims remain intact.

Frozen C5BI probe diagnostics are a maximum compact remainder `0.00009189907333771852`, maximum all-probe `C4` proxy `0.02610466242939474`, and cubic half-coupling ratios `0.9970817711917752` and `0.9999664331384399`. These diagnostics do not pass the current schedule because the deterministic allowance is zero.

## Manifest and closure

`calculations/c5x_reproducibility_manifest.py` parses Python imports with `ast` and hashes every local file in the transitive import closure. It also hashes:

- the wrapper and manifest builder;
- the manuscript, literature audit, reproducibility document, and control protocol;
- all rerun reports selected by the command.

Manifest paths are project-relative. The authoritative outputs are:

- `outputs/c5x-reproducibility-rerun/manifest.json`;
- `outputs/c5x-reproducibility-rerun/sha256_manifest.txt`;
- `outputs/c5x-reproducibility-rerun/run_summary.json`.

The project is not currently a Git repository, so this hash manifest is the correction-freeze snapshot. Submission work should add ordinary version control and an archival release.

## Failure behavior

The wrapper checks every native exit code and aborts on a missing expected decision, wrong path hash, wrong rooted constants, or an incorrect C5BH/C5BI status. A successful process exit is not by itself treated as scientific verification; the semantic assertions must also pass.
