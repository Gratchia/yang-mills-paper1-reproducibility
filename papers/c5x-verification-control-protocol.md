# C5X verification and claim-control protocol

**Effective date:** 2026-07-12  
**Applies to:** C5X/C5AX paper track and every downstream use of its retained-coordinate package  
**Current mode:** correction freeze; C5BJ and forward theorem work are paused

## 1. Claim classes

Every result must carry one of these labels.

| label | meaning | admissible evidence |
|---|---|---|
| `EXACT-FINITE` | theorem about the fixed finite block | symbolic proof, integer/rational certificate, or exhaustive path-word identity |
| `NUMERICAL-REGRESSION` | implementation or floating linear-algebra check | frozen code, seed, tolerances, environment, and expected-output assertion |
| `CONDITIONAL-CONTRACT` | conclusion under stated hypotheses not yet proved | explicit hypotheses and failure trigger |
| `OPEN` | theorem debt or unresolved obstruction | no promotion language |
| `INVALIDATED` | an earlier statement used the wrong object or failed a corrected gate | correction record and affected-dependency list |

Numerical roundoff is never described as an exact proof. A finite exact theorem may use a numerical table only as an implementation check after the proof is stated.

## 2. Semantic object identity

The following path packages are distinct objects and must have distinct names in code, prose, reports, and tables.

| object | definition | frozen inventory |
|---|---|---|
| original paths | the 24 unshifted eight-segment coarse face boundaries | 24 paths, 8 segments each |
| shifted-core paths | `cyc_1` applied to each original path | 24 paths, 8 segments each, 64 touched edges |
| rooted paths | stem + shifted core + reverse stem, rooted at `(1,0,0,0)` | 24 paths, min/max 8/16, mean 13, 84 touched edges |

The rooted package has path-word SHA256

`1EB6E829D66DCDFA6E12E18CACA3C09A3EDFC3EA56E7EBE0D7ED1E05CA01616C`.

Any calculation described as gauge-covariant in one common color frame must consume `rooted_paths`. A shifted-core result may be cited only as a shifted-core comparison. A path hash or the complete inventory must be asserted at every handoff that supplies rooted matrices to another script.

## 3. Normalization and chart gates

The manuscript and code use

\[
\bar R_{\rm face}=n^{-2}R_{\rm face},
\qquad
\bar X_i(A;g)=(gn^2)^{-1}\log U_{\gamma_i}
\]

for `g != 0`, with the continuous tangent value at `g=0`. The quotient range is unchanged by the scalar factor, but formulas and reported constants must state whether they use raw or normalized rows.

The Lie basis is `T_a=-i sigma_a/2`, so `[T_a,T_b]=epsilon_abc T_c`. The logarithm chart is the `SU(2)` chart with positive quaternion scalar part. Code must reject a holonomy with scalar part at or below zero; it must not silently replace `q` by `-q`, because those are different `SU(2)` elements.

## 4. Exact-claim gates

Before the manuscript may state the finite propositions as theorems, `calculations/c5x_exact_finite_claim_certificates.py` must pass all gates:

1. exact rational rank of the 24 raw face rows is 17;
2. the exact relation count is 7;
3. original reflection gives 12 same-order and 12 inverse-cyclic paths, with inverse cyclic shift 6;
4. the exact valid uniform shifts are 1 and 5;
5. the two-link BCH witness has nonzero row-space projection;
6. the rooted path inventory and hash match the frozen package.

Proof text must explain why each certificate implies the stated proposition. A checker output alone is insufficient.

## 5. Numerical and downstream gates

The paper reproduction wrapper must fail unless:

- C5AX reports the expected linear-pass/nonlinear-fail split;
- C5AY reports the shifted-core reflection repair;
- C5AZ reports a rooted finite-package pass, a rooted density recomputation, and failure of old-constant preservation;
- the rooted operator and covariance constants match the recomputed package within tolerance.

With `-IncludeDownstream`, it must also assert that C5BA--C5BI consume the rooted package, C5BD repeats the path hash, C5BH does not close the current-`J0` truncated margin, and C5BI does not report compatibility with the zero current-`J0` room. C5BJ must not run in correction mode.

## 6. Reproducibility freeze

Each freeze records:

- Python executable and version, NumPy version, operating-system description, and thread/hash environment;
- all entry scripts and their transitive local Python import closure;
- wrapper, manuscript, audit, and control-protocol hashes;
- rerun-output hashes using project-relative paths;
- explicit native exit-code checks and expected-decision assertions.

The workspace is not currently a Git repository. The SHA256 manifest is therefore the source snapshot for this freeze; it is not a substitute for version control at submission stage.

## 7. Literature-audit gate

Every cited source receives an access label: `FULL TEXT`, `ABSTRACT/METADATA`, or `NOT CHECKED`. A full-text row records the sections or pages actually inspected and the precise claim it supports. Abstract-only access cannot support a detailed theorem comparison.

Novelty language must account for standard maximal-tree/common-origin loop constructions and path-gauge connectors. The paper may not claim that stems, common roots, based holonomies, Bianchi connectors, or cyclic trace invariance are new. Its defensible claim is the specific reflection-compatible shift/stem design and exact obstruction certificate for this fixed retained block.

## 8. Promotion and review

A draft can move from `research draft` to `external-feedback draft` only after:

1. exact certificates, core rerun, and manifest pass;
2. all old rooted/shifted-core conflations are removed from the manuscript and state files;
3. the primary-source audit is honest about access level and prior art;
4. theorem statements, numerical tables, and limitations use the claim labels consistently;
5. the author reviews the correction audit and explicitly approves promotion.

No correction run advances the constructive route. Forward checkpoints require a separate user decision after review.
