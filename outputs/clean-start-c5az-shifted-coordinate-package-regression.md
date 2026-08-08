# Clean-Start C5AZ: Shifted-Coordinate Common-Stem Package Regression

> **SUPERSEDED 2026-07-12.** The path/group construction below remains historical evidence, but its density regression used shifted-core paths rather than the true rooted paths. Do not use its constants or downstream status. Use the corrected C5AZ JSON and `outputs/c5x-rooted-coordinate-correction-audit.md`.

<!-- ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates -->

## Claim boundary

C5AZ follows C5AY.  C5AY showed that a uniform one-segment cyclic rebase of the 24 retained face loops fixes the finite-block reflection defect of the based holonomy-log coordinate.  C5AZ asks whether that repair can be treated as a genuine common-stem, gauge-covariant retained-coordinate package rather than merely a gauge-fixed path-order trick.

Decision:

> C5AZ passes as a finite common-stem/package regression.  The C5AY shifted loops admit reflection-compatible stems from a fixed reflection-invariant root, rooted holonomies transform by simultaneous root conjugation under finite gauge transformations, and rooted reflection acts exactly as same/inverse loop reflection.  The C5AE no-\(O(g)\) Jacobian result and the C5AY shifted odd-density constants survive.  The exact nonlinear one-shell theorem, good-sector bounds, and exceptional-sector bounds remain open.

This is not a continuum construction, not a confinement proof, and not a mass-gap proof.

## 1. Common-stem path package

The common root is chosen as

\[
x_\ast=(1,0,0,0)
\]

inside the \(n=2\) block.  It is fixed by the time reflection \(t\mapsto 2-t\).

For every shifted retained face loop \(\gamma_i\), C5AZ attaches a canonical Manhattan stem \(s_i:x_\ast\to x_i\), where \(x_i\) is the shifted loop basepoint, and uses the rooted closed path

\[
\widehat\gamma_i=s_i\gamma_i s_i^{-1}.
\]

The finite path package has:

| quantity | value |
|---|---:|
| root vertex | \((1,0,0,0)\) |
| root reflection fixed | true |
| retained loops | \(24\) |
| stems | \(24\) |
| loop same-order reflections | \(12\) |
| loop inverse reflections | \(12\) |
| loop unmatched reflections | \(0\) |
| stem unmatched reflections | \(0\) |
| rooted path unmatched reflections | \(0\) |

Thus reflection sends each rooted retained path exactly to either the matching rooted path or its inverse.

## 2. Finite group covariance probes

C5AZ then tests actual \(SU(2)\)-valued link variables, not only tangent link cochains.

Under a finite gauge transformation

\[
U_e\mapsto g_{t(e)}U_e g_{h(e)}^{-1},
\]

each rooted holonomy transforms by simultaneous root conjugation:

\[
\widehat U_{\gamma_i}\mapsto
g_{x_\ast}\widehat U_{\gamma_i}g_{x_\ast}^{-1}.
\]

The diagnostic also checks group-level reflection of the rooted package.

| quantity | value |
|---|---:|
| random seed | \(20260704\) |
| random link samples | \(8\) |
| max rooted gauge covariance residual | \(1.27373599508\times10^{-15}\) |
| max rooted reflection residual | \(7.52989890787\times10^{-16}\) |

So the shifted coordinate package is gauge-covariant and reflection-compatible at finite group level in this block diagnostic.

## 3. Jacobian / half-density regression

C5AE proved that the quadratic retained coordinate correction \(N_2\) has zero divergence, so the retained coordinate change creates no \(O(g)\) Jacobian/half-density term.

C5AZ rechecks this for the shifted retained paths:

| quantity | value |
|---|---:|
| samples | \(4\) |
| max analytic divergence of \(N_2\) | \(0\) |
| max finite-difference divergence of \(N_2\) | \(0\) |

Thus the C5AE no-\(O(g)\) half-density conclusion survives the shifted coordinate.

## 4. Shifted density package regression

C5AZ uses the shifted retained paths from C5AY and rebuilds the odd density operator.

| quantity | value |
|---|---:|
| old combined reflection residual | \(0.940486836864\) |
| shifted combined reflection residual | \(4.61255124852\times10^{-14}\) |
| shifted operator norm | \(0.496353904572\) |
| shifted covariance \(L^2\) constant squared | \(0.050689424924\) |
| shifted covariance \(L^2\) constant | \(0.225143120979\) |

The normalized package should therefore use the shifted constants:

\[
\|M_{\rm odd}^{\rm shifted}\|_{\rm op}=0.496353904572,
\qquad
C_{\rm cov,shifted}^2=0.050689424924.
\]

## 5. Normalized package implication

C5AZ upgrades C5AY from “path-order repair” to “finite common-stem retained-coordinate package.”

The normalized C5AG retained shell package should be updated as follows:

| package slot | pre-C5AZ status | C5AZ regression |
|---|---|---|
| exact retained holonomy-log coordinate | blocked by C5AX reflection issue | repaired by shifted common-stem coordinate |
| \(O(g)\) coordinate Jacobian | absent by C5AE | still absent for shifted paths |
| \(O(g^2)\) retained density | retained in OS half-density slot | retained with shifted constants |
| centered odd \(KH\) density channel | C5AW finite but not reflection-compatible | reflection-compatible after C5AY/C5AZ |
| Wilson trace/conjugacy fallback | possible rescue | not needed at finite-block level |

## 6. Remaining debts

C5AZ does not prove the exact nonlinear shell theorem.  It clears a coordinate/interface obstruction and updates the finite-block constants.

The next checkpoint should be C5BA:

> formulate the shifted normalized one-shell theorem with the common-stem retained coordinate and identify the exact good-sector remainder/exceptional-sector estimates still required.

C5BA should:

1. rewrite the C5AG/C5AH theorem contract with the shifted common-stem coordinate;
2. replace old C5AW constants by shifted C5AZ constants;
3. specify the exact post-retention remainder \(R_j^{\rm post,shifted}\);
4. list the still-open bounds: nonlinear good-sector expansion, local/quasilocal conditional density, comparison-potential defect, chart/large-field exceptional sector;
5. verify no step uses confinement, area law, exponential clustering, or an existing mass gap.

## 7. Artifact

Checker: [c5az_shifted_coordinate_package_regression.py](../calculations/c5az_shifted_coordinate_package_regression.py)
