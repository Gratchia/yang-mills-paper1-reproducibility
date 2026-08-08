# Clean-Start C5AY: Reflection-Covariant Retained Coordinate Diagnostic

<!-- ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates -->

## Claim boundary

C5AY follows C5AX.  C5AX showed that the retained-coordinate reflection failure in C5AW was not a wrong linear reflection representation.  The linear 17-dimensional curvature-tile quotient was reflection-compatible, but the nonlinear based holonomy-log coordinate failed because time-reflected loops became cyclically shifted inverse loops.

C5AY tests the least disruptive repair:

> rebase every retained face loop by a uniform one-segment cyclic shift.

This does not change the tangent linear circulation row of any closed face loop, so it should preserve the 17-dimensional retained quotient.  It does change the nonlinear based log and may remove the hidden C5AX conjugation defect.

Decision:

> C5AY passes at finite-block level, with theorem debt.  A uniform one-segment cyclic rebase of the 24 face loops preserves the linear 17-dimensional curvature quotient, restores nonlinear reflection covariance of the retained holonomy-log coordinate to roundoff, and makes the retained density operator reflection-compatible to roundoff.  The Wilson-trace/conjugacy-class replacement is not needed at this stage.  Remaining debt: formulate the full gauge-covariant common-stem implementation and regression-check the normalized C5AG shell package with the shifted coordinate.

This is not a continuum construction, not a confinement proof, and not a mass-gap proof.

## 1. Candidate

The tested repair is:

\[
\gamma_p
\longmapsto
\operatorname{cyc}_1(\gamma_p),
\]

where \(\gamma_p\) is one of the 24 retained face loops and \(\operatorname{cyc}_1\) shifts the based path by one segment.

The diagnostic found two exact uniform cyclic shifts:

| valid exact uniform cyclic shifts |
|---|
| \(1,5\) |

C5AY uses shift \(1\), the smallest repair.

## 2. Exact ordered-path reflection closure

The original C5AX ordering was linearly reflection-compatible but not exactly based-path reflection-compatible.

| candidate | cyclic shift | exact same-order | exact inverse | unmatched |
|---|---:|---:|---:|---:|
| original C5AX ordering | \(0\) | \(12\) | \(0\) | \(12\) |
| uniform cyclic rebase | \(1\) | \(12\) | \(12\) | \(0\) |

Interpretation:

- the 12 spatial face loops reflect as same-order loops;
- the 12 time-containing face loops now reflect as exact inverse loops, not cyclically shifted inverse loops;
- therefore the based \(SU(2)\) log transforms by the quotient reflection action without extra conjugation.

## 3. Linear quotient preservation

Because cyclic rebasing does not change the signed linear circulation row, the tangent retained quotient is unchanged.

| quantity | value |
|---|---:|
| original row rank | \(17\) |
| shifted row rank | \(17\) |
| row difference norm | \(0\) |
| quotient reflection leak | \(2.97584013804\times10^{-15}\) |
| lift-vs-tile reflection relative difference | \(2.01097456211\times10^{-14}\) |
| retained rank | \(17\) |
| Bianchi relations | \(7\) |
| density min Hessian | \(4\) |
| density max Hessian | \(6.15384615385\) |

So the C5R/C5V tangent rank and quadratic coercivity data survive unchanged.

## 4. Nonlinear retained-log equivariance

The shifted retained coordinate was tested on the same deterministic retained-plus-positive-high probe pattern as C5AX.

| \(g\) | relative error | absolute error | retained norm |
|---:|---:|---:|---:|
| \(0\) | \(1.5569062183\times10^{-15}\) | \(2.00494812069\times10^{-15}\) | \(1.28777706526\) |
| \(10^{-3}\) | \(1.53434734698\times10^{-15}\) | \(1.97591005486\times10^{-15}\) | \(1.28778536278\) |
| \(10^{-2}\) | \(1.52833656607\times10^{-15}\) | \(1.96828059503\times10^{-15}\) | \(1.28785807965\) |
| \(10^{-1}\) | \(1.58853152424\times10^{-15}\) | \(2.04664915032\times10^{-15}\) | \(1.28839064199\) |
| \(0.3\) | \(1.59164912432\times10^{-15}\) | \(2.05051589755\times10^{-15}\) | \(1.2882964381\) |
| \(0.8\) | \(1.56852934459\times10^{-15}\) | \(2.0079051434\times10^{-15}\) | \(1.28011959121\) |

Unlike C5AX, there is no visible finite-\(g\) growth of the equivariance defect.

## 5. Density operator reflection and size

The retained density operator must be rebuilt because the based nonlinear coordinate changed.  The new shifted retained operator is reflection-compatible, but its size constants are larger than C5AW.

| source | Frobenius norm | operator norm | covariance \(L^2\) sq | covariance \(L^2\) | even residual | anti residual |
|---|---:|---:|---:|---:|---:|---:|
| old retained, pre-C5AY | \(1.45273042015\) | \(0.36684361509\) | \(0.0271265937543\) | \(0.164701529302\) | \(1.02367185521\) | \(1.71816644504\) |
| shifted retained | \(1.60355467181\) | \(0.479504156347\) | \(0.0462202859103\) | \(0.214989036721\) | \(3.90764341645\times10^{-14}\) | \(2\) |
| shifted retained \(\frac12{\rm tr}J_2\) | \(1.84049048784\) | \(0.523861740043\) | \(0.0530583044933\) | \(0.230343883125\) | \(5.86741622624\times10^{-14}\) | \(2\) |
| shifted retained \(-\frac14J_1^2\) | \(0.658273454001\) | \(0.198607086751\) | \(0.00701239684574\) | \(0.083740055205\) | \(8.84202603771\times10^{-14}\) | \(2\) |
| Haar half | \(0.277397504334\) | \(0.0573350763461\) | \(0.000897688767511\) | \(0.0299614546962\) | \(5.1567939458\times10^{-14}\) | \(2\) |
| FP half | \(0.347066193988\) | \(0.0719709558354\) | \(0.00139348387557\) | \(0.0373293969355\) | \(4.69318028554\times10^{-14}\) | \(2\) |
| old combined, pre-C5AY | \(1.5812228157\) | \(0.388456446008\) | \(0.0315650978782\) | \(0.177665691337\) | \(0.940486836864\) | \(1.76507351396\) |
| shifted combined | \(1.72082175276\) | \(0.496353904572\) | \(0.050689424924\) | \(0.225143120979\) | \(4.61255124852\times10^{-14}\) | \(2\) |

The new finite-block constants are therefore:

\[
\|M_{\rm odd}^{\rm shifted}\|_{\rm op}=0.496353904572,
\qquad
C_{\rm cov,shifted}^2=0.050689424924.
\]

These remain finite and compatible with the C5M summability model, though less sharp than the pre-C5AY constants.

## 6. Consequence for Wilson-trace/conjugacy-class replacement

C5AX proposed Wilson traces or conjugacy-class variables as a possible escape because they are insensitive to conjugation.  C5AY shows they are not needed yet:

- the retained Lie-log coordinate can be repaired at path-order/basepoint level;
- the tangent quotient and coercivity are preserved;
- the retained density becomes reflection-compatible.

Wilson traces remain a fallback if the full common-stem/gauge-covariant theorem fails.

## 7. Remaining theorem debt

C5AY is a finite-block coordinate diagnostic.  It does not yet prove the full normalized shell theorem.

Next checkpoint should be C5AZ:

> implement the shifted retained coordinate inside the full gauge-covariant common-stem package and regression-check the normalized C5AG shell ledger.

C5AZ must verify:

1. exact common-stem/gauge-covariant definition of the shifted based loops;
2. preservation of the C5AY reflection covariance in the gauge-covariant package;
3. regression of C5AD--C5AW density/action bookkeeping under the shifted coordinate;
4. updated C5AG normalized shell slots and constants;
5. no hidden use of confinement, area law, exponential clustering, or an existing mass gap.

## 8. Artifact

Checker: [c5ay_reflection_covariant_retained_coordinates.py](../calculations/c5ay_reflection_covariant_retained_coordinates.py)
