# Clean-Start C5AX: Reflection-Compatible Retained Density Diagnostic

<!-- ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates -->

## Claim boundary

C5AX follows C5AW.  C5AW built the odd centered density channel as an explicit finite-block bilinear operator

\[
B(K,H)=K^T M_{\rm odd}H,
\qquad
M_{\rm odd}=M_{\rm ret}+M_{\rm Haar}+M_{\rm FP}.
\]

C5AW found that Haar and FP are reflection-compatible to roundoff, but the retained-coordinate Jacobian piece has reflection residual \(1.02367185521\).  C5AX asks whether that bad number is merely a wrong linear reflection representation for the retained tile coordinates, or a genuine nonlinear retained-loop placement defect.

Decision:

> C5AX is a fail/pass split.  It rules out the benign explanation.  The C5AW retained reflection residual is not a linear-coordinate artifact: the lift-level and tile-quotient reflection actions agree to roundoff.  The obstruction is nonlinear and geometric: time-reflected based face loops become cyclically shifted inverse loops, so the based Lie-algebra log is conjugated by a transport segment rather than strictly reflected.  Therefore the current based holonomy-log retained coordinate cannot be used as an OS-placed half-density without redesign.

This is not a continuum construction, not a confinement proof, and not a mass-gap proof.

## 1. Linear retained-coordinate test

The two candidate scalar reflection actions are:

1. lift-level reflection, inherited from the minimum-action retained link lift \(L\);
2. tile-loop quotient reflection, inherited from the 24 coarse face-loop linear functionals and the 17-dimensional Bianchi quotient.

The diagnostic shows they are the same to roundoff:

| quantity | value |
|---|---:|
| scalar links | \(216\) |
| tile loops | \(24\) |
| retained rank | \(17\) |
| tile row rank | \(17\) |
| quotient orthogonality error | \(2.54127620691\times10^{-15}\) |
| constraint-lift residual | \(1.15064868438\times10^{-13}\) |
| tile-row reflection residual | \(1.3415637684\times10^{-15}\) |
| quotient reflection leak | \(2.97584013804\times10^{-15}\) |
| quotient reflection orthogonality error | \(1.04459705298\times10^{-14}\) |
| retained lift reflection leak | \(5.64572744853\times10^{-14}\) |
| lift-vs-tile reflection difference | \(8.29146053002\times10^{-14}\) |
| lift-vs-tile relative difference | \(2.01097456211\times10^{-14}\) |
| lift reflection orthogonality error | \(1.64988491045\times10^{-13}\) |

So C5AW did not fail because it used the wrong linear retained reflection action.

## 2. Ordered path closure

The 24 face paths are reflection-closed as linear loop functionals, but not all are reflection-closed as based ordered paths.

| quantity | value |
|---|---:|
| paths | \(24\) |
| same-order reflected paths | \(12\) |
| inverse-cyclic reflected paths | \(12\) |
| unmatched paths | \(0\) |
| maximum cyclic shift | \(2\) |

The 12 problematic paths are the time-containing face loops.  Time reflection turns them into cyclically shifted inverse loops.

For Wilson traces this is harmless, because trace is cyclic and invariant under conjugation.  For based Lie-algebra logs it is not harmless: a cyclic shift of a nonabelian loop changes the log by conjugation with the partial transport.

## 3. Nonlinear retained-log equivariance probe

The checker tests the exact retained log coordinate \(K_g(A)\) on a fixed retained-plus-positive-high field sample and compares

\[
K_g(\theta A)
\quad\text{against}\quad
R_KK_g(A),
\]

where \(\theta\) is link reflection and \(R_K\) is the retained quotient reflection matrix.

| \(g\) | relative error | absolute error | retained norm |
|---:|---:|---:|---:|
| \(0\) | \(1.55960754357\times10^{-15}\) | \(2.00842682542\times10^{-15}\) | \(1.28777706526\) |
| \(10^{-3}\) | \(1.62017858664\times10^{-4}\) | \(2.0864422782\times10^{-4}\) | \(1.28778536848\) |
| \(10^{-2}\) | \(1.6197258245\times10^{-3}\) | \(2.08597791213\times10^{-3}\) | \(1.28785864902\) |
| \(10^{-1}\) | \(1.6156096032\times10^{-2}\) | \(2.08162829885\times10^{-2}\) | \(1.28844758952\) |
| \(0.3\) | \(4.82733669574\times10^{-2}\) | \(6.22151258301\times10^{-2}\) | \(1.28880850356\) |

The linear coordinate is reflection-compatible at \(g=0\).  The finite-\(g\) retained log coordinate is not.  The observed defect is consistent with a first nonlinear basepoint/conjugation effect.

## 4. Density-source reflection residuals

Using the correct tile-quotient reflection action gives the same C5AW conclusion:

| source | Frobenius norm | even residual | anti residual | invariant projection defect ratio |
|---|---:|---:|---:|---:|
| retained | \(1.45273042015\) | \(1.02367185521\) | \(1.71816644504\) | \(0.511835927605\) |
| retained \(\frac12{\rm tr}J_2\) | \(1.63270816947\) | \(1.10331504223\) | \(1.66814145611\) | \(0.551657521114\) |
| retained \(-\frac14{\rm tr}J_1^2\) | \(0.808877725052\) | \(1.20784054217\) | \(1.59408946571\) | \(0.603920271087\) |
| Haar half | \(0.277397504334\) | \(5.1567939458\times10^{-14}\) | \(2\) | \(2.5783969729\times10^{-14}\) |
| FP half | \(0.347066193988\) | \(4.69318028554\times10^{-14}\) | \(2\) | \(2.34659014277\times10^{-14}\) |
| combined | \(1.5812228157\) | \(0.940486836864\) | \(1.76507351396\) | \(0.470243418432\) |

The retained term is neither even nor anti-even under the current based-log placement.  Projecting it onto a reflection-invariant part would remove about half of its Frobenius norm, so this is not a small harmless symmetrization.

## 5. Interpretation

C5AX preserves the C5AW size information but blocks the current OS placement:

\[
\|M_{\rm odd}\|_{\rm op}=0.388456446008,
\qquad
C_{\rm cov}^2=0.0315650978782,
\]

remain valid finite-block size data for the current coordinates.  But the current retained-coordinate Jacobian cannot be inserted as a reflection-compatible half-density.

The physical cause is clear:

- the tangent curvature-tile quotient is reflection-compatible;
- the face loops are closed under reflection as unbased linear circulations;
- but the nonabelian based logarithm remembers the basepoint/order;
- time reflection changes the basepoint/order by a cyclic shift of an inverse loop;
- cyclic shift is conjugation, invisible to Wilson traces but visible to Lie logs.

## 6. Consequence for the route

The normalized package from C5AG cannot keep using this based holonomy-log retained coordinate unless the retained-coordinate density is redesigned.

The next checkpoint should be C5AY:

> construct a reflection-covariant retained-coordinate replacement, either by adding explicit reflection-covariant stems/basepoints to the loop package or by changing retained variables to conjugacy-class/Wilson-trace data plus enough orientation data to keep the tangent 17-dimensional curvature quotient.

Acceptance standard for C5AY:

1. exact nonlinear reflection covariance of the retained coordinate map on the good chart;
2. no loss of the C5R/C5V tangent retained rank and coercive quadratic Hessian;
3. a rebuilt retained-coordinate Jacobian half-density;
4. retained/Haar/FP odd density reflection residual below numerical roundoff in the finite-block checker;
5. updated \(L^2\) constants if the coordinate replacement changes \(M_{\rm odd}\);
6. no appeal to confinement, area law, exponential clustering, or an existing gap.

## 7. Artifact

Checker: [c5ax_reflection_retained_density.py](../calculations/c5ax_reflection_retained_density.py)
