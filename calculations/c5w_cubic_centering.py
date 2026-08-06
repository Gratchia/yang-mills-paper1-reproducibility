"""Cubic first-cumulant/centering checker for C5W.

# ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates

C5W checks the O(g) cubic term

    S1(A) = sum_p Q1_p(A) . Q2_p(A)

after the split A = B(x) + xi, where B(x) is the C5V retained
minimum-action section and xi is a centered Gaussian high-mode fluctuation
with zero retained variables.

The key point is narrow:

* the xi-dependent first cumulant is zero for any color-isotropic centered
  Gaussian covariance on the zero-retained scalar link space;
* the purely retained term S1(B(x)) generally remains nonzero.

Thus the high-mode cubic first cumulant centers, but the retained-section
cubic is a coordinate/local-potential term, not a summable residual error.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from c5t_bch_incidence_expansion import plaquette_paths
from c5v_retained_quotient_bch_bounds import build_retained_lift


COLORS = 3


@dataclass(frozen=True)
class GaussianReference:
    constraint_rank: int
    zero_retained_dim: int
    positive_high_rank: int
    gauge_zero_dim: int
    scalar_cov_trace: float
    scalar_cov_symmetry_error: float
    constraint_cov_residual: float


@dataclass(frozen=True)
class CenteringResult:
    retained_samples: int
    max_abs_retained_s1: float
    mean_abs_retained_s1: float
    max_abs_tadpole_residual: float
    max_abs_conditional_mean_error: float


def signed_edge_arrays(paths: list[list[tuple[int, float]]]) -> tuple[np.ndarray, np.ndarray]:
    edges = np.array([[edge for edge, _ in path] for path in paths], dtype=int)
    signs = np.array([[sign for _, sign in path] for path in paths], dtype=float)
    return edges, signs


def q1_q2_for_signed_edges(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Q1,Q2 for local signed edge vectors x[p,4,3]."""
    q1 = np.sum(x, axis=1)
    q2 = np.zeros_like(q1)
    for i in range(4):
        for j in range(i + 1, 4):
            q2 += 0.5 * np.cross(x[:, i, :], x[:, j, :])
    return q1, q2


def s1_action(paths: list[list[tuple[int, float]]], a: np.ndarray) -> float:
    """Compute S1(A)=sum_p Q1_p.Q2_p with path signs included."""
    edges, signs = signed_edge_arrays(paths)
    signed = signs[:, :, None] * a[edges]
    q1, q2 = q1_q2_for_signed_edges(signed)
    return float(np.sum(q1 * q2))


def random_unit(rng: np.random.Generator, dim: int) -> np.ndarray:
    x = rng.normal(size=dim)
    return x / np.linalg.norm(x)


def retained_to_links(lift: np.ndarray, x: np.ndarray) -> np.ndarray:
    scalar_rank = lift.shape[1]
    return lift @ x.reshape(scalar_rank, COLORS)


def zero_retained_gaussian_reference(
    hessian: np.ndarray, constraint: np.ndarray, tol: float
) -> tuple[np.ndarray, GaussianReference]:
    """Scalar covariance on zero-retained, positive-action high modes."""
    u, s, vt = np.linalg.svd(constraint, full_matrices=True)
    constraint_rank = int(np.sum(s > tol))
    zero_retained = vt[constraint_rank:].T
    h_zero = zero_retained.T @ hessian @ zero_retained
    evals, vecs = np.linalg.eigh(h_zero)
    pos = evals > tol
    positive_modes = zero_retained @ vecs[:, pos]
    covariance = (positive_modes / evals[pos]) @ positive_modes.T
    covariance = (covariance + covariance.T) / 2.0
    ref = GaussianReference(
        constraint_rank=constraint_rank,
        zero_retained_dim=zero_retained.shape[1],
        positive_high_rank=int(np.sum(pos)),
        gauge_zero_dim=int(np.sum(~pos)),
        scalar_cov_trace=float(np.trace(covariance)),
        scalar_cov_symmetry_error=float(np.linalg.norm(covariance - covariance.T)),
        constraint_cov_residual=float(np.linalg.norm(constraint @ covariance)),
    )
    return covariance, ref


def levi_civita() -> np.ndarray:
    eps = np.zeros((COLORS, COLORS, COLORS), dtype=float)
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
    eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0
    return eps


def tadpole_residual(
    paths: list[list[tuple[int, float]]], covariance: np.ndarray, background: np.ndarray
) -> float:
    """Analytic two-high-mode part of E[S1(B+xi)]-S1(B).

    The high-mode covariance is Cov[xi_e^a,xi_f^b]=K_ef delta_ab.
    For such a color-isotropic covariance, every contraction contains a trace
    of the antisymmetric Levi-Civita tensor and vanishes.
    """
    eps = levi_civita()
    delta = np.eye(COLORS)
    residual = 0.0
    for path in paths:
        edges = [edge for edge, _ in path]
        signs = np.array([sign for _, sign in path], dtype=float)
        b_loc = signs[:, None] * background[edges]
        k_loc = signs[:, None] * signs[None, :] * covariance[np.ix_(edges, edges)]

        for t in range(4):
            for i in range(4):
                for j in range(i + 1, 4):
                    # xi_t . (xi_i x B_j)
                    residual += 0.5 * k_loc[t, i] * float(
                        np.einsum("ab,abc,c->", delta, eps, b_loc[j])
                    )
                    # xi_t . (B_i x xi_j)
                    residual += 0.5 * k_loc[t, j] * float(
                        np.einsum("ac,abc,b->", delta, eps, b_loc[i])
                    )
                    # B_t . (xi_i x xi_j)
                    residual += 0.5 * k_loc[i, j] * float(
                        np.einsum("bc,abc,a->", delta, eps, b_loc[t])
                    )
    return float(residual)


def run_centering_samples(samples: int, seed: int, tol: float) -> tuple[GaussianReference, CenteringResult]:
    data, lift, _, constraint = build_retained_lift(2)
    paths = plaquette_paths(data)
    hessian = data.d1.T @ data.d1
    covariance, reference = zero_retained_gaussian_reference(hessian, constraint, tol)

    rng = np.random.default_rng(seed)
    dim = COLORS * lift.shape[1]
    retained_values: list[float] = []
    tadpoles: list[float] = []
    mean_errors: list[float] = []
    for _ in range(samples):
        x = random_unit(rng, dim)
        background = retained_to_links(lift, x)
        retained_s1 = s1_action(paths, background)
        tadpole = tadpole_residual(paths, covariance, background)
        # For a cubic polynomial and centered Gaussian,
        # E[S1(B+xi)] = S1(B) + tadpole.
        conditional_mean_error = tadpole
        retained_values.append(retained_s1)
        tadpoles.append(tadpole)
        mean_errors.append(conditional_mean_error)

    result = CenteringResult(
        retained_samples=samples,
        max_abs_retained_s1=float(np.max(np.abs(retained_values))),
        mean_abs_retained_s1=float(np.mean(np.abs(retained_values))),
        max_abs_tadpole_residual=float(np.max(np.abs(tadpoles))),
        max_abs_conditional_mean_error=float(np.max(np.abs(mean_errors))),
    )
    return reference, result


def run_checks(reference: GaussianReference, result: CenteringResult, tol: float) -> None:
    assert reference.constraint_rank == 17
    assert reference.zero_retained_dim == 199
    assert reference.positive_high_rank == 119
    assert reference.gauge_zero_dim == 80
    assert reference.constraint_cov_residual < 1e-8
    assert result.max_abs_tadpole_residual < 10 * tol
    assert result.max_abs_retained_s1 > 1e-6


def print_report(samples: int, seed: int, tol: float) -> None:
    reference, result = run_centering_samples(samples, seed, tol)
    run_checks(reference, result, tol)

    print("ZERO-RETAINED GAUSSIAN REFERENCE")
    print(
        "| constraint rank | zero-retained scalar dim | positive high rank | "
        "gauge zero dim | tr scalar covariance | symmetry error | constraint covariance residual |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|")
    print(
        f"| {reference.constraint_rank} | {reference.zero_retained_dim} | "
        f"{reference.positive_high_rank} | {reference.gauge_zero_dim} | "
        f"{reference.scalar_cov_trace:.12g} | {reference.scalar_cov_symmetry_error:.3g} | "
        f"{reference.constraint_cov_residual:.3g} |"
    )

    print("\nCUBIC FIRST-CUMULANT CENTERING")
    print(
        "| retained samples | max |S1(B)| | mean |S1(B)| | "
        "max tadpole residual | max conditional mean error after subtracting S1(B) |"
    )
    print("|---:|---:|---:|---:|---:|")
    print(
        f"| {result.retained_samples} | {result.max_abs_retained_s1:.12g} | "
        f"{result.mean_abs_retained_s1:.12g} | "
        f"{result.max_abs_tadpole_residual:.3g} | "
        f"{result.max_abs_conditional_mean_error:.3g} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--tol", type=float, default=1e-10)
    args = parser.parse_args()
    print_report(args.samples, args.seed, args.tol)


if __name__ == "__main__":
    main()
