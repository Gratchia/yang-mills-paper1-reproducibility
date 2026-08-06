"""C5AK local holonomy-log split and disintegration diagnostic.

# ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates

C5AK is the first concrete sublemma toward C5AJ.  It tests the local coordinate
split

    microscopic link coordinates A  <->  (k, h)

where k is the rank-17 retained curvature-tile quotient and h lies in the
zero-retained high-mode kernel.  It then probes the nonlinear retained
holonomy-log map

    K_g(k,h) = g^{-1} log retained Wilson loops

and checks whether partial_K/partial_k remains invertible for small good-sector
samples.  This is a local finite-block diagnostic.  It is not a full compact
SU(2) conditional-disintegration theorem, continuum construction, or mass-gap
proof.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from c5ad_cubic_coordinate_normalization import exp_lie, log_lie, quat_mul
from c5ae_jacobian_half_density import tile_loop_paths
from c5v_retained_quotient_bch_bounds import COLORS, build_retained_lift


@dataclass(frozen=True)
class TangentSplitReport:
    scalar_links: int
    scalar_retained_rank: int
    scalar_high_rank: int
    color_links: int
    color_retained_rank: int
    color_high_rank: int
    constraint_lift_residual: float
    constraint_high_residual: float
    split_condition_number: float
    hessian_cross_block_norm: float
    positive_high_rank: int
    gauge_zero_dim: int
    min_positive_high_eigenvalue: float
    max_positive_high_eigenvalue: float


@dataclass(frozen=True)
class NonlinearProbeReport:
    samples: int
    g_probe: float
    coordinate_radius: float
    jac_step: float
    high_directions: int
    min_retained_jacobian_singular_value: float
    max_retained_jacobian_singular_value: float
    max_retained_jacobian_condition: float
    max_logdet_abs: float
    max_high_directional_coupling: float
    max_reconstruction_residual: float
    chart_failures: int


@dataclass(frozen=True)
class C5AKDecision:
    tangent_split_pass: bool
    nonlinear_local_ift_pass: bool
    full_conditional_measure_proved: bool
    decision: str


def zero_retained_basis(constraint: np.ndarray, tol: float) -> tuple[np.ndarray, int]:
    _, s, vt = np.linalg.svd(constraint, full_matrices=True)
    rank = int(np.sum(s > tol))
    return vt[rank:].T, rank


def tangent_split(tol: float) -> tuple[object, np.ndarray, np.ndarray, np.ndarray, TangentSplitReport]:
    data, lift, quotient_basis, constraint = build_retained_lift(2)
    high_basis, constraint_rank = zero_retained_basis(constraint, tol)
    transform = np.hstack([lift, high_basis])
    singular_values = np.linalg.svd(transform, compute_uv=False)

    hessian = data.d1.T @ data.d1
    high_hessian = high_basis.T @ hessian @ high_basis
    high_evals = np.linalg.eigvalsh((high_hessian + high_hessian.T) / 2.0)
    positive = high_evals > tol
    cross = lift.T @ hessian @ high_basis

    report = TangentSplitReport(
        scalar_links=len(data.cells[1]),
        scalar_retained_rank=lift.shape[1],
        scalar_high_rank=high_basis.shape[1],
        color_links=len(data.cells[1]) * COLORS,
        color_retained_rank=lift.shape[1] * COLORS,
        color_high_rank=high_basis.shape[1] * COLORS,
        constraint_lift_residual=float(np.linalg.norm(constraint @ lift - quotient_basis)),
        constraint_high_residual=float(np.linalg.norm(constraint @ high_basis)),
        split_condition_number=float(singular_values[0] / singular_values[-1]),
        hessian_cross_block_norm=float(np.linalg.norm(cross)),
        positive_high_rank=int(np.sum(positive)),
        gauge_zero_dim=int(np.sum(~positive)),
        min_positive_high_eigenvalue=float(np.min(high_evals[positive])),
        max_positive_high_eigenvalue=float(np.max(high_evals[positive])),
    )
    return data, lift, quotient_basis, high_basis, report


def links_from_split(
    k: np.ndarray,
    h: np.ndarray,
    lift: np.ndarray,
    high_basis: np.ndarray,
) -> np.ndarray:
    scalar_retained = lift.shape[1]
    scalar_high = high_basis.shape[1]
    k_matrix = k.reshape(scalar_retained, COLORS)
    h_matrix = h.reshape(scalar_high, COLORS)
    return lift @ k_matrix + high_basis @ h_matrix


def retained_log_from_links(
    links: np.ndarray,
    quotient_basis: np.ndarray,
    paths: list[list[tuple[int, float]]],
    n: int,
    g: float,
) -> np.ndarray:
    logs = np.zeros((len(paths), COLORS), dtype=float)
    if abs(g) < 1.0e-14:
        for p, path in enumerate(paths):
            total = np.zeros(COLORS)
            for edge, sign in path:
                total += sign * links[edge]
            logs[p] = total / float(n * n)
        return (quotient_basis.T @ logs).reshape(-1)

    for p, path in enumerate(paths):
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        for edge, sign in path:
            q = quat_mul(q, exp_lie(g * sign * links[edge]))
        if q[0] <= 0.0:
            raise ValueError(
                "retained holonomy left the declared positive-scalar SU(2) log chart"
            )
        logs[p] = log_lie(q) / (g * n * n)
    return (quotient_basis.T @ logs).reshape(-1)


def retained_coordinate_from_split(
    k: np.ndarray,
    h: np.ndarray,
    lift: np.ndarray,
    high_basis: np.ndarray,
    quotient_basis: np.ndarray,
    paths: list[list[tuple[int, float]]],
    n: int,
    g: float,
) -> np.ndarray:
    links = links_from_split(k, h, lift, high_basis)
    return retained_log_from_links(links, quotient_basis, paths, n, g)


def finite_difference_retained_jacobian_k(
    fn,
    k: np.ndarray,
    h: np.ndarray,
    step: float,
) -> np.ndarray:
    dim = k.size
    jac = np.zeros((dim, dim))
    for i in range(dim):
        delta = np.zeros_like(k)
        delta[i] = step
        jac[:, i] = (fn(k + delta, h) - fn(k - delta, h)) / (2.0 * step)
    return jac


def high_directional_coupling(
    fn,
    k: np.ndarray,
    h: np.ndarray,
    rng: np.random.Generator,
    directions: int,
    step: float,
) -> float:
    max_norm = 0.0
    for _ in range(directions):
        direction = rng.normal(size=h.size)
        direction /= np.linalg.norm(direction)
        derivative = (fn(k, h + step * direction) - fn(k, h - step * direction)) / (
            2.0 * step
        )
        max_norm = max(max_norm, float(np.linalg.norm(derivative)))
    return max_norm


def random_ball(rng: np.random.Generator, dim: int, radius: float) -> np.ndarray:
    x = rng.normal(size=dim)
    x /= np.linalg.norm(x)
    # Use a deterministic-radius shell inside the ball for reproducibility and
    # to probe the requested good-sector scale.
    return radius * x


def nonlinear_probe(
    data,
    lift: np.ndarray,
    quotient_basis: np.ndarray,
    high_basis: np.ndarray,
    samples: int,
    seed: int,
    g_probe: float,
    coordinate_radius: float,
    jac_step: float,
    high_directions: int,
) -> NonlinearProbeReport:
    rng = np.random.default_rng(seed)
    paths = tile_loop_paths(data)
    retained_dim = lift.shape[1] * COLORS
    high_dim = high_basis.shape[1] * COLORS

    min_singulars: list[float] = []
    max_singulars: list[float] = []
    conditions: list[float] = []
    logdets: list[float] = []
    high_couplings: list[float] = []
    reconstruction_residuals: list[float] = []
    chart_failures = 0

    for _ in range(samples):
        k = random_ball(rng, retained_dim, coordinate_radius)
        h = random_ball(rng, high_dim, coordinate_radius)

        fn = lambda kk, hh: retained_coordinate_from_split(
            kk, hh, lift, high_basis, quotient_basis, paths, data.n, g_probe
        )
        try:
            jac_k = finite_difference_retained_jacobian_k(fn, k, h, jac_step)
            singulars = np.linalg.svd(jac_k, compute_uv=False)
            sign, logdet = np.linalg.slogdet(jac_k)
            if sign <= 0:
                chart_failures += 1
                continue
            min_singulars.append(float(singulars[-1]))
            max_singulars.append(float(singulars[0]))
            conditions.append(float(singulars[0] / singulars[-1]))
            logdets.append(abs(float(logdet)))
            high_couplings.append(
                high_directional_coupling(fn, k, h, rng, high_directions, jac_step)
            )

            # Tangent reconstruction sanity: at g=0, K_0(k,h)=k exactly.
            k0 = retained_coordinate_from_split(
                k, h, lift, high_basis, quotient_basis, paths, data.n, 0.0
            )
            reconstruction_residuals.append(float(np.linalg.norm(k0 - k)))
        except (RuntimeError, np.linalg.LinAlgError):
            chart_failures += 1

    if not min_singulars:
        return NonlinearProbeReport(
            samples=samples,
            g_probe=g_probe,
            coordinate_radius=coordinate_radius,
            jac_step=jac_step,
            high_directions=high_directions,
            min_retained_jacobian_singular_value=0.0,
            max_retained_jacobian_singular_value=float("inf"),
            max_retained_jacobian_condition=float("inf"),
            max_logdet_abs=float("inf"),
            max_high_directional_coupling=float("inf"),
            max_reconstruction_residual=float("inf"),
            chart_failures=chart_failures,
        )

    return NonlinearProbeReport(
        samples=samples,
        g_probe=g_probe,
        coordinate_radius=coordinate_radius,
        jac_step=jac_step,
        high_directions=high_directions,
        min_retained_jacobian_singular_value=float(np.min(min_singulars)),
        max_retained_jacobian_singular_value=float(np.max(max_singulars)),
        max_retained_jacobian_condition=float(np.max(conditions)),
        max_logdet_abs=float(np.max(logdets)),
        max_high_directional_coupling=float(np.max(high_couplings)),
        max_reconstruction_residual=float(np.max(reconstruction_residuals)),
        chart_failures=chart_failures,
    )


def make_decision(
    tangent: TangentSplitReport,
    nonlinear: NonlinearProbeReport,
    max_condition: float,
    min_singular: float,
) -> C5AKDecision:
    tangent_pass = (
        tangent.constraint_lift_residual < 1.0e-8
        and tangent.constraint_high_residual < 1.0e-8
        and tangent.scalar_retained_rank == 17
        and tangent.scalar_high_rank == 199
        and tangent.positive_high_rank == 119
        and tangent.gauge_zero_dim == 80
    )
    nonlinear_pass = (
        nonlinear.chart_failures == 0
        and nonlinear.min_retained_jacobian_singular_value > min_singular
        and nonlinear.max_retained_jacobian_condition < max_condition
        and nonlinear.max_reconstruction_residual < 1.0e-8
    )
    return C5AKDecision(
        tangent_split_pass=tangent_pass,
        nonlinear_local_ift_pass=nonlinear_pass,
        full_conditional_measure_proved=False,
        decision=(
            "partial pass: tangent (K,H) split and sampled nonlinear local "
            "implicit-function checks pass, but gauge-zero handling and the "
            "full conditional SU(2) measure theorem remain open"
            if tangent_pass and nonlinear_pass
            else "fail or redesign: the local split/Jacobian checks did not pass"
        ),
    )


def run_checks(
    tangent: TangentSplitReport,
    nonlinear: NonlinearProbeReport,
    decision: C5AKDecision,
    max_condition: float,
    min_singular: float,
) -> None:
    assert tangent.scalar_links == 216
    assert tangent.scalar_retained_rank == 17
    assert tangent.scalar_high_rank == 199
    assert tangent.color_retained_rank == 51
    assert tangent.color_high_rank == 597
    assert tangent.constraint_lift_residual < 1.0e-8
    assert tangent.constraint_high_residual < 1.0e-8
    assert tangent.positive_high_rank == 119
    assert tangent.gauge_zero_dim == 80
    assert nonlinear.chart_failures == 0
    assert nonlinear.min_retained_jacobian_singular_value > min_singular
    assert nonlinear.max_retained_jacobian_condition < max_condition
    assert nonlinear.max_reconstruction_residual < 1.0e-8
    assert decision.tangent_split_pass
    assert decision.nonlinear_local_ift_pass
    assert not decision.full_conditional_measure_proved


def fmt(value: float | int | bool | str) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def print_report(
    tol: float,
    samples: int,
    seed: int,
    g_probe: float,
    coordinate_radius: float,
    jac_step: float,
    high_directions: int,
    max_condition: float,
    min_singular: float,
) -> None:
    data, lift, quotient_basis, high_basis, tangent = tangent_split(tol)
    nonlinear = nonlinear_probe(
        data,
        lift,
        quotient_basis,
        high_basis,
        samples,
        seed,
        g_probe,
        coordinate_radius,
        jac_step,
        high_directions,
    )
    decision = make_decision(tangent, nonlinear, max_condition, min_singular)
    run_checks(tangent, nonlinear, decision, max_condition, min_singular)

    print("C5AK LOCAL HOLONOMY-LOG DISINTEGRATION DIAGNOSTIC")
    print("\nTANGENT SPLIT")
    print("| quantity | value |")
    print("|---|---:|")
    for key, value in tangent.__dict__.items():
        print(f"| {key} | {fmt(value)} |")

    print("\nNONLINEAR RETAINED-COORDINATE PROBE")
    print("| quantity | value |")
    print("|---|---:|")
    for key, value in nonlinear.__dict__.items():
        print(f"| {key} | {fmt(value)} |")

    print("\nDECISION")
    print("| tangent split pass | nonlinear local IFT pass | full conditional measure proved | decision |")
    print("|---|---|---|---|")
    print(
        f"| {decision.tangent_split_pass} | {decision.nonlinear_local_ift_pass} | "
        f"{decision.full_conditional_measure_proved} | {decision.decision} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--g-probe", type=float, default=0.01)
    parser.add_argument("--coordinate-radius", type=float, default=0.25)
    parser.add_argument("--jac-step", type=float, default=1.0e-6)
    parser.add_argument("--high-directions", type=int, default=8)
    parser.add_argument("--max-condition", type=float, default=2.0)
    parser.add_argument("--min-singular", type=float, default=0.5)
    args = parser.parse_args()
    print_report(
        args.tol,
        args.samples,
        args.seed,
        args.g_probe,
        args.coordinate_radius,
        args.jac_step,
        args.high_directions,
        args.max_condition,
        args.min_singular,
    )


if __name__ == "__main__":
    main()
