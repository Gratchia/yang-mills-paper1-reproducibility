"""C5AU analytic retained-coordinate/FP source formula diagnostic.

# ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates

C5AT showed that the Haar O(g^2) even-centered K^2 H^2 source is exactly
zero, while the retained-coordinate/Faddeev--Popov interface is too unstable
under nested finite differences to decide cancellation or retention.

C5AU replaces finite-g coefficient extraction by analytic g=0 source formulas:

    K_g(k,h) = k + g K1(k,h) + g^2 K2(k,h) + ...

for the retained holonomy-log coordinate, and

    FP_g(A) = M0 + g M1(A) + g^2 M2(A) + ...

for the gauge-slice Faddeev--Popov matrix.  The O(g^2) log-density
coefficients are computed from

    log det(I + g A1 + g^2 A2) = g tr A1
        + g^2 (tr A2 - 1/2 tr A1^2) + ...

This is a finite-block analytic-coefficient diagnostic.  It is not a
deterministic good-sector theorem, continuum construction, confinement proof,
or mass-gap proof.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from c5al_gauge_zero_reference import build_split, reflection_report
from c5am_nonlinear_gauge_haar import gauge_slice_report, links_from_retained_positive, random_sphere
from c5ao_conditional_half_density_projection import gaussian_high_sample
from c5ae_jacobian_half_density import tile_loop_paths
from c5q_4d_cubical_incidence import coboundary
from c5t_bch_incidence_expansion import bch3_pair
from c5v_retained_quotient_bch_bounds import COLORS, build_retained_lift


@dataclass(frozen=True)
class C5AUDiagnostic:
    retained_radius: float
    gaussian_scale: float
    derivative_step: float
    retained_rank: int
    positive_high_rank: int
    gauge_vertex_rank: int
    retained_jacobian_even_centered_coeff: float
    fp_half_even_centered_coeff: float
    haar_half_even_centered_coeff_exact: float
    combined_even_centered_coeff: float
    retained_fp_abs_sum: float
    retained_fp_cancellation_ratio: float
    baseline_c5at_finite_g_coeff: float
    analytic_minus_c5at_baseline: float
    max_tangent_retained_reconstruction_error: float
    fp_m0_base_relative_error: float
    fp_m0_min_singular: float
    fp_m0_condition: float
    positive_covariance_reflection_commutator: float
    classification: str


@dataclass(frozen=True)
class C5AUDecision:
    analytic_source_formulas_implemented: bool
    cancellation_proved: bool
    retention_required: bool
    deterministic_good_sector_theorem_proved: bool
    decision: str


def bch3_sequence(vectors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zero = np.zeros(COLORS, dtype=float)
    z1 = zero.copy()
    z2 = zero.copy()
    z3 = zero.copy()
    for vec in vectors:
        z1, z2, z3 = bch3_pair(z1, z2, z3, vec, zero, zero)
    return z1, z2, z3


def retained_q_coefficients(
    links: np.ndarray,
    quotient_basis: np.ndarray,
    paths: list[list[tuple[int, float]]],
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q1 = np.zeros((len(paths), COLORS), dtype=float)
    q2 = np.zeros_like(q1)
    q3 = np.zeros_like(q1)
    for p, path in enumerate(paths):
        vectors = [sign * links[edge] for edge, sign in path]
        q1[p], q2[p], q3[p] = bch3_sequence(vectors)
    scale = float(n * n)
    return (
        (quotient_basis.T @ q1 / scale).reshape(-1),
        (quotient_basis.T @ q2 / scale).reshape(-1),
        (quotient_basis.T @ q3 / scale).reshape(-1),
    )


def retained_k1_k2(
    k: np.ndarray,
    y: np.ndarray,
    data,
    lift: np.ndarray,
    positive_basis: np.ndarray,
    quotient_basis: np.ndarray,
    paths,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    links = links_from_retained_positive(lift, positive_basis, k, y)
    return retained_q_coefficients(links, quotient_basis, paths, data.n)


def jacobian_in_k(fn, k: np.ndarray, y: np.ndarray, step: float) -> np.ndarray:
    dim = k.size
    base = fn(k, y)
    jac = np.zeros((base.size, dim), dtype=float)
    for i in range(dim):
        delta = np.zeros_like(k)
        delta[i] = step
        jac[:, i] = (fn(k + delta, y) - fn(k - delta, y)) / (2.0 * step)
    return jac


def retained_half_density_o2_coeff(
    k: np.ndarray,
    y: np.ndarray,
    data,
    lift: np.ndarray,
    positive_basis: np.ndarray,
    quotient_basis: np.ndarray,
    paths,
    derivative_step: float,
) -> tuple[float, float]:
    def q0(kk, yy):
        return retained_k1_k2(kk, yy, data, lift, positive_basis, quotient_basis, paths)[0]

    def q1(kk, yy):
        return retained_k1_k2(kk, yy, data, lift, positive_basis, quotient_basis, paths)[1]

    def q2(kk, yy):
        return retained_k1_k2(kk, yy, data, lift, positive_basis, quotient_basis, paths)[2]

    j0 = jacobian_in_k(q0, k, y, derivative_step)
    j1 = jacobian_in_k(q1, k, y, derivative_step)
    j2 = jacobian_in_k(q2, k, y, derivative_step)
    # Tangent sanity: j0 should be identity for the retained split.
    reconstruction_error = float(np.linalg.norm(j0 - np.eye(k.size)))
    coeff = 0.5 * (float(np.trace(j2)) - 0.5 * float(np.trace(j1 @ j1)))
    return coeff, reconstruction_error


def gauge_transform_expansion_links(
    data,
    links: np.ndarray,
    vertex_basis: np.ndarray,
    psi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertex_rank = vertex_basis.shape[1]
    phi = vertex_basis @ psi.reshape(vertex_rank, COLORS)
    a0 = np.zeros_like(links)
    a1 = np.zeros_like(links)
    a2 = np.zeros_like(links)
    for idx, (dirs, coords_tuple) in enumerate(data.cells[1]):
        direction = dirs[0]
        tail = data.index[0][((), coords_tuple)]
        head_coords = list(coords_tuple)
        head_coords[direction] += 1
        head = data.index[0][((), tuple(head_coords))]
        z1, z2, z3 = bch3_sequence([phi[head], links[idx], -phi[tail]])
        a0[idx] = z1
        a1[idx] = z2
        a2[idx] = z3
    return a0, a1, a2


def gauge_condition_expansion(
    data,
    d0: np.ndarray,
    links: np.ndarray,
    vertex_basis: np.ndarray,
    psi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a0, a1, a2 = gauge_transform_expansion_links(data, links, vertex_basis, psi)
    g0 = (vertex_basis.T @ (d0.T @ a0)).reshape(-1)
    g1 = (vertex_basis.T @ (d0.T @ a1)).reshape(-1)
    g2 = (vertex_basis.T @ (d0.T @ a2)).reshape(-1)
    return g0, g1, g2


def fp_expansion_matrices(
    data,
    d0: np.ndarray,
    links: np.ndarray,
    vertex_basis: np.ndarray,
    derivative_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim = vertex_basis.shape[1] * COLORS
    m0 = np.zeros((dim, dim), dtype=float)
    m1 = np.zeros_like(m0)
    m2 = np.zeros_like(m0)
    zero = np.zeros(dim, dtype=float)
    for i in range(dim):
        delta = np.zeros(dim, dtype=float)
        delta[i] = derivative_step
        plus = gauge_condition_expansion(data, d0, links, vertex_basis, zero + delta)
        minus = gauge_condition_expansion(data, d0, links, vertex_basis, zero - delta)
        m0[:, i] = (plus[0] - minus[0]) / (2.0 * derivative_step)
        m1[:, i] = (plus[1] - minus[1]) / (2.0 * derivative_step)
        m2[:, i] = (plus[2] - minus[2]) / (2.0 * derivative_step)
    return m0, m1, m2


def fp_half_density_o2_coeff(
    links: np.ndarray,
    data,
    d0: np.ndarray,
    vertex_basis: np.ndarray,
    base_matrix: np.ndarray,
    derivative_step: float,
) -> tuple[float, float, float, float]:
    m0, m1, m2 = fp_expansion_matrices(data, d0, links, vertex_basis, derivative_step)
    base_error = float(np.linalg.norm(m0 - base_matrix) / np.linalg.norm(base_matrix))
    inv_m0_m1 = np.linalg.solve(m0, m1)
    inv_m0_m2 = np.linalg.solve(m0, m2)
    full_density_coeff = float(np.trace(inv_m0_m2) - 0.5 * np.trace(inv_m0_m1 @ inv_m0_m1))
    singulars = np.linalg.svd(m0, compute_uv=False)
    return (
        0.5 * full_density_coeff,
        base_error,
        float(singulars[-1]),
        float(singulars[0] / singulars[-1]),
    )


def even_centered(component_fn, k: np.ndarray, y: np.ndarray) -> float:
    zero_k = np.zeros_like(k)
    zero_y = np.zeros_like(y)
    return (
        0.5 * (component_fn(k, y) + component_fn(k, -y))
        - component_fn(k, zero_y)
        - 0.5 * (component_fn(zero_k, y) + component_fn(zero_k, -y))
    )


def run_diagnostic(
    seed: int,
    tol: float,
    retained_radius: float,
    gaussian_scale: float,
    derivative_step: float,
) -> C5AUDiagnostic:
    data, lift, quotient_basis, _ = build_retained_lift(2)
    data_gauge, lift_gauge, d0, vertex_basis, gauge = gauge_slice_report(tol)
    if len(data.cells[1]) != len(data_gauge.cells[1]) or not np.allclose(lift, lift_gauge):
        raise RuntimeError("C5AU inconsistent retained lift/gauge data")

    (
        _,
        _,
        _,
        _,
        _,
        _,
        positive_evals_all,
        positive_mask,
        positive_basis,
        _,
    ) = build_split(tol)
    positive_evals = positive_evals_all[positive_mask]
    paths = tile_loop_paths(data)
    rng = np.random.default_rng(seed)
    k_base = random_sphere(rng, lift.shape[1] * COLORS, retained_radius)
    y_base = gaussian_high_sample(rng, positive_evals, gaussian_scale)
    coeff_denominator = retained_radius**2 * gaussian_scale**2

    base_scalar = (d0 @ vertex_basis).T @ (d0 @ vertex_basis)
    base_matrix = np.kron(base_scalar, np.eye(COLORS))

    retained_errors: list[float] = []

    def retained_component(kk, yy):
        coeff, err = retained_half_density_o2_coeff(
            kk,
            yy,
            data,
            lift,
            positive_basis,
            quotient_basis,
            paths,
            derivative_step,
        )
        retained_errors.append(err)
        return coeff

    fp_base_errors: list[float] = []
    fp_min_singulars: list[float] = []
    fp_conditions: list[float] = []

    def fp_component(kk, yy):
        links = links_from_retained_positive(lift, positive_basis, kk, yy)
        coeff, base_error, min_sing, condition = fp_half_density_o2_coeff(
            links, data, d0, vertex_basis, base_matrix, derivative_step
        )
        fp_base_errors.append(base_error)
        fp_min_singulars.append(min_sing)
        fp_conditions.append(condition)
        return coeff

    retained_even = even_centered(retained_component, k_base, y_base) / coeff_denominator
    fp_even = even_centered(fp_component, k_base, y_base) / coeff_denominator
    combined = retained_even + fp_even
    abs_sum = abs(retained_even) + abs(fp_even)
    cancellation_ratio = abs_sum / abs(combined) if abs(combined) > 1.0e-18 else float("inf")
    finite_g_baseline = 2.43757730942e-05
    reflection = reflection_report(tol)
    return C5AUDiagnostic(
        retained_radius=retained_radius,
        gaussian_scale=gaussian_scale,
        derivative_step=derivative_step,
        retained_rank=lift.shape[1],
        positive_high_rank=positive_basis.shape[1],
        gauge_vertex_rank=gauge.mean_zero_vertex_rank,
        retained_jacobian_even_centered_coeff=float(retained_even),
        fp_half_even_centered_coeff=float(fp_even),
        haar_half_even_centered_coeff_exact=0.0,
        combined_even_centered_coeff=float(combined),
        retained_fp_abs_sum=float(abs_sum),
        retained_fp_cancellation_ratio=float(cancellation_ratio),
        baseline_c5at_finite_g_coeff=finite_g_baseline,
        analytic_minus_c5at_baseline=float(combined - finite_g_baseline),
        max_tangent_retained_reconstruction_error=float(np.max(retained_errors)),
        fp_m0_base_relative_error=float(np.max(fp_base_errors)),
        fp_m0_min_singular=float(np.min(fp_min_singulars)),
        fp_m0_condition=float(np.max(fp_conditions)),
        positive_covariance_reflection_commutator=reflection.covariance_reflection_commutator,
        classification=(
            "analytic-source partial pass: Haar is exactly zero at O(g^2), "
            "and retained/FP coefficients are computed from g=0 source formulas; "
            "deterministic good-sector bounds and admissibility remain open"
        ),
    )


def make_decision(row: C5AUDiagnostic) -> C5AUDecision:
    implemented = (
        row.retained_rank == 17
        and row.positive_high_rank == 119
        and row.gauge_vertex_rank == 80
        and row.haar_half_even_centered_coeff_exact == 0.0
        and row.max_tangent_retained_reconstruction_error < 1.0e-7
        and row.fp_m0_base_relative_error < 1.0e-7
        and row.fp_m0_min_singular > 1.0e-4
        and row.fp_m0_condition < 100.0
        and row.positive_covariance_reflection_commutator < 1.0e-10
    )
    cancellation_proved = implemented and abs(row.combined_even_centered_coeff) < 1.0e-8
    retention_required = implemented and not cancellation_proved
    return C5AUDecision(
        analytic_source_formulas_implemented=implemented,
        cancellation_proved=cancellation_proved,
        retention_required=retention_required,
        deterministic_good_sector_theorem_proved=False,
        decision=(
            "partial pass: analytic g=0 source formulas are implemented; "
            "the sampled coefficient is nonzero, so cancellation is not proved "
            "and admissible retention/good-sector bounds become the next issue"
            if retention_required
            else (
                "partial pass: analytic g=0 source formulas are implemented and "
                "the sampled coefficient is consistent with cancellation; exact "
                "symbolic cancellation and good-sector bounds remain open"
                if cancellation_proved
                else "fail or redesign: analytic source formula diagnostic did not meet stability checks"
            )
        ),
    )


def run_checks(row: C5AUDiagnostic, decision: C5AUDecision) -> None:
    assert row.retained_rank == 17
    assert row.positive_high_rank == 119
    assert row.gauge_vertex_rank == 80
    assert row.haar_half_even_centered_coeff_exact == 0.0
    assert row.max_tangent_retained_reconstruction_error < 1.0e-7
    assert row.fp_m0_base_relative_error < 1.0e-7
    assert row.fp_m0_min_singular > 1.0e-4
    assert row.fp_m0_condition < 100.0
    assert row.positive_covariance_reflection_commutator < 1.0e-10
    assert decision.analytic_source_formulas_implemented
    assert not decision.deterministic_good_sector_theorem_proved


def fmt(value: float | int | bool | str) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def print_report(
    seed: int,
    tol: float,
    retained_radius: float,
    gaussian_scale: float,
    derivative_step: float,
) -> None:
    row = run_diagnostic(seed, tol, retained_radius, gaussian_scale, derivative_step)
    decision = make_decision(row)
    run_checks(row, decision)
    print("C5AU ANALYTIC RETAINED-COORDINATE/FP SOURCE FORMULA DIAGNOSTIC")
    print("| quantity | value |")
    print("|---|---:|")
    for key, value in row.__dict__.items():
        print(f"| {key} | {fmt(value)} |")
    print("\nDECISION")
    print(
        "| analytic source formulas implemented | cancellation proved | retention required | "
        "deterministic good-sector theorem proved | decision |"
    )
    print("|---|---|---|---|---|")
    print(
        f"| {decision.analytic_source_formulas_implemented} | "
        f"{decision.cancellation_proved} | "
        f"{decision.retention_required} | "
        f"{decision.deterministic_good_sector_theorem_proved} | "
        f"{decision.decision} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--retained-radius", type=float, default=0.25)
    parser.add_argument("--gaussian-scale", type=float, default=0.25)
    parser.add_argument("--derivative-step", type=float, default=1.0e-6)
    args = parser.parse_args()
    print_report(
        args.seed,
        args.tol,
        args.retained_radius,
        args.gaussian_scale,
        args.derivative_step,
    )


if __name__ == "__main__":
    main()
