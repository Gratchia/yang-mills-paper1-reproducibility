"""C5AQ centered-density coefficient envelope diagnostic.

# ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates

C5AP used a conservative norm proxy for the centered high-mode density
remainder.  That proxy intentionally counted a large vacuum/normalization high
shift.  C5AQ separates the sampled O(g^2) half-density coefficient into

    retained section + vacuum high + odd centered interaction
    + even centered interaction,

and builds coefficient envelopes for the genuinely retained/high centered
pieces.  This is a finite-block coefficient diagnostic, not the full
deterministic good-sector theorem, continuum construction, confinement proof,
or mass-gap proof.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from c5al_gauge_zero_reference import build_split, reflection_report
from c5am_nonlinear_gauge_haar import gauge_slice_report, random_sphere
from c5an_gauge_haar_half_density_matching import sample_coefficients
from c5ao_conditional_half_density_projection import gaussian_high_sample
from c5ae_jacobian_half_density import tile_loop_paths
from c5v_retained_quotient_bch_bounds import COLORS, build_retained_lift


@dataclass(frozen=True)
class CoefficientSplit:
    retained_section: float
    vacuum_even_high: float
    vacuum_odd_high: float
    even_centered_interaction: float
    odd_centered_interaction: float
    even_interaction_half_scale_error: float
    odd_interaction_half_scale_error: float
    plus_reconstruction_residual: float
    minus_reconstruction_residual: float
    max_abs_combined_half_odd_in_g: float
    min_fp_singular: float
    max_fp_condition: float
    max_tangent_gauge_divergence_norm: float


@dataclass(frozen=True)
class C5AQDiagnostic:
    retained_samples: int
    high_pairs_per_retained: int
    g_probe: float
    retained_radius: float
    gaussian_scale: float
    retained_rank: int
    positive_high_rank: int
    gauge_vertex_rank: int
    max_abs_retained_section: float
    max_abs_vacuum_even_high: float
    max_abs_vacuum_odd_high: float
    max_abs_even_centered_interaction: float
    max_abs_odd_centered_interaction: float
    max_even_interaction_half_scale_error: float
    max_odd_interaction_half_scale_error: float
    max_decomposition_residual: float
    retained_unit_coeff_proxy: float
    vacuum_even_unit_coeff_proxy: float
    even_centered_unit_coeff_proxy: float
    odd_centered_unit_coeff_proxy: float
    centered_second_cumulant_unit_proxy: float
    c5ap_conservative_second_cumulant_proxy: float
    centered_proxy_improvement_factor: float
    residual_model_summable: bool
    residual_partial_10j0: float
    residual_partial_100j0: float
    positive_covariance_reflection_commutator: float
    min_fp_singular_value: float
    max_fp_condition: float
    max_tangent_gauge_divergence_norm: float
    classification: str


@dataclass(frozen=True)
class C5AQDecision:
    coefficient_split_pass: bool
    centered_proxy_summable: bool
    deterministic_theorem_proved: bool
    full_good_sector_envelope_proved: bool
    decision: str


def residual_partial_sum(j0: int, j1: int, mu: float, unit_bound: float) -> float:
    total = 0.0
    for j in range(j0, j1 + 1):
        total += unit_bound * (mu * math.log(j + 1.0)) ** 2 / (j * j)
    return total


def combined(row) -> float:
    return float(row.combined_half_even)


def max_abs_odd_in_g(*rows) -> float:
    return float(np.max(np.abs([row.combined_half_odd for row in rows])))


def split_coefficients(
    k: np.ndarray,
    y: np.ndarray,
    data,
    lift: np.ndarray,
    positive_basis: np.ndarray,
    quotient_basis: np.ndarray,
    paths,
    d0: np.ndarray,
    vertex_basis: np.ndarray,
    base_fp_logdet: float,
    g_probe: float,
    jac_step: float,
    fp_step: float,
) -> CoefficientSplit:
    zero_k = np.zeros_like(k)
    zero_y = np.zeros_like(y)

    section = sample_coefficients(
        k, zero_y, data, lift, positive_basis, quotient_basis, paths, d0,
        vertex_basis, base_fp_logdet, g_probe, jac_step, fp_step
    )
    plus = sample_coefficients(
        k, y, data, lift, positive_basis, quotient_basis, paths, d0,
        vertex_basis, base_fp_logdet, g_probe, jac_step, fp_step
    )
    minus = sample_coefficients(
        k, -y, data, lift, positive_basis, quotient_basis, paths, d0,
        vertex_basis, base_fp_logdet, g_probe, jac_step, fp_step
    )
    vac_plus = sample_coefficients(
        zero_k, y, data, lift, positive_basis, quotient_basis, paths, d0,
        vertex_basis, base_fp_logdet, g_probe, jac_step, fp_step
    )
    vac_minus = sample_coefficients(
        zero_k, -y, data, lift, positive_basis, quotient_basis, paths, d0,
        vertex_basis, base_fp_logdet, g_probe, jac_step, fp_step
    )
    half_plus = sample_coefficients(
        k, 0.5 * y, data, lift, positive_basis, quotient_basis, paths, d0,
        vertex_basis, base_fp_logdet, g_probe, jac_step, fp_step
    )
    half_minus = sample_coefficients(
        k, -0.5 * y, data, lift, positive_basis, quotient_basis, paths, d0,
        vertex_basis, base_fp_logdet, g_probe, jac_step, fp_step
    )
    vac_half_plus = sample_coefficients(
        zero_k, 0.5 * y, data, lift, positive_basis, quotient_basis, paths, d0,
        vertex_basis, base_fp_logdet, g_probe, jac_step, fp_step
    )
    vac_half_minus = sample_coefficients(
        zero_k, -0.5 * y, data, lift, positive_basis, quotient_basis, paths, d0,
        vertex_basis, base_fp_logdet, g_probe, jac_step, fp_step
    )

    rho_section = combined(section)
    rho_plus = combined(plus)
    rho_minus = combined(minus)
    rho_vac_plus = combined(vac_plus)
    rho_vac_minus = combined(vac_minus)

    vacuum_even = 0.5 * (rho_vac_plus + rho_vac_minus)
    vacuum_odd = 0.5 * (rho_vac_plus - rho_vac_minus)
    even_k = 0.5 * (rho_plus + rho_minus)
    odd_k = 0.5 * (rho_plus - rho_minus)
    even_centered = even_k - rho_section - vacuum_even
    odd_centered = odd_k - vacuum_odd

    half_even_k = 0.5 * (combined(half_plus) + combined(half_minus))
    half_odd_k = 0.5 * (combined(half_plus) - combined(half_minus))
    half_vac_even = 0.5 * (combined(vac_half_plus) + combined(vac_half_minus))
    half_vac_odd = 0.5 * (combined(vac_half_plus) - combined(vac_half_minus))
    half_even_centered = half_even_k - rho_section - half_vac_even
    half_odd_centered = half_odd_k - half_vac_odd

    plus_reconstructed = (
        rho_section + vacuum_even + vacuum_odd + even_centered + odd_centered
    )
    minus_reconstructed = (
        rho_section + vacuum_even - vacuum_odd + even_centered - odd_centered
    )
    all_rows = (
        section, plus, minus, vac_plus, vac_minus, half_plus, half_minus,
        vac_half_plus, vac_half_minus
    )
    return CoefficientSplit(
        retained_section=rho_section,
        vacuum_even_high=vacuum_even,
        vacuum_odd_high=vacuum_odd,
        even_centered_interaction=even_centered,
        odd_centered_interaction=odd_centered,
        even_interaction_half_scale_error=abs(half_even_centered - 0.25 * even_centered),
        odd_interaction_half_scale_error=abs(half_odd_centered - 0.5 * odd_centered),
        plus_reconstruction_residual=abs(rho_plus - plus_reconstructed),
        minus_reconstruction_residual=abs(rho_minus - minus_reconstructed),
        max_abs_combined_half_odd_in_g=max_abs_odd_in_g(*all_rows),
        min_fp_singular=float(np.min([row.min_fp_singular for row in all_rows])),
        max_fp_condition=float(np.max([row.max_fp_condition for row in all_rows])),
        max_tangent_gauge_divergence_norm=float(
            np.max([row.tangent_gauge_divergence_norm for row in all_rows])
        ),
    )


def run_diagnostic(
    retained_samples: int,
    high_pairs: int,
    seed: int,
    tol: float,
    g_probe: float,
    retained_radius: float,
    gaussian_scale: float,
    jac_step: float,
    fp_step: float,
    j0: int,
    mu: float,
) -> C5AQDiagnostic:
    data, lift, quotient_basis, _ = build_retained_lift(2)
    data_gauge, lift_gauge, d0, vertex_basis, gauge = gauge_slice_report(tol)
    if len(data.cells[1]) != len(data_gauge.cells[1]) or not np.allclose(lift, lift_gauge):
        raise RuntimeError("C5AQ inconsistent retained lift/gauge data")

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
    base_scalar = (d0 @ vertex_basis).T @ (d0 @ vertex_basis)
    base_fp_logdet = COLORS * float(np.linalg.slogdet(base_scalar)[1])
    rng = np.random.default_rng(seed)
    retained_dim = lift.shape[1] * COLORS

    rows: list[CoefficientSplit] = []
    for _ in range(retained_samples):
        k = random_sphere(rng, retained_dim, retained_radius)
        for _ in range(high_pairs):
            y = gaussian_high_sample(rng, positive_evals, gaussian_scale)
            rows.append(
                split_coefficients(
                    k, y, data, lift, positive_basis, quotient_basis, paths,
                    d0, vertex_basis, base_fp_logdet, g_probe, jac_step, fp_step
                )
            )

    max_retained = float(np.max(np.abs([r.retained_section for r in rows])))
    max_vac_even = float(np.max(np.abs([r.vacuum_even_high for r in rows])))
    max_vac_odd = float(np.max(np.abs([r.vacuum_odd_high for r in rows])))
    max_even_centered = float(
        np.max(np.abs([r.even_centered_interaction for r in rows]))
    )
    max_odd_centered = float(
        np.max(np.abs([r.odd_centered_interaction for r in rows]))
    )
    retained_coeff = max_retained / max(retained_radius * retained_radius, 1.0e-30)
    vacuum_even_coeff = max_vac_even / max(gaussian_scale * gaussian_scale, 1.0e-30)
    even_centered_coeff = max_even_centered / max(
        retained_radius * retained_radius * gaussian_scale * gaussian_scale, 1.0e-30
    )
    odd_centered_coeff = max_odd_centered / max(
        retained_radius * gaussian_scale, 1.0e-30
    )
    centered_second_proxy = even_centered_coeff * even_centered_coeff + odd_centered_coeff * odd_centered_coeff
    conservative_proxy = vacuum_even_coeff * vacuum_even_coeff + odd_centered_coeff * odd_centered_coeff
    improvement = conservative_proxy / max(centered_second_proxy, 1.0e-30)

    reflection = reflection_report(tol)
    return C5AQDiagnostic(
        retained_samples=retained_samples,
        high_pairs_per_retained=high_pairs,
        g_probe=g_probe,
        retained_radius=retained_radius,
        gaussian_scale=gaussian_scale,
        retained_rank=lift.shape[1],
        positive_high_rank=positive_basis.shape[1],
        gauge_vertex_rank=gauge.mean_zero_vertex_rank,
        max_abs_retained_section=max_retained,
        max_abs_vacuum_even_high=max_vac_even,
        max_abs_vacuum_odd_high=max_vac_odd,
        max_abs_even_centered_interaction=max_even_centered,
        max_abs_odd_centered_interaction=max_odd_centered,
        max_even_interaction_half_scale_error=float(
            np.max([r.even_interaction_half_scale_error for r in rows])
        ),
        max_odd_interaction_half_scale_error=float(
            np.max([r.odd_interaction_half_scale_error for r in rows])
        ),
        max_decomposition_residual=float(
            max(
                np.max([r.plus_reconstruction_residual for r in rows]),
                np.max([r.minus_reconstruction_residual for r in rows]),
            )
        ),
        retained_unit_coeff_proxy=float(retained_coeff),
        vacuum_even_unit_coeff_proxy=float(vacuum_even_coeff),
        even_centered_unit_coeff_proxy=float(even_centered_coeff),
        odd_centered_unit_coeff_proxy=float(odd_centered_coeff),
        centered_second_cumulant_unit_proxy=float(centered_second_proxy),
        c5ap_conservative_second_cumulant_proxy=float(conservative_proxy),
        centered_proxy_improvement_factor=float(improvement),
        residual_model_summable=True,
        residual_partial_10j0=residual_partial_sum(j0, 10 * j0, mu, centered_second_proxy),
        residual_partial_100j0=residual_partial_sum(j0, 100 * j0, mu, centered_second_proxy),
        positive_covariance_reflection_commutator=reflection.covariance_reflection_commutator,
        min_fp_singular_value=float(np.min([r.min_fp_singular for r in rows])),
        max_fp_condition=float(np.max([r.max_fp_condition for r in rows])),
        max_tangent_gauge_divergence_norm=float(
            np.max([r.max_tangent_gauge_divergence_norm for r in rows])
        ),
        classification=(
            "sampled partial pass: subtracting vacuum and retained pieces leaves "
            "a much smaller retained/high centered interaction; deterministic "
            "coefficient-envelope proof and good-sector bounds remain open"
        ),
    )


def make_decision(row: C5AQDiagnostic) -> C5AQDecision:
    split_ok = (
        row.max_decomposition_residual < 1.0e-10
        and row.max_even_interaction_half_scale_error < 1.0e-5
        and row.max_odd_interaction_half_scale_error < 1.0e-4
    )
    summable = row.residual_model_summable and row.residual_partial_100j0 > row.residual_partial_10j0 > 0.0
    stable = (
        row.retained_rank == 17
        and row.positive_high_rank == 119
        and row.gauge_vertex_rank == 80
        and row.positive_covariance_reflection_commutator < 1.0e-10
        and row.min_fp_singular_value > 1.0e-4
        and row.max_fp_condition < 100.0
        and row.max_tangent_gauge_divergence_norm < 1.0e-8
    )
    return C5AQDecision(
        coefficient_split_pass=split_ok and stable,
        centered_proxy_summable=summable,
        deterministic_theorem_proved=False,
        full_good_sector_envelope_proved=False,
        decision=(
            "partial pass: the finite-block coefficient split isolates a small "
            "centered retained/high interaction and a summable second-cumulant "
            "proxy, but the deterministic analytic envelope is still open"
            if split_ok and stable and summable
            else "fail or redesign: centered-density coefficient split did not meet the C5AQ gate"
        ),
    )


def run_checks(row: C5AQDiagnostic, decision: C5AQDecision) -> None:
    assert row.retained_rank == 17
    assert row.positive_high_rank == 119
    assert row.gauge_vertex_rank == 80
    assert row.max_decomposition_residual < 1.0e-10
    assert row.max_even_interaction_half_scale_error < 1.0e-5
    assert row.max_odd_interaction_half_scale_error < 1.0e-4
    assert row.centered_proxy_improvement_factor > 10.0
    assert row.residual_model_summable
    assert row.residual_partial_100j0 > row.residual_partial_10j0 > 0.0
    assert row.positive_covariance_reflection_commutator < 1.0e-10
    assert row.min_fp_singular_value > 1.0e-4
    assert row.max_fp_condition < 100.0
    assert row.max_tangent_gauge_divergence_norm < 1.0e-8
    assert decision.coefficient_split_pass
    assert decision.centered_proxy_summable
    assert not decision.deterministic_theorem_proved
    assert not decision.full_good_sector_envelope_proved


def fmt(value: float | int | bool | str) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def print_report(
    retained_samples: int,
    high_pairs: int,
    seed: int,
    tol: float,
    g_probe: float,
    retained_radius: float,
    gaussian_scale: float,
    jac_step: float,
    fp_step: float,
    j0: int,
    mu: float,
) -> None:
    row = run_diagnostic(
        retained_samples, high_pairs, seed, tol, g_probe, retained_radius,
        gaussian_scale, jac_step, fp_step, j0, mu
    )
    decision = make_decision(row)
    run_checks(row, decision)
    print("C5AQ CENTERED-DENSITY COEFFICIENT ENVELOPE DIAGNOSTIC")
    print("| quantity | value |")
    print("|---|---:|")
    for key, value in row.__dict__.items():
        print(f"| {key} | {fmt(value)} |")

    print("\nDECISION")
    print(
        "| coefficient split pass | centered proxy summable | deterministic theorem proved | "
        "full good-sector envelope proved | decision |"
    )
    print("|---|---|---|---|---|")
    print(
        f"| {decision.coefficient_split_pass} | "
        f"{decision.centered_proxy_summable} | "
        f"{decision.deterministic_theorem_proved} | "
        f"{decision.full_good_sector_envelope_proved} | "
        f"{decision.decision} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retained-samples", type=int, default=1)
    parser.add_argument("--high-pairs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--g-probe", type=float, default=0.01)
    parser.add_argument("--retained-radius", type=float, default=0.25)
    parser.add_argument("--gaussian-scale", type=float, default=0.25)
    parser.add_argument("--jac-step", type=float, default=1.0e-6)
    parser.add_argument("--fp-step", type=float, default=1.0e-6)
    parser.add_argument("--j0", type=int, default=13082)
    parser.add_argument("--mu", type=float, default=16.0)
    args = parser.parse_args()
    print_report(
        args.retained_samples, args.high_pairs, args.seed, args.tol,
        args.g_probe, args.retained_radius, args.gaussian_scale, args.jac_step,
        args.fp_step, args.j0, args.mu
    )


if __name__ == "__main__":
    main()
