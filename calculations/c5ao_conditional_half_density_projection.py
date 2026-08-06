"""C5AO conditional half-density projection diagnostic.

# ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates

C5AN found that the combined retained-coordinate/Haar/Faddeev--Popov
O(g^2) half-density has no sampled O(g) survivor, but it is not purely
retained-only before high-mode integration.  C5AO tests the next operation:

    rho_2(K,H) -> E_0[rho_2(K,H) | K] + centered conditional remainder.

The checker uses the C5AL positive-high Gaussian reference and an antithetic
finite-block probe H, -H.  This is a projection diagnostic, not a proof of the
full conditional measure theorem, locality theorem, continuum construction, or
mass gap.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from c5al_gauge_zero_reference import build_split, reflection_report
from c5am_nonlinear_gauge_haar import gauge_slice_report, random_sphere
from c5an_gauge_haar_half_density_matching import sample_coefficients
from c5ae_jacobian_half_density import tile_loop_paths
from c5v_retained_quotient_bch_bounds import COLORS, build_retained_lift


@dataclass(frozen=True)
class PairProjection:
    section_half_even: float
    pair_projection_mean: float
    zero_retained_pair_projection_mean: float
    high_projection_shift: float
    retained_nonvacuum_projection: float
    high_centered_odd_magnitude: float
    half_scale_quadratic_error: float
    max_abs_combined_half_odd_in_g: float
    min_fp_singular: float
    max_fp_condition: float
    max_tangent_gauge_divergence_norm: float


@dataclass(frozen=True)
class C5AODiagnostic:
    retained_samples: int
    high_pairs_per_retained: int
    g_probe: float
    retained_radius: float
    gaussian_scale: float
    retained_rank: int
    positive_high_rank: int
    gauge_vertex_rank: int
    positive_covariance_reflection_commutator: float
    max_abs_section_half_even: float
    max_abs_pair_projection_mean: float
    max_abs_high_projection_shift: float
    max_abs_retained_nonvacuum_projection: float
    max_abs_centered_odd_high_remainder: float
    max_half_scale_quadratic_error: float
    max_abs_combined_half_odd_in_g: float
    min_fp_singular_value: float
    max_fp_condition: float
    max_tangent_gauge_divergence_norm: float
    projection_color_invariance_inherited: bool
    projection_reflection_compatibility_inherited: bool
    classification: str


@dataclass(frozen=True)
class C5AODecision:
    no_o_g_density_survivor: bool
    projection_has_expected_quadratic_form: bool
    retained_projection_admissible_as_half_density_data: bool
    centered_remainder_summability_proved: bool
    full_conditional_projection_theorem_proved: bool
    decision: str


def gaussian_high_sample(
    rng: np.random.Generator,
    positive_evals: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Sample positive-high coordinates with covariance scale^2 H_+^{-1} per color."""
    z = rng.normal(size=(positive_evals.size, COLORS))
    return (scale * z / np.sqrt(positive_evals[:, None])).reshape(-1)


def combined_half_even(row) -> float:
    return float(row.combined_half_even)


def max_abs_odd_in_g(*rows) -> float:
    return float(np.max(np.abs([row.combined_half_odd for row in rows])))


def min_fp_singular(*rows) -> float:
    return float(np.min([row.min_fp_singular for row in rows]))


def max_fp_condition(*rows) -> float:
    return float(np.max([row.max_fp_condition for row in rows]))


def max_gauge_divergence(*rows) -> float:
    return float(np.max([row.tangent_gauge_divergence_norm for row in rows]))


def run_pair_projection(
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
) -> PairProjection:
    zero_y = np.zeros_like(y)
    zero_k = np.zeros_like(k)

    section = sample_coefficients(
        k,
        zero_y,
        data,
        lift,
        positive_basis,
        quotient_basis,
        paths,
        d0,
        vertex_basis,
        base_fp_logdet,
        g_probe,
        jac_step,
        fp_step,
    )
    plus = sample_coefficients(
        k,
        y,
        data,
        lift,
        positive_basis,
        quotient_basis,
        paths,
        d0,
        vertex_basis,
        base_fp_logdet,
        g_probe,
        jac_step,
        fp_step,
    )
    minus = sample_coefficients(
        k,
        -y,
        data,
        lift,
        positive_basis,
        quotient_basis,
        paths,
        d0,
        vertex_basis,
        base_fp_logdet,
        g_probe,
        jac_step,
        fp_step,
    )
    half_plus = sample_coefficients(
        k,
        0.5 * y,
        data,
        lift,
        positive_basis,
        quotient_basis,
        paths,
        d0,
        vertex_basis,
        base_fp_logdet,
        g_probe,
        jac_step,
        fp_step,
    )
    half_minus = sample_coefficients(
        k,
        -0.5 * y,
        data,
        lift,
        positive_basis,
        quotient_basis,
        paths,
        d0,
        vertex_basis,
        base_fp_logdet,
        g_probe,
        jac_step,
        fp_step,
    )
    zero_plus = sample_coefficients(
        zero_k,
        y,
        data,
        lift,
        positive_basis,
        quotient_basis,
        paths,
        d0,
        vertex_basis,
        base_fp_logdet,
        g_probe,
        jac_step,
        fp_step,
    )
    zero_minus = sample_coefficients(
        zero_k,
        -y,
        data,
        lift,
        positive_basis,
        quotient_basis,
        paths,
        d0,
        vertex_basis,
        base_fp_logdet,
        g_probe,
        jac_step,
        fp_step,
    )

    pair_mean = 0.5 * (combined_half_even(plus) + combined_half_even(minus))
    half_pair_mean = 0.5 * (
        combined_half_even(half_plus) + combined_half_even(half_minus)
    )
    zero_pair_mean = 0.5 * (
        combined_half_even(zero_plus) + combined_half_even(zero_minus)
    )
    section_value = combined_half_even(section)
    high_shift = pair_mean - section_value
    half_shift = half_pair_mean - section_value

    all_rows = (section, plus, minus, half_plus, half_minus, zero_plus, zero_minus)
    return PairProjection(
        section_half_even=section_value,
        pair_projection_mean=pair_mean,
        zero_retained_pair_projection_mean=zero_pair_mean,
        high_projection_shift=high_shift,
        retained_nonvacuum_projection=pair_mean - zero_pair_mean,
        high_centered_odd_magnitude=0.5
        * abs(combined_half_even(plus) - combined_half_even(minus)),
        half_scale_quadratic_error=abs(half_shift - 0.25 * high_shift),
        max_abs_combined_half_odd_in_g=max_abs_odd_in_g(*all_rows),
        min_fp_singular=min_fp_singular(*all_rows),
        max_fp_condition=max_fp_condition(*all_rows),
        max_tangent_gauge_divergence_norm=max_gauge_divergence(*all_rows),
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
) -> C5AODiagnostic:
    data, lift, quotient_basis, _ = build_retained_lift(2)
    data_gauge, lift_gauge, d0, vertex_basis, gauge = gauge_slice_report(tol)
    if len(data.cells[1]) != len(data_gauge.cells[1]) or not np.allclose(lift, lift_gauge):
        raise RuntimeError("C5AO inconsistent retained lift/gauge data")

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

    projections: list[PairProjection] = []
    for _ in range(retained_samples):
        k = random_sphere(rng, retained_dim, retained_radius)
        for _ in range(high_pairs):
            y = gaussian_high_sample(rng, positive_evals, gaussian_scale)
            projections.append(
                run_pair_projection(
                    k,
                    y,
                    data,
                    lift,
                    positive_basis,
                    quotient_basis,
                    paths,
                    d0,
                    vertex_basis,
                    base_fp_logdet,
                    g_probe,
                    jac_step,
                    fp_step,
                )
            )

    reflection = reflection_report(tol)
    classification = (
        "sampled partial pass: the antithetic C5AL Gaussian projection has no "
        "O(g) density survivor and the even high-mode projection is quadratic "
        "to the tested tolerance; the centered conditional remainder is "
        "identified but its uniform summability is not proved"
    )

    return C5AODiagnostic(
        retained_samples=retained_samples,
        high_pairs_per_retained=high_pairs,
        g_probe=g_probe,
        retained_radius=retained_radius,
        gaussian_scale=gaussian_scale,
        retained_rank=lift.shape[1],
        positive_high_rank=positive_basis.shape[1],
        gauge_vertex_rank=gauge.mean_zero_vertex_rank,
        positive_covariance_reflection_commutator=reflection.covariance_reflection_commutator,
        max_abs_section_half_even=float(
            np.max(np.abs([p.section_half_even for p in projections]))
        ),
        max_abs_pair_projection_mean=float(
            np.max(np.abs([p.pair_projection_mean for p in projections]))
        ),
        max_abs_high_projection_shift=float(
            np.max(np.abs([p.high_projection_shift for p in projections]))
        ),
        max_abs_retained_nonvacuum_projection=float(
            np.max(np.abs([p.retained_nonvacuum_projection for p in projections]))
        ),
        max_abs_centered_odd_high_remainder=float(
            np.max(np.abs([p.high_centered_odd_magnitude for p in projections]))
        ),
        max_half_scale_quadratic_error=float(
            np.max([p.half_scale_quadratic_error for p in projections])
        ),
        max_abs_combined_half_odd_in_g=float(
            np.max([p.max_abs_combined_half_odd_in_g for p in projections])
        ),
        min_fp_singular_value=float(np.min([p.min_fp_singular for p in projections])),
        max_fp_condition=float(np.max([p.max_fp_condition for p in projections])),
        max_tangent_gauge_divergence_norm=float(
            np.max([p.max_tangent_gauge_divergence_norm for p in projections])
        ),
        projection_color_invariance_inherited=True,
        projection_reflection_compatibility_inherited=(
            reflection.covariance_reflection_commutator < 1.0e-10
        ),
        classification=classification,
    )


def make_decision(row: C5AODiagnostic) -> C5AODecision:
    no_o_g = row.max_abs_combined_half_odd_in_g < 1.0e-4
    quadratic = row.max_half_scale_quadratic_error < 2.0e-4
    admissible = (
        no_o_g
        and quadratic
        and row.retained_rank == 17
        and row.positive_high_rank == 119
        and row.gauge_vertex_rank == 80
        and row.projection_color_invariance_inherited
        and row.projection_reflection_compatibility_inherited
        and row.min_fp_singular_value > 1.0e-4
        and row.max_fp_condition < 100.0
        and row.max_tangent_gauge_divergence_norm < 1.0e-8
    )
    return C5AODecision(
        no_o_g_density_survivor=no_o_g,
        projection_has_expected_quadratic_form=quadratic,
        retained_projection_admissible_as_half_density_data=admissible,
        centered_remainder_summability_proved=False,
        full_conditional_projection_theorem_proved=False,
        decision=(
            "partial pass: the sampled conditional projection creates retained "
            "half-density data plus a centered high-mode remainder, with no "
            "O(g) survivor; exact summability/locality/uniform-sector bounds "
            "for the centered remainder remain open"
            if admissible
            else "fail or redesign: the sampled conditional projection did not meet the C5AO gate"
        ),
    )


def run_checks(row: C5AODiagnostic, decision: C5AODecision) -> None:
    assert row.retained_rank == 17
    assert row.positive_high_rank == 119
    assert row.gauge_vertex_rank == 80
    assert row.max_abs_combined_half_odd_in_g < 1.0e-4
    assert row.max_half_scale_quadratic_error < 2.0e-4
    assert row.projection_color_invariance_inherited
    assert row.projection_reflection_compatibility_inherited
    assert row.min_fp_singular_value > 1.0e-4
    assert row.max_fp_condition < 100.0
    assert row.max_tangent_gauge_divergence_norm < 1.0e-8
    assert decision.no_o_g_density_survivor
    assert decision.projection_has_expected_quadratic_form
    assert decision.retained_projection_admissible_as_half_density_data
    assert not decision.centered_remainder_summability_proved
    assert not decision.full_conditional_projection_theorem_proved


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
) -> None:
    row = run_diagnostic(
        retained_samples,
        high_pairs,
        seed,
        tol,
        g_probe,
        retained_radius,
        gaussian_scale,
        jac_step,
        fp_step,
    )
    decision = make_decision(row)
    run_checks(row, decision)

    print("C5AO CONDITIONAL HALF-DENSITY PROJECTION DIAGNOSTIC")
    print("| quantity | value |")
    print("|---|---:|")
    for key, value in row.__dict__.items():
        print(f"| {key} | {fmt(value)} |")

    print("\nDECISION")
    print(
        "| no O(g) density survivor | expected quadratic projection | "
        "retained projection admissible | centered summability proved | "
        "full theorem proved | decision |"
    )
    print("|---|---|---|---|---|---|")
    print(
        f"| {decision.no_o_g_density_survivor} | "
        f"{decision.projection_has_expected_quadratic_form} | "
        f"{decision.retained_projection_admissible_as_half_density_data} | "
        f"{decision.centered_remainder_summability_proved} | "
        f"{decision.full_conditional_projection_theorem_proved} | "
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
    args = parser.parse_args()
    print_report(
        args.retained_samples,
        args.high_pairs,
        args.seed,
        args.tol,
        args.g_probe,
        args.retained_radius,
        args.gaussian_scale,
        args.jac_step,
        args.fp_step,
    )


if __name__ == "__main__":
    main()
