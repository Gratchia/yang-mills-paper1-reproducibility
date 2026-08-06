"""C5AN gauge/Haar half-density matching diagnostic.

# ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates

C5AN combines the density pieces that were separated in C5AF, C5AK,
C5AL, and C5AM:

    * the retained-coordinate Jacobian for K_g(k,h);
    * the product link-Haar density in exponential coordinates;
    * the Faddeev--Popov determinant on the mean-zero Landau quotient.

The diagnostic extracts the O(g) and O(g^2) coefficients of the OS
half-density logarithm on the C5AK/C5AL/C5AM gauge-fixed positive-high slice.

This is a finite-block matching check.  It is not a proof of global compact
gauge fixing, continuum existence, confinement, exponential clustering, or a
mass gap.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from c5af_density_output_classification import random_so3, rotate_color
from c5ak_local_holonomy_disintegration import (
    finite_difference_retained_jacobian_k,
    retained_coordinate_from_split,
)
from c5ae_jacobian_half_density import tile_loop_paths
from c5am_nonlinear_gauge_haar import (
    finite_difference_fp_matrix,
    gauge_slice_report,
    link_haar_log_density,
    links_from_retained_positive,
    logdet_with_orientation,
    positive_high_basis,
    random_sphere,
)
from c5v_retained_quotient_bch_bounds import COLORS, build_retained_lift


@dataclass(frozen=True)
class SampleCoefficients:
    retained_half_even: float
    retained_half_odd: float
    retained_section_half_even: float
    retained_high_dependence: float
    haar_full_even: float
    haar_full_odd: float
    fp_full_even: float
    fp_full_odd: float
    combined_half_even: float
    combined_half_odd: float
    full_log_density_even: float
    min_fp_singular: float
    max_fp_condition: float
    tangent_gauge_divergence_norm: float
    link_norm: float


@dataclass(frozen=True)
class C5ANDiagnostic:
    samples: int
    g_probe: float
    coordinate_radius: float
    retained_rank: int
    positive_high_rank: int
    gauge_vertex_rank: int
    max_abs_retained_half_even: float
    max_abs_retained_high_dependence: float
    max_abs_haar_full_even: float
    max_abs_fp_full_even: float
    max_abs_combined_half_even: float
    max_abs_combined_half_odd: float
    max_quadratic_scaling_error: float
    max_color_rotation_error: float
    max_tangent_gauge_divergence_norm: float
    min_fp_singular_value: float
    max_fp_condition: float
    fp_orientation_failures: int
    classification: str


@dataclass(frozen=True)
class C5ANDecision:
    no_o_g_density_survivor: bool
    sampled_o2_density_admissible: bool
    retained_only_matching_proved: bool
    full_uniform_half_density_theorem_proved: bool
    decision: str


def retained_half_density_coefficients(
    k: np.ndarray,
    y: np.ndarray,
    lift: np.ndarray,
    positive_basis: np.ndarray,
    quotient_basis: np.ndarray,
    paths,
    n: int,
    g_probe: float,
    jac_step: float,
) -> tuple[float, float, float]:
    """Return conditional half-density even/odd coefficients and h=0 even coefficient."""
    fn = lambda kk, yy, gg: retained_coordinate_from_split(
        kk, yy, lift, positive_basis, quotient_basis, paths, n, gg
    )

    jac_plus = finite_difference_retained_jacobian_k(
        lambda kk, yy: fn(kk, yy, g_probe), k, y, jac_step
    )
    jac_minus = finite_difference_retained_jacobian_k(
        lambda kk, yy: fn(kk, yy, -g_probe), k, y, jac_step
    )
    sign_plus, logdet_plus = np.linalg.slogdet(jac_plus)
    sign_minus, logdet_minus = np.linalg.slogdet(jac_minus)
    if sign_plus <= 0.0 or sign_minus <= 0.0:
        raise RuntimeError("retained K_g(k,h) Jacobian lost positive orientation")

    retained_half_even = float((logdet_plus + logdet_minus) / (4.0 * g_probe * g_probe))
    retained_half_odd = float((logdet_plus - logdet_minus) / (4.0 * g_probe))

    zero_y = np.zeros_like(y)
    jac_section_plus = finite_difference_retained_jacobian_k(
        lambda kk, yy: fn(kk, yy, g_probe), k, zero_y, jac_step
    )
    jac_section_minus = finite_difference_retained_jacobian_k(
        lambda kk, yy: fn(kk, yy, -g_probe), k, zero_y, jac_step
    )
    sign_section_plus, logdet_section_plus = np.linalg.slogdet(jac_section_plus)
    sign_section_minus, logdet_section_minus = np.linalg.slogdet(jac_section_minus)
    if sign_section_plus <= 0.0 or sign_section_minus <= 0.0:
        raise RuntimeError("retained section Jacobian lost positive orientation")

    retained_section_half_even = float(
        (logdet_section_plus + logdet_section_minus) / (4.0 * g_probe * g_probe)
    )
    return retained_half_even, retained_half_odd, retained_section_half_even


def fp_and_haar_coefficients(
    data,
    d0: np.ndarray,
    vertex_basis: np.ndarray,
    links: np.ndarray,
    g_probe: float,
    fp_step: float,
    base_fp_logdet: float,
) -> tuple[float, float, float, float, float, float]:
    fp_plus = finite_difference_fp_matrix(data, d0, links, vertex_basis, g_probe, fp_step)
    fp_minus = finite_difference_fp_matrix(data, d0, links, vertex_basis, -g_probe, fp_step)
    fp_logdet_plus, orient_plus, singulars_plus = logdet_with_orientation(fp_plus)
    fp_logdet_minus, orient_minus, singulars_minus = logdet_with_orientation(fp_minus)
    if not orient_plus or not orient_minus:
        raise RuntimeError("FP determinant lost positive orientation")

    fp_full_even = float(
        (fp_logdet_plus + fp_logdet_minus - 2.0 * base_fp_logdet)
        / (2.0 * g_probe * g_probe)
    )
    fp_full_odd = float((fp_logdet_plus - fp_logdet_minus) / (2.0 * g_probe))
    min_fp_singular = float(min(singulars_plus[-1], singulars_minus[-1]))
    max_fp_condition = float(
        max(singulars_plus[0] / singulars_plus[-1], singulars_minus[0] / singulars_minus[-1])
    )

    haar_plus = link_haar_log_density(links, g_probe)
    haar_minus = link_haar_log_density(links, -g_probe)
    haar_full_even = float((haar_plus + haar_minus) / (2.0 * g_probe * g_probe))
    haar_full_odd = float((haar_plus - haar_minus) / (2.0 * g_probe))

    return (
        haar_full_even,
        haar_full_odd,
        fp_full_even,
        fp_full_odd,
        min_fp_singular,
        max_fp_condition,
    )


def sample_coefficients(
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
) -> SampleCoefficients:
    links = links_from_retained_positive(lift, positive_basis, k, y)
    retained_half_even, retained_half_odd, retained_section_half_even = (
        retained_half_density_coefficients(
            k,
            y,
            lift,
            positive_basis,
            quotient_basis,
            paths,
            data.n,
            g_probe,
            jac_step,
        )
    )
    (
        haar_full_even,
        haar_full_odd,
        fp_full_even,
        fp_full_odd,
        min_fp_singular,
        max_fp_condition,
    ) = fp_and_haar_coefficients(
        data, d0, vertex_basis, links, g_probe, fp_step, base_fp_logdet
    )

    # Haar and FP are full density logs.  Their OS half-density contribution is half.
    combined_half_even = retained_half_even + 0.5 * (haar_full_even + fp_full_even)
    combined_half_odd = retained_half_odd + 0.5 * (haar_full_odd + fp_full_odd)
    full_log_density_even = 2.0 * combined_half_even

    tangent_divergence = d0.T @ links
    tangent_gauge_divergence_norm = float(np.linalg.norm(vertex_basis.T @ tangent_divergence))

    return SampleCoefficients(
        retained_half_even=retained_half_even,
        retained_half_odd=retained_half_odd,
        retained_section_half_even=retained_section_half_even,
        retained_high_dependence=retained_half_even - retained_section_half_even,
        haar_full_even=haar_full_even,
        haar_full_odd=haar_full_odd,
        fp_full_even=fp_full_even,
        fp_full_odd=fp_full_odd,
        combined_half_even=combined_half_even,
        combined_half_odd=combined_half_odd,
        full_log_density_even=full_log_density_even,
        min_fp_singular=min_fp_singular,
        max_fp_condition=max_fp_condition,
        tangent_gauge_divergence_norm=tangent_gauge_divergence_norm,
        link_norm=float(np.linalg.norm(links)),
    )


def run_diagnostic(
    samples: int,
    seed: int,
    tol: float,
    g_probe: float,
    coordinate_radius: float,
    jac_step: float,
    fp_step: float,
) -> C5ANDiagnostic:
    data, lift, quotient_basis, _ = build_retained_lift(2)
    data_gauge, lift_gauge, d0, vertex_basis, gauge_report = gauge_slice_report(tol)
    if len(data.cells[1]) != len(data_gauge.cells[1]) or not np.allclose(lift, lift_gauge):
        raise RuntimeError("C5AN inconsistent retained lift/gauge data")

    positive_basis = positive_high_basis(tol)
    paths = tile_loop_paths(data)
    retained_dim = lift.shape[1] * COLORS
    positive_dim = positive_basis.shape[1] * COLORS
    base_scalar = (d0 @ vertex_basis).T @ (d0 @ vertex_basis)
    base_fp_logdet = COLORS * float(np.linalg.slogdet(base_scalar)[1])

    rng = np.random.default_rng(seed)
    rows: list[SampleCoefficients] = []
    scale_errors: list[float] = []
    rotation_errors: list[float] = []
    failures = 0

    for _ in range(samples):
        k = random_sphere(rng, retained_dim, coordinate_radius)
        y = random_sphere(rng, positive_dim, coordinate_radius)
        try:
            row = sample_coefficients(
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
            rows.append(row)

            scale = 0.5
            scaled = sample_coefficients(
                scale * k,
                scale * y,
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
            scale_errors.append(
                abs(scaled.combined_half_even - scale * scale * row.combined_half_even)
            )

            rotation = random_so3(rng)
            k_rot = rotate_color(k, lift.shape[1], rotation)
            y_rot = rotate_color(y, positive_basis.shape[1], rotation)
            rotated = sample_coefficients(
                k_rot,
                y_rot,
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
            rotation_errors.append(abs(rotated.combined_half_even - row.combined_half_even))
        except (RuntimeError, np.linalg.LinAlgError):
            failures += 1

    if not rows:
        return C5ANDiagnostic(
            samples=samples,
            g_probe=g_probe,
            coordinate_radius=coordinate_radius,
            retained_rank=lift.shape[1],
            positive_high_rank=positive_basis.shape[1],
            gauge_vertex_rank=gauge_report.mean_zero_vertex_rank,
            max_abs_retained_half_even=float("inf"),
            max_abs_retained_high_dependence=float("inf"),
            max_abs_haar_full_even=float("inf"),
            max_abs_fp_full_even=float("inf"),
            max_abs_combined_half_even=float("inf"),
            max_abs_combined_half_odd=float("inf"),
            max_quadratic_scaling_error=float("inf"),
            max_color_rotation_error=float("inf"),
            max_tangent_gauge_divergence_norm=float("inf"),
            min_fp_singular_value=0.0,
            max_fp_condition=float("inf"),
            fp_orientation_failures=failures,
            classification="fail: no successful C5AN samples",
        )

    high_dependence = np.array([r.retained_high_dependence for r in rows])
    classification = (
        "sampled partial pass: no O(g) density survivor; the combined O(g^2) "
        "coefficient is half-density/conditional-density data.  The nonzero "
        "high-dependence means it is not yet a retained-only C5AF function "
        "before conditional integration."
        if np.max(np.abs(high_dependence)) > 1.0e-10
        else "sampled partial pass: no O(g) density survivor; no high-dependence seen"
    )

    return C5ANDiagnostic(
        samples=samples,
        g_probe=g_probe,
        coordinate_radius=coordinate_radius,
        retained_rank=lift.shape[1],
        positive_high_rank=positive_basis.shape[1],
        gauge_vertex_rank=gauge_report.mean_zero_vertex_rank,
        max_abs_retained_half_even=float(np.max(np.abs([r.retained_half_even for r in rows]))),
        max_abs_retained_high_dependence=float(np.max(np.abs(high_dependence))),
        max_abs_haar_full_even=float(np.max(np.abs([r.haar_full_even for r in rows]))),
        max_abs_fp_full_even=float(np.max(np.abs([r.fp_full_even for r in rows]))),
        max_abs_combined_half_even=float(np.max(np.abs([r.combined_half_even for r in rows]))),
        max_abs_combined_half_odd=float(np.max(np.abs([r.combined_half_odd for r in rows]))),
        max_quadratic_scaling_error=float(np.max(scale_errors)),
        max_color_rotation_error=float(np.max(rotation_errors)),
        max_tangent_gauge_divergence_norm=float(
            np.max([r.tangent_gauge_divergence_norm for r in rows])
        ),
        min_fp_singular_value=float(np.min([r.min_fp_singular for r in rows])),
        max_fp_condition=float(np.max([r.max_fp_condition for r in rows])),
        fp_orientation_failures=failures,
        classification=classification,
    )


def make_decision(row: C5ANDiagnostic) -> C5ANDecision:
    no_o_g = row.max_abs_combined_half_odd < 1.0e-4 and row.fp_orientation_failures == 0
    admissible = (
        no_o_g
        and row.retained_rank == 17
        and row.positive_high_rank == 119
        and row.gauge_vertex_rank == 80
        and row.max_quadratic_scaling_error < 5.0e-4
        and row.max_color_rotation_error < 5.0e-4
        and row.max_tangent_gauge_divergence_norm < 1.0e-8
        and row.min_fp_singular_value > 1.0e-4
        and row.max_fp_condition < 100.0
    )
    return C5ANDecision(
        no_o_g_density_survivor=no_o_g,
        sampled_o2_density_admissible=admissible,
        retained_only_matching_proved=False,
        full_uniform_half_density_theorem_proved=False,
        decision=(
            "partial pass: sampled combined half-density has no O(g) survivor "
            "and has admissible O(g^2) symmetry/scaling fingerprints, but exact "
            "retained-only matching after conditional integration, locality, "
            "reflection placement, and uniform good-sector bounds remain open"
            if admissible
            else "fail or redesign: sampled combined density did not meet the C5AN gate"
        ),
    )


def run_checks(row: C5ANDiagnostic, decision: C5ANDecision) -> None:
    assert row.retained_rank == 17
    assert row.positive_high_rank == 119
    assert row.gauge_vertex_rank == 80
    assert row.fp_orientation_failures == 0
    assert row.max_abs_combined_half_odd < 1.0e-4
    assert row.max_quadratic_scaling_error < 5.0e-4
    assert row.max_color_rotation_error < 5.0e-4
    assert row.max_tangent_gauge_divergence_norm < 1.0e-8
    assert row.min_fp_singular_value > 1.0e-4
    assert row.max_fp_condition < 100.0
    assert decision.no_o_g_density_survivor
    assert decision.sampled_o2_density_admissible
    assert not decision.retained_only_matching_proved
    assert not decision.full_uniform_half_density_theorem_proved


def fmt(value: float | int | bool | str) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def print_report(
    samples: int,
    seed: int,
    tol: float,
    g_probe: float,
    coordinate_radius: float,
    jac_step: float,
    fp_step: float,
) -> None:
    row = run_diagnostic(samples, seed, tol, g_probe, coordinate_radius, jac_step, fp_step)
    decision = make_decision(row)
    run_checks(row, decision)

    print("C5AN GAUGE/HAAR HALF-DENSITY MATCHING DIAGNOSTIC")
    print("| quantity | value |")
    print("|---|---:|")
    for key, value in row.__dict__.items():
        print(f"| {key} | {fmt(value)} |")

    print("\nDECISION")
    print(
        "| no O(g) density survivor | sampled O(g^2) density admissible | "
        "retained-only matching proved | full uniform theorem proved | decision |"
    )
    print("|---|---|---|---|---|")
    print(
        f"| {decision.no_o_g_density_survivor} | "
        f"{decision.sampled_o2_density_admissible} | "
        f"{decision.retained_only_matching_proved} | "
        f"{decision.full_uniform_half_density_theorem_proved} | "
        f"{decision.decision} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--g-probe", type=float, default=0.01)
    parser.add_argument("--coordinate-radius", type=float, default=0.25)
    parser.add_argument("--jac-step", type=float, default=1.0e-6)
    parser.add_argument("--fp-step", type=float, default=1.0e-6)
    args = parser.parse_args()
    print_report(
        args.samples,
        args.seed,
        args.tol,
        args.g_probe,
        args.coordinate_radius,
        args.jac_step,
        args.fp_step,
    )


if __name__ == "__main__":
    main()
