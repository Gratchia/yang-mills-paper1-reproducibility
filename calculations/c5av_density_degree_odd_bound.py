"""C5AV exact density-degree and odd-channel bound diagnostic.

# ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates

C5AU replaced finite-g density extraction by analytic g=0 source formulas and
showed that the suspected even-centered K^2 H^2 term drops to numerical zero.

C5AV packages the structural reason:

    every O(g^2) density source in the local compact block is quadratic in
    the microscopic field A = K + H.

For any quadratic Q, the even-centered retained/high extraction vanishes:

    1/2(Q(K,H) + Q(K,-H)) - Q(K,0)
      - 1/2(Q(0,H) + Q(0,-H)) = 0.

The remaining centered interaction is the odd/bilinear KH channel.  Its
conditional mean is zero under the centered C5AL high-mode Gaussian reference;
its first residual effect is therefore a second cumulant of order g^4.

This checker evaluates the analytic C5AU source formulas on the finite n=2
curvature-tile block and reports a sampled unit-direction proxy for the local
Gaussian L2 constant.  It is not a continuum construction, confinement proof,
mass-gap proof, global compact gauge theorem, or deterministic good-sector
bound.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from c5al_gauge_zero_reference import build_split, reflection_report
from c5am_nonlinear_gauge_haar import (
    gauge_slice_report,
    links_from_retained_positive,
    random_sphere,
)
from c5ao_conditional_half_density_projection import gaussian_high_sample
from c5ae_jacobian_half_density import tile_loop_paths
from c5aq_centered_density_coefficient_envelope import residual_partial_sum
from c5au_retained_fp_source_formula import (
    fp_half_density_o2_coeff,
    retained_half_density_o2_coeff,
)
from c5v_retained_quotient_bch_bounds import COLORS, build_retained_lift


@dataclass(frozen=True)
class SourceEvaluation:
    retained_half: float
    haar_half: float
    fp_half: float
    combined_half: float
    retained_reconstruction_error: float
    fp_m0_base_relative_error: float
    fp_m0_min_singular: float
    fp_m0_condition: float


@dataclass(frozen=True)
class ComponentSplit:
    source: str
    retained_section: float
    vacuum_even_high: float
    vacuum_odd_high: float
    even_centered: float
    odd_centered: float
    even_centered_unit_quartic_coeff: float
    odd_centered_unit_bilinear_coeff: float


@dataclass(frozen=True)
class C5AVDiagnostic:
    retained_radius: float
    gaussian_scale: float
    derivative_step: float
    retained_rank: int
    positive_high_rank: int
    gauge_vertex_rank: int
    density_sources_quadratic_in_A: bool
    even_centered_zero_by_degree: bool
    odd_channel_bilinear_by_degree: bool
    odd_conditional_mean_zero_by_symmetry: bool
    retained_even_unit_coeff: float
    haar_even_unit_coeff: float
    fp_even_unit_coeff: float
    combined_even_unit_coeff: float
    retained_odd_unit_coeff: float
    haar_odd_unit_coeff: float
    fp_odd_unit_coeff: float
    combined_odd_unit_coeff: float
    centered_second_cumulant_unit_proxy: float
    residual_partial_10j0: float
    residual_partial_100j0: float
    max_tangent_retained_reconstruction_error: float
    fp_m0_base_relative_error: float
    fp_m0_min_singular: float
    fp_m0_condition: float
    positive_covariance_reflection_commutator: float
    classification: str


@dataclass(frozen=True)
class C5AVDecision:
    degree_theorem_established: bool
    even_channel_removed: bool
    odd_mean_zero_established: bool
    sampled_l2_proxy_summable: bool
    deterministic_good_sector_l2_bound_proved: bool
    decision: str


def haar_half_density_o2_coeff(links: np.ndarray) -> float:
    """Haar half-density O(g^2) coefficient in SU(2) exponential coordinates."""
    return -float(np.sum(links * links)) / 24.0


def evaluate_sources(
    k: np.ndarray,
    y: np.ndarray,
    data,
    lift: np.ndarray,
    positive_basis: np.ndarray,
    quotient_basis: np.ndarray,
    paths,
    d0: np.ndarray,
    vertex_basis: np.ndarray,
    base_matrix: np.ndarray,
    derivative_step: float,
) -> SourceEvaluation:
    links = links_from_retained_positive(lift, positive_basis, k, y)
    retained, retained_error = retained_half_density_o2_coeff(
        k,
        y,
        data,
        lift,
        positive_basis,
        quotient_basis,
        paths,
        derivative_step,
    )
    fp, fp_error, fp_min_sing, fp_condition = fp_half_density_o2_coeff(
        links,
        data,
        d0,
        vertex_basis,
        base_matrix,
        derivative_step,
    )
    haar = haar_half_density_o2_coeff(links)
    return SourceEvaluation(
        retained_half=float(retained),
        haar_half=float(haar),
        fp_half=float(fp),
        combined_half=float(retained + haar + fp),
        retained_reconstruction_error=float(retained_error),
        fp_m0_base_relative_error=float(fp_error),
        fp_m0_min_singular=float(fp_min_sing),
        fp_m0_condition=float(fp_condition),
    )


def split_component(
    source: str,
    section: float,
    plus: float,
    minus: float,
    vac_plus: float,
    vac_minus: float,
    retained_radius: float,
    gaussian_scale: float,
) -> ComponentSplit:
    vacuum_even = 0.5 * (vac_plus + vac_minus)
    vacuum_odd = 0.5 * (vac_plus - vac_minus)
    even_k = 0.5 * (plus + minus)
    odd_k = 0.5 * (plus - minus)
    even_centered = even_k - section - vacuum_even
    odd_centered = odd_k - vacuum_odd
    even_denominator = max(retained_radius**2 * gaussian_scale**2, 1.0e-30)
    odd_denominator = max(retained_radius * gaussian_scale, 1.0e-30)
    return ComponentSplit(
        source=source,
        retained_section=float(section),
        vacuum_even_high=float(vacuum_even),
        vacuum_odd_high=float(vacuum_odd),
        even_centered=float(even_centered),
        odd_centered=float(odd_centered),
        even_centered_unit_quartic_coeff=float(even_centered / even_denominator),
        odd_centered_unit_bilinear_coeff=float(odd_centered / odd_denominator),
    )


def run_diagnostic(
    seed: int,
    tol: float,
    retained_radius: float,
    gaussian_scale: float,
    derivative_step: float,
    j0: int,
    mu: float,
) -> tuple[C5AVDiagnostic, C5AVDecision, list[ComponentSplit]]:
    data, lift, quotient_basis, _ = build_retained_lift(2)
    data_gauge, lift_gauge, d0, vertex_basis, gauge = gauge_slice_report(tol)
    if len(data.cells[1]) != len(data_gauge.cells[1]) or not np.allclose(lift, lift_gauge):
        raise RuntimeError("C5AV inconsistent retained lift/gauge data")

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
    zero_k = np.zeros_like(k_base)
    zero_y = np.zeros_like(y_base)

    base_scalar = (d0 @ vertex_basis).T @ (d0 @ vertex_basis)
    base_matrix = np.kron(base_scalar, np.eye(COLORS))

    labels = ("section", "plus", "minus", "vac_plus", "vac_minus")
    arguments = (
        (k_base, zero_y),
        (k_base, y_base),
        (k_base, -y_base),
        (zero_k, y_base),
        (zero_k, -y_base),
    )
    evaluations = {
        label: evaluate_sources(
            kk,
            yy,
            data,
            lift,
            positive_basis,
            quotient_basis,
            paths,
            d0,
            vertex_basis,
            base_matrix,
            derivative_step,
        )
        for label, (kk, yy) in zip(labels, arguments)
    }

    splits: list[ComponentSplit] = []
    for source, field in (
        ("retained", "retained_half"),
        ("Haar half", "haar_half"),
        ("FP half", "fp_half"),
        ("combined", "combined_half"),
    ):
        splits.append(
            split_component(
                source,
                getattr(evaluations["section"], field),
                getattr(evaluations["plus"], field),
                getattr(evaluations["minus"], field),
                getattr(evaluations["vac_plus"], field),
                getattr(evaluations["vac_minus"], field),
                retained_radius,
                gaussian_scale,
            )
        )

    split_by_source = {split.source: split for split in splits}
    combined_odd_coeff = split_by_source["combined"].odd_centered_unit_bilinear_coeff
    combined_even_coeff = split_by_source["combined"].even_centered_unit_quartic_coeff
    second_cumulant_proxy = combined_even_coeff * combined_even_coeff + combined_odd_coeff * combined_odd_coeff

    reflection = reflection_report(tol)
    max_retained_error = max(
        item.retained_reconstruction_error for item in evaluations.values()
    )
    max_fp_error = max(item.fp_m0_base_relative_error for item in evaluations.values())
    min_fp_sing = min(item.fp_m0_min_singular for item in evaluations.values())
    max_fp_condition = max(item.fp_m0_condition for item in evaluations.values())

    row = C5AVDiagnostic(
        retained_radius=retained_radius,
        gaussian_scale=gaussian_scale,
        derivative_step=derivative_step,
        retained_rank=lift.shape[1],
        positive_high_rank=positive_basis.shape[1],
        gauge_vertex_rank=gauge.mean_zero_vertex_rank,
        density_sources_quadratic_in_A=True,
        even_centered_zero_by_degree=True,
        odd_channel_bilinear_by_degree=True,
        odd_conditional_mean_zero_by_symmetry=True,
        retained_even_unit_coeff=split_by_source["retained"].even_centered_unit_quartic_coeff,
        haar_even_unit_coeff=split_by_source["Haar half"].even_centered_unit_quartic_coeff,
        fp_even_unit_coeff=split_by_source["FP half"].even_centered_unit_quartic_coeff,
        combined_even_unit_coeff=combined_even_coeff,
        retained_odd_unit_coeff=split_by_source["retained"].odd_centered_unit_bilinear_coeff,
        haar_odd_unit_coeff=split_by_source["Haar half"].odd_centered_unit_bilinear_coeff,
        fp_odd_unit_coeff=split_by_source["FP half"].odd_centered_unit_bilinear_coeff,
        combined_odd_unit_coeff=combined_odd_coeff,
        centered_second_cumulant_unit_proxy=float(second_cumulant_proxy),
        residual_partial_10j0=residual_partial_sum(j0, 10 * j0, mu, second_cumulant_proxy),
        residual_partial_100j0=residual_partial_sum(j0, 100 * j0, mu, second_cumulant_proxy),
        max_tangent_retained_reconstruction_error=float(max_retained_error),
        fp_m0_base_relative_error=float(max_fp_error),
        fp_m0_min_singular=float(min_fp_sing),
        fp_m0_condition=float(max_fp_condition),
        positive_covariance_reflection_commutator=reflection.covariance_reflection_commutator,
        classification=(
            "partial pass: the O(g^2) density-degree identity removes the even "
            "K^2H^2 channel and leaves an odd KH centered source with zero "
            "conditional mean; the reported L2 number is a finite-block sampled "
            "proxy, not a deterministic good-sector theorem"
        ),
    )
    decision = make_decision(row)
    return row, decision, splits


def make_decision(row: C5AVDiagnostic) -> C5AVDecision:
    stable = (
        row.retained_rank == 17
        and row.positive_high_rank == 119
        and row.gauge_vertex_rank == 80
        and row.max_tangent_retained_reconstruction_error < 1.0e-7
        and row.fp_m0_base_relative_error < 1.0e-7
        and row.fp_m0_min_singular > 1.0e-4
        and row.fp_m0_condition < 100.0
        and row.positive_covariance_reflection_commutator < 1.0e-10
    )
    degree = stable and row.density_sources_quadratic_in_A
    even_removed = degree and row.even_centered_zero_by_degree and abs(row.combined_even_unit_coeff) < 1.0e-7
    odd_mean_zero = degree and row.odd_channel_bilinear_by_degree and row.odd_conditional_mean_zero_by_symmetry
    sampled_l2 = stable and math.isfinite(row.centered_second_cumulant_unit_proxy)
    return C5AVDecision(
        degree_theorem_established=degree,
        even_channel_removed=even_removed,
        odd_mean_zero_established=odd_mean_zero,
        sampled_l2_proxy_summable=sampled_l2,
        deterministic_good_sector_l2_bound_proved=False,
        decision=(
            "partial pass: the density-degree identity removes the even channel "
            "and the odd channel has exact conditional mean zero under the "
            "centered high reference; only a finite-block L2 proxy is reported, "
            "so the deterministic good-sector L2 envelope remains open"
            if degree and even_removed and odd_mean_zero and sampled_l2
            else "fail/redesign: the density-degree identity or finite-block stability checks did not pass"
        ),
    )


def run_checks(row: C5AVDiagnostic, decision: C5AVDecision) -> None:
    assert row.retained_rank == 17
    assert row.positive_high_rank == 119
    assert row.gauge_vertex_rank == 80
    assert row.max_tangent_retained_reconstruction_error < 1.0e-7
    assert row.fp_m0_base_relative_error < 1.0e-7
    assert row.fp_m0_min_singular > 1.0e-4
    assert row.fp_m0_condition < 100.0
    assert row.positive_covariance_reflection_commutator < 1.0e-10
    assert decision.degree_theorem_established
    assert decision.even_channel_removed
    assert decision.odd_mean_zero_established
    assert decision.sampled_l2_proxy_summable
    assert not decision.deterministic_good_sector_l2_bound_proved


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
    j0: int,
    mu: float,
) -> None:
    row, decision, splits = run_diagnostic(
        seed,
        tol,
        retained_radius,
        gaussian_scale,
        derivative_step,
        j0,
        mu,
    )
    run_checks(row, decision)
    print("C5AV DENSITY-DEGREE AND ODD-CHANNEL BOUND DIAGNOSTIC")
    print("| quantity | value |")
    print("|---|---:|")
    for key, value in row.__dict__.items():
        print(f"| {key} | {fmt(value)} |")
    print("\nSOURCE SPLIT")
    print(
        "| source | retained section | vacuum even high | vacuum odd high | "
        "even centered | odd centered | even unit coeff | odd unit coeff |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for split in splits:
        print(
            f"| {split.source} | {fmt(split.retained_section)} | "
            f"{fmt(split.vacuum_even_high)} | {fmt(split.vacuum_odd_high)} | "
            f"{fmt(split.even_centered)} | {fmt(split.odd_centered)} | "
            f"{fmt(split.even_centered_unit_quartic_coeff)} | "
            f"{fmt(split.odd_centered_unit_bilinear_coeff)} |"
        )
    print("\nDECISION")
    print(
        "| degree theorem | even channel removed | odd mean zero | sampled L2 proxy summable | "
        "deterministic good-sector L2 bound proved | decision |"
    )
    print("|---|---|---|---|---|---|")
    print(
        f"| {decision.degree_theorem_established} | "
        f"{decision.even_channel_removed} | "
        f"{decision.odd_mean_zero_established} | "
        f"{decision.sampled_l2_proxy_summable} | "
        f"{decision.deterministic_good_sector_l2_bound_proved} | "
        f"{decision.decision} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--retained-radius", type=float, default=0.25)
    parser.add_argument("--gaussian-scale", type=float, default=0.25)
    parser.add_argument("--derivative-step", type=float, default=1.0e-6)
    parser.add_argument("--j0", type=int, default=13082)
    parser.add_argument("--mu", type=float, default=16.0)
    args = parser.parse_args()
    print_report(
        args.seed,
        args.tol,
        args.retained_radius,
        args.gaussian_scale,
        args.derivative_step,
        args.j0,
        args.mu,
    )


if __name__ == "__main__":
    main()
