"""C5AL gauge-zero quotient and conditional reference diagnostic.

# ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates

C5AL follows C5AK by separating the zero-retained high sector into:

    positive-action high modes  plus  gauge-zero directions.

At tangent level the gauge-zero directions should be exactly the image of the
vertex-to-link coboundary D0.  The positive high sector then gives the natural
Gaussian conditional reference measure modulo gauge.

This checker is a finite-block tangent/reference-measure diagnostic.  It does
not prove the compact SU(2) Haar disintegration, nonlinear gauge fixing,
continuum construction, or mass gap.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from c5ak_local_holonomy_disintegration import zero_retained_basis
from c5q_4d_cubical_incidence import coboundary
from c5v_retained_quotient_bch_bounds import COLORS, build_retained_lift


@dataclass(frozen=True)
class GaugeZeroReport:
    vertices: int
    scalar_links: int
    retained_rank: int
    high_rank: int
    d0_rank: int
    expected_global_gauge_kernel: int
    high_zero_rank: int
    d1_d0_residual: float
    constraint_d0_residual: float
    gauge_vs_zero_projector_diff: float
    min_principal_cosine: float
    max_principal_angle_degrees: float


@dataclass(frozen=True)
class PositiveReferenceReport:
    scalar_positive_high_rank: int
    scalar_gauge_zero_rank: int
    color_positive_high_rank: int
    color_gauge_zero_rank: int
    min_positive_eigenvalue: float
    max_positive_eigenvalue: float
    covariance_trace_scalar: float
    covariance_symmetry_error: float
    constraint_covariance_residual: float
    gauge_covariance_residual: float
    positive_gauge_orthogonality: float


@dataclass(frozen=True)
class ReflectionReport:
    reflection_orthogonality_error: float
    hessian_reflection_commutator: float
    high_reflection_leak: float
    positive_reflection_leak: float
    gauge_reflection_leak: float
    covariance_reflection_commutator: float


@dataclass(frozen=True)
class C5ALDecision:
    gauge_zero_identified: bool
    tangent_reference_constructed: bool
    reflection_compatible_tangent: bool
    compact_haar_disintegration_proved: bool
    decision: str


def orthonormal_column_space(mat: np.ndarray, tol: float) -> tuple[np.ndarray, int]:
    if mat.size == 0:
        return np.zeros((mat.shape[0], 0)), 0
    u, s, _ = np.linalg.svd(mat, full_matrices=False)
    rank = int(np.sum(s > tol))
    return u[:, :rank], rank


def link_reflection_matrix(data) -> np.ndarray:
    """Scalar link-cochain time reflection.

    Time links reverse orientation and pick up a minus sign.  Spatial links are
    reflected without orientation reversal.
    """
    links = len(data.cells[1])
    reflection = np.zeros((links, links))
    for idx, (dirs, coords_tuple) in enumerate(data.cells[1]):
        direction = dirs[0]
        coords = list(coords_tuple)
        if direction == 0:
            coords[0] = data.n - coords[0] - 1
            sign = -1.0
        else:
            coords[0] = data.n - coords[0]
            sign = 1.0
        reflected = data.index[1][((direction,), tuple(coords))]
        reflection[reflected, idx] = sign
    return reflection


def projector(basis: np.ndarray) -> np.ndarray:
    return basis @ basis.T


def subspace_leak(operator: np.ndarray, basis: np.ndarray) -> float:
    proj = projector(basis)
    return float(np.linalg.norm((np.eye(operator.shape[0]) - proj) @ operator @ proj))


def build_split(tol: float):
    data, lift, quotient_basis, constraint = build_retained_lift(2)
    high_basis, _ = zero_retained_basis(constraint, tol)
    hessian = data.d1.T @ data.d1
    high_hessian = (high_basis.T @ hessian @ high_basis)
    high_hessian = (high_hessian + high_hessian.T) / 2.0
    evals, vecs = np.linalg.eigh(high_hessian)
    positive = evals > tol
    positive_basis = high_basis @ vecs[:, positive]
    zero_basis = high_basis @ vecs[:, ~positive]
    return data, lift, quotient_basis, constraint, high_basis, hessian, evals, positive, positive_basis, zero_basis


def gauge_zero_report(tol: float) -> GaugeZeroReport:
    (
        data,
        lift,
        _,
        constraint,
        high_basis,
        hessian,
        evals,
        positive,
        _,
        zero_basis,
    ) = build_split(tol)
    d0 = coboundary(0, data.cells[0], data.index[0], data.cells[1])
    gauge_basis, d0_rank = orthonormal_column_space(d0, tol)
    overlap = np.linalg.svd(gauge_basis.T @ zero_basis, compute_uv=False)
    min_cos = float(np.min(overlap))
    max_angle = float(np.degrees(np.arccos(np.clip(min_cos, -1.0, 1.0))))
    return GaugeZeroReport(
        vertices=len(data.cells[0]),
        scalar_links=len(data.cells[1]),
        retained_rank=lift.shape[1],
        high_rank=high_basis.shape[1],
        d0_rank=d0_rank,
        expected_global_gauge_kernel=1,
        high_zero_rank=int(np.sum(~positive)),
        d1_d0_residual=float(np.linalg.norm(data.d1 @ d0)),
        constraint_d0_residual=float(np.linalg.norm(constraint @ d0)),
        gauge_vs_zero_projector_diff=float(
            np.linalg.norm(projector(gauge_basis) - projector(zero_basis))
        ),
        min_principal_cosine=min_cos,
        max_principal_angle_degrees=max_angle,
    )


def positive_reference_report(tol: float) -> PositiveReferenceReport:
    (
        data,
        _,
        _,
        constraint,
        _,
        hessian,
        evals,
        positive,
        positive_basis,
        zero_basis,
    ) = build_split(tol)
    pos_evals = evals[positive]
    covariance = (positive_basis / pos_evals) @ positive_basis.T
    covariance = (covariance + covariance.T) / 2.0
    d0 = coboundary(0, data.cells[0], data.index[0], data.cells[1])
    gauge_basis, _ = orthonormal_column_space(d0, tol)
    return PositiveReferenceReport(
        scalar_positive_high_rank=positive_basis.shape[1],
        scalar_gauge_zero_rank=zero_basis.shape[1],
        color_positive_high_rank=positive_basis.shape[1] * COLORS,
        color_gauge_zero_rank=zero_basis.shape[1] * COLORS,
        min_positive_eigenvalue=float(np.min(pos_evals)),
        max_positive_eigenvalue=float(np.max(pos_evals)),
        covariance_trace_scalar=float(np.trace(covariance)),
        covariance_symmetry_error=float(np.linalg.norm(covariance - covariance.T)),
        constraint_covariance_residual=float(np.linalg.norm(constraint @ covariance)),
        gauge_covariance_residual=float(np.linalg.norm(gauge_basis.T @ covariance)),
        positive_gauge_orthogonality=float(np.linalg.norm(positive_basis.T @ gauge_basis)),
    )


def reflection_report(tol: float) -> ReflectionReport:
    (
        data,
        _,
        _,
        _,
        high_basis,
        hessian,
        evals,
        positive,
        positive_basis,
        zero_basis,
    ) = build_split(tol)
    reflection = link_reflection_matrix(data)
    pos_evals = evals[positive]
    covariance = (positive_basis / pos_evals) @ positive_basis.T
    covariance = (covariance + covariance.T) / 2.0
    return ReflectionReport(
        reflection_orthogonality_error=float(
            np.linalg.norm(reflection.T @ reflection - np.eye(reflection.shape[0]))
        ),
        hessian_reflection_commutator=float(
            np.linalg.norm(reflection.T @ hessian @ reflection - hessian)
        ),
        high_reflection_leak=subspace_leak(reflection, high_basis),
        positive_reflection_leak=subspace_leak(reflection, positive_basis),
        gauge_reflection_leak=subspace_leak(reflection, zero_basis),
        covariance_reflection_commutator=float(
            np.linalg.norm(reflection @ covariance @ reflection.T - covariance)
        ),
    )


def make_decision(
    gauge: GaugeZeroReport,
    positive: PositiveReferenceReport,
    reflection: ReflectionReport,
    tol: float,
) -> C5ALDecision:
    gauge_ok = (
        gauge.d0_rank == 80
        and gauge.high_zero_rank == 80
        and gauge.d1_d0_residual < 1.0e-12
        and gauge.constraint_d0_residual < 1.0e-12
        and gauge.gauge_vs_zero_projector_diff < 1.0e-8
    )
    reference_ok = (
        positive.scalar_positive_high_rank == 119
        and positive.scalar_gauge_zero_rank == 80
        and positive.constraint_covariance_residual < 1.0e-8
        and positive.gauge_covariance_residual < 1.0e-8
        and positive.positive_gauge_orthogonality < 1.0e-8
        and positive.min_positive_eigenvalue > tol
    )
    reflection_ok = (
        reflection.reflection_orthogonality_error < 1.0e-12
        and reflection.hessian_reflection_commutator < 1.0e-12
        and reflection.high_reflection_leak < 1.0e-10
        and reflection.positive_reflection_leak < 1.0e-10
        and reflection.gauge_reflection_leak < 1.0e-10
        and reflection.covariance_reflection_commutator < 1.0e-10
    )
    return C5ALDecision(
        gauge_zero_identified=gauge_ok,
        tangent_reference_constructed=reference_ok,
        reflection_compatible_tangent=reflection_ok,
        compact_haar_disintegration_proved=False,
        decision=(
            "partial pass: gauge-zero modes equal im(D0), the positive high "
            "Gaussian reference is clean at tangent level, and time reflection "
            "preserves the split; compact Haar/gauge-fixed disintegration remains open"
            if gauge_ok and reference_ok and reflection_ok
            else "fail or redesign: gauge-zero/reference/reflection tangent checks did not pass"
        ),
    )


def run_checks(
    gauge: GaugeZeroReport,
    positive: PositiveReferenceReport,
    reflection: ReflectionReport,
    decision: C5ALDecision,
) -> None:
    assert gauge.vertices == 81
    assert gauge.scalar_links == 216
    assert gauge.retained_rank == 17
    assert gauge.high_rank == 199
    assert gauge.d0_rank == 80
    assert gauge.high_zero_rank == 80
    assert gauge.d1_d0_residual < 1.0e-12
    assert gauge.constraint_d0_residual < 1.0e-12
    assert gauge.gauge_vs_zero_projector_diff < 1.0e-8
    assert positive.scalar_positive_high_rank == 119
    assert positive.scalar_gauge_zero_rank == 80
    assert positive.color_positive_high_rank == 357
    assert positive.color_gauge_zero_rank == 240
    assert positive.constraint_covariance_residual < 1.0e-8
    assert positive.gauge_covariance_residual < 1.0e-8
    assert positive.positive_gauge_orthogonality < 1.0e-8
    assert reflection.reflection_orthogonality_error < 1.0e-12
    assert reflection.hessian_reflection_commutator < 1.0e-12
    assert reflection.high_reflection_leak < 1.0e-10
    assert reflection.positive_reflection_leak < 1.0e-10
    assert reflection.gauge_reflection_leak < 1.0e-10
    assert reflection.covariance_reflection_commutator < 1.0e-10
    assert decision.gauge_zero_identified
    assert decision.tangent_reference_constructed
    assert decision.reflection_compatible_tangent
    assert not decision.compact_haar_disintegration_proved


def fmt(value: float | int | bool | str) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def print_table(title: str, row) -> None:
    print(f"\n{title}")
    print("| quantity | value |")
    print("|---|---:|")
    for key, value in row.__dict__.items():
        print(f"| {key} | {fmt(value)} |")


def print_report(tol: float) -> None:
    gauge = gauge_zero_report(tol)
    positive = positive_reference_report(tol)
    reflection = reflection_report(tol)
    decision = make_decision(gauge, positive, reflection, tol)
    run_checks(gauge, positive, reflection, decision)

    print("C5AL GAUGE-ZERO QUOTIENT AND CONDITIONAL REFERENCE DIAGNOSTIC")
    print_table("GAUGE-ZERO IDENTIFICATION", gauge)
    print_table("POSITIVE HIGH-MODE REFERENCE", positive)
    print_table("TIME-REFLECTION COMPATIBILITY", reflection)

    print("\nDECISION")
    print(
        "| gauge-zero identified | tangent reference constructed | "
        "reflection-compatible tangent | compact Haar disintegration proved | decision |"
    )
    print("|---|---|---|---|---|")
    print(
        f"| {decision.gauge_zero_identified} | "
        f"{decision.tangent_reference_constructed} | "
        f"{decision.reflection_compatible_tangent} | "
        f"{decision.compact_haar_disintegration_proved} | {decision.decision} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tol", type=float, default=1.0e-10)
    args = parser.parse_args()
    print_report(args.tol)


if __name__ == "__main__":
    main()
