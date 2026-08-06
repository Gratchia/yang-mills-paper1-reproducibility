"""C5AY reflection-covariant retained-coordinate redesign.

# ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates

C5AX localized the retained-density reflection obstruction: the linear
curvature-tile quotient is reflection-compatible, but the nonlinear based
holonomy-log coordinate is not, because time-reflected face loops become
cyclically shifted inverse loops.

C5AY tests the least disruptive repair: rebase every face loop by a uniform
one-segment cyclic shift.  Since cyclic shifts do not change the tangent
linear circulation, this candidate should preserve the 17-dimensional retained
quotient.  But at nonlinear level it changes the based log, and may remove the
C5AX hidden conjugation defect.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from c5ae_jacobian_half_density import tile_loop_paths
from c5al_gauge_zero_reference import link_reflection_matrix
from c5aw_odd_bilinear_envelope import (
    build_all_matrices,
    color_lift_matrix,
    covariance_l2_squared,
    retained_odd_matrix,
)
from c5ax_reflection_retained_density import (
    exact_retained_coordinate_from_links,
    path_row_matrix,
    reflected_same_path_sequence,
    reversed_path,
)
from c5r_curvature_tile_schur import analyze
from c5v_retained_quotient_bch_bounds import COLORS, build_retained_lift


@dataclass(frozen=True)
class ClosureReport:
    label: str
    cyclic_shift: int
    exact_same_order_count: int
    exact_inverse_count: int
    unmatched_count: int


@dataclass(frozen=True)
class LinearPreservationReport:
    original_row_rank: int
    shifted_row_rank: int
    row_difference_norm: float
    quotient_reflection_leak: float
    lift_vs_tile_reflection_relative_difference: float
    retained_rank: int
    bianchi_relations: int
    density_min_hessian: float
    density_max_hessian: float


@dataclass(frozen=True)
class EquivarianceProbe:
    g: float
    relative_error: float
    absolute_error: float
    retained_norm: float


@dataclass(frozen=True)
class DensityReflectionRow:
    source: str
    frobenius_norm: float
    operator_norm: float
    covariance_l2_constant_squared: float
    covariance_l2_constant: float
    even_reflection_residual: float
    anti_reflection_residual: float


@dataclass(frozen=True)
class C5AYDecision:
    stem_basepoint_repair_passes_finite_block_test: bool
    wilson_trace_replacement_needed_now: bool
    full_common_stem_theorem_proved: bool
    decision: str


def cyclic_shift_path(path: list[tuple[int, float]], shift: int) -> list[tuple[int, float]]:
    length = len(path)
    return [path[(idx + shift) % length] for idx in range(length)]


def shifted_paths(
    paths: list[list[tuple[int, float]]], shift: int
) -> list[list[tuple[int, float]]]:
    return [cyclic_shift_path(path, shift) for path in paths]


def exact_closure_report(
    label: str,
    paths: list[list[tuple[int, float]]],
    reflection: np.ndarray,
    cyclic_shift: int,
) -> ClosureReport:
    same_count = 0
    inverse_count = 0
    unmatched = 0
    for path in paths:
        reflected = reflected_same_path_sequence(path, reflection)
        matched = False
        for reference in paths:
            if reflected == reference:
                same_count += 1
                matched = True
                break
            if reflected == reversed_path(reference):
                inverse_count += 1
                matched = True
                break
        if not matched:
            unmatched += 1
    return ClosureReport(
        label=label,
        cyclic_shift=cyclic_shift,
        exact_same_order_count=same_count,
        exact_inverse_count=inverse_count,
        unmatched_count=unmatched,
    )


def valid_uniform_shifts(
    paths: list[list[tuple[int, float]]], reflection: np.ndarray
) -> list[int]:
    length = len(paths[0])
    out: list[int] = []
    for shift in range(length):
        report = exact_closure_report(
            f"uniform shift {shift}", shifted_paths(paths, shift), reflection, shift
        )
        if report.unmatched_count == 0:
            out.append(shift)
    return out


def quotient_reflection(
    data,
    paths: list[list[tuple[int, float]]],
    quotient_basis: np.ndarray,
    reflection: np.ndarray,
) -> np.ndarray:
    rows = path_row_matrix(data, paths)
    tile_reflection = (rows @ reflection) @ np.linalg.pinv(rows)
    return quotient_basis.T @ tile_reflection @ quotient_basis


def linear_preservation_report(
    data,
    lift: np.ndarray,
    quotient_basis: np.ndarray,
    original_paths: list[list[tuple[int, float]]],
    shifted: list[list[tuple[int, float]]],
    reflection: np.ndarray,
) -> LinearPreservationReport:
    original_rows = path_row_matrix(data, original_paths)
    shifted_rows = path_row_matrix(data, shifted)
    tile_reflection = (shifted_rows @ reflection) @ np.linalg.pinv(shifted_rows)
    quotient_reflection = quotient_basis.T @ tile_reflection @ quotient_basis
    lift_reflection = np.linalg.pinv(lift) @ reflection @ lift
    quotient_leak = np.linalg.norm(
        (np.eye(quotient_basis.shape[0]) - quotient_basis @ quotient_basis.T)
        @ tile_reflection
        @ quotient_basis
    )
    hessian = analyze(data.n)
    diff = np.linalg.norm(lift_reflection - quotient_reflection) / max(
        np.linalg.norm(quotient_reflection), 1.0e-30
    )
    return LinearPreservationReport(
        original_row_rank=int(np.linalg.matrix_rank(original_rows)),
        shifted_row_rank=int(np.linalg.matrix_rank(shifted_rows)),
        row_difference_norm=float(np.linalg.norm(original_rows - shifted_rows)),
        quotient_reflection_leak=float(quotient_leak),
        lift_vs_tile_reflection_relative_difference=float(diff),
        retained_rank=hessian.retained_rank,
        bianchi_relations=hessian.bianchi_relations,
        density_min_hessian=hessian.density_min_hessian,
        density_max_hessian=hessian.density_max_hessian,
    )


def nonlinear_equivariance_probes(
    data,
    lift: np.ndarray,
    positive_basis: np.ndarray,
    quotient_basis: np.ndarray,
    paths: list[list[tuple[int, float]]],
    link_reflection: np.ndarray,
    quotient_reflection_matrix: np.ndarray,
    seed: int,
    amplitude: float,
    g_values: tuple[float, ...],
) -> list[EquivarianceProbe]:
    rng = np.random.default_rng(seed)
    retained_color = color_lift_matrix(lift)
    positive_color = color_lift_matrix(positive_basis)
    link_color_reflection = np.kron(link_reflection, np.eye(COLORS))
    quotient_color_reflection = np.kron(quotient_reflection_matrix, np.eye(COLORS))

    k = rng.normal(size=retained_color.shape[1]) * amplitude
    h = rng.normal(size=positive_color.shape[1]) * amplitude
    field = (retained_color @ k + positive_color @ h).reshape(
        len(data.cells[1]), COLORS
    )
    reflected_field = (link_color_reflection @ field.reshape(-1)).reshape(
        len(data.cells[1]), COLORS
    )

    probes: list[EquivarianceProbe] = []
    for g in g_values:
        retained = exact_retained_coordinate_from_links(
            field, quotient_basis, paths, data.n, g
        )
        retained_reflected = exact_retained_coordinate_from_links(
            reflected_field, quotient_basis, paths, data.n, g
        )
        defect = retained_reflected - quotient_color_reflection @ retained
        probes.append(
            EquivarianceProbe(
                g=g,
                relative_error=float(
                    np.linalg.norm(defect) / max(np.linalg.norm(retained), 1.0e-30)
                ),
                absolute_error=float(np.linalg.norm(defect)),
                retained_norm=float(np.linalg.norm(retained)),
            )
        )
    return probes


def reflection_residuals(
    matrix: np.ndarray,
    retained_reflection: np.ndarray,
    positive_reflection: np.ndarray,
) -> tuple[float, float]:
    transformed = retained_reflection.T @ matrix @ positive_reflection
    norm = max(np.linalg.norm(matrix), 1.0e-30)
    return (
        float(np.linalg.norm(transformed - matrix) / norm),
        float(np.linalg.norm(transformed + matrix) / norm),
    )


def density_reflection_rows(
    data,
    lift: np.ndarray,
    positive_basis: np.ndarray,
    positive_evals: np.ndarray,
    quotient_basis: np.ndarray,
    shifted: list[list[tuple[int, float]]],
    link_reflection: np.ndarray,
    quotient_reflection_matrix: np.ndarray,
    old_matrices: dict[str, np.ndarray],
) -> list[DensityReflectionRow]:
    shifted_retained, shifted_trace_j2, shifted_j1_square = retained_odd_matrix(
        data, lift, positive_basis, quotient_basis, shifted
    )
    shifted_combined = (
        shifted_retained + old_matrices["Haar half"] + old_matrices["FP half"]
    )

    link_color_reflection = np.kron(link_reflection, np.eye(COLORS))
    retained_reflection = np.kron(quotient_reflection_matrix, np.eye(COLORS))
    positive_reflection = (
        color_lift_matrix(positive_basis).T
        @ link_color_reflection
        @ color_lift_matrix(positive_basis)
    )

    matrices = [
        ("old retained, pre-C5AY", old_matrices["retained"]),
        ("shifted retained", shifted_retained),
        ("shifted retained trace J2", shifted_trace_j2),
        ("shifted retained -J1^2/4", shifted_j1_square),
        ("Haar half", old_matrices["Haar half"]),
        ("FP half", old_matrices["FP half"]),
        ("old combined, pre-C5AY", old_matrices["combined"]),
        ("shifted combined", shifted_combined),
    ]
    rows: list[DensityReflectionRow] = []
    for source, matrix in matrices:
        even, anti = reflection_residuals(
            matrix, retained_reflection, positive_reflection
        )
        cov_sq = covariance_l2_squared(matrix, positive_evals)
        rows.append(
            DensityReflectionRow(
                source=source,
                frobenius_norm=float(np.linalg.norm(matrix)),
                operator_norm=float(np.linalg.svd(matrix, compute_uv=False)[0]),
                covariance_l2_constant_squared=cov_sq,
                covariance_l2_constant=float(np.sqrt(max(cov_sq, 0.0))),
                even_reflection_residual=even,
                anti_reflection_residual=anti,
            )
        )
    return rows


def make_decision(
    repaired_closure: ClosureReport,
    linear: LinearPreservationReport,
    probes: list[EquivarianceProbe],
    density_rows: list[DensityReflectionRow],
) -> C5AYDecision:
    shifted_retained = next(row for row in density_rows if row.source == "shifted retained")
    shifted_combined = next(row for row in density_rows if row.source == "shifted combined")
    finite_pass = (
        repaired_closure.unmatched_count == 0
        and linear.row_difference_norm < 1.0e-12
        and linear.shifted_row_rank == 17
        and linear.retained_rank == 17
        and max(probe.relative_error for probe in probes) < 1.0e-10
        and shifted_retained.even_reflection_residual < 1.0e-10
        and shifted_combined.even_reflection_residual < 1.0e-10
    )
    if finite_pass:
        decision = (
            "C5AY finite-block pass with theorem debt: a uniform one-segment "
            "cyclic rebase of the 24 face loops preserves the linear "
            "17-dimensional curvature quotient and restores exact nonlinear "
            "reflection covariance in the tested holonomy-log coordinate.  The "
            "retained density operator becomes reflection-compatible to "
            "roundoff.  The Wilson-trace/conjugacy-class replacement is not "
            "needed at this stage.  Remaining debt: formulate the full "
            "gauge-covariant common-stem implementation and regression-check "
            "the normalized C5AG shell package with the shifted coordinate."
        )
    else:
        decision = (
            "C5AY partial/fail: the simple cyclic basepoint repair did not "
            "simultaneously preserve rank, nonlinear reflection covariance, "
            "and density reflection placement; Wilson-trace/conjugacy variables "
            "or a larger retained-variable redesign must be tested next."
        )
    return C5AYDecision(
        stem_basepoint_repair_passes_finite_block_test=finite_pass,
        wilson_trace_replacement_needed_now=not finite_pass,
        full_common_stem_theorem_proved=False,
        decision=decision,
    )


def markdown_report(
    valid_shifts: list[int],
    original_closure: ClosureReport,
    repaired_closure: ClosureReport,
    linear: LinearPreservationReport,
    probes: list[EquivarianceProbe],
    density_rows: list[DensityReflectionRow],
    decision: C5AYDecision,
) -> str:
    lines: list[str] = []
    lines.append("# C5AY Reflection-Covariant Retained Coordinate Diagnostic")
    lines.append("")
    lines.append("## Candidate")
    lines.append("")
    lines.append(
        "The tested repair is a uniform one-segment cyclic rebase of every retained face loop."
    )
    lines.append("")
    lines.append(f"Valid exact uniform cyclic shifts: `{valid_shifts}`.")
    lines.append("")
    lines.append("## Exact ordered-path reflection closure")
    lines.append("")
    lines.append("| candidate | cyclic shift | exact same-order | exact inverse | unmatched |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in (original_closure, repaired_closure):
        lines.append(
            f"| {row.label} | {row.cyclic_shift} | {row.exact_same_order_count} | "
            f"{row.exact_inverse_count} | {row.unmatched_count} |"
        )

    lines.append("")
    lines.append("## Linear quotient preservation")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("|---|---:|")
    for key, value in linear.__dict__.items():
        if isinstance(value, int):
            lines.append(f"| {key} | {value} |")
        else:
            lines.append(f"| {key} | {value:.12g} |")

    lines.append("")
    lines.append("## Nonlinear retained-log equivariance")
    lines.append("")
    lines.append("| g | relative error | absolute error | retained norm |")
    lines.append("|---:|---:|---:|---:|")
    for probe in probes:
        lines.append(
            f"| {probe.g:.12g} | {probe.relative_error:.12g} | "
            f"{probe.absolute_error:.12g} | {probe.retained_norm:.12g} |"
        )

    lines.append("")
    lines.append("## Density operator reflection and size")
    lines.append("")
    lines.append(
        "| source | Frobenius norm | operator norm | covariance L2 sq | covariance L2 | even residual | anti residual |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in density_rows:
        lines.append(
            f"| {row.source} | {row.frobenius_norm:.12g} | "
            f"{row.operator_norm:.12g} | {row.covariance_l2_constant_squared:.12g} | "
            f"{row.covariance_l2_constant:.12g} | "
            f"{row.even_reflection_residual:.12g} | {row.anti_reflection_residual:.12g} |"
        )

    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(
        "| stem/basepoint finite-block repair passes | Wilson trace replacement needed now | full common-stem theorem proved |"
    )
    lines.append("|---|---|---|")
    lines.append(
        f"| {decision.stem_basepoint_repair_passes_finite_block_test} | "
        f"{decision.wilson_trace_replacement_needed_now} | "
        f"{decision.full_common_stem_theorem_proved} |"
    )
    lines.append("")
    lines.append(decision.decision)
    return "\n".join(lines)


def run(
    tol: float,
    chunk_scalar: int,
    seed: int,
    amplitude: float,
    g_values: tuple[float, ...],
    repair_shift: int,
) -> str:
    data, lift, quotient_basis, _ = build_retained_lift(2)
    original_paths = tile_loop_paths(data)
    link_reflection = link_reflection_matrix(data)
    repaired_paths = shifted_paths(original_paths, repair_shift)
    valid_shifts = valid_uniform_shifts(original_paths, link_reflection)

    original_closure = exact_closure_report(
        "original C5AX ordering", original_paths, link_reflection, 0
    )
    repaired_closure = exact_closure_report(
        "uniform cyclic rebase", repaired_paths, link_reflection, repair_shift
    )
    linear = linear_preservation_report(
        data, lift, quotient_basis, original_paths, repaired_paths, link_reflection
    )

    old_matrices = build_all_matrices(tol, chunk_scalar)
    quotient_reflection_matrix = quotient_reflection(
        data, repaired_paths, quotient_basis, link_reflection
    )
    probes = nonlinear_equivariance_probes(
        data,
        lift,
        old_matrices["positive_basis"],
        quotient_basis,
        repaired_paths,
        link_reflection,
        quotient_reflection_matrix,
        seed,
        amplitude,
        g_values,
    )
    density_rows = density_reflection_rows(
        data,
        lift,
        old_matrices["positive_basis"],
        old_matrices["positive_evals"],
        quotient_basis,
        repaired_paths,
        link_reflection,
        quotient_reflection_matrix,
        old_matrices,
    )
    decision = make_decision(repaired_closure, linear, probes, density_rows)
    return markdown_report(
        valid_shifts,
        original_closure,
        repaired_closure,
        linear,
        probes,
        density_rows,
        decision,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--chunk-scalar", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--amplitude", type=float, default=0.2)
    parser.add_argument("--repair-shift", type=int, default=1)
    parser.add_argument(
        "--g-values",
        type=float,
        nargs="+",
        default=[0.0, 1.0e-3, 1.0e-2, 1.0e-1, 3.0e-1, 8.0e-1],
    )
    args = parser.parse_args()
    print(
        run(
            tol=args.tol,
            chunk_scalar=args.chunk_scalar,
            seed=args.seed,
            amplitude=args.amplitude,
            g_values=tuple(args.g_values),
            repair_shift=args.repair_shift,
        )
    )


if __name__ == "__main__":
    main()
