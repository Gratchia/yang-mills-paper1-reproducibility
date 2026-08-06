"""C5AX reflection-compatible retained-coordinate density placement.

# ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates

C5AX follows C5AW.  C5AW found a large reflection residual in the
retained-coordinate Jacobian part of the odd bilinear density, while Haar and
FP half-density parts were reflection compatible.

This checkpoint asks whether that failure is only a wrong linear coordinate
representation, or a genuine nonlinear retained-loop placement defect.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from c5ae_jacobian_half_density import (
    exp_lie,
    log_lie,
    quat_mul,
    signed_edge_arrays,
    tile_loop_paths,
)
from c5al_gauge_zero_reference import link_reflection_matrix
from c5aw_odd_bilinear_envelope import (
    build_all_matrices,
    color_lift_matrix,
)
from c5v_retained_quotient_bch_bounds import COLORS, build_retained_lift


@dataclass(frozen=True)
class LinearReflectionReport:
    scalar_links: int
    tile_loops: int
    retained_rank: int
    tile_row_rank: int
    quotient_orthogonality_error: float
    constraint_lift_residual: float
    tile_row_reflection_residual: float
    quotient_reflection_leak: float
    quotient_reflection_orthogonality_error: float
    retained_lift_reflection_leak: float
    lift_vs_tile_reflection_difference: float
    lift_vs_tile_reflection_relative_difference: float
    lift_reflection_orthogonality_error: float


@dataclass(frozen=True)
class PathClosureReport:
    paths: int
    same_order_count: int
    inverse_cyclic_count: int
    unmatched_count: int
    max_cyclic_shift: int


@dataclass(frozen=True)
class EquivarianceProbe:
    g: float
    relative_error: float
    absolute_error: float
    retained_norm: float


@dataclass(frozen=True)
class SourceReflectionReport:
    source: str
    frobenius_norm: float
    even_residual: float
    anti_residual: float
    invariant_projection_defect_ratio: float


@dataclass(frozen=True)
class C5AXDecision:
    linear_coordinate_artifact: bool
    nonlinear_basepoint_defect_detected: bool
    retained_density_reflection_compatible: bool
    decision: str


def path_row_matrix(data, paths: list[list[tuple[int, float]]]) -> np.ndarray:
    rows = np.zeros((len(paths), len(data.cells[1])), dtype=float)
    scale = float(data.n * data.n)
    for row, path in enumerate(paths):
        for edge, sign in path:
            rows[row, edge] += sign / scale
    return rows


def quotient_reflection_data(data, lift: np.ndarray, quotient_basis: np.ndarray):
    paths = tile_loop_paths(data)
    rows = path_row_matrix(data, paths)
    reflection = link_reflection_matrix(data)
    tile_reflection = (rows @ reflection) @ np.linalg.pinv(rows)
    quotient_reflection = quotient_basis.T @ tile_reflection @ quotient_basis
    lift_reflection = np.linalg.pinv(lift) @ reflection @ lift
    return paths, rows, reflection, tile_reflection, quotient_reflection, lift_reflection


def linear_reflection_report(
    data,
    lift: np.ndarray,
    quotient_basis: np.ndarray,
    constraint: np.ndarray,
) -> tuple[LinearReflectionReport, np.ndarray, np.ndarray, list[list[tuple[int, float]]]]:
    (
        paths,
        rows,
        reflection,
        tile_reflection,
        quotient_reflection,
        lift_reflection,
    ) = quotient_reflection_data(data, lift, quotient_basis)

    quotient_leak = np.linalg.norm(
        (np.eye(quotient_basis.shape[0]) - quotient_basis @ quotient_basis.T)
        @ tile_reflection
        @ quotient_basis
    )
    lift_tile_diff = np.linalg.norm(lift_reflection - quotient_reflection)
    return (
        LinearReflectionReport(
            scalar_links=len(data.cells[1]),
            tile_loops=len(paths),
            retained_rank=lift.shape[1],
            tile_row_rank=int(np.linalg.matrix_rank(rows)),
            quotient_orthogonality_error=float(
                np.linalg.norm(
                    quotient_basis.T @ quotient_basis - np.eye(quotient_basis.shape[1])
                )
            ),
            constraint_lift_residual=float(np.linalg.norm(constraint @ lift - quotient_basis)),
            tile_row_reflection_residual=float(
                np.linalg.norm(rows @ reflection - tile_reflection @ rows)
                / max(np.linalg.norm(rows @ reflection), 1.0e-30)
            ),
            quotient_reflection_leak=float(quotient_leak),
            quotient_reflection_orthogonality_error=float(
                np.linalg.norm(
                    quotient_reflection.T @ quotient_reflection
                    - np.eye(quotient_reflection.shape[0])
                )
            ),
            retained_lift_reflection_leak=float(
                np.linalg.norm(reflection @ lift - lift @ lift_reflection)
                / max(np.linalg.norm(lift), 1.0e-30)
            ),
            lift_vs_tile_reflection_difference=float(lift_tile_diff),
            lift_vs_tile_reflection_relative_difference=float(
                lift_tile_diff / max(np.linalg.norm(quotient_reflection), 1.0e-30)
            ),
            lift_reflection_orthogonality_error=float(
                np.linalg.norm(
                    lift_reflection.T @ lift_reflection - np.eye(lift_reflection.shape[0])
                )
            ),
        ),
        reflection,
        quotient_reflection,
        paths,
    )


def reversed_path(path: list[tuple[int, float]]) -> list[tuple[int, float]]:
    return [(edge, -sign) for edge, sign in reversed(path)]


def cyclic_shift_match(
    candidate: list[tuple[int, float]], reference: list[tuple[int, float]]
) -> tuple[bool, int]:
    if len(candidate) != len(reference):
        return False, -1
    length = len(reference)
    for shift in range(length):
        if all(candidate[(idx + shift) % length] == reference[idx] for idx in range(length)):
            return True, shift
    return False, -1


def reflected_same_path_sequence(
    path: list[tuple[int, float]], reflection: np.ndarray
) -> list[tuple[int, float]]:
    """Return the ordered sequence for evaluating this path on reflected fields.

    The link reflection matrix has one nonzero entry in each row.  If
    A'_e = s A_f, then a signed path segment sign*A'_e contributes
    (f, sign*s) as a functional of the original field.
    """
    reflected: list[tuple[int, float]] = []
    for edge, sign in path:
        source_edges = np.nonzero(np.abs(reflection[edge]) > 0.5)[0]
        if len(source_edges) != 1:
            raise RuntimeError("reflection row is not a signed permutation row")
        source = int(source_edges[0])
        reflected.append((source, float(sign * reflection[edge, source])))
    return reflected


def path_closure_report(
    paths: list[list[tuple[int, float]]], reflection: np.ndarray
) -> PathClosureReport:
    same_order_count = 0
    inverse_cyclic_count = 0
    unmatched_count = 0
    max_shift = 0

    for path in paths:
        reflected = reflected_same_path_sequence(path, reflection)
        matched = False
        for reference in paths:
            same, same_shift = cyclic_shift_match(reflected, reference)
            if same:
                same_order_count += 1
                max_shift = max(max_shift, same_shift)
                matched = True
                break
            inverse, inverse_shift = cyclic_shift_match(reflected, reversed_path(reference))
            if inverse:
                inverse_cyclic_count += 1
                max_shift = max(max_shift, inverse_shift)
                matched = True
                break
        if not matched:
            unmatched_count += 1

    return PathClosureReport(
        paths=len(paths),
        same_order_count=same_order_count,
        inverse_cyclic_count=inverse_cyclic_count,
        unmatched_count=unmatched_count,
        max_cyclic_shift=max_shift,
    )


def exact_retained_coordinate_from_links(
    links_color: np.ndarray,
    quotient_basis: np.ndarray,
    paths: list[list[tuple[int, float]]],
    n: int,
    g: float,
) -> np.ndarray:
    logs = np.zeros((len(paths), COLORS), dtype=float)
    if abs(g) < 1.0e-14:
        edges, signs = signed_edge_arrays(paths)
        signed = signs[:, :, None] * links_color[edges]
        linear = np.sum(signed, axis=1) / float(n * n)
        return (quotient_basis.T @ linear).reshape(-1)

    for path_index, path in enumerate(paths):
        group_element = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        for edge, sign in path:
            group_element = quat_mul(
                group_element, exp_lie(g * sign * links_color[edge])
            )
        logs[path_index] = log_lie(group_element) / (g * n * n)
    return (quotient_basis.T @ logs).reshape(-1)


def nonlinear_equivariance_probes(
    data,
    lift: np.ndarray,
    positive_basis: np.ndarray,
    quotient_basis: np.ndarray,
    paths: list[list[tuple[int, float]]],
    link_reflection: np.ndarray,
    quotient_reflection: np.ndarray,
    seed: int,
    amplitude: float,
    g_values: tuple[float, ...],
) -> list[EquivarianceProbe]:
    rng = np.random.default_rng(seed)
    retained_color = color_lift_matrix(lift)
    positive_color = color_lift_matrix(positive_basis)
    link_color_reflection = np.kron(link_reflection, np.eye(COLORS))
    quotient_color_reflection = np.kron(quotient_reflection, np.eye(COLORS))

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


def source_reflection_reports(
    matrices: dict[str, np.ndarray],
    link_reflection: np.ndarray,
    lift: np.ndarray,
    quotient_reflection: np.ndarray,
    positive_basis: np.ndarray,
) -> list[SourceReflectionReport]:
    link_color_reflection = np.kron(link_reflection, np.eye(COLORS))
    retained_reflection = np.kron(quotient_reflection, np.eye(COLORS))
    positive_reflection = (
        color_lift_matrix(positive_basis).T
        @ link_color_reflection
        @ color_lift_matrix(positive_basis)
    )

    reports: list[SourceReflectionReport] = []
    for source in (
        "retained",
        "retained_trace_j2",
        "retained_j1_square",
        "Haar half",
        "FP half",
        "combined",
    ):
        matrix = matrices[source]
        transformed = retained_reflection.T @ matrix @ positive_reflection
        norm = max(np.linalg.norm(matrix), 1.0e-30)
        even_residual = np.linalg.norm(transformed - matrix) / norm
        reports.append(
            SourceReflectionReport(
                source=source,
                frobenius_norm=float(np.linalg.norm(matrix)),
                even_residual=float(even_residual),
                anti_residual=float(np.linalg.norm(transformed + matrix) / norm),
                invariant_projection_defect_ratio=float(0.5 * even_residual),
            )
        )
    return reports


def make_decision(
    linear: LinearReflectionReport,
    closure: PathClosureReport,
    probes: list[EquivarianceProbe],
    sources: list[SourceReflectionReport],
) -> C5AXDecision:
    retained = next(row for row in sources if row.source == "retained")
    linear_artifact = not (
        linear.tile_row_reflection_residual < 1.0e-10
        and linear.quotient_reflection_leak < 1.0e-10
        and linear.lift_vs_tile_reflection_relative_difference < 1.0e-10
    )
    nonlinear_defect = (
        closure.inverse_cyclic_count > 0
        and max(probe.relative_error for probe in probes if probe.g > 0.0) > 1.0e-4
    )
    retained_ok = retained.even_residual < 1.0e-8
    if (not linear_artifact) and nonlinear_defect and (not retained_ok):
        decision = (
            "C5AX fail/pass split: the C5AW retained reflection residual is not "
            "a wrong linear retained-coordinate representation.  The linear "
            "lift and tile-quotient reflection actions agree to roundoff.  The "
            "failure is a genuine nonlinear based-loop placement defect: "
            "time-reflected loops become cyclically shifted inverse loops, so "
            "the based Lie log is conjugated rather than strictly reflected.  "
            "Redesign retained coordinates with reflection-covariant stems or "
            "move to conjugacy-class/Wilson-trace coordinates before using the "
            "retained Jacobian as an OS-placed density."
        )
    else:
        decision = (
            "C5AX inconclusive: one of the linear, nonlinear, or source-level "
            "diagnostics did not match the expected obstruction pattern."
        )
    return C5AXDecision(
        linear_coordinate_artifact=linear_artifact,
        nonlinear_basepoint_defect_detected=nonlinear_defect,
        retained_density_reflection_compatible=retained_ok,
        decision=decision,
    )


def markdown_report(
    linear: LinearReflectionReport,
    closure: PathClosureReport,
    probes: list[EquivarianceProbe],
    sources: list[SourceReflectionReport],
    decision: C5AXDecision,
) -> str:
    lines: list[str] = []
    lines.append("# C5AX Reflection-Compatible Retained Density Diagnostic")
    lines.append("")
    lines.append("## Linear retained-coordinate check")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("|---|---:|")
    for key, value in linear.__dict__.items():
        if isinstance(value, int):
            lines.append(f"| {key} | {value} |")
        else:
            lines.append(f"| {key} | {value:.12g} |")

    lines.append("")
    lines.append("## Ordered path closure")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("|---|---:|")
    for key, value in closure.__dict__.items():
        lines.append(f"| {key} | {value} |")

    lines.append("")
    lines.append("## Nonlinear retained-log equivariance probe")
    lines.append("")
    lines.append("| g | relative error | absolute error | retained norm |")
    lines.append("|---:|---:|---:|---:|")
    for probe in probes:
        lines.append(
            f"| {probe.g:.12g} | {probe.relative_error:.12g} | "
            f"{probe.absolute_error:.12g} | {probe.retained_norm:.12g} |"
        )

    lines.append("")
    lines.append("## Density-source reflection residuals")
    lines.append("")
    lines.append(
        "| source | Frobenius norm | even residual | anti residual | invariant projection defect ratio |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for row in sources:
        lines.append(
            f"| {row.source} | {row.frobenius_norm:.12g} | "
            f"{row.even_residual:.12g} | {row.anti_residual:.12g} | "
            f"{row.invariant_projection_defect_ratio:.12g} |"
        )

    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("| linear artifact | nonlinear basepoint defect | retained density reflection compatible |")
    lines.append("|---|---|---|")
    lines.append(
        f"| {decision.linear_coordinate_artifact} | "
        f"{decision.nonlinear_basepoint_defect_detected} | "
        f"{decision.retained_density_reflection_compatible} |"
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
) -> str:
    data, lift, quotient_basis, constraint = build_retained_lift(2)
    linear, link_reflection, quotient_reflection, paths = linear_reflection_report(
        data, lift, quotient_basis, constraint
    )
    closure = path_closure_report(paths, link_reflection)
    matrices = build_all_matrices(tol, chunk_scalar)
    probes = nonlinear_equivariance_probes(
        data,
        lift,
        matrices["positive_basis"],
        quotient_basis,
        paths,
        link_reflection,
        quotient_reflection,
        seed,
        amplitude,
        g_values,
    )
    sources = source_reflection_reports(
        matrices,
        link_reflection,
        lift,
        quotient_reflection,
        matrices["positive_basis"],
    )
    decision = make_decision(linear, closure, probes, sources)
    return markdown_report(linear, closure, probes, sources, decision)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--chunk-scalar", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--amplitude", type=float, default=0.2)
    parser.add_argument(
        "--g-values",
        type=float,
        nargs="+",
        default=[0.0, 1.0e-3, 1.0e-2, 1.0e-1, 3.0e-1],
    )
    args = parser.parse_args()
    print(
        run(
            tol=args.tol,
            chunk_scalar=args.chunk_scalar,
            seed=args.seed,
            amplitude=args.amplitude,
            g_values=tuple(args.g_values),
        )
    )


if __name__ == "__main__":
    main()
