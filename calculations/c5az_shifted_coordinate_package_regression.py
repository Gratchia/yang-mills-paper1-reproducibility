"""C5AZ shifted-coordinate common-stem package regression.

# ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates

C5AY repaired the C5AX retained-coordinate reflection obstruction by shifting
each retained face-loop basepoint by one segment.  C5AZ asks whether this is
only a gauge-fixed path-order trick, or whether it can be embedded into a
common-stem gauge-covariant retained holonomy package.

This remains a finite-block diagnostic and package regression.  It does not
prove the exact nonlinear shell theorem, continuum construction, confinement,
or mass gap.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from c5ae_jacobian_half_density import (
    analytic_divergence_by_color_trace,
    finite_difference_divergence,
    quat_mul,
    retained_n2,
    tile_loop_paths,
)
from c5al_gauge_zero_reference import link_reflection_matrix
from c5ax_reflection_retained_density import reflected_same_path_sequence, reversed_path
from c5ay_reflection_covariant_retained_coordinates import (
    cyclic_shift_path,
    density_reflection_rows,
    quotient_reflection,
)
from c5v_retained_quotient_bch_bounds import COLORS, build_retained_lift
from c5aw_odd_bilinear_envelope import build_all_matrices


@dataclass(frozen=True)
class StemPackageReport:
    root_vertex: tuple[int, int, int, int]
    root_reflection_fixed: bool
    retained_loops: int
    stems: int
    loop_same_order_reflections: int
    loop_inverse_reflections: int
    loop_unmatched_reflections: int
    stem_unmatched_reflections: int
    rooted_path_unmatched_reflections: int
    shifted_core_min_segments: int
    shifted_core_max_segments: int
    rooted_min_segments: int
    rooted_max_segments: int
    rooted_mean_segments: float
    shifted_core_unique_edges: int
    rooted_unique_edges: int
    path_word_sha256: str


@dataclass(frozen=True)
class CommonStemPathPackage:
    shifted_core_paths: list[list[tuple[int, float]]]
    stems: list[list[tuple[int, float]]]
    rooted_paths: list[list[tuple[int, float]]]
    relations: list[tuple[int, str]]
    report: StemPackageReport


@dataclass(frozen=True)
class GroupCovarianceReport:
    random_seed: int
    random_link_samples: int
    max_rooted_gauge_covariance_residual: float
    max_rooted_reflection_residual: float


@dataclass(frozen=True)
class JacobianRegressionReport:
    samples: int
    max_abs_analytic_divergence_n2: float
    max_abs_fd_divergence_n2: float


@dataclass(frozen=True)
class DensityRegressionReport:
    old_combined_reflection_residual: float
    shifted_core_combined_reflection_residual: float
    rooted_combined_reflection_residual: float
    shifted_core_operator_norm: float
    rooted_operator_norm: float
    shifted_core_covariance_l2_squared: float
    rooted_covariance_l2_squared: float
    shifted_core_covariance_l2: float
    rooted_covariance_l2: float
    rooted_to_shifted_core_operator_ratio: float
    rooted_to_shifted_core_covariance_squared_ratio: float


@dataclass(frozen=True)
class C5AZDecision:
    common_stem_finite_package_passes: bool
    rooted_density_recomputed: bool
    previous_shifted_core_constants_preserved: bool
    exact_shell_theorem_proved: bool
    decision: str


def quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float) / float(q @ q)


def quat_norm(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def quat_product(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    return quat_norm(quat_mul(quat_norm(q), quat_norm(r)))


def quat_distance(q: np.ndarray, r: np.ndarray) -> float:
    return float(np.linalg.norm(quat_norm(q) - quat_norm(r)))


def random_quaternion(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def edge_tail_head(data, edge: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    dirs, coords_tuple = data.cells[1][edge]
    direction = dirs[0]
    tail = tuple(coords_tuple)
    head = list(coords_tuple)
    head[direction] += 1
    return tail, tuple(head)


def signed_segment_vertices(
    data, segment: tuple[int, float]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    edge, sign = segment
    tail, head = edge_tail_head(data, edge)
    return (tail, head) if sign > 0.0 else (head, tail)


def path_basepoint(data, path: list[tuple[int, float]]) -> tuple[int, int, int, int]:
    return tuple(int(v) for v in signed_segment_vertices(data, path[0])[0])


def reverse_path(path: list[tuple[int, float]]) -> list[tuple[int, float]]:
    return [(edge, -sign) for edge, sign in reversed(path)]


def stem_path(
    data,
    root: tuple[int, int, int, int],
    target: tuple[int, int, int, int],
) -> list[tuple[int, float]]:
    """Canonical Manhattan stem from root to target."""
    current = list(root)
    out: list[tuple[int, float]] = []
    for direction in range(4):
        while current[direction] < target[direction]:
            key = ((direction,), tuple(current))
            out.append((data.index[1][key], 1.0))
            current[direction] += 1
        while current[direction] > target[direction]:
            next_coords = current.copy()
            next_coords[direction] -= 1
            key = ((direction,), tuple(next_coords))
            out.append((data.index[1][key], -1.0))
            current[direction] -= 1
    if tuple(current) != target:
        raise RuntimeError("stem construction failed")
    return out


def reflected_vertex(data, vertex: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    coords = list(vertex)
    coords[0] = data.n - coords[0]
    return tuple(coords)


def find_loop_relation(
    path: list[tuple[int, float]],
    paths: list[list[tuple[int, float]]],
    reflection: np.ndarray,
) -> tuple[int, str]:
    reflected = reflected_same_path_sequence(path, reflection)
    for index, candidate in enumerate(paths):
        if reflected == candidate:
            return index, "same"
        if reflected == reversed_path(candidate):
            return index, "inverse"
    return -1, "unmatched"


def stem_package_report(data, paths, reflection) -> tuple[StemPackageReport, list, list, list]:
    root = (data.n // 2, 0, 0, 0)
    stems = [stem_path(data, root, path_basepoint(data, path)) for path in paths]
    rooted_paths = [stems[i] + paths[i] + reverse_path(stems[i]) for i in range(len(paths))]
    relations = [find_loop_relation(path, paths, reflection) for path in paths]

    loop_same = sum(1 for _, relation in relations if relation == "same")
    loop_inverse = sum(1 for _, relation in relations if relation == "inverse")
    loop_unmatched = sum(1 for _, relation in relations if relation == "unmatched")

    stem_unmatched = 0
    rooted_unmatched = 0
    for index, stem in enumerate(stems):
        target, relation = relations[index]
        if target < 0 or reflected_same_path_sequence(stem, reflection) != stems[target]:
            stem_unmatched += 1
        reflected_rooted = reflected_same_path_sequence(rooted_paths[index], reflection)
        if relation == "same":
            expected = rooted_paths[target]
        elif relation == "inverse":
            expected = reverse_path(rooted_paths[target])
        else:
            expected = None
        if expected is None or reflected_rooted != expected:
            rooted_unmatched += 1

    shifted_lengths = [len(path) for path in paths]
    rooted_lengths = [len(path) for path in rooted_paths]
    shifted_edges = {edge for path in paths for edge, _ in path}
    rooted_edges = {edge for path in rooted_paths for edge, _ in path}
    path_payload = {
        "root": root,
        "shifted_core_paths": [
            [[int(edge), int(sign)] for edge, sign in path] for path in paths
        ],
        "stems": [
            [[int(edge), int(sign)] for edge, sign in path] for path in stems
        ],
        "rooted_paths": [
            [[int(edge), int(sign)] for edge, sign in path] for path in rooted_paths
        ],
    }
    path_word_sha256 = hashlib.sha256(
        json.dumps(path_payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest().upper()

    report = StemPackageReport(
        root_vertex=root,
        root_reflection_fixed=reflected_vertex(data, root) == root,
        retained_loops=len(paths),
        stems=len(stems),
        loop_same_order_reflections=loop_same,
        loop_inverse_reflections=loop_inverse,
        loop_unmatched_reflections=loop_unmatched,
        stem_unmatched_reflections=stem_unmatched,
        rooted_path_unmatched_reflections=rooted_unmatched,
        shifted_core_min_segments=min(shifted_lengths),
        shifted_core_max_segments=max(shifted_lengths),
        rooted_min_segments=min(rooted_lengths),
        rooted_max_segments=max(rooted_lengths),
        rooted_mean_segments=float(np.mean(rooted_lengths)),
        shifted_core_unique_edges=len(shifted_edges),
        rooted_unique_edges=len(rooted_edges),
        path_word_sha256=path_word_sha256,
    )
    if data.n == 2:
        expected = (
            report.retained_loops == 24
            and report.shifted_core_min_segments == 8
            and report.shifted_core_max_segments == 8
            and report.rooted_min_segments == 8
            and report.rooted_max_segments == 16
            and abs(report.rooted_mean_segments - 13.0) < 1.0e-12
            and report.shifted_core_unique_edges == 64
            and report.rooted_unique_edges == 84
        )
        if not expected:
            raise RuntimeError("unexpected n=2 shifted-core/rooted path inventory")

    return (
        report,
        stems,
        rooted_paths,
        relations,
    )


def build_common_stem_path_package(
    data, reflection: np.ndarray, shift: int = 1
) -> CommonStemPathPackage:
    shifted_core_paths = [
        cyclic_shift_path(path, shift) for path in tile_loop_paths(data)
    ]
    report, stems, rooted_paths, relations = stem_package_report(
        data, shifted_core_paths, reflection
    )
    return CommonStemPathPackage(
        shifted_core_paths=shifted_core_paths,
        stems=stems,
        rooted_paths=rooted_paths,
        relations=relations,
        report=report,
    )


def path_holonomy(path: list[tuple[int, float]], links: np.ndarray) -> np.ndarray:
    out = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    for edge, sign in path:
        link = links[edge] if sign > 0.0 else quat_inv(links[edge])
        out = quat_product(out, link)
    return quat_norm(out)


def gauge_transform_links(data, links: np.ndarray, gauges: dict[tuple[int, ...], np.ndarray]) -> np.ndarray:
    transformed = np.zeros_like(links)
    for edge in range(len(data.cells[1])):
        tail, head = edge_tail_head(data, edge)
        transformed[edge] = quat_product(
            quat_product(gauges[tail], links[edge]), quat_inv(gauges[head])
        )
    return transformed


def reflect_links(reflection: np.ndarray, links: np.ndarray) -> np.ndarray:
    reflected = np.zeros_like(links)
    for edge in range(reflection.shape[0]):
        sources = np.nonzero(np.abs(reflection[edge]) > 0.5)[0]
        if len(sources) != 1:
            raise RuntimeError("reflection is not a signed permutation")
        source = int(sources[0])
        sign = float(reflection[edge, source])
        reflected[edge] = links[source] if sign > 0.0 else quat_inv(links[source])
    return reflected


def group_covariance_report(data, rooted_paths, relations, reflection, seed: int, samples: int) -> GroupCovarianceReport:
    rng = np.random.default_rng(seed)
    root = (data.n // 2, 0, 0, 0)
    vertices = [tuple(cell[1]) for cell in data.cells[0]]
    max_gauge = 0.0
    max_reflection = 0.0

    for _ in range(samples):
        links = np.array([random_quaternion(rng) for _ in range(len(data.cells[1]))])
        gauges = {vertex: random_quaternion(rng) for vertex in vertices}
        gauged_links = gauge_transform_links(data, links, gauges)

        holonomies = [path_holonomy(path, links) for path in rooted_paths]
        gauged_holonomies = [path_holonomy(path, gauged_links) for path in rooted_paths]
        for base, gauged in zip(holonomies, gauged_holonomies):
            expected = quat_product(quat_product(gauges[root], base), quat_inv(gauges[root]))
            max_gauge = max(max_gauge, quat_distance(gauged, expected))

        reflected = reflect_links(reflection, links)
        reflected_holonomies = [path_holonomy(path, reflected) for path in rooted_paths]
        for index, reflected_holonomy in enumerate(reflected_holonomies):
            target, relation = relations[index]
            if relation == "same":
                expected = holonomies[target]
            elif relation == "inverse":
                expected = quat_inv(holonomies[target])
            else:
                continue
            max_reflection = max(max_reflection, quat_distance(reflected_holonomy, expected))

    return GroupCovarianceReport(
        random_seed=seed,
        random_link_samples=samples,
        max_rooted_gauge_covariance_residual=float(max_gauge),
        max_rooted_reflection_residual=float(max_reflection),
    )


def jacobian_regression_report(data, lift, quotient_basis, paths, seed: int, samples: int) -> JacobianRegressionReport:
    rng = np.random.default_rng(seed + 17)
    dim = lift.shape[1] * COLORS
    analytic: list[float] = []
    finite_difference: list[float] = []
    for _ in range(samples):
        x = rng.normal(size=dim)
        x /= np.linalg.norm(x)
        analytic.append(
            abs(analytic_divergence_by_color_trace(x, lift, quotient_basis, paths, data.n))
        )
        finite_difference.append(
            abs(
                finite_difference_divergence(
                    lambda y: retained_n2(y, lift, quotient_basis, paths, data.n),
                    x,
                    1.0e-6,
                )
            )
        )
    return JacobianRegressionReport(
        samples=samples,
        max_abs_analytic_divergence_n2=float(max(analytic)),
        max_abs_fd_divergence_n2=float(max(finite_difference)),
    )


def density_regression_report(
    data,
    lift,
    quotient_basis,
    shifted_core_paths,
    rooted_paths,
    reflection,
    tol: float,
    chunk_scalar: int,
) -> DensityRegressionReport:
    old_matrices = build_all_matrices(tol, chunk_scalar)
    def combined_row(path_package):
        q_reflection = quotient_reflection(
            data, path_package, quotient_basis, reflection
        )
        rows = density_reflection_rows(
            data,
            lift,
            old_matrices["positive_basis"],
            old_matrices["positive_evals"],
            quotient_basis,
            path_package,
            reflection,
            q_reflection,
            old_matrices,
        )
        return rows, next(row for row in rows if row.source == "shifted combined")

    core_rows, shifted_core_combined = combined_row(shifted_core_paths)
    _, rooted_combined = combined_row(rooted_paths)
    old_combined = next(
        row for row in core_rows if row.source == "old combined, pre-C5AY"
    )
    return DensityRegressionReport(
        old_combined_reflection_residual=old_combined.even_reflection_residual,
        shifted_core_combined_reflection_residual=shifted_core_combined.even_reflection_residual,
        rooted_combined_reflection_residual=rooted_combined.even_reflection_residual,
        shifted_core_operator_norm=shifted_core_combined.operator_norm,
        rooted_operator_norm=rooted_combined.operator_norm,
        shifted_core_covariance_l2_squared=shifted_core_combined.covariance_l2_constant_squared,
        rooted_covariance_l2_squared=rooted_combined.covariance_l2_constant_squared,
        shifted_core_covariance_l2=shifted_core_combined.covariance_l2_constant,
        rooted_covariance_l2=rooted_combined.covariance_l2_constant,
        rooted_to_shifted_core_operator_ratio=(
            rooted_combined.operator_norm / shifted_core_combined.operator_norm
        ),
        rooted_to_shifted_core_covariance_squared_ratio=(
            rooted_combined.covariance_l2_constant_squared
            / shifted_core_combined.covariance_l2_constant_squared
        ),
    )


def make_decision(
    stem: StemPackageReport,
    group: GroupCovarianceReport,
    jacobian: JacobianRegressionReport,
    density: DensityRegressionReport,
) -> C5AZDecision:
    common_stem_ok = (
        stem.root_reflection_fixed
        and stem.loop_unmatched_reflections == 0
        and stem.stem_unmatched_reflections == 0
        and stem.rooted_path_unmatched_reflections == 0
        and group.max_rooted_gauge_covariance_residual < 1.0e-12
        and group.max_rooted_reflection_residual < 1.0e-12
    )
    rooted_density_ok = (
        jacobian.max_abs_analytic_divergence_n2 < 1.0e-12
        and jacobian.max_abs_fd_divergence_n2 < 1.0e-10
        and density.rooted_combined_reflection_residual < 1.0e-10
        and np.isfinite(density.rooted_operator_norm)
        and np.isfinite(density.rooted_covariance_l2_squared)
    )
    constants_preserved = (
        abs(density.rooted_operator_norm - density.shifted_core_operator_norm)
        < 1.0e-10
        and abs(
            density.rooted_covariance_l2_squared
            - density.shifted_core_covariance_l2_squared
        )
        < 1.0e-10
    )
    if common_stem_ok and rooted_density_ok:
        decision = (
            "C5AZ corrected finite-package pass: the C5AY "
            "shifted loops admit reflection-compatible stems from a fixed root, "
            "rooted holonomies transform by simultaneous root conjugation under "
            "finite gauge transformations, and rooted reflection acts exactly as "
            "same/inverse loop reflection.  The rooted no-O(g) Jacobian result and "
            "rooted density reflection placement pass, but the old shifted-core "
            "density constants do not survive rooting and downstream constants "
            "must be rebaselined.  The exact nonlinear one-shell theorem remains open."
        )
    else:
        decision = (
            "C5AZ partial/fail: at least one common-stem, Jacobian, or shifted "
            "density regression check did not pass.  The C5AY repair should not "
            "be promoted to the normalized package without redesign."
        )
    return C5AZDecision(
        common_stem_finite_package_passes=common_stem_ok,
        rooted_density_recomputed=rooted_density_ok,
        previous_shifted_core_constants_preserved=constants_preserved,
        exact_shell_theorem_proved=False,
        decision=decision,
    )


def markdown_report(stem, group, jacobian, density, decision) -> str:
    lines: list[str] = []
    lines.append("# C5AZ Shifted-Coordinate Common-Stem Package Regression")
    lines.append("")
    lines.append("## Common-stem path package")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("|---|---:|")
    for key, value in stem.__dict__.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append("## Finite group covariance probes")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("|---|---:|")
    for key, value in group.__dict__.items():
        if isinstance(value, int):
            lines.append(f"| {key} | {value} |")
        else:
            lines.append(f"| {key} | {value:.12g} |")
    lines.append("")
    lines.append("## Rooted Jacobian / half-density regression")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("|---|---:|")
    for key, value in jacobian.__dict__.items():
        if isinstance(value, int):
            lines.append(f"| {key} | {value} |")
        else:
            lines.append(f"| {key} | {value:.12g} |")
    lines.append("")
    lines.append("## Shifted-core versus rooted density package")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("|---|---:|")
    for key, value in density.__dict__.items():
        lines.append(f"| {key} | {value:.12g} |")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(
        "| common-stem finite package passes | rooted density recomputed | previous shifted-core constants preserved | exact shell theorem proved |"
    )
    lines.append("|---|---|---|---|")
    lines.append(
        f"| {decision.common_stem_finite_package_passes} | "
        f"{decision.rooted_density_recomputed} | "
        f"{decision.previous_shifted_core_constants_preserved} | "
        f"{decision.exact_shell_theorem_proved} |"
    )
    lines.append("")
    lines.append(decision.decision)
    return "\n".join(lines)


def run(tol: float, chunk_scalar: int, seed: int, group_samples: int, jacobian_samples: int) -> str:
    data, lift, quotient_basis, _ = build_retained_lift(2)
    reflection = link_reflection_matrix(data)
    path_package = build_common_stem_path_package(data, reflection, shift=1)
    group = group_covariance_report(
        data,
        path_package.rooted_paths,
        path_package.relations,
        reflection,
        seed,
        group_samples,
    )
    jacobian = jacobian_regression_report(
        data,
        lift,
        quotient_basis,
        path_package.rooted_paths,
        seed,
        jacobian_samples,
    )
    density = density_regression_report(
        data,
        lift,
        quotient_basis,
        path_package.shifted_core_paths,
        path_package.rooted_paths,
        reflection,
        tol,
        chunk_scalar,
    )
    decision = make_decision(path_package.report, group, jacobian, density)
    return markdown_report(path_package.report, group, jacobian, density, decision)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--chunk-scalar", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--group-samples", type=int, default=8)
    parser.add_argument("--jacobian-samples", type=int, default=4)
    args = parser.parse_args()
    print(
        run(
            tol=args.tol,
            chunk_scalar=args.chunk_scalar,
            seed=args.seed,
            group_samples=args.group_samples,
            jacobian_samples=args.jacobian_samples,
        )
    )


if __name__ == "__main__":
    main()
