"""Exact finite certificates used by the C5X paper.

The checker separates integer/combinatorial claims from floating-point
regressions.  It certifies the coarse-face rank, an explicit BCH obstruction
witness, the valid uniform cyclic shifts, and the rooted path-word package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

from c5ae_jacobian_half_density import tile_loop_paths
from c5al_gauge_zero_reference import link_reflection_matrix
from c5ax_reflection_retained_density import (
    cyclic_shift_match,
    reflected_same_path_sequence,
    reversed_path,
)
from c5ay_reflection_covariant_retained_coordinates import (
    exact_closure_report,
    shifted_paths,
    valid_uniform_shifts,
)
from c5az_shifted_coordinate_package_regression import (
    build_common_stem_path_package,
)
from c5q_4d_cubical_incidence import build_data


EXPECTED_ROOTED_PATH_HASH = (
    "1EB6E829D66DCDFA6E12E18CACA3C09A3EDFC3EA56E7EBE0D7ED1E05CA01616C"
)


@dataclass(frozen=True)
class ExactCertificate:
    raw_face_rows: int
    raw_face_rank_over_q: int
    exact_bianchi_relation_count: int
    original_same_order_paths: int
    original_inverse_cyclic_paths: int
    original_inverse_cyclic_shift: int
    valid_exact_uniform_shifts: list[int]
    witness_edge_one: int
    witness_edge_two: int
    witness_nonzero_defect_rows: list[list[object]]
    witness_nonzero_rt_defect_entries: list[list[object]]
    rooted_path_count: int
    rooted_min_segments: int
    rooted_max_segments: int
    rooted_mean_segments: float
    rooted_unique_edges: int
    rooted_path_word_sha256: str
    all_exact_gates_pass: bool


def integer_path_rows(paths: list[list[tuple[int, float]]], links: int) -> list[list[int]]:
    rows = [[0 for _ in range(links)] for _ in paths]
    for row_index, path in enumerate(paths):
        for edge, sign in path:
            rows[row_index][edge] += int(sign)
    return rows


def exact_rational_rank(rows: list[list[int]]) -> int:
    matrix = [[Fraction(value) for value in row] for row in rows]
    row_count = len(matrix)
    column_count = len(matrix[0])
    pivot_row = 0
    for column in range(column_count):
        pivot = next(
            (row for row in range(pivot_row, row_count) if matrix[row][column]),
            None,
        )
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        pivot_value = matrix[pivot_row][column]
        for entry in range(column, column_count):
            matrix[pivot_row][entry] /= pivot_value
        for row in range(row_count):
            if row == pivot_row or not matrix[row][column]:
                continue
            factor = matrix[row][column]
            for entry in range(column, column_count):
                matrix[row][entry] -= factor * matrix[pivot_row][entry]
        pivot_row += 1
        if pivot_row == row_count:
            break
    return pivot_row


def raw_reflection_permutation(paths, reflection: np.ndarray) -> tuple[np.ndarray, list]:
    permutation = np.zeros((len(paths), len(paths)), dtype=int)
    relations: list[tuple[int, int, int]] = []
    for path_index, path in enumerate(paths):
        reflected = reflected_same_path_sequence(path, reflection)
        relation = None
        for target, reference in enumerate(paths):
            same, returned_shift = cyclic_shift_match(reflected, reference)
            if same:
                relation = (target, 1, (-returned_shift) % len(path))
                break
            inverse, returned_shift = cyclic_shift_match(
                reflected, reversed_path(reference)
            )
            if inverse:
                relation = (target, -1, (-returned_shift) % len(path))
                break
        if relation is None:
            raise RuntimeError("unmatched original reflected path")
        target, sign, geometric_shift = relation
        permutation[path_index, target] = sign
        relations.append((target, sign, geometric_shift))
    return permutation, relations


def twice_bch_q2(paths, field: np.ndarray) -> np.ndarray:
    out = np.zeros((len(paths), 3), dtype=int)
    for path_index, path in enumerate(paths):
        signed = [int(sign) * field[edge] for edge, sign in path]
        for first in range(len(signed)):
            for second in range(first + 1, len(signed)):
                out[path_index] += np.cross(signed[first], signed[second])
    return out


def exact_obstruction(data, paths, reflection, raw_rows):
    edge_one = data.index[1][((0,), (0, 0, 0, 0))]
    edge_two = data.index[1][((0,), (0, 0, 0, 2))]
    field = np.zeros((len(data.cells[1]), 3), dtype=int)
    field[edge_one, 0] = 1
    field[edge_two, 1] = 1

    permutation, relations = raw_reflection_permutation(paths, reflection)
    reflected_field = reflection.astype(int) @ field
    defect = twice_bch_q2(paths, reflected_field) - permutation @ twice_bch_q2(
        paths, field
    )
    retained_certificate = np.asarray(raw_rows, dtype=int).T @ defect

    defect_rows = [
        [int(row), [int(value) for value in defect[row]]]
        for row in range(defect.shape[0])
        if np.any(defect[row])
    ]
    certificate_entries = [
        [
            int(edge),
            int(color),
            int(retained_certificate[edge, color]),
        ]
        for edge in range(retained_certificate.shape[0])
        for color in range(3)
        if retained_certificate[edge, color]
    ]
    return edge_one, edge_two, relations, defect_rows, certificate_entries


def build_certificate() -> ExactCertificate:
    data = build_data(2)
    paths = tile_loop_paths(data)
    reflection = link_reflection_matrix(data)
    raw_rows = integer_path_rows(paths, len(data.cells[1]))
    rank = exact_rational_rank(raw_rows)

    edge_one, edge_two, relations, defect_rows, certificate_entries = exact_obstruction(
        data, paths, reflection, raw_rows
    )
    same_count = sum(sign == 1 for _, sign, _ in relations)
    inverse_relations = [relation for relation in relations if relation[1] == -1]
    inverse_shifts = sorted({shift for _, _, shift in inverse_relations})

    shifts = valid_uniform_shifts(paths, reflection)
    shifted = shifted_paths(paths, 1)
    closure = exact_closure_report("uniform shift 1", shifted, reflection, 1)
    rooted = build_common_stem_path_package(data, reflection, shift=1).report

    gates = (
        len(paths) == 24
        and rank == 17
        and 24 - rank == 7
        and same_count == 12
        and len(inverse_relations) == 12
        and inverse_shifts == [6]
        and shifts == [1, 5]
        and closure.exact_same_order_count == 12
        and closure.exact_inverse_count == 12
        and closure.unmatched_count == 0
        and defect_rows == [[8, [0, 0, -2]]]
        and certificate_entries
        and rooted.retained_loops == 24
        and rooted.rooted_min_segments == 8
        and rooted.rooted_max_segments == 16
        and abs(rooted.rooted_mean_segments - 13.0) < 1.0e-12
        and rooted.rooted_unique_edges == 84
        and rooted.path_word_sha256 == EXPECTED_ROOTED_PATH_HASH
    )
    return ExactCertificate(
        raw_face_rows=len(paths),
        raw_face_rank_over_q=rank,
        exact_bianchi_relation_count=len(paths) - rank,
        original_same_order_paths=same_count,
        original_inverse_cyclic_paths=len(inverse_relations),
        original_inverse_cyclic_shift=inverse_shifts[0],
        valid_exact_uniform_shifts=shifts,
        witness_edge_one=edge_one,
        witness_edge_two=edge_two,
        witness_nonzero_defect_rows=defect_rows,
        witness_nonzero_rt_defect_entries=certificate_entries,
        rooted_path_count=rooted.retained_loops,
        rooted_min_segments=rooted.rooted_min_segments,
        rooted_max_segments=rooted.rooted_max_segments,
        rooted_mean_segments=rooted.rooted_mean_segments,
        rooted_unique_edges=rooted.rooted_unique_edges,
        rooted_path_word_sha256=rooted.path_word_sha256,
        all_exact_gates_pass=bool(gates),
    )


def main() -> None:
    certificate = build_certificate()
    if not certificate.all_exact_gates_pass:
        raise RuntimeError("one or more C5X exact finite certificate gates failed")
    output = {
        "checkpoint": "C5X-EXACT",
        "title": "exact finite claim certificates",
        "certificate": asdict(certificate),
    }
    out_path = Path("outputs/c5x_exact_finite_claim_certificates.json")
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
