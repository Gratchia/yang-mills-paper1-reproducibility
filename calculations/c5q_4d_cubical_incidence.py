"""4D cubical incidence and Schur-complement diagnostics for C5Q.

# ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates

This script builds the tangent-model cellular coboundary matrices on an
n^4 cubical block:

    links --D1--> plaquettes --D2--> cubes.

It splits links into topological-boundary links and interior links, computes
the Schur complement on boundary links after optimizing interior links, and
checks the discrete Bianchi identity D2 D1 = 0.

The calculation is abelianized/quadratic.  It is an incidence diagnostic, not
a non-Abelian Yang--Mills proof.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass

import numpy as np


D = 4


@dataclass(frozen=True)
class CubicalData:
    n: int
    cells: dict[int, list[tuple[tuple[int, ...], tuple[int, ...]]]]
    index: dict[int, dict[tuple[tuple[int, ...], tuple[int, ...]], int]]
    d1: np.ndarray
    d2: np.ndarray
    boundary_links: list[int]
    interior_links: list[int]


@dataclass(frozen=True)
class RankRow:
    n: int
    links: int
    plaquettes: int
    cubes: int
    interior_links: int
    rank_d1: int
    rank_d2: int
    rank_dint: int
    schur_rank: int
    schur_complement_dim: int
    bianchi_residual: float


@dataclass(frozen=True)
class LoopRow:
    n: int
    orientation: str
    hessian: float
    kernel_component: float


def make_cells(k: int, n: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    out: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for dirs in itertools.combinations(range(D), k):
        ranges = [range(n) if mu in dirs else range(n + 1) for mu in range(D)]
        for coords in itertools.product(*ranges):
            out.append((tuple(dirs), tuple(coords)))
    return out


def coboundary(
    k: int,
    cells_k: list[tuple[tuple[int, ...], tuple[int, ...]]],
    index_k: dict[tuple[tuple[int, ...], tuple[int, ...]], int],
    cells_next: list[tuple[tuple[int, ...], tuple[int, ...]]],
) -> np.ndarray:
    mat = np.zeros((len(cells_next), len(cells_k)), dtype=float)
    for row, (dirs, coords) in enumerate(cells_next):
        for r, mu in enumerate(dirs):
            face_dirs = tuple(nu for nu in dirs if nu != mu)
            upper = list(coords)
            upper[mu] += 1
            lower_key = (face_dirs, tuple(coords))
            upper_key = (face_dirs, tuple(upper))
            sign = -1.0 if r % 2 else 1.0
            mat[row, index_k[upper_key]] += sign
            mat[row, index_k[lower_key]] -= sign
    return mat


def is_boundary_cell(
    cell: tuple[tuple[int, ...], tuple[int, ...]], n: int
) -> bool:
    dirs, coords = cell
    dirs_set = set(dirs)
    return any(coords[mu] in (0, n) for mu in range(D) if mu not in dirs_set)


def build_data(n: int) -> CubicalData:
    cells = {k: make_cells(k, n) for k in [0, 1, 2, 3]}
    index = {k: {cell: i for i, cell in enumerate(cells[k])} for k in cells}
    d1 = coboundary(1, cells[1], index[1], cells[2])
    d2 = coboundary(2, cells[2], index[2], cells[3])
    boundary_links: list[int] = []
    interior_links: list[int] = []
    for idx, cell in enumerate(cells[1]):
        if is_boundary_cell(cell, n):
            boundary_links.append(idx)
        else:
            interior_links.append(idx)
    return CubicalData(n, cells, index, d1, d2, boundary_links, interior_links)


def rank(mat: np.ndarray, tol: float = 1e-10) -> int:
    if mat.size == 0:
        return 0
    return int(np.linalg.matrix_rank(mat, tol=tol))


def projector_onto_complement(mat: np.ndarray, tol: float = 1e-10) -> tuple[np.ndarray, int]:
    rows = mat.shape[0]
    if mat.size == 0 or mat.shape[1] == 0:
        return np.eye(rows), 0
    u, s, _ = np.linalg.svd(mat, full_matrices=True)
    r = int(np.sum(s > tol))
    if r == 0:
        return np.eye(rows), 0
    q = u[:, :r]
    return np.eye(rows) - q @ q.T, r


def boundary_schur(data: CubicalData) -> tuple[np.ndarray, int, int]:
    d_int = data.d1[:, data.interior_links]
    d_bd = data.d1[:, data.boundary_links]
    p_perp, rank_int = projector_onto_complement(d_int)
    h_bd = d_bd.T @ p_perp @ d_bd
    return h_bd, rank_int, rank(h_bd)


def rank_row(n: int) -> RankRow:
    data = build_data(n)
    h_bd, rank_int, schur_rank = boundary_schur(data)
    bianchi = float(np.linalg.norm(data.d2 @ data.d1, ord="fro"))
    return RankRow(
        n=n,
        links=len(data.cells[1]),
        plaquettes=len(data.cells[2]),
        cubes=len(data.cells[3]),
        interior_links=len(data.interior_links),
        rank_d1=rank(data.d1),
        rank_d2=rank(data.d2),
        rank_dint=rank_int,
        schur_rank=schur_rank,
        schur_complement_dim=data.d1.shape[0] - rank_int,
        bianchi_residual=bianchi,
    )


def loop_functional(data: CubicalData, mu: int, nu: int) -> np.ndarray:
    """Boundary loop around an n x n square in the mu-nu plane at other coords 0."""
    n = data.n
    r_all = np.zeros(len(data.cells[1]), dtype=float)

    def add_edge(direction: int, coords: list[int], coeff: float) -> None:
        key = ((direction,), tuple(coords))
        r_all[data.index[1][key]] += coeff

    other = [rho for rho in range(D) if rho not in (mu, nu)]

    for i in range(n):
        coords = [0] * D
        coords[mu] = i
        coords[nu] = 0
        for rho in other:
            coords[rho] = 0
        add_edge(mu, coords, 1.0)

    for j in range(n):
        coords = [0] * D
        coords[mu] = n
        coords[nu] = j
        for rho in other:
            coords[rho] = 0
        add_edge(nu, coords, 1.0)

    for i in range(n - 1, -1, -1):
        coords = [0] * D
        coords[mu] = i
        coords[nu] = n
        for rho in other:
            coords[rho] = 0
        add_edge(mu, coords, -1.0)

    for j in range(n - 1, -1, -1):
        coords = [0] * D
        coords[mu] = 0
        coords[nu] = j
        for rho in other:
            coords[rho] = 0
        add_edge(nu, coords, -1.0)

    return r_all[data.boundary_links]


def minimized_hessian_for_functional(
    h: np.ndarray, r: np.ndarray, tol: float = 1e-10
) -> tuple[float, float]:
    """Return Hessian for retained b=r.a after minimizing boundary shapes.

    If r has a component along ker(h), the retained functional can change at
    zero energy, so the Hessian is zero.
    """
    evals, vecs = np.linalg.eigh(h)
    pos = evals > tol
    coeff = vecs.T @ r
    kernel_component = float(np.linalg.norm(coeff[~pos]))
    if kernel_component > 1e-8:
        return 0.0, kernel_component
    denom = float(np.sum((coeff[pos] ** 2) / evals[pos]))
    if denom <= 0:
        return 0.0, kernel_component
    return 1.0 / denom, kernel_component


def loop_rows(max_n: int) -> list[LoopRow]:
    rows: list[LoopRow] = []
    for n in range(1, max_n + 1):
        data = build_data(n)
        h_bd, _, _ = boundary_schur(data)
        for mu, nu in [(0, 1), (0, 2), (0, 3)]:
            r = loop_functional(data, mu, nu)
            hessian, kernel_component = minimized_hessian_for_functional(h_bd, r)
            rows.append(
                LoopRow(
                    n=n,
                    orientation=f"{mu}{nu}",
                    hessian=hessian,
                    kernel_component=kernel_component,
                )
            )
    return rows


def run_checks(max_n: int) -> None:
    for n in range(1, max_n + 1):
        row = rank_row(n)
        assert row.bianchi_residual == 0.0
        assert row.rank_d1 + row.rank_d2 == row.plaquettes
        assert row.schur_rank == row.rank_d1 - row.rank_dint
    for item in loop_rows(min(max_n, 3)):
        assert item.kernel_component < 1e-8
        assert item.hessian > 0.0


def print_tables(max_n: int, max_loop_n: int) -> None:
    print("4D CUBICAL RANK AND BIANCHI DIAGNOSTIC")
    print(
        "| n | links | plaquettes | cubes | interior links | rank D1 | "
        "rank D2 | rank D_int | Schur rank | complement dim | ||D2 D1|| |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n in range(1, max_n + 1):
        row = rank_row(n)
        print(
            f"| {row.n} | {row.links} | {row.plaquettes} | {row.cubes} | "
            f"{row.interior_links} | {row.rank_d1} | {row.rank_d2} | "
            f"{row.rank_dint} | {row.schur_rank} | "
            f"{row.schur_complement_dim} | {row.bianchi_residual:.1g} |"
        )

    print(f"\nMINIMIZED COARSE BOUNDARY-LOOP HESSIAN (shown for n <= {max_loop_n})")
    print("| n | loop plane | Hessian for b=loop circulation | kernel component |")
    print("|---:|---:|---:|---:|")
    for item in loop_rows(min(max_n, max_loop_n)):
        print(
            f"| {item.n} | {item.orientation} | "
            f"{item.hessian:.12g} | {item.kernel_component:.3g} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-n", type=int, default=3)
    parser.add_argument("--max-loop-n", type=int, default=3)
    args = parser.parse_args()
    run_checks(args.max_n)
    print_tables(args.max_n, args.max_loop_n)


if __name__ == "__main__":
    main()
