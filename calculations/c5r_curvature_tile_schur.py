"""Curvature-tile retained-variable Schur diagnostics for C5R.

# ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates

Retained variables are the 24 coarse plaquette-face circulations on the
boundary of one n^4 fixed-ratio cubical block: choose an orientation mu<nu and
fix the two transverse coordinates to 0 or n.  These variables satisfy seven
coarse Bianchi relations, leaving a rank-17 retained curvature quotient.

The script computes the minimized quadratic Wilson-action Hessian on that
quotient after interior and boundary-shape modes are optimized away.  It also
reports the Hessian after curvature-density normalization, i.e. dividing each
coarse face circulation by n^2.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass

import numpy as np

from c5q_4d_cubical_incidence import D, boundary_schur, build_data, rank


@dataclass(frozen=True)
class TileRow:
    n: int
    retained_faces: int
    retained_rank: int
    bianchi_relations: int
    boundary_kernel_leak: float
    flux_min_hessian: float
    flux_max_hessian: float
    flux_condition: float
    density_min_hessian: float
    density_max_hessian: float


def face_loop_functional(data, mu: int, nu: int, fixed: dict[int, int]) -> np.ndarray:
    """Loop around an n x n coordinate square in plane (mu,nu)."""
    n = data.n
    r_all = np.zeros(len(data.cells[1]), dtype=float)

    def coords_with(mu_value: int | None = None, nu_value: int | None = None) -> list[int]:
        coords = [0] * D
        for rho, value in fixed.items():
            coords[rho] = value
        if mu_value is not None:
            coords[mu] = mu_value
        if nu_value is not None:
            coords[nu] = nu_value
        return coords

    def add_edge(direction: int, coords: list[int], coeff: float) -> None:
        key = ((direction,), tuple(coords))
        r_all[data.index[1][key]] += coeff

    for i in range(n):
        add_edge(mu, coords_with(mu_value=i, nu_value=0), 1.0)
    for j in range(n):
        add_edge(nu, coords_with(mu_value=n, nu_value=j), 1.0)
    for i in range(n - 1, -1, -1):
        add_edge(mu, coords_with(mu_value=i, nu_value=n), -1.0)
    for j in range(n - 1, -1, -1):
        add_edge(nu, coords_with(mu_value=0, nu_value=j), -1.0)

    return r_all[data.boundary_links]


def tile_matrix(data) -> np.ndarray:
    rows = []
    for mu, nu in itertools.combinations(range(D), 2):
        transverse = [rho for rho in range(D) if rho not in (mu, nu)]
        for values in itertools.product([0, data.n], repeat=2):
            fixed = dict(zip(transverse, values))
            rows.append(face_loop_functional(data, mu, nu, fixed))
    return np.vstack(rows)


def effective_hessian_spectrum(h_boundary: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Return positive eigenvalues of K=(R H^+ R^T)^+ and kernel leak."""
    evals, vecs = np.linalg.eigh(h_boundary)
    pos = evals > 1e-10
    null = ~pos
    leak = 0.0
    if np.any(null):
        leak = float(np.linalg.norm(r @ vecs[:, null], ord=2))

    hplus = (vecs[:, pos] / evals[pos]) @ vecs[:, pos].T
    gram = r @ hplus @ r.T
    gram_evals = np.linalg.eigvalsh(gram)
    gram_pos = gram_evals[gram_evals > 1e-9]
    k_evals = np.sort(1.0 / gram_pos)[::-1]
    return k_evals, leak, len(gram_pos)


def analyze(n: int) -> TileRow:
    data = build_data(n)
    h_boundary, _, _ = boundary_schur(data)
    r_flux = tile_matrix(data)
    k_flux, leak, retained_rank = effective_hessian_spectrum(h_boundary, r_flux)
    density_scale = float(n**4)
    k_density = density_scale * k_flux
    return TileRow(
        n=n,
        retained_faces=r_flux.shape[0],
        retained_rank=retained_rank,
        bianchi_relations=r_flux.shape[0] - retained_rank,
        boundary_kernel_leak=leak,
        flux_min_hessian=float(np.min(k_flux)),
        flux_max_hessian=float(np.max(k_flux)),
        flux_condition=float(np.max(k_flux) / np.min(k_flux)),
        density_min_hessian=float(np.min(k_density)),
        density_max_hessian=float(np.max(k_density)),
    )


def run_checks(max_n: int) -> None:
    for n in range(1, max_n + 1):
        row = analyze(n)
        assert row.retained_faces == 24
        assert row.retained_rank == 17
        assert row.bianchi_relations == 7
        assert row.boundary_kernel_leak < 1e-8
        assert row.flux_min_hessian > 0
        assert row.density_min_hessian > 0


def print_table(max_n: int) -> None:
    print("CURVATURE-TILE SCHUR-COMPLEMENT DIAGNOSTIC")
    print(
        "| n | retained faces | retained rank | Bianchi relations | "
        "kernel leak | flux min H | flux max H | flux cond | "
        "density min H | density max H |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n in range(1, max_n + 1):
        row = analyze(n)
        print(
            f"| {row.n} | {row.retained_faces} | {row.retained_rank} | "
            f"{row.bianchi_relations} | {row.boundary_kernel_leak:.3g} | "
            f"{row.flux_min_hessian:.12g} | {row.flux_max_hessian:.12g} | "
            f"{row.flux_condition:.12g} | {row.density_min_hessian:.12g} | "
            f"{row.density_max_hessian:.12g} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-n", type=int, default=3)
    args = parser.parse_args()
    run_checks(args.max_n)
    print_table(args.max_n)


if __name__ == "__main__":
    main()

