"""Retained-quotient BCH/incidence bounds for C5V.

# ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates

This checker sharpens C5U by restricting the BCH/incidence Hessian-loss
calculation to the 17-dimensional scalar retained curvature-tile quotient
from C5R, with three SU(2) color coordinates per scalar quotient direction.

The retained section used here is the minimum quadratic-action lift of the
24 face-density variables, restricted to their rank-17 Bianchi-compatible
row space.  The output is a coefficient-Frobenius deterministic envelope for
the O(g) cubic and O(g^2) quartic Wilson-action Hessian losses on the resulting
51-dimensional real quotient-coordinate space.

This is still not the full nonlinear block theorem: high-mode centering,
quartic running-potential/reflection-positivity admission, chart failure, and
full conditional tails remain separate obligations.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from c5q_4d_cubical_incidence import build_data
from c5r_curvature_tile_schur import tile_matrix
from c5t_bch_incidence_expansion import plaquette_paths


LOCAL_DIM = 12
COLORS = 3


@dataclass(frozen=True)
class QuotientGeometry:
    scalar_rank: int
    color_rank: int
    constraint_residual: float
    lift_operator_norm: float
    lift_frobenius_norm: float
    quadratic_hessian_min: float
    quadratic_hessian_max: float
    plaquette_lift_norm_min: float
    plaquette_lift_norm_max: float


@dataclass(frozen=True)
class Bounds:
    local_cubic_frobenius: float
    local_bch_quartic_frobenius: float
    local_wilson_quartic_frobenius: float
    quotient_cubic_frobenius: float
    quotient_bch_quartic_frobenius: float
    quotient_wilson_quartic_frobenius: float
    c5s_half_margin: float


@dataclass(frozen=True)
class ShellRow:
    j: int
    g: float
    m_j: float
    g_m: float
    cubic_loss: float
    quartic_loss: float
    total_loss: float
    margin: float
    passes: bool


def cross_dual(
    av: np.ndarray,
    ag: np.ndarray,
    ah: np.ndarray,
    bv: np.ndarray,
    bg: np.ndarray,
    bh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross product for vector-valued functions through second derivatives."""
    val = np.cross(av, bv)
    grad = np.cross(ag, bv) + np.cross(av, bg)
    hess = (
        np.cross(ah, bv[None, None, :])
        + np.cross(ag[:, None, :], bg[None, :, :])
        + np.cross(ag[None, :, :], bg[:, None, :])
        + np.cross(av[None, None, :], bh)
    )
    return val, grad, hess


def add_dual(
    a: tuple[np.ndarray, np.ndarray, np.ndarray],
    b: tuple[np.ndarray, np.ndarray, np.ndarray],
    scale_b: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return a[0] + scale_b * b[0], a[1] + scale_b * b[1], a[2] + scale_b * b[2]


def scale_dual(
    a: tuple[np.ndarray, np.ndarray, np.ndarray], scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return scale * a[0], scale * a[1], scale * a[2]


def local_q_derivatives(
    z: np.ndarray,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Return q1,q2,q3 value/gradient/Hessian for one four-edge plaquette."""
    d = LOCAL_DIM
    zero_v = np.zeros(COLORS)
    zero_g = np.zeros((d, COLORS))
    zero_h = np.zeros((d, d, COLORS))
    q1 = (zero_v.copy(), zero_g.copy(), zero_h.copy())
    q2 = (zero_v.copy(), zero_g.copy(), zero_h.copy())
    q3 = (zero_v.copy(), zero_g.copy(), zero_h.copy())

    for edge in range(4):
        val = z[COLORS * edge : COLORS * (edge + 1)].copy()
        grad = np.zeros((d, COLORS))
        for color in range(COLORS):
            grad[COLORS * edge + color, color] = 1.0
        y1 = (val, grad, zero_h.copy())

        c11 = cross_dual(*q1, *y1)
        new_q1 = add_dual(q1, y1)
        new_q2 = add_dual(q2, scale_dual(c11, 0.5))

        c21 = cross_dual(*q2, *y1)
        q1_cross_c11 = cross_dual(*q1, *c11)
        y1_cross_q1 = cross_dual(*y1, *q1)
        y1_cross_y1_cross_q1 = cross_dual(*y1, *y1_cross_q1)
        new_q3 = add_dual(
            add_dual(q3, scale_dual(c21, 0.5)),
            scale_dual(add_dual(q1_cross_c11, y1_cross_y1_cross_q1), 1.0 / 12.0),
        )

        q1, q2, q3 = new_q1, new_q2, new_q3

    return q1, q2, q3


def dot_hessian(
    a: tuple[np.ndarray, np.ndarray, np.ndarray],
    b: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    av, ag, ah = a
    bv, bg, bh = b
    hess = np.einsum("c,cij->ij", av, np.transpose(bh, (2, 0, 1)), optimize=True)
    hess += np.einsum("c,cij->ij", bv, np.transpose(ah, (2, 0, 1)), optimize=True)
    hess += np.einsum("ic,jc->ij", ag, bg, optimize=True)
    hess += np.einsum("ic,jc->ij", bg, ag, optimize=True)
    return hess


def local_hessians(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hessians of S1, S2_BCH, and S2_Wilson for one plaquette."""
    q1, q2, q3 = local_q_derivatives(z)
    h1 = dot_hessian(q1, q2)
    h2_bch = dot_hessian(q1, q3)

    q2v, q2g, q2h = q2
    h2_bch += np.einsum(
        "c,cij->ij", q2v, np.transpose(q2h, (2, 0, 1)), optimize=True
    )
    h2_bch += np.einsum("ic,jc->ij", q2g, q2g, optimize=True)

    q1v, q1g, _ = q1
    q1_square = float(np.dot(q1v, q1v))
    grad_q1_square = 2.0 * np.einsum("c,ic->i", q1v, q1g, optimize=True)
    hess_q1_square = 2.0 * np.einsum("ic,jc->ij", q1g, q1g, optimize=True)
    h2_wilson = h2_bch - (
        np.outer(grad_q1_square, grad_q1_square)
        + q1_square * hess_q1_square
    ) / 12.0

    return (h1 + h1.T) / 2.0, (h2_bch + h2_bch.T) / 2.0, (h2_wilson + h2_wilson.T) / 2.0


def local_coefficient_tensors() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local Hessian coefficient tensors.

    H_S1(z) = sum_i z_i A_i.
    H_S2(z) = sum_{i,j} z_i z_j B_{ij}.
    """
    d = LOCAL_DIM
    a_tensor = np.zeros((d, d, d))
    bch_tensor = np.zeros((d, d, d, d))
    wilson_tensor = np.zeros_like(bch_tensor)

    basis_hess_bch: list[np.ndarray] = []
    basis_hess_wilson: list[np.ndarray] = []
    for i in range(d):
        e = np.zeros(d)
        e[i] = 1.0
        h1, h2_bch, h2_wilson = local_hessians(e)
        a_tensor[i] = h1
        basis_hess_bch.append(h2_bch)
        basis_hess_wilson.append(h2_wilson)
        bch_tensor[i, i] = h2_bch
        wilson_tensor[i, i] = h2_wilson

    for i in range(d):
        for j in range(i + 1, d):
            e = np.zeros(d)
            e[i] = 1.0
            e[j] = 1.0
            _, h2_bch, h2_wilson = local_hessians(e)
            bch_ij = (h2_bch - basis_hess_bch[i] - basis_hess_bch[j]) / 2.0
            wilson_ij = (
                h2_wilson - basis_hess_wilson[i] - basis_hess_wilson[j]
            ) / 2.0
            bch_tensor[i, j] = bch_ij
            bch_tensor[j, i] = bch_ij
            wilson_tensor[i, j] = wilson_ij
            wilson_tensor[j, i] = wilson_ij

    return a_tensor, bch_tensor, wilson_tensor


def build_retained_lift(n: int) -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    """Minimum quadratic-action lift from rank-17 density quotient to links."""
    data = build_data(n)
    r_boundary = tile_matrix(data) / float(n * n)
    constraint = np.zeros((r_boundary.shape[0], len(data.cells[1])))
    constraint[:, data.boundary_links] = r_boundary

    u, s, _ = np.linalg.svd(constraint, full_matrices=False)
    rank = int(np.sum(s > 1e-10))
    quotient_basis = u[:, :rank]

    hessian = data.d1.T @ data.d1
    kkt = np.block(
        [
            [hessian, constraint.T],
            [constraint, np.zeros((constraint.shape[0], constraint.shape[0]))],
        ]
    )

    lift = np.zeros((len(data.cells[1]), rank))
    for i in range(rank):
        rhs = np.concatenate([np.zeros(len(data.cells[1])), quotient_basis[:, i]])
        sol = np.linalg.lstsq(kkt, rhs, rcond=1e-10)[0]
        lift[:, i] = sol[: len(data.cells[1])]

    return data, lift, quotient_basis, constraint


def local_edge_map(path: list[tuple[int, float]], lift: np.ndarray) -> np.ndarray:
    """Map 51 quotient coordinates to one plaquette's 12 signed edge coordinates."""
    scalar_rank = lift.shape[1]
    dim = COLORS * scalar_rank
    edge_map = np.zeros((LOCAL_DIM, dim))
    row = 0
    for edge, sign in path:
        for color in range(COLORS):
            edge_map[row, color::COLORS] = sign * lift[edge]
            row += 1
    return edge_map


def assemble_quotient_tensors(
    data,
    lift: np.ndarray,
    local_a: np.ndarray,
    local_bch: np.ndarray,
    local_wilson: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble global quotient Hessian coefficient tensors."""
    dim = COLORS * lift.shape[1]
    global_a = np.zeros((dim, dim, dim))
    global_bch = np.zeros((dim, dim, dim, dim))
    global_wilson = np.zeros_like(global_bch)
    plaquette_norms: list[float] = []

    for path in plaquette_paths(data):
        edge_map = local_edge_map(path, lift)
        plaquette_norms.append(float(np.linalg.svd(edge_map, compute_uv=False)[0]))

        pulled_a = np.einsum(
            "xa,lxy,yb->lab", edge_map, local_a, edge_map, optimize=True
        )
        global_a += np.einsum("li,lab->iab", edge_map, pulled_a, optimize=True)

        pulled_bch = np.einsum(
            "xa,lmxy,yb->lmab", edge_map, local_bch, edge_map, optimize=True
        )
        global_bch += np.einsum(
            "li,mj,lmab->ijab", edge_map, edge_map, pulled_bch, optimize=True
        )

        pulled_wilson = np.einsum(
            "xa,lmxy,yb->lmab", edge_map, local_wilson, edge_map, optimize=True
        )
        global_wilson += np.einsum(
            "li,mj,lmab->ijab",
            edge_map,
            edge_map,
            pulled_wilson,
            optimize=True,
        )

    global_a = (global_a + np.transpose(global_a, (0, 2, 1))) / 2.0
    global_bch = (global_bch + np.transpose(global_bch, (1, 0, 2, 3))) / 2.0
    global_bch = (global_bch + np.transpose(global_bch, (0, 1, 3, 2))) / 2.0
    global_wilson = (
        global_wilson + np.transpose(global_wilson, (1, 0, 2, 3))
    ) / 2.0
    global_wilson = (
        global_wilson + np.transpose(global_wilson, (0, 1, 3, 2))
    ) / 2.0

    return global_a, global_bch, global_wilson, np.array(plaquette_norms)


def frobenius_coefficient_norm_cubic(tensor: np.ndarray) -> float:
    return float(np.linalg.norm(tensor.reshape(tensor.shape[0], -1), ord="fro"))


def frobenius_coefficient_norm_quartic(tensor: np.ndarray) -> float:
    dim = tensor.shape[0]
    return float(np.linalg.norm(tensor.reshape(dim * dim, -1), ord="fro"))


def build_geometry_and_bounds(margin: float) -> tuple[QuotientGeometry, Bounds]:
    data, lift, quotient_basis, constraint = build_retained_lift(2)
    local_a, local_bch, local_wilson = local_coefficient_tensors()
    global_a, global_bch, global_wilson, plaquette_norms = assemble_quotient_tensors(
        data, lift, local_a, local_bch, local_wilson
    )

    quadratic = lift.T @ (data.d1.T @ data.d1) @ lift
    q_eigs = np.linalg.eigvalsh(quadratic)
    lift_singulars = np.linalg.svd(lift, compute_uv=False)

    geom = QuotientGeometry(
        scalar_rank=lift.shape[1],
        color_rank=COLORS * lift.shape[1],
        constraint_residual=float(np.linalg.norm(constraint @ lift - quotient_basis)),
        lift_operator_norm=float(lift_singulars[0]),
        lift_frobenius_norm=float(np.linalg.norm(lift, ord="fro")),
        quadratic_hessian_min=float(q_eigs[0]),
        quadratic_hessian_max=float(q_eigs[-1]),
        plaquette_lift_norm_min=float(np.min(plaquette_norms)),
        plaquette_lift_norm_max=float(np.max(plaquette_norms)),
    )
    bounds = Bounds(
        local_cubic_frobenius=frobenius_coefficient_norm_cubic(local_a),
        local_bch_quartic_frobenius=frobenius_coefficient_norm_quartic(local_bch),
        local_wilson_quartic_frobenius=frobenius_coefficient_norm_quartic(
            local_wilson
        ),
        quotient_cubic_frobenius=frobenius_coefficient_norm_cubic(global_a),
        quotient_bch_quartic_frobenius=frobenius_coefficient_norm_quartic(
            global_bch
        ),
        quotient_wilson_quartic_frobenius=frobenius_coefficient_norm_quartic(
            global_wilson
        ),
        c5s_half_margin=margin,
    )
    return geom, bounds


def shell_row(j: int, mu: float, bounds: Bounds) -> ShellRow:
    g = 1.0 / math.sqrt(j)
    m_j = math.sqrt(mu * math.log(j + 1.0))
    gm = g * m_j
    cubic = bounds.quotient_cubic_frobenius * gm
    quartic = bounds.quotient_wilson_quartic_frobenius * gm * gm
    total = cubic + quartic
    return ShellRow(
        j=j,
        g=g,
        m_j=m_j,
        g_m=gm,
        cubic_loss=cubic,
        quartic_loss=quartic,
        total_loss=total,
        margin=bounds.c5s_half_margin,
        passes=total <= bounds.c5s_half_margin,
    )


def first_passing_shell(mu: float, bounds: Bounds, max_j: int) -> int | None:
    for j in range(1, max_j + 1):
        if shell_row(j, mu, bounds).passes:
            return j
    return None


def run_checks(geom: QuotientGeometry, bounds: Bounds, mu: float) -> None:
    assert geom.scalar_rank == 17
    assert geom.color_rank == 51
    assert geom.constraint_residual < 1e-9
    assert abs(geom.quadratic_hessian_min - 4.0) < 1e-9
    assert bounds.quotient_cubic_frobenius < 93.5307436087
    assert bounds.quotient_wilson_quartic_frobenius < 3451.84804771
    assert shell_row(100_000, mu, bounds).passes


def print_report(mu: float, margin: float, max_search: int) -> None:
    geom, bounds = build_geometry_and_bounds(margin)
    run_checks(geom, bounds, mu)
    first = first_passing_shell(mu, bounds, max_search)

    print("RETAINED-QUOTIENT GEOMETRY")
    print(
        "| scalar rank | color rank | constraint residual | ||lift|| | "
        "||lift||_F | quadratic H min | quadratic H max | min plaquette ||E_p|| | max plaquette ||E_p|| |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    print(
        f"| {geom.scalar_rank} | {geom.color_rank} | {geom.constraint_residual:.3g} | "
        f"{geom.lift_operator_norm:.12g} | {geom.lift_frobenius_norm:.12g} | "
        f"{geom.quadratic_hessian_min:.12g} | {geom.quadratic_hessian_max:.12g} | "
        f"{geom.plaquette_lift_norm_min:.12g} | {geom.plaquette_lift_norm_max:.12g} |"
    )

    print("\nCOEFFICIENT-FROBENIUS HESSIAN BOUNDS")
    print(
        "| level | cubic S1 | quartic S2_BCH | quartic S2_Wilson | C5S half-margin |"
    )
    print("|---|---:|---:|---:|---:|")
    print(
        f"| local one-plaquette | {bounds.local_cubic_frobenius:.12g} | "
        f"{bounds.local_bch_quartic_frobenius:.12g} | "
        f"{bounds.local_wilson_quartic_frobenius:.12g} | {bounds.c5s_half_margin:.12g} |"
    )
    print(
        f"| retained quotient | {bounds.quotient_cubic_frobenius:.12g} | "
        f"{bounds.quotient_bch_quartic_frobenius:.12g} | "
        f"{bounds.quotient_wilson_quartic_frobenius:.12g} | {bounds.c5s_half_margin:.12g} |"
    )

    print("\nULTRAVIOLET MARGIN SCHEDULE")
    print(f"mu={mu:g}, first passing shell up to search limit {max_search}: {first}")
    print(
        "| j | g | M_j | g M_j | cubic loss | quartic Wilson loss | total loss | margin | pass |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for j in [10**2, 10**3, 10**4, 10**5, 10**6, 10**7]:
        row = shell_row(j, mu, bounds)
        print(
            f"| {row.j} | {row.g:.10g} | {row.m_j:.10g} | {row.g_m:.10g} | "
            f"{row.cubic_loss:.12g} | {row.quartic_loss:.12g} | "
            f"{row.total_loss:.12g} | {row.margin:.12g} | {row.passes} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", type=float, default=16.0)
    parser.add_argument("--margin", type=float, default=1.08060461174)
    parser.add_argument("--max-search", type=int, default=10_000_000)
    args = parser.parse_args()
    print_report(args.mu, args.margin, args.max_search)


if __name__ == "__main__":
    main()
