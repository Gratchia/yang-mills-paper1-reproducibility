"""C5AW deterministic odd-channel bilinear envelope.

# ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates

C5AV showed that the analytic O(g^2) density source is quadratic in
A=K+H.  Therefore the only centered retained/high density source is an odd
bilinear form

    B(K,H) = K^T M_odd H.

C5AW builds M_odd explicitly for the finite n=2 SU(2) curvature-tile block.
It separates retained-coordinate, Haar, and Faddeev--Popov contributions,
computes operator and C5AL-covariance weighted L2 constants, and validates the
matrix against the C5AV sampled odd coefficient.

This is still a finite-block/tangent good-sector operator diagnostic.  It is
not a continuum construction, confinement proof, mass-gap proof, global compact
gauge theorem, or exceptional-sector estimate.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from c5ae_jacobian_half_density import tile_loop_paths
from c5al_gauge_zero_reference import (
    build_split,
    link_reflection_matrix,
    reflection_report,
)
from c5am_nonlinear_gauge_haar import gauge_slice_report, random_sphere
from c5ao_conditional_half_density_projection import gaussian_high_sample
from c5aq_centered_density_coefficient_envelope import residual_partial_sum
from c5v_retained_quotient_bch_bounds import (
    COLORS,
    add_dual,
    build_retained_lift,
    cross_dual,
    scale_dual,
)


@dataclass(frozen=True)
class MatrixNorms:
    source: str
    operator_norm: float
    frobenius_norm: float
    covariance_l2_constant_squared: float
    covariance_l2_constant: float
    sampled_raw_odd: float
    sampled_unit_odd_coeff: float


@dataclass(frozen=True)
class C5AWDiagnostic:
    retained_rank: int
    positive_high_rank: int
    gauge_vertex_rank: int
    retained_color_dim: int
    positive_high_color_dim: int
    gauge_color_dim: int
    combined_operator_norm: float
    combined_frobenius_norm: float
    combined_covariance_l2_constant_squared: float
    combined_covariance_l2_constant: float
    combined_sampled_raw_odd: float
    combined_sampled_unit_odd_coeff: float
    c5av_sampled_unit_odd_reference: float
    c5av_sampled_unit_odd_error: float
    bounded_good_sector_unit_proxy: float
    covariance_good_sector_unit_proxy: float
    bounded_residual_partial_10j0: float
    bounded_residual_partial_100j0: float
    covariance_residual_partial_10j0: float
    covariance_residual_partial_100j0: float
    retained_reflection_leak: float
    positive_reflection_leak: float
    retained_operator_reflection_residual: float
    haar_operator_reflection_residual: float
    fp_operator_reflection_residual: float
    operator_reflection_residual: float
    positive_covariance_reflection_commutator: float
    fp_m0_min_singular: float
    fp_m0_condition: float
    classification: str


@dataclass(frozen=True)
class C5AWDecision:
    explicit_bilinear_operator_built: bool
    c5av_sample_reproduced: bool
    covariance_l2_bound_computed: bool
    bounded_good_sector_bound_computed: bool
    reflection_compatible: bool
    deterministic_exceptional_sector_proved: bool
    decision: str


def color_lift_matrix(scalar_basis: np.ndarray) -> np.ndarray:
    return np.kron(scalar_basis, np.eye(COLORS))


def local_basis_for_path(path: list[tuple[int, float]], scalar_basis: np.ndarray) -> np.ndarray:
    """Map scalar link coordinates into the unsigned local path slots.

    The path orientation signs are already built into ``path_tensors(signs)``.
    Applying them here as well would double-count orientation on reversed
    segments.
    """
    local_dim = len(path) * COLORS
    color_dim = scalar_basis.shape[1] * COLORS
    out = np.zeros((local_dim, color_dim), dtype=float)
    for segment, (edge, _sign) in enumerate(path):
        for color in range(COLORS):
            out[COLORS * segment + color, color::COLORS] = scalar_basis[edge]
    return out


def path_q_derivatives(
    z: np.ndarray,
    signs: tuple[float, ...],
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Return q1,q2,q3 value/gradient/Hessian for one signed path."""
    d = z.size
    zero_v = np.zeros(COLORS)
    zero_g = np.zeros((d, COLORS))
    zero_h = np.zeros((d, d, COLORS))
    q1 = (zero_v.copy(), zero_g.copy(), zero_h.copy())
    q2 = (zero_v.copy(), zero_g.copy(), zero_h.copy())
    q3 = (zero_v.copy(), zero_g.copy(), zero_h.copy())

    for segment, sign in enumerate(signs):
        val = sign * z[COLORS * segment : COLORS * (segment + 1)].copy()
        grad = np.zeros((d, COLORS))
        for color in range(COLORS):
            grad[COLORS * segment + color, color] = sign
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


@lru_cache(maxsize=None)
def path_tensors(signs: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return q2 Hessian and q3 third derivative tensors for a path sign pattern."""
    d = COLORS * len(signs)
    zero = np.zeros(d)
    _, q2, _ = path_q_derivatives(zero, signs)
    h2 = np.moveaxis(q2[2], -1, 0)
    h2 = 0.5 * (h2 + np.transpose(h2, (0, 2, 1)))

    d3 = np.zeros((COLORS, d, d, d), dtype=float)
    for local_axis in range(d):
        z = np.zeros(d)
        z[local_axis] = 1.0
        _, _, q3 = path_q_derivatives(z, signs)
        d3[:, :, :, local_axis] = np.moveaxis(q3[2], -1, 0)

    # Symmetrize the three input slots.  The BCH polynomial is cubic, so this
    # removes only roundoff/asymmetry from the derivative construction.
    d3 = (
        d3
        + np.transpose(d3, (0, 1, 3, 2))
        + np.transpose(d3, (0, 2, 1, 3))
        + np.transpose(d3, (0, 2, 3, 1))
        + np.transpose(d3, (0, 3, 1, 2))
        + np.transpose(d3, (0, 3, 2, 1))
    ) / 6.0
    return h2, d3


def retained_odd_matrix(
    data,
    lift: np.ndarray,
    positive_basis: np.ndarray,
    quotient_basis: np.ndarray,
    paths: list[list[tuple[int, float]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    retained_scalar = lift.shape[1]
    retained_dim = retained_scalar * COLORS
    high_dim = positive_basis.shape[1] * COLORS
    scale = float(data.n * data.n)
    trace_j2_bilinear = np.zeros((retained_dim, high_dim), dtype=float)
    j1_retained = np.zeros((retained_dim, retained_dim, retained_dim), dtype=float)
    j1_high = np.zeros((high_dim, retained_dim, retained_dim), dtype=float)

    for path_index, path in enumerate(paths):
        signs = tuple(sign for _, sign in path)
        h2, d3 = path_tensors(signs)
        rloc = local_basis_for_path(path, lift)
        zloc = local_basis_for_path(path, positive_basis)

        for color in range(COLORS):
            h2_color = h2[color]
            rr = rloc.T @ h2_color @ rloc
            rz = rloc.T @ h2_color @ zloc
            d3_color = d3[color]

            for scalar_out in range(retained_scalar):
                weight = quotient_basis[path_index, scalar_out] / scale
                if abs(weight) < 1.0e-15:
                    continue
                out = scalar_out * COLORS + color

                # J1(A)_{out, retained_direction} = D K1_out(A)[R_direction].
                j1_retained[:, out, :] += weight * rr.T
                j1_high[:, out, :] += weight * rz.T

                # Q_ret term from 1/2 tr J2.  For cubic K2, the bilinearization
                # of D K2_out(A)[R_out] contributes 1/2 D^3 K2_out(R_out,K,H).
                rtrace = rloc[:, out]
                contracted = np.einsum("a,abc->bc", rtrace, d3_color, optimize=True)
                trace_j2_bilinear += 0.5 * weight * (rloc.T @ contracted @ zloc)

    j1_square_bilinear = -0.5 * np.einsum(
        "iab,jba->ij", j1_retained, j1_high, optimize=True
    )
    return (
        trace_j2_bilinear + j1_square_bilinear,
        trace_j2_bilinear,
        j1_square_bilinear,
    )


def gauge_edge_data(data, vertex_basis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertex_rank = vertex_basis.shape[1]
    gauge_dim = vertex_rank * COLORS
    vertices = vertex_basis.shape[0]
    edges = len(data.cells[1])
    phi_basis = np.zeros((gauge_dim, vertices, COLORS), dtype=float)
    for scalar in range(vertex_rank):
        for color in range(COLORS):
            phi_basis[scalar * COLORS + color, :, color] = vertex_basis[:, scalar]

    head_phi = np.zeros((gauge_dim, edges, COLORS), dtype=float)
    tail_phi = np.zeros_like(head_phi)
    for edge, (dirs, coords_tuple) in enumerate(data.cells[1]):
        direction = dirs[0]
        tail = data.index[0][((), coords_tuple)]
        head_coords = list(coords_tuple)
        head_coords[direction] += 1
        head = data.index[0][((), tuple(head_coords))]
        head_phi[:, edge, :] = phi_basis[:, head, :]
        tail_phi[:, edge, :] = phi_basis[:, tail, :]
    delta_phi = head_phi - tail_phi
    return head_phi, tail_phi, delta_phi


def fp_precompute(data, d0: np.ndarray, vertex_basis: np.ndarray):
    vertex_rank = vertex_basis.shape[1]
    gauge_dim = vertex_rank * COLORS
    edges = len(data.cells[1])
    g_scalar = vertex_basis.T @ d0.T
    g_flat = np.kron(g_scalar, np.eye(COLORS))
    g_edge = g_flat.reshape(gauge_dim, edges, COLORS)
    base_scalar = (d0 @ vertex_basis).T @ (d0 @ vertex_basis)
    m0 = np.kron(base_scalar, np.eye(COLORS))
    inv_m0 = np.linalg.inv(m0)
    singulars = np.linalg.svd(m0, compute_uv=False)

    head_phi, tail_phi, delta_phi = gauge_edge_data(data, vertex_basis)
    color_eye = np.eye(COLORS)
    a1_kernel = np.zeros((gauge_dim, edges, COLORS, COLORS), dtype=float)
    for color in range(COLORS):
        unit = color_eye[color]
        a1_kernel[:, :, color, :] = 0.5 * (
            np.cross(head_phi, unit) - np.cross(unit, tail_phi)
        )

    w_edge = (inv_m0 @ g_flat).reshape(gauge_dim, edges, COLORS)
    return g_edge, a1_kernel, delta_phi, w_edge, inv_m0, float(singulars[-1]), float(singulars[0] / singulars[-1])


def n1_coordinate_matrices(
    scalar_basis: np.ndarray,
    g_edge: np.ndarray,
    a1_kernel: np.ndarray,
    inv_m0: np.ndarray,
) -> np.ndarray:
    """Return N1=M0^{-1}M1 for each color coordinate of scalar_basis."""
    m1 = np.einsum(
        "rev,qecv,eb->rqbc",
        g_edge,
        a1_kernel,
        scalar_basis,
        optimize=True,
    )
    coord_count = scalar_basis.shape[1] * COLORS
    m1_coord = np.transpose(m1, (2, 3, 0, 1)).reshape(
        coord_count, inv_m0.shape[0], inv_m0.shape[1]
    )
    return np.einsum("ab,ibc->iac", inv_m0, m1_coord, optimize=True)


def fp_m2_bilinear_edge_tensors(delta_phi: np.ndarray, w_edge: np.ndarray) -> np.ndarray:
    gauge_dim, edges, _ = delta_phi.shape
    tensors = np.zeros((edges, COLORS, COLORS), dtype=float)
    eye = np.eye(COLORS)
    for edge in range(edges):
        for gauge_axis in range(gauge_dim):
            delta = delta_phi[gauge_axis, edge]
            weight = w_edge[gauge_axis, edge]
            for left_color in range(COLORS):
                left = eye[left_color]
                for right_color in range(COLORS):
                    right = eye[right_color]
                    value = (
                        np.cross(left, np.cross(right, delta))
                        + np.cross(right, np.cross(left, delta))
                    ) / 24.0
                    tensors[edge, left_color, right_color] += float(np.dot(weight, value))
    return tensors


def scalar_color_bilinear_from_edge_tensors(
    left_basis: np.ndarray,
    right_basis: np.ndarray,
    edge_tensors: np.ndarray,
) -> np.ndarray:
    left_rank = left_basis.shape[1]
    right_rank = right_basis.shape[1]
    out = np.zeros((left_rank * COLORS, right_rank * COLORS), dtype=float)
    for edge in range(left_basis.shape[0]):
        scalar_outer = np.outer(left_basis[edge], right_basis[edge])
        out += np.einsum("ij,ab->iajb", scalar_outer, edge_tensors[edge], optimize=True).reshape(
            left_rank * COLORS, right_rank * COLORS
        )
    return out


def fp_odd_matrix(
    data,
    d0: np.ndarray,
    vertex_basis: np.ndarray,
    lift: np.ndarray,
    positive_basis: np.ndarray,
    chunk_scalar: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    g_edge, a1_kernel, delta_phi, w_edge, inv_m0, min_singular, condition = fp_precompute(
        data, d0, vertex_basis
    )
    retained_dim = lift.shape[1] * COLORS
    high_dim = positive_basis.shape[1] * COLORS

    edge_tensors = fp_m2_bilinear_edge_tensors(delta_phi, w_edge)
    m2_term = scalar_color_bilinear_from_edge_tensors(lift, positive_basis, edge_tensors)

    n1_retained = n1_coordinate_matrices(lift, g_edge, a1_kernel, inv_m0)
    n1_square_term = np.zeros((retained_dim, high_dim), dtype=float)
    for start in range(0, positive_basis.shape[1], chunk_scalar):
        stop = min(start + chunk_scalar, positive_basis.shape[1])
        n1_high = n1_coordinate_matrices(
            positive_basis[:, start:stop], g_edge, a1_kernel, inv_m0
        )
        n1_square_term[:, start * COLORS : stop * COLORS] = -0.5 * np.einsum(
            "iab,jba->ij", n1_retained, n1_high, optimize=True
        )

    return m2_term + n1_square_term, m2_term, n1_square_term, min_singular, condition


def haar_odd_matrix(lift: np.ndarray, positive_basis: np.ndarray) -> np.ndarray:
    return -(1.0 / 12.0) * np.kron(lift.T @ positive_basis, np.eye(COLORS))


def covariance_l2_squared(matrix: np.ndarray, positive_evals: np.ndarray) -> float:
    cov_diag = np.repeat(1.0 / positive_evals, COLORS)
    weighted = matrix * np.sqrt(cov_diag)[None, :]
    gram = weighted @ weighted.T
    gram = 0.5 * (gram + gram.T)
    return float(np.linalg.eigvalsh(gram)[-1])


def matrix_norm_row(
    source: str,
    matrix: np.ndarray,
    positive_evals: np.ndarray,
    k_sample: np.ndarray,
    y_sample: np.ndarray,
    retained_radius: float,
    gaussian_scale: float,
) -> MatrixNorms:
    cov_sq = covariance_l2_squared(matrix, positive_evals)
    sampled_raw = float(k_sample @ matrix @ y_sample)
    return MatrixNorms(
        source=source,
        operator_norm=float(np.linalg.svd(matrix, compute_uv=False)[0]),
        frobenius_norm=float(np.linalg.norm(matrix)),
        covariance_l2_constant_squared=cov_sq,
        covariance_l2_constant=math.sqrt(max(cov_sq, 0.0)),
        sampled_raw_odd=sampled_raw,
        sampled_unit_odd_coeff=float(sampled_raw / (retained_radius * gaussian_scale)),
    )


def reflection_coordinate_report(
    data,
    lift: np.ndarray,
    positive_basis: np.ndarray,
    matrix: np.ndarray,
) -> tuple[float, float, float]:
    reflection = link_reflection_matrix(data)
    retained_color = color_lift_matrix(lift)
    positive_color = color_lift_matrix(positive_basis)
    reflection_color = np.kron(reflection, np.eye(COLORS))

    retained_rep = np.linalg.pinv(retained_color) @ reflection_color @ retained_color
    positive_rep = positive_color.T @ reflection_color @ positive_color
    retained_leak = float(
        np.linalg.norm(reflection_color @ retained_color - retained_color @ retained_rep)
        / max(np.linalg.norm(retained_color), 1.0e-30)
    )
    positive_leak = float(
        np.linalg.norm(reflection_color @ positive_color - positive_color @ positive_rep)
        / max(np.linalg.norm(positive_color), 1.0e-30)
    )
    operator_residual = float(
        np.linalg.norm(retained_rep.T @ matrix @ positive_rep - matrix)
        / max(np.linalg.norm(matrix), 1.0e-30)
    )
    return retained_leak, positive_leak, operator_residual


def build_all_matrices(tol: float, chunk_scalar: int):
    data, lift, quotient_basis, _ = build_retained_lift(2)
    data_gauge, lift_gauge, d0, vertex_basis, gauge = gauge_slice_report(tol)
    if len(data.cells[1]) != len(data_gauge.cells[1]) or not np.allclose(lift, lift_gauge):
        raise RuntimeError("C5AW inconsistent retained lift/gauge data")
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

    retained_matrix, retained_trace_j2, retained_j1_square = retained_odd_matrix(
        data, lift, positive_basis, quotient_basis, paths
    )
    fp_matrix, fp_m2, fp_n1_square, fp_min_singular, fp_condition = fp_odd_matrix(
        data, d0, vertex_basis, lift, positive_basis, chunk_scalar
    )
    haar_matrix = haar_odd_matrix(lift, positive_basis)
    combined_matrix = retained_matrix + haar_matrix + fp_matrix
    return {
        "data": data,
        "lift": lift,
        "positive_basis": positive_basis,
        "positive_evals": positive_evals,
        "gauge": gauge,
        "retained": retained_matrix,
        "retained_trace_j2": retained_trace_j2,
        "retained_j1_square": retained_j1_square,
        "Haar half": haar_matrix,
        "FP half": fp_matrix,
        "FP m2": fp_m2,
        "FP n1 square": fp_n1_square,
        "combined": combined_matrix,
        "fp_min_singular": fp_min_singular,
        "fp_condition": fp_condition,
    }


def run_diagnostic(
    seed: int,
    tol: float,
    retained_radius: float,
    gaussian_scale: float,
    j0: int,
    mu: float,
    chunk_scalar: int,
) -> tuple[C5AWDiagnostic, C5AWDecision, list[MatrixNorms]]:
    matrices = build_all_matrices(tol, chunk_scalar)
    lift = matrices["lift"]
    positive_basis = matrices["positive_basis"]
    positive_evals = matrices["positive_evals"]
    gauge = matrices["gauge"]

    rng = np.random.default_rng(seed)
    k_sample = random_sphere(rng, lift.shape[1] * COLORS, retained_radius)
    y_sample = gaussian_high_sample(rng, positive_evals, gaussian_scale)

    rows = [
        matrix_norm_row(
            source,
            matrices[source],
            positive_evals,
            k_sample,
            y_sample,
            retained_radius,
            gaussian_scale,
        )
        for source in ("retained", "Haar half", "FP half", "combined")
    ]
    row_by_source = {row.source: row for row in rows}
    combined = row_by_source["combined"]

    reflection = reflection_report(tol)
    retained_leak, positive_leak, operator_reflection = reflection_coordinate_report(
        matrices["data"], lift, positive_basis, matrices["combined"]
    )
    _, _, retained_operator_reflection = reflection_coordinate_report(
        matrices["data"], lift, positive_basis, matrices["retained"]
    )
    _, _, haar_operator_reflection = reflection_coordinate_report(
        matrices["data"], lift, positive_basis, matrices["Haar half"]
    )
    _, _, fp_operator_reflection = reflection_coordinate_report(
        matrices["data"], lift, positive_basis, matrices["FP half"]
    )

    bounded_proxy = combined.operator_norm * combined.operator_norm
    covariance_proxy = combined.covariance_l2_constant_squared
    c5av_reference = 0.0225939725197
    diagnostic = C5AWDiagnostic(
        retained_rank=lift.shape[1],
        positive_high_rank=positive_basis.shape[1],
        gauge_vertex_rank=gauge.mean_zero_vertex_rank,
        retained_color_dim=lift.shape[1] * COLORS,
        positive_high_color_dim=positive_basis.shape[1] * COLORS,
        gauge_color_dim=gauge.mean_zero_vertex_rank * COLORS,
        combined_operator_norm=combined.operator_norm,
        combined_frobenius_norm=combined.frobenius_norm,
        combined_covariance_l2_constant_squared=combined.covariance_l2_constant_squared,
        combined_covariance_l2_constant=combined.covariance_l2_constant,
        combined_sampled_raw_odd=combined.sampled_raw_odd,
        combined_sampled_unit_odd_coeff=combined.sampled_unit_odd_coeff,
        c5av_sampled_unit_odd_reference=c5av_reference,
        c5av_sampled_unit_odd_error=abs(combined.sampled_unit_odd_coeff - c5av_reference),
        bounded_good_sector_unit_proxy=bounded_proxy,
        covariance_good_sector_unit_proxy=covariance_proxy,
        bounded_residual_partial_10j0=residual_partial_sum(j0, 10 * j0, mu, bounded_proxy),
        bounded_residual_partial_100j0=residual_partial_sum(j0, 100 * j0, mu, bounded_proxy),
        covariance_residual_partial_10j0=residual_partial_sum(j0, 10 * j0, mu, covariance_proxy),
        covariance_residual_partial_100j0=residual_partial_sum(j0, 100 * j0, mu, covariance_proxy),
        retained_reflection_leak=retained_leak,
        positive_reflection_leak=positive_leak,
        retained_operator_reflection_residual=retained_operator_reflection,
        haar_operator_reflection_residual=haar_operator_reflection,
        fp_operator_reflection_residual=fp_operator_reflection,
        operator_reflection_residual=operator_reflection,
        positive_covariance_reflection_commutator=reflection.covariance_reflection_commutator,
        fp_m0_min_singular=matrices["fp_min_singular"],
        fp_m0_condition=matrices["fp_condition"],
        classification=(
            "partial pass: an explicit finite-block odd bilinear operator is built "
            "and its covariance/good-sector L2 constants are finite and summable "
            "under the C5M schedule; retained-coordinate reflection placement, "
            "uniform nonlinear good-sector control, and exceptional-sector proofs remain open"
        ),
    )
    decision = make_decision(diagnostic)
    return diagnostic, decision, rows


def make_decision(row: C5AWDiagnostic) -> C5AWDecision:
    built = (
        row.retained_rank == 17
        and row.positive_high_rank == 119
        and row.gauge_vertex_rank == 80
        and row.retained_color_dim == 51
        and row.positive_high_color_dim == 357
        and row.gauge_color_dim == 240
        and row.fp_m0_min_singular > 1.0e-4
        and row.fp_m0_condition < 100.0
        and math.isfinite(row.combined_operator_norm)
        and row.combined_operator_norm > 0.0
    )
    reproduced = built and row.c5av_sampled_unit_odd_error < 1.0e-10
    cov = built and math.isfinite(row.combined_covariance_l2_constant_squared)
    bounded = built and math.isfinite(row.bounded_good_sector_unit_proxy)
    reflection_ok = (
        row.retained_reflection_leak < 1.0e-10
        and row.positive_reflection_leak < 1.0e-10
        and row.operator_reflection_residual < 1.0e-8
        and row.positive_covariance_reflection_commutator < 1.0e-10
    )
    return C5AWDecision(
        explicit_bilinear_operator_built=built,
        c5av_sample_reproduced=reproduced,
        covariance_l2_bound_computed=cov,
        bounded_good_sector_bound_computed=bounded,
        reflection_compatible=reflection_ok,
        deterministic_exceptional_sector_proved=False,
        decision=(
            "partial pass: finite-block M_odd is explicit and reproduces C5AV; "
            "operator and covariance L2 bounds are summable, while retained-coordinate "
            "reflection placement, nonlinear uniformity, and exceptional-sector estimates remain open"
            if built and reproduced and cov and bounded
            else "fail/redesign: the explicit odd-operator construction did not pass validation"
        ),
    )


def run_checks(row: C5AWDiagnostic, decision: C5AWDecision) -> None:
    assert row.retained_rank == 17
    assert row.positive_high_rank == 119
    assert row.gauge_vertex_rank == 80
    assert row.retained_color_dim == 51
    assert row.positive_high_color_dim == 357
    assert row.gauge_color_dim == 240
    assert row.fp_m0_min_singular > 1.0e-4
    assert row.fp_m0_condition < 100.0
    assert decision.explicit_bilinear_operator_built
    assert decision.c5av_sample_reproduced
    assert decision.covariance_l2_bound_computed
    assert decision.bounded_good_sector_bound_computed
    assert not decision.deterministic_exceptional_sector_proved


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
    j0: int,
    mu: float,
    chunk_scalar: int,
) -> None:
    row, decision, rows = run_diagnostic(
        seed, tol, retained_radius, gaussian_scale, j0, mu, chunk_scalar
    )
    run_checks(row, decision)
    print("C5AW ODD-CHANNEL BILINEAR ENVELOPE DIAGNOSTIC")
    print("| quantity | value |")
    print("|---|---:|")
    for key, value in row.__dict__.items():
        print(f"| {key} | {fmt(value)} |")
    print("\nSOURCE OPERATOR TABLE")
    print(
        "| source | operator norm | Frobenius norm | covariance L2 constant^2 | "
        "covariance L2 constant | sampled raw odd | sampled unit odd coeff |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for item in rows:
        print(
            f"| {item.source} | {fmt(item.operator_norm)} | {fmt(item.frobenius_norm)} | "
            f"{fmt(item.covariance_l2_constant_squared)} | "
            f"{fmt(item.covariance_l2_constant)} | {fmt(item.sampled_raw_odd)} | "
            f"{fmt(item.sampled_unit_odd_coeff)} |"
        )
    print("\nDECISION")
    print(
        "| explicit operator built | C5AV sample reproduced | covariance L2 bound | "
        "bounded good-sector bound | reflection compatible | exceptional sector proved | decision |"
    )
    print("|---|---|---|---|---|---|---|")
    print(
        f"| {decision.explicit_bilinear_operator_built} | "
        f"{decision.c5av_sample_reproduced} | "
        f"{decision.covariance_l2_bound_computed} | "
        f"{decision.bounded_good_sector_bound_computed} | "
        f"{decision.reflection_compatible} | "
        f"{decision.deterministic_exceptional_sector_proved} | "
        f"{decision.decision} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--retained-radius", type=float, default=0.25)
    parser.add_argument("--gaussian-scale", type=float, default=0.25)
    parser.add_argument("--j0", type=int, default=13082)
    parser.add_argument("--mu", type=float, default=16.0)
    parser.add_argument("--chunk-scalar", type=int, default=24)
    args = parser.parse_args()
    print_report(
        args.seed,
        args.tol,
        args.retained_radius,
        args.gaussian_scale,
        args.j0,
        args.mu,
        args.chunk_scalar,
    )


if __name__ == "__main__":
    main()
