"""C5AM nonlinear gauge quotient and Haar/Faddeev--Popov diagnostic.

# ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates

C5AM probes a concrete nonlinear local gauge slice near the identity:

    D0^T A^phi = 0, modulo the global constant gauge mode.

Here A^phi is obtained by compact SU(2) lattice gauge transformation followed
by the principal logarithm.  The tangent Faddeev--Popov operator is the scalar
vertex Laplacian D0^T D0 on the mean-zero vertex subspace, tensored with color.

The checker tests:

* the tangent gauge slice is reflection-compatible;
* the nonlinear FP matrix remains invertible on small finite-block samples;
* the FP and link-Haar logarithmic densities have no O(g) odd term in the
  sampled probes and begin as O(g^2) density data.

This is not a proof of the full compact SU(2) gauge-fixed Haar disintegration,
continuum construction, or mass gap.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from c5ad_cubic_coordinate_normalization import exp_lie, log_lie, quat_mul
from c5al_gauge_zero_reference import link_reflection_matrix, orthonormal_column_space
from c5q_4d_cubical_incidence import coboundary
from c5v_retained_quotient_bch_bounds import COLORS, build_retained_lift


@dataclass(frozen=True)
class GaugeSliceReport:
    vertices: int
    scalar_links: int
    mean_zero_vertex_rank: int
    color_gauge_rank: int
    global_gauge_kernel_dim: int
    fp_base_min_eigenvalue: float
    fp_base_max_eigenvalue: float
    fp_base_condition: float
    vertex_reflection_orthogonality_error: float
    link_vertex_reflection_residual: float
    fp_reflection_commutator: float


@dataclass(frozen=True)
class NonlinearFPReport:
    samples: int
    g_probe: float
    coordinate_radius: float
    fp_step: float
    min_fp_singular_value: float
    max_fp_singular_value: float
    max_fp_condition: float
    max_fp_logdet_even_coeff_abs: float
    max_fp_logdet_odd_coeff_abs: float
    max_fp_logdet_relative_abs: float
    fp_orientation_failures: int


@dataclass(frozen=True)
class HaarDensityReport:
    samples: int
    g_probe: float
    max_link_haar_even_coeff_abs: float
    max_link_haar_odd_coeff_abs: float
    max_link_haar_quadratic_model_error: float
    max_sample_link_norm: float
    classification: str


@dataclass(frozen=True)
class C5AMDecision:
    tangent_gauge_slice_pass: bool
    nonlinear_fp_sample_pass: bool
    density_classified_as_half_density: bool
    compact_haar_disintegration_proved: bool
    decision: str


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def vertex_reflection_matrix(data) -> np.ndarray:
    vertices = len(data.cells[0])
    reflection = np.zeros((vertices, vertices))
    for idx, (_, coords_tuple) in enumerate(data.cells[0]):
        coords = list(coords_tuple)
        coords[0] = data.n - coords[0]
        reflected = data.index[0][((), tuple(coords))]
        reflection[reflected, idx] = 1.0
    return reflection


def mean_zero_vertex_basis(data, tol: float) -> tuple[np.ndarray, int]:
    vertices = len(data.cells[0])
    ones = np.ones((1, vertices)) / math.sqrt(float(vertices))
    basis, rank = orthonormal_column_space(np.eye(vertices) - ones.T @ ones, tol)
    return basis, rank


def gauge_slice_report(tol: float) -> tuple[object, np.ndarray, np.ndarray, np.ndarray, GaugeSliceReport]:
    data, lift, _, _ = build_retained_lift(2)
    d0 = coboundary(0, data.cells[0], data.index[0], data.cells[1])
    vertex_basis, vertex_rank = mean_zero_vertex_basis(data, tol)
    d0_reduced = d0 @ vertex_basis
    fp_base = d0_reduced.T @ d0_reduced
    evals = np.linalg.eigvalsh((fp_base + fp_base.T) / 2.0)

    r_link = link_reflection_matrix(data)
    r_vertex = vertex_reflection_matrix(data)
    reflection_residual = float(np.linalg.norm(r_link @ d0 - d0 @ r_vertex))
    r_vertex_reduced = vertex_basis.T @ r_vertex @ vertex_basis
    fp_reflection_commutator = float(
        np.linalg.norm(r_vertex_reduced.T @ fp_base @ r_vertex_reduced - fp_base)
    )

    report = GaugeSliceReport(
        vertices=len(data.cells[0]),
        scalar_links=len(data.cells[1]),
        mean_zero_vertex_rank=vertex_rank,
        color_gauge_rank=vertex_rank * COLORS,
        global_gauge_kernel_dim=1,
        fp_base_min_eigenvalue=float(np.min(evals)),
        fp_base_max_eigenvalue=float(np.max(evals)),
        fp_base_condition=float(np.max(evals) / np.min(evals)),
        vertex_reflection_orthogonality_error=float(
            np.linalg.norm(r_vertex.T @ r_vertex - np.eye(r_vertex.shape[0]))
        ),
        link_vertex_reflection_residual=reflection_residual,
        fp_reflection_commutator=fp_reflection_commutator,
    )
    return data, lift, d0, vertex_basis, report


def positive_high_basis(tol: float) -> np.ndarray:
    data, _, _, constraint = build_retained_lift(2)
    _, _, vt = np.linalg.svd(constraint, full_matrices=True)
    constraint_rank = int(np.linalg.matrix_rank(constraint, tol=tol))
    high_basis = vt[constraint_rank:].T
    hessian = data.d1.T @ data.d1
    high_hessian = high_basis.T @ hessian @ high_basis
    high_hessian = (high_hessian + high_hessian.T) / 2.0
    evals, vecs = np.linalg.eigh(high_hessian)
    return high_basis @ vecs[:, evals > tol]


def links_from_retained_positive(
    lift: np.ndarray,
    positive_basis: np.ndarray,
    k: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    scalar_retained = lift.shape[1]
    scalar_positive = positive_basis.shape[1]
    return lift @ k.reshape(scalar_retained, COLORS) + positive_basis @ y.reshape(
        scalar_positive, COLORS
    )


def random_sphere(rng: np.random.Generator, dim: int, radius: float) -> np.ndarray:
    x = rng.normal(size=dim)
    x /= np.linalg.norm(x)
    return radius * x


def transformed_links(
    data,
    links: np.ndarray,
    vertex_basis: np.ndarray,
    psi: np.ndarray,
    g: float,
) -> np.ndarray:
    """Compact gauge transform followed by principal logarithm.

    Convention: U_e -> G_head U_e G_tail^{-1}; tangent action is
    A_e -> A_e + (D0 phi)_e.
    """
    vertex_rank = vertex_basis.shape[1]
    phi = vertex_basis @ psi.reshape(vertex_rank, COLORS)
    out = np.zeros_like(links)
    if abs(g) < 1.0e-14:
        d0 = coboundary(0, data.cells[0], data.index[0], data.cells[1])
        return links + d0 @ phi

    for idx, (dirs, coords_tuple) in enumerate(data.cells[1]):
        direction = dirs[0]
        tail = data.index[0][((), coords_tuple)]
        head_coords = list(coords_tuple)
        head_coords[direction] += 1
        head = data.index[0][((), tuple(head_coords))]
        q_head = exp_lie(g * phi[head])
        q_link = exp_lie(g * links[idx])
        q_tail_inv = quat_conj(exp_lie(g * phi[tail]))
        q_new = quat_mul(quat_mul(q_head, q_link), q_tail_inv)
        out[idx] = log_lie(q_new) / g
    return out


def gauge_condition(
    data,
    d0: np.ndarray,
    links: np.ndarray,
    vertex_basis: np.ndarray,
    psi: np.ndarray,
    g: float,
) -> np.ndarray:
    a_phi = transformed_links(data, links, vertex_basis, psi, g)
    divergence = d0.T @ a_phi
    return (vertex_basis.T @ divergence).reshape(-1)


def finite_difference_fp_matrix(
    data,
    d0: np.ndarray,
    links: np.ndarray,
    vertex_basis: np.ndarray,
    g: float,
    step: float,
) -> np.ndarray:
    dim = vertex_basis.shape[1] * COLORS
    matrix = np.zeros((dim, dim))
    zero = np.zeros(dim)
    for i in range(dim):
        delta = np.zeros(dim)
        delta[i] = step
        fp_plus = gauge_condition(data, d0, links, vertex_basis, zero + delta, g)
        fp_minus = gauge_condition(data, d0, links, vertex_basis, zero - delta, g)
        matrix[:, i] = (fp_plus - fp_minus) / (2.0 * step)
    return matrix


def logdet_with_orientation(matrix: np.ndarray) -> tuple[float, bool, np.ndarray]:
    singulars = np.linalg.svd(matrix, compute_uv=False)
    sign, logdet = np.linalg.slogdet(matrix)
    return float(logdet), bool(sign > 0.0), singulars


def link_haar_log_density(links: np.ndarray, g: float) -> float:
    """Product Haar density in exponential A-coordinates, up to a constant."""
    if abs(g) < 1.0e-14:
        return 0.0
    total = 0.0
    for vec in links:
        radius = abs(g) * float(np.linalg.norm(vec))
        if radius < 1.0e-8:
            # 2 log(sin(r/2)/(r/2)) = -r^2/12 - r^4/1440 + ...
            total += -radius * radius / 12.0 - radius**4 / 1440.0
        else:
            ratio = math.sin(0.5 * radius) / (0.5 * radius)
            total += 2.0 * math.log(ratio)
    return total


def nonlinear_fp_and_haar_report(
    tol: float,
    samples: int,
    seed: int,
    g_probe: float,
    coordinate_radius: float,
    fp_step: float,
) -> tuple[NonlinearFPReport, HaarDensityReport]:
    data, lift, d0, vertex_basis, _ = gauge_slice_report(tol)
    pos_basis = positive_high_basis(tol)
    rng = np.random.default_rng(seed)
    retained_dim = lift.shape[1] * COLORS
    positive_dim = pos_basis.shape[1] * COLORS
    base_scalar = (d0 @ vertex_basis).T @ (d0 @ vertex_basis)
    base_logdet = COLORS * float(np.linalg.slogdet(base_scalar)[1])

    min_singulars: list[float] = []
    max_singulars: list[float] = []
    conditions: list[float] = []
    fp_even_coeffs: list[float] = []
    fp_odd_coeffs: list[float] = []
    fp_relative: list[float] = []
    haar_even_coeffs: list[float] = []
    haar_odd_coeffs: list[float] = []
    haar_model_errors: list[float] = []
    link_norms: list[float] = []
    failures = 0

    for _ in range(samples):
        k = random_sphere(rng, retained_dim, coordinate_radius)
        y = random_sphere(rng, positive_dim, coordinate_radius)
        links = links_from_retained_positive(lift, pos_basis, k, y)
        link_norms.append(float(np.linalg.norm(links)))

        fp_plus = finite_difference_fp_matrix(
            data, d0, links, vertex_basis, g_probe, fp_step
        )
        fp_minus = finite_difference_fp_matrix(
            data, d0, links, vertex_basis, -g_probe, fp_step
        )
        logdet_plus, orient_plus, singulars_plus = logdet_with_orientation(fp_plus)
        logdet_minus, orient_minus, singulars_minus = logdet_with_orientation(fp_minus)
        if not orient_plus or not orient_minus:
            failures += 1
            continue
        min_singulars.append(float(min(singulars_plus[-1], singulars_minus[-1])))
        max_singulars.append(float(max(singulars_plus[0], singulars_minus[0])))
        conditions.append(
            float(max(singulars_plus[0] / singulars_plus[-1], singulars_minus[0] / singulars_minus[-1]))
        )
        fp_even = (logdet_plus + logdet_minus - 2.0 * base_logdet) / (
            2.0 * g_probe * g_probe
        )
        fp_odd = (logdet_plus - logdet_minus) / (2.0 * g_probe)
        fp_even_coeffs.append(fp_even)
        fp_odd_coeffs.append(fp_odd)
        fp_relative.append(max(abs(logdet_plus - base_logdet), abs(logdet_minus - base_logdet)))

        haar_plus = link_haar_log_density(links, g_probe)
        haar_minus = link_haar_log_density(links, -g_probe)
        haar_even = (haar_plus + haar_minus) / (2.0 * g_probe * g_probe)
        haar_odd = (haar_plus - haar_minus) / (2.0 * g_probe)
        quadratic_model = -float(np.sum(links * links)) / 12.0
        haar_even_coeffs.append(haar_even)
        haar_odd_coeffs.append(haar_odd)
        haar_model_errors.append(abs(haar_even - quadratic_model))

    if not min_singulars:
        fp_report = NonlinearFPReport(
            samples=samples,
            g_probe=g_probe,
            coordinate_radius=coordinate_radius,
            fp_step=fp_step,
            min_fp_singular_value=0.0,
            max_fp_singular_value=float("inf"),
            max_fp_condition=float("inf"),
            max_fp_logdet_even_coeff_abs=float("inf"),
            max_fp_logdet_odd_coeff_abs=float("inf"),
            max_fp_logdet_relative_abs=float("inf"),
            fp_orientation_failures=failures,
        )
    else:
        fp_report = NonlinearFPReport(
            samples=samples,
            g_probe=g_probe,
            coordinate_radius=coordinate_radius,
            fp_step=fp_step,
            min_fp_singular_value=float(np.min(min_singulars)),
            max_fp_singular_value=float(np.max(max_singulars)),
            max_fp_condition=float(np.max(conditions)),
            max_fp_logdet_even_coeff_abs=float(np.max(np.abs(fp_even_coeffs))),
            max_fp_logdet_odd_coeff_abs=float(np.max(np.abs(fp_odd_coeffs))),
            max_fp_logdet_relative_abs=float(np.max(fp_relative)),
            fp_orientation_failures=failures,
        )

    haar_report = HaarDensityReport(
        samples=samples,
        g_probe=g_probe,
        max_link_haar_even_coeff_abs=float(np.max(np.abs(haar_even_coeffs))),
        max_link_haar_odd_coeff_abs=float(np.max(np.abs(haar_odd_coeffs))),
        max_link_haar_quadratic_model_error=float(np.max(haar_model_errors)),
        max_sample_link_norm=float(np.max(link_norms)),
        classification=(
            "O(g^2) local density data; belongs to the exact half-density/gauge-density slot, not residual"
        ),
    )
    return fp_report, haar_report


def make_decision(
    gauge: GaugeSliceReport,
    fp: NonlinearFPReport,
    haar: HaarDensityReport,
) -> C5AMDecision:
    tangent_ok = (
        gauge.mean_zero_vertex_rank == 80
        and gauge.color_gauge_rank == 240
        and gauge.fp_base_min_eigenvalue > 1.0e-8
        and gauge.link_vertex_reflection_residual < 1.0e-12
        and gauge.fp_reflection_commutator < 1.0e-10
    )
    fp_ok = (
        fp.fp_orientation_failures == 0
        and fp.min_fp_singular_value > 1.0e-4
        and fp.max_fp_condition < 100.0
        and fp.max_fp_logdet_odd_coeff_abs < 1.0e-4
    )
    density_ok = (
        haar.max_link_haar_odd_coeff_abs < 1.0e-12
        and haar.max_link_haar_quadratic_model_error < 1.0e-6
    )
    return C5AMDecision(
        tangent_gauge_slice_pass=tangent_ok,
        nonlinear_fp_sample_pass=fp_ok,
        density_classified_as_half_density=density_ok,
        compact_haar_disintegration_proved=False,
        decision=(
            "partial pass: the nonlinear mean-zero Landau slice has invertible "
            "sampled FP matrix, reflection-compatible tangent operator, and "
            "only O(g^2) sampled Haar/FP density data; full compact gauge-fixed "
            "disintegration remains open"
            if tangent_ok and fp_ok and density_ok
            else "fail or redesign: nonlinear gauge-slice/Haar diagnostic did not pass"
        ),
    )


def run_checks(
    gauge: GaugeSliceReport,
    fp: NonlinearFPReport,
    haar: HaarDensityReport,
    decision: C5AMDecision,
) -> None:
    assert gauge.vertices == 81
    assert gauge.scalar_links == 216
    assert gauge.mean_zero_vertex_rank == 80
    assert gauge.color_gauge_rank == 240
    assert gauge.global_gauge_kernel_dim == 1
    assert gauge.fp_base_min_eigenvalue > 1.0e-8
    assert gauge.link_vertex_reflection_residual < 1.0e-12
    assert gauge.fp_reflection_commutator < 1.0e-10
    assert fp.fp_orientation_failures == 0
    assert fp.min_fp_singular_value > 1.0e-4
    assert fp.max_fp_condition < 100.0
    assert fp.max_fp_logdet_odd_coeff_abs < 1.0e-4
    assert haar.max_link_haar_odd_coeff_abs < 1.0e-12
    assert haar.max_link_haar_quadratic_model_error < 1.0e-6
    assert decision.tangent_gauge_slice_pass
    assert decision.nonlinear_fp_sample_pass
    assert decision.density_classified_as_half_density
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


def print_report(
    tol: float,
    samples: int,
    seed: int,
    g_probe: float,
    coordinate_radius: float,
    fp_step: float,
) -> None:
    _, _, _, _, gauge = gauge_slice_report(tol)
    fp, haar = nonlinear_fp_and_haar_report(
        tol, samples, seed, g_probe, coordinate_radius, fp_step
    )
    decision = make_decision(gauge, fp, haar)
    run_checks(gauge, fp, haar, decision)

    print("C5AM NONLINEAR GAUGE QUOTIENT AND HAAR/FP DIAGNOSTIC")
    print_table("TANGENT GAUGE SLICE", gauge)
    print_table("NONLINEAR FADDEEV-POPOV SAMPLE", fp)
    print_table("LINK HAAR DENSITY SAMPLE", haar)

    print("\nDECISION")
    print(
        "| tangent gauge slice pass | nonlinear FP sample pass | "
        "density classified as half-density | compact Haar disintegration proved | decision |"
    )
    print("|---|---|---|---|---|")
    print(
        f"| {decision.tangent_gauge_slice_pass} | "
        f"{decision.nonlinear_fp_sample_pass} | "
        f"{decision.density_classified_as_half_density} | "
        f"{decision.compact_haar_disintegration_proved} | {decision.decision} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--g-probe", type=float, default=0.01)
    parser.add_argument("--coordinate-radius", type=float, default=0.25)
    parser.add_argument("--fp-step", type=float, default=1.0e-6)
    args = parser.parse_args()
    print_report(
        args.tol,
        args.samples,
        args.seed,
        args.g_probe,
        args.coordinate_radius,
        args.fp_step,
    )


if __name__ == "__main__":
    main()
