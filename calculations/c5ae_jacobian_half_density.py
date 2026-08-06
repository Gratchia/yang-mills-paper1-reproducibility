"""C5AE retained holonomy-coordinate Jacobian/half-density diagnostic.

# ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates

C5AD removed the retained O(g) cubic from the action by moving from
linearized retained curvature variables B to exact retained holonomy-log
variables

    K_g(B) = B + g N2(B) + O(g^2).

C5AE checks the first possible measure obstruction: the Jacobian term

    log det D K_g(B) = g div N2(B) + O(g^2).

For the C5R/C5V retained tile quotient, N2 is built from cross products of
linear color fields.  Its divergence vanishes by color antisymmetry, so the
coordinate normalization introduces no O(g) Lebesgue Jacobian or half-density
term at the retained-section level.

This is not the full nonlinear disintegration theorem; it does not prove the
all-order Jacobian, high-mode conditional density, exceptional-sector bound,
or mass gap.
"""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass

import numpy as np

from c5ad_cubic_coordinate_normalization import exp_lie, log_lie, quat_mul
from c5q_4d_cubical_incidence import D
from c5v_retained_quotient_bch_bounds import build_retained_lift
from c5w_cubic_centering import retained_to_links


COLORS = 3


@dataclass(frozen=True)
class JacobianResult:
    samples: int
    g_probe: float
    scalar_rank: int
    color_rank: int
    retained_tile_paths: int
    tile_path_length: int
    max_analytic_divergence: float
    max_finite_difference_divergence: float
    max_exact_logdet_odd_coefficient: float
    max_exact_logdet_odd_scaled_by_g2: float


def tile_loop_paths(data) -> list[list[tuple[int, float]]]:
    """Return the 24 coarse boundary face-loop paths used by C5R.

    The order matches c5r_curvature_tile_schur.tile_matrix: for each
    orientation mu<nu, fix the two transverse coordinates at 0 or n and run
    around the n x n square.
    """
    paths: list[list[tuple[int, float]]] = []
    for mu, nu in itertools.combinations(range(D), 2):
        transverse = [rho for rho in range(D) if rho not in (mu, nu)]
        for values in itertools.product([0, data.n], repeat=2):
            fixed = dict(zip(transverse, values))
            path: list[tuple[int, float]] = []

            def coords_with(
                mu_value: int | None = None, nu_value: int | None = None
            ) -> list[int]:
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
                path.append((data.index[1][key], coeff))

            for i in range(data.n):
                add_edge(mu, coords_with(mu_value=i, nu_value=0), 1.0)
            for j in range(data.n):
                add_edge(nu, coords_with(mu_value=data.n, nu_value=j), 1.0)
            for i in range(data.n - 1, -1, -1):
                add_edge(mu, coords_with(mu_value=i, nu_value=data.n), -1.0)
            for j in range(data.n - 1, -1, -1):
                add_edge(nu, coords_with(mu_value=0, nu_value=j), -1.0)

            paths.append(path)
    return paths


def signed_edge_arrays(paths: list[list[tuple[int, float]]]) -> tuple[np.ndarray, np.ndarray]:
    edges = np.array([[edge for edge, _ in path] for path in paths], dtype=int)
    signs = np.array([[sign for _, sign in path] for path in paths], dtype=float)
    return edges, signs


def retained_n2(
    x: np.ndarray,
    lift: np.ndarray,
    quotient_basis: np.ndarray,
    paths: list[list[tuple[int, float]]],
    n: int,
) -> np.ndarray:
    """Quadratic retained BCH correction N2 on the color quotient."""
    scalar_rank = lift.shape[1]
    background = retained_to_links(lift, x.reshape(scalar_rank, COLORS))
    q2 = np.zeros((len(paths), COLORS), dtype=float)
    for path_index, path in enumerate(paths):
        signed = [sign * background[edge] for edge, sign in path]
        for i in range(len(signed)):
            for j in range(i + 1, len(signed)):
                q2[path_index] += 0.5 * np.cross(signed[i], signed[j])
    q2_density = q2 / float(n * n)
    return (quotient_basis.T @ q2_density).reshape(-1)


def exact_retained_log_coordinate(
    x: np.ndarray,
    lift: np.ndarray,
    quotient_basis: np.ndarray,
    paths: list[list[tuple[int, float]]],
    n: int,
    g: float,
) -> np.ndarray:
    """Exact retained quotient coordinate K_g(x)."""
    scalar_rank = lift.shape[1]
    background = retained_to_links(lift, x.reshape(scalar_rank, COLORS))
    logs = np.zeros((len(paths), COLORS), dtype=float)
    if abs(g) < 1.0e-14:
        linear = np.zeros((len(paths), COLORS), dtype=float)
        for path_index, path in enumerate(paths):
            for edge, sign in path:
                linear[path_index] += sign * background[edge]
        linear /= float(n * n)
        return (quotient_basis.T @ linear).reshape(-1)

    for p, path in enumerate(paths):
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        for edge, sign in path:
            q = quat_mul(q, exp_lie(g * sign * background[edge]))
        if q[0] <= 0.0:
            raise ValueError(
                "retained holonomy left the declared positive-scalar SU(2) log chart"
            )
        logs[p] = log_lie(q) / (g * n * n)
    return (quotient_basis.T @ logs).reshape(-1)


def finite_difference_divergence(
    fn, x: np.ndarray, h: float
) -> float:
    total = 0.0
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = h
        total += (fn(x + step)[i] - fn(x - step)[i]) / (2.0 * h)
    return float(total)


def finite_difference_jacobian(fn, x: np.ndarray, h: float) -> np.ndarray:
    dim = x.size
    jac = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        step = np.zeros_like(x)
        step[i] = h
        jac[:, i] = (fn(x + step) - fn(x - step)) / (2.0 * h)
    return jac


def analytic_divergence_by_color_trace(
    x: np.ndarray,
    lift: np.ndarray,
    quotient_basis: np.ndarray,
    paths: list[list[tuple[int, float]]],
    n: int,
) -> float:
    """Directly evaluate the first divergence contraction.

    The returned value is zero because every derivative of a cross-product
    component contracts the Levi-Civita tensor with a repeated color index.
    This routine keeps the geometry loops explicit as a reproducible check.
    """
    scalar_rank = lift.shape[1]
    x_matrix = x.reshape(scalar_rank, COLORS)
    eps = np.zeros((COLORS, COLORS, COLORS), dtype=float)
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
    eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0

    total = 0.0
    scale = 1.0 / float(n * n)
    for p, path in enumerate(paths):
        signed_lifts = [sign * lift[edge] for edge, sign in path]
        for i in range(len(path)):
            for j in range(i + 1, len(path)):
                li = signed_lifts[i]
                lj = signed_lifts[j]
                vi = li @ x_matrix
                vj = lj @ x_matrix
                for r in range(scalar_rank):
                    weight = scale * quotient_basis[p, r]
                    for color in range(COLORS):
                        # derivative of (vi x vj)_color with respect to x_{r,color}
                        for c in range(COLORS):
                            total += 0.5 * weight * eps[color, color, c] * li[r] * vj[c]
                        for b in range(COLORS):
                            total += 0.5 * weight * eps[color, b, color] * vi[b] * lj[r]
    return float(total)


def random_unit(rng: np.random.Generator, dim: int) -> np.ndarray:
    x = rng.normal(size=dim)
    return x / np.linalg.norm(x)


def run_diagnostic(
    samples: int,
    seed: int,
    g_probe: float,
    div_step: float,
    jac_step: float,
) -> JacobianResult:
    data, lift, quotient_basis, _ = build_retained_lift(2)
    paths = tile_loop_paths(data)
    scalar_rank = lift.shape[1]
    color_rank = scalar_rank * COLORS
    rng = np.random.default_rng(seed)

    analytic_divs: list[float] = []
    fd_divs: list[float] = []
    odd_coefficients: list[float] = []

    for _ in range(samples):
        x = random_unit(rng, color_rank)
        n2_fn = lambda y: retained_n2(y, lift, quotient_basis, paths, data.n)
        analytic_divs.append(
            abs(analytic_divergence_by_color_trace(x, lift, quotient_basis, paths, data.n))
        )
        fd_divs.append(abs(finite_difference_divergence(n2_fn, x, div_step)))

        k_plus = lambda y: exact_retained_log_coordinate(
            y, lift, quotient_basis, paths, data.n, g_probe
        )
        k_minus = lambda y: exact_retained_log_coordinate(
            y, lift, quotient_basis, paths, data.n, -g_probe
        )
        jac_plus = finite_difference_jacobian(k_plus, x, jac_step)
        jac_minus = finite_difference_jacobian(k_minus, x, jac_step)
        sign_p, logdet_p = np.linalg.slogdet(jac_plus)
        sign_m, logdet_m = np.linalg.slogdet(jac_minus)
        if sign_p <= 0 or sign_m <= 0:
            raise RuntimeError("Jacobian lost positive orientation at probe scale")
        odd_coefficients.append(abs((logdet_p - logdet_m) / (2.0 * g_probe)))

    max_odd = float(np.max(odd_coefficients))
    return JacobianResult(
        samples=samples,
        g_probe=g_probe,
        scalar_rank=scalar_rank,
        color_rank=color_rank,
        retained_tile_paths=len(paths),
        tile_path_length=len(paths[0]),
        max_analytic_divergence=float(np.max(analytic_divs)),
        max_finite_difference_divergence=float(np.max(fd_divs)),
        max_exact_logdet_odd_coefficient=max_odd,
        max_exact_logdet_odd_scaled_by_g2=max_odd / (g_probe * g_probe),
    )


def run_checks(result: JacobianResult) -> None:
    assert result.scalar_rank == 17
    assert result.color_rank == 51
    assert result.retained_tile_paths == 24
    assert result.tile_path_length == 8
    assert result.max_analytic_divergence < 1.0e-14
    assert result.max_finite_difference_divergence < 1.0e-8
    assert result.max_exact_logdet_odd_coefficient < 1.0e-3


def print_report(
    samples: int,
    seed: int,
    g_probe: float,
    div_step: float,
    jac_step: float,
) -> None:
    result = run_diagnostic(samples, seed, g_probe, div_step, jac_step)
    run_checks(result)

    print("C5AE RETAINED HOLONOMY-COORDINATE JACOBIAN DIAGNOSTIC")
    print("| quantity | value |")
    print("|---|---:|")
    print(f"| samples | {result.samples} |")
    print(f"| g probe | {result.g_probe:.12g} |")
    print(f"| scalar retained rank | {result.scalar_rank} |")
    print(f"| color retained rank | {result.color_rank} |")
    print(f"| retained tile paths | {result.retained_tile_paths} |")
    print(f"| tile path length | {result.tile_path_length} |")
    print(f"| max analytic div N2 | {result.max_analytic_divergence:.12g} |")
    print(f"| max finite-difference div N2 | {result.max_finite_difference_divergence:.12g} |")
    print(f"| max exact logdet odd coefficient | {result.max_exact_logdet_odd_coefficient:.12g} |")
    print(
        "| max exact logdet odd coefficient / g^2 | "
        f"{result.max_exact_logdet_odd_scaled_by_g2:.12g} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--g-probe", type=float, default=0.01)
    parser.add_argument("--div-step", type=float, default=1.0e-6)
    parser.add_argument("--jac-step", type=float, default=1.0e-6)
    args = parser.parse_args()
    print_report(args.samples, args.seed, args.g_probe, args.div_step, args.jac_step)


if __name__ == "__main__":
    main()
