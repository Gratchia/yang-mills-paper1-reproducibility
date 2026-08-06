"""C5AF O(g^2) density-output classification in normalized coordinates.

# ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates

C5AE proved that the retained holonomy-log coordinate change has no O(g)
Jacobian/half-density term.  C5AF inspects the first nonzero retained-section
density correction,

    (1/2) log det D K_g(B) = g^2 rho_2(B) + O(g^3),

using exact finite-g retained holonomy-log coordinates.

The goal is classification, not proof of the full conditional disintegration:
rho_2 is a local closed-loop retained density output.  Since g_j^2 rho_2 is
not summable if discarded, it must be retained/admitted as running half-density
data or included exactly; it is not a residual error.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from c5ae_jacobian_half_density import (
    exact_retained_log_coordinate,
    finite_difference_jacobian,
    tile_loop_paths,
)
from c5v_retained_quotient_bch_bounds import COLORS, build_retained_lift


@dataclass(frozen=True)
class DensityClassification:
    samples: int
    g_probe: float
    scalar_rank: int
    color_rank: int
    retained_tile_paths: int
    max_abs_half_density_rho2: float
    mean_abs_half_density_rho2: float
    max_scale_quadratic_error: float
    max_color_rotation_error: float
    max_probe_stability_error: float
    residual_model_summable: bool
    residual_partial_10j0: float
    residual_partial_100j0: float
    classification: str


def random_unit(rng: np.random.Generator, dim: int) -> np.ndarray:
    x = rng.normal(size=dim)
    return x / np.linalg.norm(x)


def random_so3(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    q = q * signs
    if np.linalg.det(q) < 0.0:
        q[:, 0] *= -1.0
    return q


def rotate_color(x: np.ndarray, scalar_rank: int, rotation: np.ndarray) -> np.ndarray:
    matrix = x.reshape(scalar_rank, COLORS)
    return (matrix @ rotation.T).reshape(-1)


def logdet_coordinate_jacobian(
    x: np.ndarray,
    g: float,
    lift: np.ndarray,
    quotient_basis: np.ndarray,
    paths,
    n: int,
    jac_step: float,
) -> float:
    fn = lambda y: exact_retained_log_coordinate(y, lift, quotient_basis, paths, n, g)
    jac = finite_difference_jacobian(fn, x, jac_step)
    sign, logdet = np.linalg.slogdet(jac)
    if sign <= 0.0:
        raise RuntimeError("retained coordinate Jacobian lost positive orientation")
    return float(logdet)


def half_density_rho2(
    x: np.ndarray,
    g_probe: float,
    lift: np.ndarray,
    quotient_basis: np.ndarray,
    paths,
    n: int,
    jac_step: float,
) -> float:
    """Even O(g^2) coefficient of the OS half-density log."""
    logdet_plus = logdet_coordinate_jacobian(
        x, g_probe, lift, quotient_basis, paths, n, jac_step
    )
    logdet_minus = logdet_coordinate_jacobian(
        x, -g_probe, lift, quotient_basis, paths, n, jac_step
    )
    # Since K_0 is the identity on quotient coordinates, log det DK_0 = 0.
    # Half-density coefficient: (1/2)*(even logdet coefficient).
    return float((logdet_plus + logdet_minus) / (4.0 * g_probe * g_probe))


def residual_partial_sum(j0: int, j1: int, mu: float, unit_bound: float) -> float:
    # rho_2 is quadratic, so on |B| <= M_j the residual model is
    # unit_bound * g_j^2 M_j^2 = unit_bound * mu log(j+1) / j.
    total = 0.0
    for j in range(j0, j1 + 1):
        total += unit_bound * (mu * math.log(j + 1.0)) / j
    return total


def run_diagnostic(
    samples: int,
    seed: int,
    g_probe: float,
    jac_step: float,
    j0: int,
    mu: float,
) -> DensityClassification:
    data, lift, quotient_basis, _ = build_retained_lift(2)
    paths = tile_loop_paths(data)
    scalar_rank = lift.shape[1]
    color_rank = scalar_rank * COLORS
    rng = np.random.default_rng(seed)

    rho_values: list[float] = []
    scale_errors: list[float] = []
    rotation_errors: list[float] = []
    stability_errors: list[float] = []

    for _ in range(samples):
        x = random_unit(rng, color_rank)
        rho = half_density_rho2(x, g_probe, lift, quotient_basis, paths, data.n, jac_step)
        rho_values.append(rho)

        scale = 0.5
        rho_scaled = half_density_rho2(
            scale * x, g_probe, lift, quotient_basis, paths, data.n, jac_step
        )
        scale_errors.append(abs(rho_scaled - scale * scale * rho))

        rotation = random_so3(rng)
        x_rot = rotate_color(x, scalar_rank, rotation)
        rho_rot = half_density_rho2(
            x_rot, g_probe, lift, quotient_basis, paths, data.n, jac_step
        )
        rotation_errors.append(abs(rho_rot - rho))

        rho_half_probe = half_density_rho2(
            x, g_probe / 2.0, lift, quotient_basis, paths, data.n, jac_step
        )
        stability_errors.append(abs(rho_half_probe - rho))

    max_abs = float(np.max(np.abs(rho_values)))
    return DensityClassification(
        samples=samples,
        g_probe=g_probe,
        scalar_rank=scalar_rank,
        color_rank=color_rank,
        retained_tile_paths=len(paths),
        max_abs_half_density_rho2=max_abs,
        mean_abs_half_density_rho2=float(np.mean(np.abs(rho_values))),
        max_scale_quadratic_error=float(np.max(scale_errors)),
        max_color_rotation_error=float(np.max(rotation_errors)),
        max_probe_stability_error=float(np.max(stability_errors)),
        residual_model_summable=False,
        residual_partial_10j0=residual_partial_sum(j0, 10 * j0, mu, max_abs),
        residual_partial_100j0=residual_partial_sum(j0, 100 * j0, mu, max_abs),
        classification=(
            "retain as exact local closed-loop half-density/running density data; "
            "not a residual"
        ),
    )


def run_checks(row: DensityClassification) -> None:
    assert row.scalar_rank == 17
    assert row.color_rank == 51
    assert row.retained_tile_paths == 24
    assert row.max_abs_half_density_rho2 > 0.0
    assert row.max_scale_quadratic_error < 1.0e-3
    assert row.max_color_rotation_error < 1.0e-3
    assert row.max_probe_stability_error < 1.0e-2
    assert not row.residual_model_summable
    assert row.residual_partial_100j0 > row.residual_partial_10j0 > 0.0


def print_report(
    samples: int,
    seed: int,
    g_probe: float,
    jac_step: float,
    j0: int,
    mu: float,
) -> None:
    row = run_diagnostic(samples, seed, g_probe, jac_step, j0, mu)
    run_checks(row)

    print("C5AF O(g^2) DENSITY-OUTPUT CLASSIFICATION")
    print("| quantity | value |")
    print("|---|---:|")
    print(f"| samples | {row.samples} |")
    print(f"| g probe | {row.g_probe:.12g} |")
    print(f"| scalar retained rank | {row.scalar_rank} |")
    print(f"| color retained rank | {row.color_rank} |")
    print(f"| retained tile paths | {row.retained_tile_paths} |")
    print(f"| max abs half-density rho2 on unit samples | {row.max_abs_half_density_rho2:.12g} |")
    print(f"| mean abs half-density rho2 on unit samples | {row.mean_abs_half_density_rho2:.12g} |")
    print(f"| max quadratic scaling error | {row.max_scale_quadratic_error:.12g} |")
    print(f"| max global color-rotation error | {row.max_color_rotation_error:.12g} |")
    print(f"| max probe-stability error | {row.max_probe_stability_error:.12g} |")

    print("\nRESIDUAL SUMMABILITY CHECK")
    print(
        "| model | summable | partial to 10 j0 | partial to 100 j0 | classification |"
    )
    print("|---|---|---:|---:|---|")
    print(
        "| C g_j^2 M_j^2 density term left residual | "
        f"{row.residual_model_summable} | {row.residual_partial_10j0:.12g} | "
        f"{row.residual_partial_100j0:.12g} | {row.classification} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--g-probe", type=float, default=0.01)
    parser.add_argument("--jac-step", type=float, default=1.0e-6)
    parser.add_argument("--j0", type=int, default=13082)
    parser.add_argument("--mu", type=float, default=16.0)
    args = parser.parse_args()
    print_report(args.samples, args.seed, args.g_probe, args.jac_step, args.j0, args.mu)


if __name__ == "__main__":
    main()
