"""C5AD retained cubic coordinate-normalization diagnostic.

# ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates

The C5W retained cubic

    S1(B) = sum_p Q1_p(B) . Q2_p(B)

is nonzero in the linear retained-section coordinates.  C5AD checks the
local-action interpretation: the cubic is the pullback of the exact Wilson
plaquette action through the nonlinear plaquette-holonomy logarithm

    Theta_g(A) = g^{-1} log prod_e exp(g sign_e A_e)
               = Q1(A) + g Q2(A) + g^2 Q3(A) + ...

When the Wilson action is written in the exact log variable Theta, it is
even in g and has no O(g) cubic term.  Thus S1 is not a summable residual and
should not be admitted as a standalone potential; it is coordinate data to be
removed by using exact holonomy-log retained coordinates, subject to the
separate Jacobian/half-density and nonlinear conditional-measure obligations.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from c5t_bch_incidence_expansion import plaquette_paths, q_expansion
from c5v_retained_quotient_bch_bounds import build_retained_lift
from c5w_cubic_centering import retained_to_links, s1_action


@dataclass(frozen=True)
class CubicNormalizationResult:
    samples: int
    g_probe: float
    max_abs_retained_s1: float
    mean_abs_retained_s1: float
    max_raw_odd_minus_s1_error: float
    max_bch_q3_vector_error: float
    max_fixed_log_odd_coefficient: float
    max_action_coordinate_identity_error: float


def quat_mul(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Hamilton product for unit quaternions."""
    w, x, y, z = q
    a, b, c, d = r
    return np.array(
        [
            w * a - x * b - y * c - z * d,
            w * b + x * a + y * d - z * c,
            w * c - x * d + y * a + z * b,
            w * d + x * c - y * b + z * a,
        ],
        dtype=float,
    )


def exp_lie(v: np.ndarray) -> np.ndarray:
    """Exponential for the abstract SU(2) chart with bracket x cross y.

    The half-angle quaternion convention makes the BCH bracket in vector
    coordinates equal to the ordinary cross product used throughout C5T--C5V.
    """
    radius = float(np.linalg.norm(v))
    if radius < 1.0e-14:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    half = 0.5 * radius
    return np.concatenate(
        [[math.cos(half)], math.sin(half) * v / radius],
    )


def log_lie(q: np.ndarray) -> np.ndarray:
    """Logarithm inverse to exp_lie on the positive-scalar SU(2) chart."""
    q = q / np.linalg.norm(q)
    if q[0] <= 0.0:
        raise ValueError(
            "SU(2) element is outside the declared positive-scalar log chart"
        )
    w = float(np.clip(q[0], -1.0, 1.0))
    spatial = q[1:]
    spatial_norm = float(np.linalg.norm(spatial))
    if spatial_norm < 1.0e-14:
        return np.zeros(3, dtype=float)
    half = math.atan2(spatial_norm, w)
    return 2.0 * half * spatial / spatial_norm


def holonomy_logs(
    paths: list[list[tuple[int, float]]], background: np.ndarray, g: float
) -> np.ndarray:
    """Return Theta_g = g^{-1} log product exp(g sign A_e) for each plaquette."""
    out = np.zeros((len(paths), 3), dtype=float)
    for p, path in enumerate(paths):
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        for edge, sign in path:
            q = quat_mul(q, exp_lie(g * sign * background[edge]))
        out[p] = log_lie(q) / g
    return out


def wilson_action_from_logs(theta: np.ndarray, g: float) -> float:
    radii = np.linalg.norm(theta, axis=1)
    return float(np.sum((1.0 - np.cos(g * radii)) / (g * g)))


def raw_wilson_action(
    paths: list[list[tuple[int, float]]], background: np.ndarray, g: float
) -> float:
    return wilson_action_from_logs(holonomy_logs(paths, background, g), g)


def random_unit(rng: np.random.Generator, dim: int) -> np.ndarray:
    x = rng.normal(size=dim)
    return x / np.linalg.norm(x)


def run_diagnostic(samples: int, seed: int, g_probe: float) -> CubicNormalizationResult:
    data, lift, _, _ = build_retained_lift(2)
    paths = plaquette_paths(data)
    rng = np.random.default_rng(seed)
    dim = 3 * lift.shape[1]

    retained_s1_values: list[float] = []
    raw_odd_errors: list[float] = []
    q3_errors: list[float] = []
    fixed_log_odd_coefficients: list[float] = []
    identity_errors: list[float] = []

    for _ in range(samples):
        x = random_unit(rng, dim)
        background = retained_to_links(lift, x)

        retained_s1 = s1_action(paths, background)
        retained_s1_values.append(retained_s1)

        raw_odd = (
            raw_wilson_action(paths, background, g_probe)
            - raw_wilson_action(paths, background, -g_probe)
        ) / (2.0 * g_probe)
        raw_odd_errors.append(abs(raw_odd - retained_s1))

        q1, q2, q3 = q_expansion(data, paths, background)
        theta = holonomy_logs(paths, background, g_probe)
        q3_errors.append(float(np.linalg.norm((theta - q1 - g_probe * q2) / (g_probe * g_probe) - q3)))

        # At fixed holonomy-log coordinate the Wilson action is exactly even
        # in g.  This is the coordinate-normalization mechanism.
        fixed_plus = wilson_action_from_logs(theta, g_probe)
        fixed_minus = wilson_action_from_logs(theta, -g_probe)
        fixed_log_odd_coefficients.append(abs((fixed_plus - fixed_minus) / (2.0 * g_probe)))

        # The raw action is the exact pullback of the log-coordinate action.
        identity_errors.append(abs(raw_wilson_action(paths, background, g_probe) - fixed_plus))

    return CubicNormalizationResult(
        samples=samples,
        g_probe=g_probe,
        max_abs_retained_s1=float(np.max(np.abs(retained_s1_values))),
        mean_abs_retained_s1=float(np.mean(np.abs(retained_s1_values))),
        max_raw_odd_minus_s1_error=float(np.max(raw_odd_errors)),
        max_bch_q3_vector_error=float(np.max(q3_errors)),
        max_fixed_log_odd_coefficient=float(np.max(fixed_log_odd_coefficients)),
        max_action_coordinate_identity_error=float(np.max(identity_errors)),
    )


def run_checks(result: CubicNormalizationResult) -> None:
    assert result.samples > 0
    assert 0.0 < result.g_probe < 0.1
    assert result.max_abs_retained_s1 > 1.0e-6
    assert result.max_raw_odd_minus_s1_error < 1.0e-5
    assert result.max_bch_q3_vector_error < 1.0e-4
    assert result.max_fixed_log_odd_coefficient < 1.0e-12
    assert result.max_action_coordinate_identity_error < 1.0e-12


def print_report(samples: int, seed: int, g_probe: float) -> None:
    result = run_diagnostic(samples, seed, g_probe)
    run_checks(result)

    print("C5AD RETAINED CUBIC COORDINATE-NORMALIZATION DIAGNOSTIC")
    print("| quantity | value |")
    print("|---|---:|")
    print(f"| retained samples | {result.samples} |")
    print(f"| g probe | {result.g_probe:.12g} |")
    print(f"| max abs S1(B) | {result.max_abs_retained_s1:.12g} |")
    print(f"| mean abs S1(B) | {result.mean_abs_retained_s1:.12g} |")
    print(f"| max raw odd coefficient minus S1(B) | {result.max_raw_odd_minus_s1_error:.12g} |")
    print(f"| max BCH Q3 vector error | {result.max_bch_q3_vector_error:.12g} |")
    print(f"| max fixed-log odd coefficient | {result.max_fixed_log_odd_coefficient:.12g} |")
    print(
        "| max raw/log-coordinate action identity error | "
        f"{result.max_action_coordinate_identity_error:.12g} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--g-probe", type=float, default=0.01)
    args = parser.parse_args()
    print_report(args.samples, args.seed, args.g_probe)


if __name__ == "__main__":
    main()
