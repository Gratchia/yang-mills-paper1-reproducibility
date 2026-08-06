"""BCH/incidence expansion diagnostics for the C5T n=2 tile block.

# ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates

For each plaquette path exp(g X1) exp(g X2) exp(g X3) exp(g X4),
compute

    log(product) / g = Q1 + g Q2 + g^2 Q3 + O(g^3)

using the BCH formula through degree three.  The checker verifies Q1 equals
the oriented incidence D1 A, tests parity of the action expansion terms,
and samples raw Hessian-size constants for the O(g) and O(g^2) pieces.

This is not the final nonlinear block proof.  It identifies which terms must
be centered, absorbed into running local potentials, or bounded against the
C5S margin.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from c5q_4d_cubical_incidence import build_data


@dataclass(frozen=True)
class ParityRow:
    q1_odd_error: float
    q2_even_error: float
    q3_odd_error: float
    s1_odd_error: float
    s2_even_error: float


@dataclass(frozen=True)
class SampleRow:
    samples: int
    q2_norm_const: float
    q3_norm_const: float
    s1_cubic_const: float
    s2_bch_quartic_const: float
    s2_wilson_quartic_const: float
    s1_hessian_const: float
    s2_bch_hessian_const_at_radius: float
    s2_wilson_hessian_const_at_radius: float
    test_radius: float
    c5s_half_margin: float


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b)


def bch3_pair(
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    y3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BCH coefficients through g^3 for X(g)+Y(g)."""
    z1 = x1 + y1
    z2 = x2 + y2 + 0.5 * cross(x1, y1)
    z3 = (
        x3
        + y3
        + 0.5 * (cross(x1, y2) + cross(x2, y1))
        + (cross(x1, cross(x1, y1)) + cross(y1, cross(y1, x1))) / 12.0
    )
    return z1, z2, z3


def plaquette_paths(data) -> list[list[tuple[int, float]]]:
    paths: list[list[tuple[int, float]]] = []
    for dirs, coords in data.cells[2]:
        mu, nu = dirs
        c = list(coords)
        c_mu = list(coords)
        c_mu[mu] += 1
        c_nu = list(coords)
        c_nu[nu] += 1
        e_mu_bottom = data.index[1][((mu,), tuple(c))]
        e_nu_right = data.index[1][((nu,), tuple(c_mu))]
        e_mu_top = data.index[1][((mu,), tuple(c_nu))]
        e_nu_left = data.index[1][((nu,), tuple(c))]
        paths.append(
            [
                (e_mu_bottom, 1.0),
                (e_nu_right, 1.0),
                (e_mu_top, -1.0),
                (e_nu_left, -1.0),
            ]
        )
    return paths


def q_expansion(data, paths, a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q1 = np.zeros((len(paths), 3), dtype=float)
    q2 = np.zeros_like(q1)
    q3 = np.zeros_like(q1)
    zero = np.zeros(3, dtype=float)
    for p, path in enumerate(paths):
        z1 = zero.copy()
        z2 = zero.copy()
        z3 = zero.copy()
        for edge, sign in path:
            x1 = sign * a[edge]
            z1, z2, z3 = bch3_pair(z1, z2, z3, x1, zero, zero)
        q1[p] = z1
        q2[p] = z2
        q3[p] = z3
    return q1, q2, q3


def linear_incidence(data, a: np.ndarray) -> np.ndarray:
    out = np.zeros((data.d1.shape[0], 3), dtype=float)
    for color in range(3):
        out[:, color] = data.d1 @ a[:, color]
    return out


def action_terms(data, paths, a: np.ndarray) -> tuple[float, float, float, float]:
    q1, q2, q3 = q_expansion(data, paths, a)
    s0 = 0.5 * float(np.sum(q1 * q1))
    s1 = float(np.sum(q1 * q2))
    s2_bch = float(np.sum(q1 * q3) + 0.5 * np.sum(q2 * q2))
    # Wilson action contribution:
    # g^{-2}(1-cos(g|Y|)) = 1/2 |Y|^2 - g^2 |Y|^4/24 + O(g^4).
    s2_wilson = s2_bch - float(np.sum(np.sum(q1 * q1, axis=1) ** 2)) / 24.0
    return s0, s1, s2_bch, s2_wilson


def norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(a.reshape(-1)))


def random_field(rng: np.random.Generator, links: int, radius: float = 1.0) -> np.ndarray:
    a = rng.normal(size=(links, 3))
    nrm = norm(a)
    return radius * a / nrm


def parity_diagnostics(data, paths, rng: np.random.Generator) -> ParityRow:
    a = random_field(rng, len(data.cells[1]), radius=1.0)
    q1, q2, q3 = q_expansion(data, paths, a)
    mq1, mq2, mq3 = q_expansion(data, paths, -a)
    _, s1, s2_bch, s2_wilson = action_terms(data, paths, a)
    _, ms1, ms2_bch, ms2_wilson = action_terms(data, paths, -a)
    return ParityRow(
        q1_odd_error=float(np.linalg.norm(mq1 + q1)),
        q2_even_error=float(np.linalg.norm(mq2 - q2)),
        q3_odd_error=float(np.linalg.norm(mq3 + q3)),
        s1_odd_error=abs(ms1 + s1),
        s2_even_error=max(abs(ms2_bch - s2_bch), abs(ms2_wilson - s2_wilson)),
    )


def sampled_constants(
    data,
    paths,
    samples: int,
    seed: int,
    test_radius: float,
    c5s_half_margin: float,
) -> SampleRow:
    rng = np.random.default_rng(seed)
    q2_const = 0.0
    q3_const = 0.0
    s1_const = 0.0
    s2_bch_const = 0.0
    s2_wilson_const = 0.0
    s1_hess_const = 0.0
    s2_bch_hess_const = 0.0
    s2_wilson_hess_const = 0.0

    for _ in range(samples):
        a_unit = random_field(rng, len(data.cells[1]), radius=1.0)
        q1, q2, q3 = q_expansion(data, paths, a_unit)
        q2_const = max(q2_const, float(np.linalg.norm(q2)))
        q3_const = max(q3_const, float(np.linalg.norm(q3)))
        _, s1, s2_bch, s2_wilson = action_terms(data, paths, a_unit)
        s1_const = max(s1_const, abs(s1))
        s2_bch_const = max(s2_bch_const, abs(s2_bch))
        s2_wilson_const = max(s2_wilson_const, abs(s2_wilson))

        v = random_field(rng, len(data.cells[1]), radius=1.0)
        # S1 is homogeneous cubic, so the exact second directional derivative is
        # S1(A+v)+S1(A-v)-2S1(A) at unit radius.
        _, s1_plus, _, _ = action_terms(data, paths, a_unit + v)
        _, s1_minus, _, _ = action_terms(data, paths, a_unit - v)
        s1_hess_const = max(s1_hess_const, abs(s1_plus + s1_minus - 2.0 * s1))

        # S2 is homogeneous quartic; use a small finite-difference step at the
        # requested radius to capture the local Hessian scale.
        a = test_radius * a_unit
        _, _, s2_bch_center, s2_wilson_center = action_terms(data, paths, a)
        eps = 1e-4
        _, _, s2_bch_plus, s2_wilson_plus = action_terms(data, paths, a + eps * v)
        _, _, s2_bch_minus, s2_wilson_minus = action_terms(data, paths, a - eps * v)
        s2_bch_hess = abs(s2_bch_plus + s2_bch_minus - 2.0 * s2_bch_center) / (eps * eps)
        s2_wilson_hess = abs(s2_wilson_plus + s2_wilson_minus - 2.0 * s2_wilson_center) / (eps * eps)
        s2_bch_hess_const = max(s2_bch_hess_const, s2_bch_hess)
        s2_wilson_hess_const = max(s2_wilson_hess_const, s2_wilson_hess)

    return SampleRow(
        samples=samples,
        q2_norm_const=q2_const,
        q3_norm_const=q3_const,
        s1_cubic_const=s1_const,
        s2_bch_quartic_const=s2_bch_const,
        s2_wilson_quartic_const=s2_wilson_const,
        s1_hessian_const=s1_hess_const,
        s2_bch_hessian_const_at_radius=s2_bch_hess_const,
        s2_wilson_hessian_const_at_radius=s2_wilson_hess_const,
        test_radius=test_radius,
        c5s_half_margin=c5s_half_margin,
    )


def run_checks() -> None:
    data = build_data(2)
    paths = plaquette_paths(data)
    rng = np.random.default_rng(123)
    a = random_field(rng, len(data.cells[1]), radius=1.0)
    q1, _, _ = q_expansion(data, paths, a)
    lin = linear_incidence(data, a)
    assert np.linalg.norm(q1 - lin) < 1e-12
    parity = parity_diagnostics(data, paths, rng)
    assert parity.q1_odd_error < 1e-12
    assert parity.q2_even_error < 1e-12
    assert parity.q3_odd_error < 1e-12
    assert parity.s1_odd_error < 1e-12
    assert parity.s2_even_error < 1e-12


def print_report(samples: int, seed: int, test_radius: float, c5s_half_margin: float) -> None:
    data = build_data(2)
    paths = plaquette_paths(data)
    rng = np.random.default_rng(seed)
    a = random_field(rng, len(data.cells[1]), radius=1.0)
    q1, _, _ = q_expansion(data, paths, a)
    lin_error = float(np.linalg.norm(q1 - linear_incidence(data, a)))
    parity = parity_diagnostics(data, paths, rng)
    constants = sampled_constants(data, paths, samples, seed + 1, test_radius, c5s_half_margin)

    print("BCH INCIDENCE STRUCTURE")
    print("| quantity | value |")
    print("|---|---:|")
    print(f"| n | 2 |")
    print(f"| links | {len(data.cells[1])} |")
    print(f"| plaquettes | {len(data.cells[2])} |")
    print(f"| Q1-D1A check | {lin_error:.3g} |")

    print("\nPARITY CHECKS")
    print("| check | error |")
    print("|---|---:|")
    print(f"| Q1 odd | {parity.q1_odd_error:.3g} |")
    print(f"| Q2 even | {parity.q2_even_error:.3g} |")
    print(f"| Q3 odd | {parity.q3_odd_error:.3g} |")
    print(f"| S1=<Q1,Q2> odd | {parity.s1_odd_error:.3g} |")
    print(f"| S2_BCH and S2_Wilson even | {parity.s2_even_error:.3g} |")

    print("\nSAMPLED RAW CONSTANTS")
    print("| samples | ||Q2||/||A||^2 | ||Q3||/||A||^3 | |S1|/||A||^3 | |S2_BCH|/||A||^4 | |S2_Wilson|/||A||^4 | S1 Hess/||A|| | S2_BCH Hess | S2_Wilson Hess | C5S half margin |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    print(
        f"| {constants.samples} | {constants.q2_norm_const:.12g} | "
        f"{constants.q3_norm_const:.12g} | {constants.s1_cubic_const:.12g} | "
        f"{constants.s2_bch_quartic_const:.12g} | "
        f"{constants.s2_wilson_quartic_const:.12g} | "
        f"{constants.s1_hessian_const:.12g} | "
        f"{constants.s2_bch_hessian_const_at_radius:.12g} | "
        f"{constants.s2_wilson_hessian_const_at_radius:.12g} | "
        f"{constants.c5s_half_margin:.12g} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--test-radius", type=float, default=1.0)
    parser.add_argument("--c5s-half-margin", type=float, default=1.08060461174)
    args = parser.parse_args()
    run_checks()
    print_report(args.samples, args.seed, args.test_radius, args.c5s_half_margin)


if __name__ == "__main__":
    main()
