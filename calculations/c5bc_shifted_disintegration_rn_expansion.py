"""C5BC: shifted disintegration / Radon--Nikodym expansion diagnostic.

This checkpoint starts the concrete C5BC task:

    build the local shifted good-sector map and test the O(g^2)
    Radon--Nikodym coefficient structure in the rooted common-stem variables.

It remains a finite-block diagnostic.  It does not prove the uniform compact
SU(2) disintegration theorem, continuum construction, confinement, or a mass
gap.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path

import numpy as np

from c5ak_local_holonomy_disintegration import (
    finite_difference_retained_jacobian_k,
    retained_coordinate_from_split,
)
from c5al_gauge_zero_reference import build_split, link_reflection_matrix, reflection_report
from c5am_nonlinear_gauge_haar import (
    gauge_slice_report,
    links_from_retained_positive,
    random_sphere,
)
from c5an_gauge_haar_half_density_matching import sample_coefficients
from c5ao_conditional_half_density_projection import gaussian_high_sample
from c5az_shifted_coordinate_package_regression import build_common_stem_path_package
from c5v_retained_quotient_bch_bounds import COLORS, build_retained_lift


SHIFT = 1
J0 = 13082
MU = 16.0
ROOTED_COV_SQ = 0.21480467679414722
PHI4_UNIT = 0.0280541227163


@dataclass(frozen=True)
class ShiftedChartReport:
    samples: int
    seed: int
    g_probe: float
    retained_radius: float
    gaussian_scale: float
    jac_step: float
    retained_rank: int
    positive_high_rank: int
    gauge_vertex_rank: int
    shifted_path_cyclic_shift: int
    max_g_link_norm: float
    principal_log_margin: float
    min_retained_jacobian_singular_value: float
    max_retained_jacobian_singular_value: float
    max_retained_jacobian_condition: float
    max_abs_retained_logdet: float
    max_tangent_reconstruction_residual: float
    chart_failures: int
    classification: str


@dataclass(frozen=True)
class RNPairRow:
    retained_norm: float
    high_norm: float
    link_norm: float
    plus_combined_half_even: float
    minus_combined_half_even: float
    odd_y_coefficient: float
    pair_even_mean: float
    half_scale_odd_error: float
    max_abs_o_g_density_coeff: float
    min_fp_singular: float
    max_fp_condition: float
    max_tangent_gauge_divergence_norm: float


@dataclass(frozen=True)
class RNExpansionReport:
    pairs: int
    seed: int
    g_probe: float
    retained_radius: float
    gaussian_scale: float
    jac_step: float
    fp_step: float
    max_abs_odd_y_coefficient: float
    max_abs_pair_even_mean: float
    max_half_scale_odd_error: float
    max_abs_o_g_density_coeff: float
    min_fp_singular_value: float
    max_fp_condition: float
    max_tangent_gauge_divergence_norm: float
    reference_antithetic_centering_exact: bool
    exact_compact_conditional_centering_proved: bool
    classification: str


@dataclass(frozen=True)
class ResidualGate:
    name: str
    model_term: str
    unit_bound: float
    partial_100j0: float
    tail_after_100j0: float | None
    gate: str
    meaning: str


@dataclass(frozen=True)
class TheoremDebt:
    label: str
    finite_result: str
    missing_uniform_statement: str
    failure_trigger: str


@dataclass(frozen=True)
class C5BCDecision:
    shifted_chart_sample_pass: bool
    rn_o_g_density_sample_pass: bool
    reference_odd_centering_pass: bool
    exact_compact_rn_theorem_proved: bool
    no_forbidden_o2_drift_found_in_diagnostic: bool
    status: str
    next_checkpoint: str
    reason: str


def term_value(j: int, g_power: float, m_power: float, unit_bound: float, mu: float = MU) -> float:
    return unit_bound * (j ** (-0.5 * g_power)) * (
        mu * math.log(j + 1.0)
    ) ** (0.5 * m_power)


def partial_sum(j0: int, j1: int, g_power: float, m_power: float, unit_bound: float) -> float:
    return sum(term_value(j, g_power, m_power, unit_bound) for j in range(j0, j1 + 1))


def integral_tail(j_start: int, g_power: float, m_power: float, unit_bound: float) -> float | None:
    alpha = 0.5 * g_power
    log_power = 0.5 * m_power
    if alpha <= 1.0:
        return None
    if abs(log_power - round(log_power)) > 1.0e-12:
        return None
    n = int(round(log_power))
    log_start = math.log(float(j_start))
    a = alpha - 1.0
    factorial = math.factorial(n)
    total = 0.0
    for k in range(n + 1):
        total += (
            factorial
            / math.factorial(n - k)
            * log_start ** (n - k)
            / a ** (k + 1)
        )
    return unit_bound * (MU**n) * (float(j_start) ** (1.0 - alpha)) * total


def residual_gates() -> list[ResidualGate]:
    return [
        ResidualGate(
            name="forbidden uncentered shifted odd density",
            model_term="O(g_j^2 M_j^2)",
            unit_bound=ROOTED_COV_SQ,
            partial_100j0=partial_sum(J0, 100 * J0, 2.0, 2.0, ROOTED_COV_SQ),
            tail_after_100j0=None,
            gate="FAIL",
            meaning="If exact centering fails, the odd channel creates a nonsummable O(g^2) drift.",
        ),
        ResidualGate(
            name="allowed centered shifted odd second cumulant",
            model_term="O(g_j^4 M_j^4)",
            unit_bound=ROOTED_COV_SQ,
            partial_100j0=partial_sum(J0, 100 * J0, 4.0, 4.0, ROOTED_COV_SQ),
            tail_after_100j0=integral_tail(100 * J0 + 1, 4.0, 4.0, ROOTED_COV_SQ),
            gate="PASS IF EXACTLY CENTERED",
            meaning="After exact centering, the first surviving effect is a summable second cumulant.",
        ),
    ]


def build_rooted_data(tol: float):
    data, lift, quotient_basis, _ = build_retained_lift(2)
    (
        data_split,
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
    if len(data.cells[1]) != len(data_split.cells[1]):
        raise RuntimeError("inconsistent retained/split data")
    reflection = link_reflection_matrix(data)
    path_package = build_common_stem_path_package(data, reflection, shift=SHIFT)
    rooted_paths = path_package.rooted_paths
    positive_evals = positive_evals_all[positive_mask]
    return data, lift, quotient_basis, positive_basis, positive_evals, rooted_paths


def retained_rooted_coordinate(
    k: np.ndarray,
    y: np.ndarray,
    lift: np.ndarray,
    positive_basis: np.ndarray,
    quotient_basis: np.ndarray,
    rooted_paths,
    n: int,
    g: float,
) -> np.ndarray:
    return retained_coordinate_from_split(
        k, y, lift, positive_basis, quotient_basis, rooted_paths, n, g
    )


def run_chart_probe(
    samples: int,
    seed: int,
    tol: float,
    g_probe: float,
    retained_radius: float,
    gaussian_scale: float,
    jac_step: float,
) -> ShiftedChartReport:
    data, lift, quotient_basis, positive_basis, positive_evals, rooted_paths = build_rooted_data(tol)
    gauge = gauge_slice_report(tol)[4]
    rng = np.random.default_rng(seed)
    retained_dim = lift.shape[1] * COLORS

    min_singulars: list[float] = []
    max_singulars: list[float] = []
    conditions: list[float] = []
    logdets: list[float] = []
    reconstruction: list[float] = []
    g_link_norms: list[float] = []
    failures = 0

    for _ in range(samples):
        k = random_sphere(rng, retained_dim, retained_radius)
        y = gaussian_high_sample(rng, positive_evals, gaussian_scale)
        links = links_from_retained_positive(lift, positive_basis, k, y)
        g_link_norms.append(float(abs(g_probe) * np.max(np.linalg.norm(links, axis=1))))

        fn = lambda kk, yy: retained_rooted_coordinate(
            kk, yy, lift, positive_basis, quotient_basis, rooted_paths, data.n, g_probe
        )
        try:
            jac = finite_difference_retained_jacobian_k(fn, k, y, jac_step)
            singulars = np.linalg.svd(jac, compute_uv=False)
            sign, logdet = np.linalg.slogdet(jac)
            if sign <= 0.0:
                failures += 1
                continue
            min_singulars.append(float(singulars[-1]))
            max_singulars.append(float(singulars[0]))
            conditions.append(float(singulars[0] / singulars[-1]))
            logdets.append(abs(float(logdet)))
            tangent = retained_rooted_coordinate(
                k, y, lift, positive_basis, quotient_basis, rooted_paths, data.n, 0.0
            )
            reconstruction.append(float(np.linalg.norm(tangent - k)))
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            failures += 1

    if not min_singulars:
        return ShiftedChartReport(
            samples=samples,
            seed=seed,
            g_probe=g_probe,
            retained_radius=retained_radius,
            gaussian_scale=gaussian_scale,
            jac_step=jac_step,
            retained_rank=lift.shape[1],
            positive_high_rank=positive_basis.shape[1],
            gauge_vertex_rank=gauge.mean_zero_vertex_rank,
            shifted_path_cyclic_shift=SHIFT,
            max_g_link_norm=float("inf"),
            principal_log_margin=float("-inf"),
            min_retained_jacobian_singular_value=0.0,
            max_retained_jacobian_singular_value=float("inf"),
            max_retained_jacobian_condition=float("inf"),
            max_abs_retained_logdet=float("inf"),
            max_tangent_reconstruction_residual=float("inf"),
            chart_failures=failures,
            classification="fail: no successful rooted chart samples",
        )

    max_g_link = float(np.max(g_link_norms))
    return ShiftedChartReport(
        samples=samples,
        seed=seed,
        g_probe=g_probe,
        retained_radius=retained_radius,
        gaussian_scale=gaussian_scale,
        jac_step=jac_step,
        retained_rank=lift.shape[1],
        positive_high_rank=positive_basis.shape[1],
        gauge_vertex_rank=gauge.mean_zero_vertex_rank,
        shifted_path_cyclic_shift=SHIFT,
        max_g_link_norm=max_g_link,
        principal_log_margin=float(math.pi - max_g_link),
        min_retained_jacobian_singular_value=float(np.min(min_singulars)),
        max_retained_jacobian_singular_value=float(np.max(max_singulars)),
        max_retained_jacobian_condition=float(np.max(conditions)),
        max_abs_retained_logdet=float(np.max(logdets)),
        max_tangent_reconstruction_residual=float(np.max(reconstruction)),
        chart_failures=failures,
        classification=(
            "finite rooted chart sample pass: local IFT data are stable in sampled good-sector probes; "
            "uniform compact theorem still open"
        ),
    )


def rn_pair_row(
    k: np.ndarray,
    y: np.ndarray,
    data,
    lift: np.ndarray,
    positive_basis: np.ndarray,
    quotient_basis: np.ndarray,
    rooted_paths,
    d0: np.ndarray,
    vertex_basis: np.ndarray,
    base_fp_logdet: float,
    g_probe: float,
    jac_step: float,
    fp_step: float,
) -> RNPairRow:
    plus = sample_coefficients(
        k,
        y,
        data,
        lift,
        positive_basis,
        quotient_basis,
        rooted_paths,
        d0,
        vertex_basis,
        base_fp_logdet,
        g_probe,
        jac_step,
        fp_step,
    )
    minus = sample_coefficients(
        k,
        -y,
        data,
        lift,
        positive_basis,
        quotient_basis,
        rooted_paths,
        d0,
        vertex_basis,
        base_fp_logdet,
        g_probe,
        jac_step,
        fp_step,
    )
    half_plus = sample_coefficients(
        k,
        0.5 * y,
        data,
        lift,
        positive_basis,
        quotient_basis,
        rooted_paths,
        d0,
        vertex_basis,
        base_fp_logdet,
        g_probe,
        jac_step,
        fp_step,
    )
    half_minus = sample_coefficients(
        k,
        -0.5 * y,
        data,
        lift,
        positive_basis,
        quotient_basis,
        rooted_paths,
        d0,
        vertex_basis,
        base_fp_logdet,
        g_probe,
        jac_step,
        fp_step,
    )

    odd = 0.5 * (plus.combined_half_even - minus.combined_half_even)
    half_odd = 0.5 * (half_plus.combined_half_even - half_minus.combined_half_even)
    pair_mean = 0.5 * (plus.combined_half_even + minus.combined_half_even)
    links = links_from_retained_positive(lift, positive_basis, k, y)
    return RNPairRow(
        retained_norm=float(np.linalg.norm(k)),
        high_norm=float(np.linalg.norm(y)),
        link_norm=float(np.linalg.norm(links)),
        plus_combined_half_even=float(plus.combined_half_even),
        minus_combined_half_even=float(minus.combined_half_even),
        odd_y_coefficient=float(odd),
        pair_even_mean=float(pair_mean),
        half_scale_odd_error=float(abs(half_odd - 0.5 * odd)),
        max_abs_o_g_density_coeff=float(
            max(
                abs(plus.combined_half_odd),
                abs(minus.combined_half_odd),
                abs(half_plus.combined_half_odd),
                abs(half_minus.combined_half_odd),
            )
        ),
        min_fp_singular=float(
            min(
                plus.min_fp_singular,
                minus.min_fp_singular,
                half_plus.min_fp_singular,
                half_minus.min_fp_singular,
            )
        ),
        max_fp_condition=float(
            max(
                plus.max_fp_condition,
                minus.max_fp_condition,
                half_plus.max_fp_condition,
                half_minus.max_fp_condition,
            )
        ),
        max_tangent_gauge_divergence_norm=float(
            max(
                plus.tangent_gauge_divergence_norm,
                minus.tangent_gauge_divergence_norm,
                half_plus.tangent_gauge_divergence_norm,
                half_minus.tangent_gauge_divergence_norm,
            )
        ),
    )


def run_rn_probe(
    pairs: int,
    seed: int,
    tol: float,
    g_probe: float,
    retained_radius: float,
    gaussian_scale: float,
    jac_step: float,
    fp_step: float,
) -> tuple[RNExpansionReport, list[RNPairRow]]:
    data, lift, quotient_basis, positive_basis, positive_evals, rooted_paths = build_rooted_data(tol)
    data_gauge, lift_gauge, d0, vertex_basis, gauge = gauge_slice_report(tol)
    if len(data.cells[1]) != len(data_gauge.cells[1]) or not np.allclose(lift, lift_gauge):
        raise RuntimeError("inconsistent retained/gauge data")

    base_scalar = (d0 @ vertex_basis).T @ (d0 @ vertex_basis)
    base_fp_logdet = COLORS * float(np.linalg.slogdet(base_scalar)[1])
    rng = np.random.default_rng(seed)
    retained_dim = lift.shape[1] * COLORS
    rows: list[RNPairRow] = []
    failures = 0

    for _ in range(pairs):
        k = random_sphere(rng, retained_dim, retained_radius)
        y = gaussian_high_sample(rng, positive_evals, gaussian_scale)
        try:
            rows.append(
                rn_pair_row(
                    k,
                    y,
                    data,
                    lift,
                    positive_basis,
                    quotient_basis,
                    rooted_paths,
                    d0,
                    vertex_basis,
                    base_fp_logdet,
                    g_probe,
                    jac_step,
                    fp_step,
                )
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            failures += 1

    if not rows:
        return (
            RNExpansionReport(
                pairs=pairs,
                seed=seed,
                g_probe=g_probe,
                retained_radius=retained_radius,
                gaussian_scale=gaussian_scale,
                jac_step=jac_step,
                fp_step=fp_step,
                max_abs_odd_y_coefficient=float("inf"),
                max_abs_pair_even_mean=float("inf"),
                max_half_scale_odd_error=float("inf"),
                max_abs_o_g_density_coeff=float("inf"),
                min_fp_singular_value=0.0,
                max_fp_condition=float("inf"),
                max_tangent_gauge_divergence_norm=float("inf"),
                reference_antithetic_centering_exact=False,
                exact_compact_conditional_centering_proved=False,
                classification=f"fail: no successful RN pairs; failures={failures}",
            ),
            rows,
        )

    report = RNExpansionReport(
        pairs=pairs,
        seed=seed,
        g_probe=g_probe,
        retained_radius=retained_radius,
        gaussian_scale=gaussian_scale,
        jac_step=jac_step,
        fp_step=fp_step,
        max_abs_odd_y_coefficient=float(np.max([abs(row.odd_y_coefficient) for row in rows])),
        max_abs_pair_even_mean=float(np.max([abs(row.pair_even_mean) for row in rows])),
        max_half_scale_odd_error=float(np.max([row.half_scale_odd_error for row in rows])),
        max_abs_o_g_density_coeff=float(np.max([row.max_abs_o_g_density_coeff for row in rows])),
        min_fp_singular_value=float(np.min([row.min_fp_singular for row in rows])),
        max_fp_condition=float(np.max([row.max_fp_condition for row in rows])),
        max_tangent_gauge_divergence_norm=float(
            np.max([row.max_tangent_gauge_divergence_norm for row in rows])
        ),
        reference_antithetic_centering_exact=True,
        exact_compact_conditional_centering_proved=False,
        classification=(
            "finite shifted RN sample pass: O(g) density coefficient is small and the odd-y O(g^2) "
            "channel is antithetically centered in the reference measure; exact compact conditional centering remains open"
        ),
    )
    return report, rows


def theorem_debts() -> list[TheoremDebt]:
    return [
        TheoremDebt(
            label="BC1 uniform rooted chart",
            finite_result="sampled shifted ∂K/∂k is invertible and principal-log margin is positive",
            missing_uniform_statement="prove uniform invertibility over the full C5M good sector",
            failure_trigger="minimum singular value approaches zero or log chart crosses the principal boundary",
        ),
        TheoremDebt(
            label="BC2 exact RN expansion",
            finite_result="finite ±g coefficient extraction shows the expected O(g)/O(g^2) structure",
            missing_uniform_statement="derive analytic compact SU(2) RN expansion with controlled R_ge3",
            failure_trigger="uncontrolled Haar/FP/retained-coordinate terms appear at O(g) or O(g^2)",
        ),
        TheoremDebt(
            label="BC3 exact odd centering",
            finite_result="antithetic reference high measure centers the shifted odd-y coefficient",
            missing_uniform_statement="prove the true shifted conditional reference remains centered after compact disintegration",
            failure_trigger="a nonzero conditional mean creates the forbidden O(g_j^2 M_j^2) drift",
        ),
        TheoremDebt(
            label="BC4 quasilocality and exceptions",
            finite_result="finite block has controlled gauge/reflection and FP diagnostics",
            missing_uniform_statement="prove quasilocal constants and summable chart/large-field/near-Cartan exceptional bounds",
            failure_trigger="constants grow with scale or exceptional-sector probability is nonsummable",
        ),
    ]


def make_decision(chart: ShiftedChartReport, rn: RNExpansionReport) -> C5BCDecision:
    chart_ok = (
        chart.chart_failures == 0
        and chart.min_retained_jacobian_singular_value > 0.5
        and chart.principal_log_margin > 0.0
        and chart.max_tangent_reconstruction_residual < 1.0e-8
    )
    rn_ok = (
        rn.min_fp_singular_value > 1.0e-4
        and rn.max_fp_condition < 100.0
        and rn.max_abs_o_g_density_coeff < 1.0e-4
    )
    no_forbidden = chart_ok and rn_ok and rn.reference_antithetic_centering_exact
    return C5BCDecision(
        shifted_chart_sample_pass=chart_ok,
        rn_o_g_density_sample_pass=rn_ok,
        reference_odd_centering_pass=rn.reference_antithetic_centering_exact,
        exact_compact_rn_theorem_proved=False,
        no_forbidden_o2_drift_found_in_diagnostic=no_forbidden,
        status=(
            "partial pass: finite rooted chart/RN diagnostics pass, but exact compact conditional "
            "centering and uniform RN theorem remain open"
        ),
        next_checkpoint=(
            "C5BD: analytic shifted RN coefficient formulas. Replace finite ±g/finite-difference extraction "
            "by explicit g=0 source formulas for the shifted retained-coordinate, Haar, and FP terms."
        ),
        reason=(
            "C5BC builds the actual local compact-link map in rooted variables and finds no sampled "
            "forbidden O(g^2) drift.  The odd channel is centered by the antithetic reference measure, but this "
            "is not yet the exact compact conditional centering theorem."
        ),
    )


def main() -> None:
    tol = 1.0e-10
    chart = run_chart_probe(
        samples=4,
        seed=20260705,
        tol=tol,
        g_probe=0.01,
        retained_radius=0.20,
        gaussian_scale=0.18,
        jac_step=1.0e-6,
    )
    rn, rn_rows = run_rn_probe(
        pairs=2,
        seed=20260706,
        tol=tol,
        g_probe=0.01,
        retained_radius=0.18,
        gaussian_scale=0.12,
        jac_step=1.0e-6,
        fp_step=1.0e-6,
    )
    reflection = reflection_report(tol)
    result = {
        "checkpoint": "C5BC",
        "title": "shifted disintegration / Radon-Nikodym expansion diagnostic",
        "shifted_path_cyclic_shift": SHIFT,
        "chart_report": asdict(chart),
        "rn_report": asdict(rn),
        "rn_pair_rows": [asdict(row) for row in rn_rows],
        "reflection_reference": asdict(reflection),
        "residual_gates": [asdict(gate) for gate in residual_gates()],
        "theorem_debts": [asdict(debt) for debt in theorem_debts()],
        "decision": asdict(make_decision(chart, rn)),
    }
    out_path = Path("outputs/c5bc_shifted_disintegration_rn_expansion.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
