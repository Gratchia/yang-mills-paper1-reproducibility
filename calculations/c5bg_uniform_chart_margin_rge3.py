"""C5BG: uniform rooted chart-margin and compact Taylor R_ge3 contract.

# ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates

C5BF bounded the leading same-fiber parity defect by using the rooted Q2
bilinear tensor.  That bound was conditional on a chart-margin assumption:

    ||partial_K K_g - I|| <= eta < 1.

C5BG computes deterministic finite-block constants for the truncated chart
margin through Q3:

    K_g(K,H) = K + g Q2(K,H) + g^2 Q3(K,H) + R_ge4.

It also formulates the remaining compact Taylor/Radon--Nikodym R_ge3
contract.  The full compact theorem is not proved here, because the actual
principal-log/BCH remainder, singular strata, and exceptional sectors still
need uniform estimates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np

from c5ak_local_holonomy_disintegration import finite_difference_retained_jacobian_k
from c5bd_analytic_shifted_rn_coefficients import (
    J0,
    MU,
    build_rooted_matrices,
    integral_tail,
    scale_partial_sum,
)
from c5be_compact_conditional_centering import retained_rooted_coordinate
from c5am_nonlinear_gauge_haar import random_sphere
from c5ao_conditional_half_density_projection import gaussian_high_sample
from c5aw_odd_bilinear_envelope import COLORS, local_basis_for_path, path_tensors


@dataclass(frozen=True)
class ChartTensorBounds:
    retained_dim: int
    high_dim: int
    combined_dim: int
    q2_dk_frobenius_operator_bound: float
    q3_dk_frobenius_operator_bound: float
    eta_target: float
    j0: int
    g_j0: float
    m_j0: float
    combined_a_bound_at_j0: float
    truncated_margin_bound_at_j0: float
    truncated_margin_passes_at_j0: bool
    first_j_meeting_eta: int
    delayed_start_factor: float


@dataclass(frozen=True)
class ChartValidationRow:
    sample: int
    g_probe: float
    retained_norm: float
    high_norm: float
    actual_jacobian_margin: float
    q2_predicted_margin: float
    truncated_bound: float
    actual_over_bound: float
    q2_prediction_error: float


@dataclass(frozen=True)
class Rge3ContractRow:
    channel: str
    model_term: str
    unit_bound: float
    partial_100j0: float
    tail_after_100j0: float | None
    status: str
    meaning: str


@dataclass(frozen=True)
class C5BGReport:
    seed: int
    samples: int
    g_values: tuple[float, float]
    retained_radius: float
    gaussian_scale: float
    retained_rank: int
    positive_high_rank: int
    gauge_vertex_rank: int
    q2_chart_tensor_built: bool
    q3_chart_tensor_bound_built: bool
    truncated_chart_margin_passes_at_j0: bool
    full_compact_chart_margin_proved: bool
    max_actual_jacobian_margin: float
    max_q2_prediction_error: float
    max_actual_over_bound: float
    r_ge3_contract_formulated: bool
    compact_taylor_remainder_proved: bool
    exceptional_sector_estimate_proved: bool
    classification: str


@dataclass(frozen=True)
class C5BGDecision:
    truncated_chart_margin_bound_available: bool
    current_j0_margin_closes_with_truncated_bound: bool
    r_ge3_contract_formulated: bool
    full_uniform_compact_theorem_proved: bool
    status: str
    next_checkpoint: str
    reason: str


def combined_local_basis(path, lift: np.ndarray, positive_basis: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [local_basis_for_path(path, lift), local_basis_for_path(path, positive_basis)],
        axis=1,
    )


def q2_dk_matrix(matrices: dict[str, object], k: np.ndarray, h: np.ndarray) -> np.ndarray:
    data = matrices["data"]
    lift = matrices["lift"]
    positive_basis = matrices["positive_basis"]
    quotient_basis = matrices["quotient_basis"]
    paths = matrices["rooted_paths"]
    retained_dim = lift.shape[1] * COLORS
    out = np.zeros((retained_dim, retained_dim), dtype=float)
    scale = float(data.n * data.n)
    for path_index, path in enumerate(paths):
        signs = tuple(sign for _, sign in path)
        h2, _ = path_tensors(signs)
        rloc = local_basis_for_path(path, lift)
        zloc = local_basis_for_path(path, positive_basis)
        local_a = rloc @ k + zloc @ h
        for color in range(COLORS):
            h2a = h2[color] @ local_a
            for scalar_out in range(lift.shape[1]):
                weight = quotient_basis[path_index, scalar_out] / scale
                if abs(weight) < 1.0e-15:
                    continue
                output = scalar_out * COLORS + color
                out[output] += weight * (rloc.T @ h2a)
    return out


def chart_tensor_bounds(matrices: dict[str, object], eta: float) -> ChartTensorBounds:
    data = matrices["data"]
    lift = matrices["lift"]
    positive_basis = matrices["positive_basis"]
    quotient_basis = matrices["quotient_basis"]
    paths = matrices["rooted_paths"]
    retained_dim = lift.shape[1] * COLORS
    high_dim = positive_basis.shape[1] * COLORS
    combined_dim = retained_dim + high_dim
    scale = float(data.n * data.n)

    q2_sq = 0.0
    q3_sq = 0.0
    for path_index, path in enumerate(paths):
        signs = tuple(sign for _, sign in path)
        h2, d3 = path_tensors(signs)
        rloc = local_basis_for_path(path, lift)
        cloc = combined_local_basis(path, lift, positive_basis)
        gram = cloc @ cloc.T
        for color in range(COLORS):
            h2_color = h2[color]
            d3_color = d3[color]
            for scalar_out in range(lift.shape[1]):
                weight = quotient_basis[path_index, scalar_out] / scale
                if abs(weight) < 1.0e-15:
                    continue
                for retained_axis in range(retained_dim):
                    rvec = rloc[:, retained_axis]
                    coeff = h2_color @ rvec
                    q2_sq += float(weight * weight * coeff.T @ gram @ coeff)

                    # For q3, D_K Q3(A)[delta K] = 1/2 D^3Q3(delta K, A, A).
                    bilocal = 0.5 * weight * np.einsum(
                        "a,abc->bc", rvec, d3_color, optimize=True
                    )
                    q3_sq += float(np.trace(bilocal.T @ gram @ bilocal @ gram))

    q2_bound = math.sqrt(max(q2_sq, 0.0))
    q3_bound = math.sqrt(max(q3_sq, 0.0))

    def m_of_j(j: int) -> float:
        return math.sqrt(MU * math.log(j + 1.0))

    def truncated_margin(j: int) -> float:
        g = 1.0 / math.sqrt(float(j))
        a_bound = math.sqrt(2.0) * m_of_j(j)
        return g * q2_bound * a_bound + (g * g) * q3_bound * a_bound * a_bound

    g_j0 = 1.0 / math.sqrt(float(J0))
    m_j0 = m_of_j(J0)
    a_j0 = math.sqrt(2.0) * m_j0
    margin_j0 = truncated_margin(J0)
    if margin_j0 <= eta:
        first = J0
    else:
        lo = J0
        hi = J0
        while truncated_margin(hi) > eta:
            lo = hi
            hi *= 2
            if hi > 10**12:
                break
        while hi - lo > 1:
            mid = (hi + lo) // 2
            if truncated_margin(mid) <= eta:
                hi = mid
            else:
                lo = mid
        first = hi
    return ChartTensorBounds(
        retained_dim=retained_dim,
        high_dim=high_dim,
        combined_dim=combined_dim,
        q2_dk_frobenius_operator_bound=q2_bound,
        q3_dk_frobenius_operator_bound=q3_bound,
        eta_target=eta,
        j0=J0,
        g_j0=g_j0,
        m_j0=m_j0,
        combined_a_bound_at_j0=a_j0,
        truncated_margin_bound_at_j0=margin_j0,
        truncated_margin_passes_at_j0=margin_j0 <= eta,
        first_j_meeting_eta=first,
        delayed_start_factor=float(first / J0),
    )


def validation_rows(
    matrices: dict[str, object],
    bounds: ChartTensorBounds,
    seed: int,
    samples: int,
    g_values: tuple[float, float],
    retained_radius: float,
    gaussian_scale: float,
    jac_step: float,
) -> list[ChartValidationRow]:
    rng = np.random.default_rng(seed)
    rows: list[ChartValidationRow] = []

    def fn(kk: np.ndarray, yy: np.ndarray, g_probe: float) -> np.ndarray:
        return retained_rooted_coordinate(kk, yy, matrices, g_probe)

    for sample in range(1, samples + 1):
        k = random_sphere(rng, matrices["lift"].shape[1] * COLORS, retained_radius)
        h = gaussian_high_sample(rng, matrices["positive_evals"], gaussian_scale)
        q2_jac = q2_dk_matrix(matrices, k, h)
        q2_norm = float(np.linalg.svd(q2_jac, compute_uv=False)[0])
        a_norm = math.sqrt(float(np.linalg.norm(k) ** 2 + np.linalg.norm(h) ** 2))
        for g_probe in g_values:
            jac = finite_difference_retained_jacobian_k(
                lambda kk, yy: fn(kk, yy, g_probe), k, h, jac_step
            )
            actual = float(np.linalg.svd(jac - np.eye(k.size), compute_uv=False)[0])
            q2_pred = abs(g_probe) * q2_norm
            truncated_bound = (
                abs(g_probe) * bounds.q2_dk_frobenius_operator_bound * a_norm
                + (g_probe * g_probe)
                * bounds.q3_dk_frobenius_operator_bound
                * a_norm
                * a_norm
            )
            rows.append(
                ChartValidationRow(
                    sample=sample,
                    g_probe=g_probe,
                    retained_norm=float(np.linalg.norm(k)),
                    high_norm=float(np.linalg.norm(h)),
                    actual_jacobian_margin=actual,
                    q2_predicted_margin=q2_pred,
                    truncated_bound=float(truncated_bound),
                    actual_over_bound=float(actual / max(truncated_bound, 1.0e-30)),
                    q2_prediction_error=float(abs(actual - q2_pred)),
                )
            )
    return rows


def rge3_contract_rows() -> list[Rge3ContractRow]:
    return [
        Rge3ContractRow(
            channel="forbidden unmatched O(g^2) local survivor",
            model_term="O(g_j^2 M_j^p)",
            unit_bound=1.0,
            partial_100j0=scale_partial_sum(J0, 100 * J0, 2.0, 2.0, 1.0),
            tail_after_100j0=None,
            status="FAIL IF PRESENT",
            meaning="Any unmatched O(g^2) local density/action term is nonsummable and forces redesign.",
        ),
        Rge3ContractRow(
            channel="compact Taylor R_ge3 per-unit contract",
            model_term="O(g_j^3 M_j^6)",
            unit_bound=1.0,
            partial_100j0=scale_partial_sum(J0, 100 * J0, 3.0, 6.0, 1.0),
            tail_after_100j0=integral_tail(100 * J0 + 1, 3.0, 6.0, 1.0),
            status="SUMMABLE PER UNIT",
            meaning="A genuine third-order compact remainder is summable for each fixed local constant.",
        ),
        Rge3ContractRow(
            channel="fiber-parity higher correction per-unit contract",
            model_term="O(g_j^3 M_j^4)",
            unit_bound=1.0,
            partial_100j0=scale_partial_sum(J0, 100 * J0, 3.0, 4.0, 1.0),
            tail_after_100j0=integral_tail(100 * J0 + 1, 3.0, 4.0, 1.0),
            status="SUMMABLE PER UNIT",
            meaning="Higher same-fiber skew is allowed only if it gains one extra g power.",
        ),
    ]


def run_diagnostic(
    seed: int = 20260708,
    samples: int = 3,
    g_values: tuple[float, float] = (0.01, 0.005),
    retained_radius: float = 0.18,
    gaussian_scale: float = 0.12,
    tol: float = 1.0e-10,
    chunk_scalar: int = 64,
    eta: float = 0.5,
    jac_step: float = 1.0e-6,
) -> tuple[C5BGReport, C5BGDecision, ChartTensorBounds, list[ChartValidationRow], list[Rge3ContractRow]]:
    matrices = build_rooted_matrices(tol, chunk_scalar)
    bounds = chart_tensor_bounds(matrices, eta)
    rows = validation_rows(
        matrices,
        bounds,
        seed,
        samples,
        g_values,
        retained_radius,
        gaussian_scale,
        jac_step,
    )
    contracts = rge3_contract_rows()
    report = C5BGReport(
        seed=seed,
        samples=samples,
        g_values=g_values,
        retained_radius=retained_radius,
        gaussian_scale=gaussian_scale,
        retained_rank=matrices["lift"].shape[1],
        positive_high_rank=matrices["positive_basis"].shape[1],
        gauge_vertex_rank=matrices["gauge"].mean_zero_vertex_rank,
        q2_chart_tensor_built=True,
        q3_chart_tensor_bound_built=True,
        truncated_chart_margin_passes_at_j0=bounds.truncated_margin_passes_at_j0,
        full_compact_chart_margin_proved=False,
        max_actual_jacobian_margin=max(row.actual_jacobian_margin for row in rows),
        max_q2_prediction_error=max(row.q2_prediction_error for row in rows),
        max_actual_over_bound=max(row.actual_over_bound for row in rows),
        r_ge3_contract_formulated=True,
        compact_taylor_remainder_proved=False,
        exceptional_sector_estimate_proved=False,
        classification=(
            "partial pass: Q2/Q3 deterministic chart-margin constants and "
            "per-unit R_ge3 summability contracts are explicit; the current "
            "J0 margin closes for the truncated bound only if indicated below, "
            "and the full compact Taylor/exceptions theorem remains open"
        ),
    )
    decision = make_decision(report, bounds)
    return report, decision, bounds, rows, contracts


def make_decision(report: C5BGReport, bounds: ChartTensorBounds) -> C5BGDecision:
    available = (
        report.retained_rank == 17
        and report.positive_high_rank == 119
        and report.gauge_vertex_rank == 80
        and report.q2_chart_tensor_built
        and report.q3_chart_tensor_bound_built
        and math.isfinite(bounds.truncated_margin_bound_at_j0)
    )
    return C5BGDecision(
        truncated_chart_margin_bound_available=bool(available),
        current_j0_margin_closes_with_truncated_bound=bool(
            available and bounds.truncated_margin_passes_at_j0
        ),
        r_ge3_contract_formulated=bool(report.r_ge3_contract_formulated),
        full_uniform_compact_theorem_proved=False,
        status=(
            "partial pass: truncated Q2/Q3 chart-margin bound is explicit and "
            "closes at the current J0, but full compact R_ge3 and exceptional "
            "estimates remain open"
            if available and bounds.truncated_margin_passes_at_j0
            else (
                "partial pass with warning: truncated Q2/Q3 chart-margin bound "
                "is explicit but too crude to close at the current J0; either "
                "constants must be sharpened or the UV start must be delayed, "
                "and full compact R_ge3/exceptions remain open"
                if available
                else "fail/redesign: chart-margin constants could not be made explicit"
            )
        ),
        next_checkpoint=(
            "C5BH: sharpen rooted chart constants or prove delayed-start "
            "UV margin plus compact R_ge3 locality"
        ),
        reason=(
            "C5BG converts the chart-margin assumption into explicit Q2/Q3 "
            "constants and a summability contract; the remaining question is "
            "whether the constants close uniformly without hiding exceptional "
            "or compact-log failures."
        ),
    )


def main() -> None:
    report, decision, bounds, rows, contracts = run_diagnostic()
    output = {
        "checkpoint": "C5BG",
        "title": "uniform rooted chart-margin and compact Taylor R_ge3 proof contract",
        "report": asdict(report),
        "chart_tensor_bounds": asdict(bounds),
        "chart_validation_rows": [asdict(row) for row in rows],
        "rge3_contract_rows": [asdict(row) for row in contracts],
        "decision": asdict(decision),
    }
    out_path = Path("outputs/c5bg_uniform_chart_margin_rge3.json")
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
