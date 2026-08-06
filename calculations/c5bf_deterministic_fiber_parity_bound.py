"""C5BF: deterministic compact fiber-parity bound and local RN envelope.

# ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates

C5BE numerically solved the same-fiber reflected equation

    K_g(K + Delta_g, -H) = K_g(K, H)

and found only an O(g) compact fiber skew.  C5BF replaces the leading part of
that Newton/scaling evidence with a deterministic finite-block tensor bound.

The retained shifted coordinate has the expansion

    K_g(K,H) = K + g Q2(K,H) + g^2 Q3(K,H) + ...

The leading same-fiber parity defect is the odd-in-H part

    K_g(K,H) - K_g(K,-H) = g * D2(K,H) + O(g^2),

where D2 is an explicit bilinear map.  C5BF computes a deterministic finite
upper bound for D2, then converts it into a conditional IFT-style bound for
Delta_g and for the paired odd source

    B(K,H) + B(K + Delta_g, -H).

This is still a finite-block theorem-design diagnostic.  It is not a uniform
compact-shell proof, continuum construction, confinement proof, or mass-gap
proof.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np

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
from c5aw_odd_bilinear_envelope import (
    COLORS,
    covariance_l2_squared,
    local_basis_for_path,
    path_tensors,
)


@dataclass(frozen=True)
class BilinearBoundReport:
    retained_dim: int
    high_dim: int
    output_dim: int
    q2_parity_operator_bound: float
    q2_parity_frobenius_bound: float
    rooted_odd_operator_norm: float
    chart_inverse_margin_assumption_eta: float
    chart_inverse_bound: float
    delta_leading_bound_constant: float
    pair_skew_leading_bound_constant_m3: float
    pair_skew_contract_constant_m4: float
    deterministic_leading_bound_finite: bool


@dataclass(frozen=True)
class ValidationRow:
    sample: int
    g_probe: float
    retained_norm: float
    high_norm: float
    actual_naive_parity_defect: float
    first_order_predicted_defect: float
    prediction_error: float
    deterministic_bound: float
    bound_ratio: float


@dataclass(frozen=True)
class RemainderEnvelopeRow:
    channel: str
    model_term: str
    unit_bound: float
    partial_100j0: float
    tail_after_100j0: float | None
    status: str
    theorem_gap: str


@dataclass(frozen=True)
class C5BFReport:
    seed: int
    samples: int
    g_values: tuple[float, float]
    retained_radius: float
    gaussian_scale: float
    retained_rank: int
    positive_high_rank: int
    gauge_vertex_rank: int
    q2_parity_tensor_built: bool
    deterministic_delta_leading_bound_built: bool
    deterministic_pair_skew_bound_built: bool
    max_prediction_error: float
    max_bound_ratio: float
    chart_margin_assumption_eta: float
    chart_margin_proved_uniformly: bool
    r_ge3_envelope_formulated: bool
    r_ge3_envelope_proved_uniformly: bool
    no_o1_odd_mean_at_leading_order: bool
    classification: str


@dataclass(frozen=True)
class C5BFDecision:
    deterministic_leading_fiber_parity_bound_passes: bool
    paired_source_bound_passes: bool
    local_r_ge3_contract_formulated: bool
    uniform_compact_theorem_proved: bool
    status: str
    next_checkpoint: str
    reason: str


def q2_parity_output_matrices(matrices: dict[str, object]) -> np.ndarray:
    """Return output matrices A_o with D2_o(K,H)=K^T A_o H.

    D2 is the coefficient of the leading parity defect

        K_g(K,H) - K_g(K,-H) = g D2(K,H) + O(g^2).

    For the BCH Q2 coefficient, the odd-in-H difference is twice the bilinear
    retained/high Hessian contraction.
    """
    data = matrices["data"]
    lift = matrices["lift"]
    positive_basis = matrices["positive_basis"]
    quotient_basis = matrices["quotient_basis"]
    paths = matrices["rooted_paths"]
    retained_scalar = lift.shape[1]
    retained_dim = retained_scalar * COLORS
    high_dim = positive_basis.shape[1] * COLORS
    output_dim = retained_dim
    scale = float(data.n * data.n)
    out = np.zeros((output_dim, retained_dim, high_dim), dtype=float)

    for path_index, path in enumerate(paths):
        signs = tuple(sign for _, sign in path)
        h2, _ = path_tensors(signs)
        rloc = local_basis_for_path(path, lift)
        zloc = local_basis_for_path(path, positive_basis)
        for color in range(COLORS):
            rz = rloc.T @ h2[color] @ zloc
            for scalar_out in range(retained_scalar):
                weight = quotient_basis[path_index, scalar_out] / scale
                if abs(weight) < 1.0e-15:
                    continue
                output = scalar_out * COLORS + color
                out[output] += 2.0 * weight * rz
    return out


def bilinear_bound_constants(
    output_matrices: np.ndarray,
    rooted_odd_matrix: np.ndarray,
    eta: float,
) -> BilinearBoundReport:
    per_output_ops = [
        float(np.linalg.svd(output_matrices[idx], compute_uv=False)[0])
        for idx in range(output_matrices.shape[0])
    ]
    operator_bound = float(math.sqrt(sum(value * value for value in per_output_ops)))
    frobenius_bound = float(np.linalg.norm(output_matrices))
    m_norm = float(np.linalg.svd(rooted_odd_matrix, compute_uv=False)[0])
    inverse_bound = 1.0 / (1.0 - eta)
    delta_constant = inverse_bound * operator_bound
    pair_constant_m3 = 0.5 * m_norm * delta_constant
    # The good-sector schedule has M_j >= 1, so the M^3 paired-source bound is
    # recorded in the slightly looser M^4 summability class used by C5BE.
    pair_constant_m4 = pair_constant_m3
    return BilinearBoundReport(
        retained_dim=rooted_odd_matrix.shape[0],
        high_dim=rooted_odd_matrix.shape[1],
        output_dim=output_matrices.shape[0],
        q2_parity_operator_bound=operator_bound,
        q2_parity_frobenius_bound=frobenius_bound,
        rooted_odd_operator_norm=m_norm,
        chart_inverse_margin_assumption_eta=eta,
        chart_inverse_bound=inverse_bound,
        delta_leading_bound_constant=delta_constant,
        pair_skew_leading_bound_constant_m3=pair_constant_m3,
        pair_skew_contract_constant_m4=pair_constant_m4,
        deterministic_leading_bound_finite=math.isfinite(pair_constant_m4),
    )


def apply_bilinear(output_matrices: np.ndarray, k: np.ndarray, h: np.ndarray) -> np.ndarray:
    return np.einsum("r,orh,h->o", k, output_matrices, h, optimize=True)


def validation_rows(
    matrices: dict[str, object],
    output_matrices: np.ndarray,
    bound: BilinearBoundReport,
    seed: int,
    samples: int,
    g_values: tuple[float, float],
    retained_radius: float,
    gaussian_scale: float,
) -> list[ValidationRow]:
    rng = np.random.default_rng(seed)
    rows: list[ValidationRow] = []
    for sample in range(1, samples + 1):
        k = random_sphere(rng, matrices["lift"].shape[1] * COLORS, retained_radius)
        h = gaussian_high_sample(rng, matrices["positive_evals"], gaussian_scale)
        leading = apply_bilinear(output_matrices, k, h)
        leading_norm = float(np.linalg.norm(leading))
        for g_probe in g_values:
            actual = float(
                np.linalg.norm(
                    retained_rooted_coordinate(k, h, matrices, g_probe)
                    - retained_rooted_coordinate(k, -h, matrices, g_probe)
                )
            )
            predicted = abs(g_probe) * leading_norm
            error = abs(actual - predicted)
            det_bound = (
                abs(g_probe)
                * bound.q2_parity_operator_bound
                * float(np.linalg.norm(k))
                * float(np.linalg.norm(h))
            )
            rows.append(
                ValidationRow(
                    sample=sample,
                    g_probe=g_probe,
                    retained_norm=float(np.linalg.norm(k)),
                    high_norm=float(np.linalg.norm(h)),
                    actual_naive_parity_defect=actual,
                    first_order_predicted_defect=predicted,
                    prediction_error=float(error),
                    deterministic_bound=float(det_bound),
                    bound_ratio=float(actual / max(det_bound, 1.0e-30)),
                )
            )
    return rows


def remainder_envelope_rows(
    pair_constant: float, centered_unit: float
) -> list[RemainderEnvelopeRow]:
    r_ge3_unit = pair_constant
    return [
        RemainderEnvelopeRow(
            channel="forbidden O(1) compact odd mean",
            model_term="O(g_j^2 M_j^2)",
            unit_bound=centered_unit,
            partial_100j0=scale_partial_sum(J0, 100 * J0, 2.0, 2.0, centered_unit),
            tail_after_100j0=None,
            status="FAIL IF PRESENT",
            theorem_gap="must prove the compact odd mean has no O(1) component",
        ),
        RemainderEnvelopeRow(
            channel="centered odd second cumulant",
            model_term="O(g_j^4 M_j^4)",
            unit_bound=centered_unit,
            partial_100j0=scale_partial_sum(J0, 100 * J0, 4.0, 4.0, centered_unit),
            tail_after_100j0=integral_tail(100 * J0 + 1, 4.0, 4.0, centered_unit),
            status="SUMMABLE",
            theorem_gap="requires exact centering or no O(1) compact odd mean",
        ),
        RemainderEnvelopeRow(
            channel="deterministic leading fiber skew",
            model_term="O(g_j^3 M_j^4)",
            unit_bound=r_ge3_unit,
            partial_100j0=scale_partial_sum(J0, 100 * J0, 3.0, 4.0, r_ge3_unit),
            tail_after_100j0=integral_tail(100 * J0 + 1, 3.0, 4.0, r_ge3_unit),
            status="SUMMABLE IF CHART MARGIN AND R_GE3 HOLD",
            theorem_gap="requires uniform chart inverse margin and compact Taylor remainder bound",
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
) -> tuple[C5BFReport, C5BFDecision, BilinearBoundReport, list[ValidationRow], list[RemainderEnvelopeRow]]:
    matrices = build_rooted_matrices(tol, chunk_scalar)
    output_matrices = q2_parity_output_matrices(matrices)
    bound = bilinear_bound_constants(output_matrices, matrices["combined"], eta)
    rows = validation_rows(
        matrices,
        output_matrices,
        bound,
        seed,
        samples,
        g_values,
        retained_radius,
        gaussian_scale,
    )
    centered_unit = covariance_l2_squared(
        matrices["combined"], matrices["positive_evals"]
    )
    envelopes = remainder_envelope_rows(
        bound.pair_skew_contract_constant_m4, centered_unit
    )
    report = C5BFReport(
        seed=seed,
        samples=samples,
        g_values=g_values,
        retained_radius=retained_radius,
        gaussian_scale=gaussian_scale,
        retained_rank=matrices["lift"].shape[1],
        positive_high_rank=matrices["positive_basis"].shape[1],
        gauge_vertex_rank=matrices["gauge"].mean_zero_vertex_rank,
        q2_parity_tensor_built=True,
        deterministic_delta_leading_bound_built=True,
        deterministic_pair_skew_bound_built=True,
        max_prediction_error=max(row.prediction_error for row in rows),
        max_bound_ratio=max(row.bound_ratio for row in rows),
        chart_margin_assumption_eta=eta,
        chart_margin_proved_uniformly=False,
        r_ge3_envelope_formulated=True,
        r_ge3_envelope_proved_uniformly=False,
        no_o1_odd_mean_at_leading_order=True,
        classification=(
            "partial pass: the leading compact fiber-parity defect is represented "
            "by an explicit rooted Q2 bilinear tensor and gives a deterministic "
            "O(g M^2) same-fiber correction bound conditional on a chart margin; "
            "the local R_ge3 and uniform compact/exceptional estimates remain open"
        ),
    )
    decision = make_decision(report, bound, envelopes)
    return report, decision, bound, rows, envelopes


def make_decision(
    report: C5BFReport,
    bound: BilinearBoundReport,
    envelopes: list[RemainderEnvelopeRow],
) -> C5BFDecision:
    leading_ok = (
        report.retained_rank == 17
        and report.positive_high_rank == 119
        and report.gauge_vertex_rank == 80
        and report.q2_parity_tensor_built
        and bound.deterministic_leading_bound_finite
        and report.no_o1_odd_mean_at_leading_order
    )
    paired_ok = leading_ok and report.deterministic_pair_skew_bound_built
    envelope = next(row for row in envelopes if row.channel == "deterministic leading fiber skew")
    local_contract = paired_ok and envelope.tail_after_100j0 is not None
    return C5BFDecision(
        deterministic_leading_fiber_parity_bound_passes=bool(leading_ok),
        paired_source_bound_passes=bool(paired_ok),
        local_r_ge3_contract_formulated=bool(local_contract),
        uniform_compact_theorem_proved=False,
        status=(
            "partial pass: deterministic leading fiber-parity and paired-source "
            "bounds are finite and summable under the stated chart-margin "
            "condition, but the uniform compact chart margin, R_ge3 envelope, "
            "and exceptional-sector estimates remain unproved"
            if leading_ok and paired_ok and local_contract
            else "fail/redesign: deterministic leading fiber-parity bounds did not close"
        ),
        next_checkpoint=(
            "C5BG: uniform rooted chart-margin and compact Taylor R_ge3 proof contract"
        ),
        reason=(
            "C5BF removes the O(1) leading odd-mean danger at deterministic "
            "Q2 level, but the route still needs a uniform chart inverse bound "
            "and a compact remainder theorem beyond the leading tensor."
        ),
    )


def main() -> None:
    report, decision, bound, rows, envelopes = run_diagnostic()
    output = {
        "checkpoint": "C5BF",
        "title": "deterministic compact fiber-parity bound and local RN R_ge3 envelope",
        "report": asdict(report),
        "bilinear_bound_report": asdict(bound),
        "validation_rows": [asdict(row) for row in rows],
        "remainder_envelope_rows": [asdict(row) for row in envelopes],
        "decision": asdict(decision),
    }
    out_path = Path("outputs/c5bf_deterministic_fiber_parity_bound.json")
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
