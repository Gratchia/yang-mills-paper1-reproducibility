"""C5BH: sharpen rooted chart constants or justify delayed UV start.

# ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates

C5BG made the rooted chart-margin constants explicit but too conservative:

    ||d_K K_g - I|| <= g C2 ||A|| + g^2 C3 ||A||^2

did not close at the current J0 when C2 was computed by summing local
Frobenius contributions before cancellations.

C5BH keeps the same finite n=2 SU(2) rooted common-stem block, but computes
the exact global Q2 derivative tensor after all path and quotient cancellations.
For Q2 this gives a rigorous adapted upper bound:

    ||D_K Q2(A)||_op <= ||D_K Q2(A)||_F
                         <= sqrt(lambda_max(G_Q2)) ||A||,

where G_Q2 is the Gram matrix of the exact coefficient slices.  This is a
finite-dimensional operator/Frobenius-map bound, not a Monte Carlo estimate.

The Q3 constant is deliberately left at the older conservative C5BG value, so
any improvement reported here comes only from the Q2 sharpening.

This still does not prove the full compact one-shell theorem: compact Taylor
remainders, singular strata, near-Cartan sectors, chart/large-field exceptions,
and exceptional-sector probabilities remain separate theorem debts.
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
from c5bg_uniform_chart_margin_rge3 import chart_tensor_bounds
from c5aw_odd_bilinear_envelope import COLORS, local_basis_for_path, path_tensors


@dataclass(frozen=True)
class Q2SharpeningReport:
    retained_dim: int
    high_dim: int
    combined_dim: int
    c5bg_uncancelled_q2_bound: float
    exact_q2_tensor_frobenius_bound: float
    exact_q2_gram_frobenius_map_bound: float
    alternating_operator_lower_proxy: float
    gram_vs_uncancelled_factor: float
    lower_proxy_vs_gram_factor: float
    exact_path_cancellations_used: bool
    rigorous_q2_bound_used_for_decision: str


@dataclass(frozen=True)
class ChartMarginScenario:
    scenario: str
    q2_constant: float
    q3_constant: float
    j_start: int
    delayed_start_factor: float
    g_start: float
    m_start: float
    a_bound_start: float
    truncated_margin: float
    eta_target: float
    margin_room_for_chart_remainder: float
    closes_eta: bool
    interpretation: str


@dataclass(frozen=True)
class RemainderLocalityRow:
    channel: str
    model_term: str
    unit_or_allowance: float
    partial_100j0: float | None
    tail_after_100j0: float | None
    status: str
    meaning: str


@dataclass(frozen=True)
class C5BHReport:
    tol: float
    chunk_scalar: int
    eta_target: float
    q2_restarts: int
    q2_iterations: int
    retained_rank: int
    positive_high_rank: int
    gauge_vertex_rank: int
    adapted_q2_bound_built: bool
    adapted_margin_closes_at_current_j0: bool
    delayed_start_needed_after_q2_sharpening: bool
    q3_bound_sharpened: bool
    compact_chart_remainder_proved: bool
    compact_rge3_locality_proved: bool
    exceptional_sector_estimate_proved: bool
    classification: str


@dataclass(frozen=True)
class C5BHDecision:
    q2_constants_sharpened: bool
    current_j0_truncated_chart_margin_closes: bool
    delayed_start_required_for_truncated_chart_margin: bool
    full_compact_chart_theorem_proved: bool
    status: str
    next_checkpoint: str
    reason: str


def m_of_j(j: int) -> float:
    return math.sqrt(MU * math.log(j + 1.0))


def g_of_j(j: int) -> float:
    return 1.0 / math.sqrt(float(j))


def a_bound_of_j(j: int) -> float:
    return math.sqrt(2.0) * m_of_j(j)


def truncated_margin(j: int, c2: float, c3: float) -> float:
    g = g_of_j(j)
    a_bound = a_bound_of_j(j)
    return g * c2 * a_bound + (g * g) * c3 * a_bound * a_bound


def first_j_meeting_eta(j0: int, c2: float, c3: float, eta: float) -> int:
    if truncated_margin(j0, c2, c3) <= eta:
        return j0
    lo = j0
    hi = j0
    while truncated_margin(hi, c2, c3) > eta:
        lo = hi
        hi *= 2
        if hi > 10**12:
            return hi
    while hi - lo > 1:
        mid = (hi + lo) // 2
        if truncated_margin(mid, c2, c3) <= eta:
            hi = mid
        else:
            lo = mid
    return hi


def exact_q2_derivative_slices(matrices: dict[str, object]) -> np.ndarray:
    """Return exact coefficient slices T[a] for D_K Q2(A).

    The output has shape (combined_dim, retained_dim, retained_dim) and satisfies

        D_K Q2(A) = sum_a A_a T[a].

    Unlike C5BG's initial constant, path/quotient cancellations are kept before
    taking norms.
    """
    data = matrices["data"]
    lift = matrices["lift"]
    positive_basis = matrices["positive_basis"]
    quotient_basis = matrices["quotient_basis"]
    paths = matrices["rooted_paths"]

    retained_dim = lift.shape[1] * COLORS
    high_dim = positive_basis.shape[1] * COLORS
    combined_dim = retained_dim + high_dim
    scale = float(data.n * data.n)
    tensor = np.zeros((combined_dim, retained_dim, retained_dim), dtype=float)

    for path_index, path in enumerate(paths):
        signs = tuple(sign for _, sign in path)
        h2, _ = path_tensors(signs)
        rloc = local_basis_for_path(path, lift)
        zloc = local_basis_for_path(path, positive_basis)
        cloc = np.concatenate([rloc, zloc], axis=1)
        for color in range(COLORS):
            local_block = rloc.T @ h2[color] @ cloc
            for scalar_out in range(lift.shape[1]):
                weight = quotient_basis[path_index, scalar_out] / scale
                if abs(weight) < 1.0e-15:
                    continue
                output = scalar_out * COLORS + color
                tensor[:, output, :] += weight * local_block.T
    return tensor


def q2_gram_bound(tensor: np.ndarray) -> float:
    flat = tensor.reshape(tensor.shape[0], -1)
    gram = flat @ flat.T
    gram = 0.5 * (gram + gram.T)
    return math.sqrt(max(float(np.linalg.eigvalsh(gram)[-1]), 0.0))


def alternating_operator_lower_proxy(
    tensor: np.ndarray,
    seed: int,
    restarts: int,
    iterations: int,
) -> float:
    """Return a non-rigorous lower proxy for the true operator-valued norm.

    This is included only to measure slack between the rigorous Gram/Frobenius
    upper bound and observed extremizers.  It is not used in the pass/fail
    decision.
    """
    rng = np.random.default_rng(seed)
    combined_dim = tensor.shape[0]

    def mat_of(a: np.ndarray) -> np.ndarray:
        return np.tensordot(a, tensor, axes=(0, 0))

    best = 0.0
    for _ in range(restarts):
        a = rng.normal(size=combined_dim)
        a /= max(float(np.linalg.norm(a)), 1.0e-30)
        for _ in range(iterations):
            mat = mat_of(a)
            u, _, vh = np.linalg.svd(mat, full_matrices=False)
            coeff = np.einsum("i,aij,j->a", u[:, 0], tensor, vh[0], optimize=True)
            coeff_norm = float(np.linalg.norm(coeff))
            if coeff_norm <= 1.0e-30:
                break
            a = coeff / coeff_norm
        value = float(np.linalg.svd(mat_of(a), compute_uv=False)[0])
        best = max(best, value)
    return best


def scenario_row(
    label: str,
    q2_constant: float,
    q3_constant: float,
    j_start: int,
    eta: float,
    interpretation: str,
) -> ChartMarginScenario:
    margin = truncated_margin(j_start, q2_constant, q3_constant)
    return ChartMarginScenario(
        scenario=label,
        q2_constant=float(q2_constant),
        q3_constant=float(q3_constant),
        j_start=int(j_start),
        delayed_start_factor=float(j_start / J0),
        g_start=g_of_j(j_start),
        m_start=m_of_j(j_start),
        a_bound_start=a_bound_of_j(j_start),
        truncated_margin=float(margin),
        eta_target=eta,
        margin_room_for_chart_remainder=float(eta - margin),
        closes_eta=bool(margin <= eta),
        interpretation=interpretation,
    )


def remainder_locality_rows(adapted_margin: ChartMarginScenario) -> list[RemainderLocalityRow]:
    g = adapted_margin.g_start
    a_bound = adapted_margin.a_bound_start
    room = max(adapted_margin.margin_room_for_chart_remainder, 0.0)
    # If the chart derivative remainder begins as C4 g^3 ||A||^3, this is the
    # largest allowed C4 at the current start.  It is only a margin allowance,
    # not a proof that such a bound holds.
    c4_allowance = room / max((g**3) * (a_bound**3), 1.0e-30)
    return [
        RemainderLocalityRow(
            channel="chart derivative remainder allowance",
            model_term="C4 g_j^3 ||A||^3",
            unit_or_allowance=float(c4_allowance),
            partial_100j0=None,
            tail_after_100j0=None,
            status="ALLOWANCE ONLY",
            meaning=(
                "If the compact chart derivative remainder has this scale form, "
                "its local C4 must stay below this allowance at the current J0."
            ),
        ),
        RemainderLocalityRow(
            channel="forbidden unmatched O(g^2) local survivor",
            model_term="O(g_j^2 M_j^2)",
            unit_or_allowance=1.0,
            partial_100j0=scale_partial_sum(J0, 100 * J0, 2.0, 2.0, 1.0),
            tail_after_100j0=None,
            status="FAIL IF PRESENT",
            meaning="Any unmatched O(g^2) local action/density survivor remains nonsummable.",
        ),
        RemainderLocalityRow(
            channel="compact Taylor/RN R_ge3 per fixed local unit",
            model_term="O(g_j^3 M_j^6)",
            unit_or_allowance=1.0,
            partial_100j0=scale_partial_sum(J0, 100 * J0, 3.0, 6.0, 1.0),
            tail_after_100j0=integral_tail(100 * J0 + 1, 3.0, 6.0, 1.0),
            status="SUMMABLE PER FIXED UNIT",
            meaning=(
                "A genuine third-order compact/RN remainder is summable for each "
                "fixed local constant, but C5BH does not prove the constant."
            ),
        ),
        RemainderLocalityRow(
            channel="fiber-parity higher correction per fixed local unit",
            model_term="O(g_j^3 M_j^4)",
            unit_or_allowance=1.0,
            partial_100j0=scale_partial_sum(J0, 100 * J0, 3.0, 4.0, 1.0),
            tail_after_100j0=integral_tail(100 * J0 + 1, 3.0, 4.0, 1.0),
            status="SUMMABLE PER FIXED UNIT",
            meaning="The same-fiber skew remains acceptable only with one extra g power.",
        ),
    ]


def run_diagnostic(
    tol: float = 1.0e-10,
    chunk_scalar: int = 64,
    eta: float = 0.5,
    q2_seed: int = 20260708,
    q2_restarts: int = 48,
    q2_iterations: int = 24,
) -> tuple[
    C5BHReport,
    C5BHDecision,
    Q2SharpeningReport,
    list[ChartMarginScenario],
    list[RemainderLocalityRow],
]:
    matrices = build_rooted_matrices(tol, chunk_scalar)
    old_bounds = chart_tensor_bounds(matrices, eta)
    q2_tensor = exact_q2_derivative_slices(matrices)
    q2_exact_frobenius = float(np.linalg.norm(q2_tensor))
    q2_adapted = q2_gram_bound(q2_tensor)
    q2_lower_proxy = alternating_operator_lower_proxy(
        q2_tensor, q2_seed, q2_restarts, q2_iterations
    )

    q2_report = Q2SharpeningReport(
        retained_dim=old_bounds.retained_dim,
        high_dim=old_bounds.high_dim,
        combined_dim=old_bounds.combined_dim,
        c5bg_uncancelled_q2_bound=old_bounds.q2_dk_frobenius_operator_bound,
        exact_q2_tensor_frobenius_bound=q2_exact_frobenius,
        exact_q2_gram_frobenius_map_bound=q2_adapted,
        alternating_operator_lower_proxy=q2_lower_proxy,
        gram_vs_uncancelled_factor=float(
            q2_adapted / old_bounds.q2_dk_frobenius_operator_bound
        ),
        lower_proxy_vs_gram_factor=float(q2_lower_proxy / max(q2_adapted, 1.0e-30)),
        exact_path_cancellations_used=True,
        rigorous_q2_bound_used_for_decision="exact_q2_gram_frobenius_map_bound",
    )

    q3_constant = old_bounds.q3_dk_frobenius_operator_bound
    old_first = old_bounds.first_j_meeting_eta
    adapted_first = first_j_meeting_eta(J0, q2_adapted, q3_constant, eta)
    scenarios = [
        scenario_row(
            "C5BG uncancelled Q2/Q3 at current J0",
            old_bounds.q2_dk_frobenius_operator_bound,
            q3_constant,
            J0,
            eta,
            "baseline from C5BG; does not close at current J0",
        ),
        scenario_row(
            "C5BG uncancelled Q2/Q3 at delayed start",
            old_bounds.q2_dk_frobenius_operator_bound,
            q3_constant,
            old_first,
            eta,
            "old conservative route would close only after delayed UV start",
        ),
        scenario_row(
            "C5BH adapted Q2 plus old Q3 at current J0",
            q2_adapted,
            q3_constant,
            J0,
            eta,
            "C5BH adapted-bound decision scenario at the current UV start",
        ),
        scenario_row(
            "C5BH adapted Q2 plus old Q3 first closing start",
            q2_adapted,
            q3_constant,
            adapted_first,
            eta,
            "first delayed UV start at which the adapted truncated bound closes",
        ),
    ]
    adapted_current = scenarios[2]
    locality = remainder_locality_rows(adapted_current)

    closes = adapted_current.closes_eta
    report = C5BHReport(
        tol=tol,
        chunk_scalar=chunk_scalar,
        eta_target=eta,
        q2_restarts=q2_restarts,
        q2_iterations=q2_iterations,
        retained_rank=matrices["lift"].shape[1],
        positive_high_rank=matrices["positive_basis"].shape[1],
        gauge_vertex_rank=matrices["gauge"].mean_zero_vertex_rank,
        adapted_q2_bound_built=True,
        adapted_margin_closes_at_current_j0=bool(closes),
        delayed_start_needed_after_q2_sharpening=not bool(closes),
        q3_bound_sharpened=False,
        compact_chart_remainder_proved=False,
        compact_rge3_locality_proved=False,
        exceptional_sector_estimate_proved=False,
        classification=(
            "partial pass: exact Q2 path cancellations give a rigorous adapted "
            "Gram/Frobenius-map bound that closes the truncated Q2/Q3 chart "
            "margin at the current J0 using the old conservative Q3 constant; "
            "full compact chart remainder, R_ge3 locality, and exceptional "
            "sectors remain open"
            if closes
            else (
                "partial pass with warning: Q2 constants are sharpened but the "
                "current J0 margin still does not close, so delayed-start "
                "compatibility must be used"
            )
        ),
    )
    decision = make_decision(report)
    return report, decision, q2_report, scenarios, locality


def make_decision(report: C5BHReport) -> C5BHDecision:
    q2_ok = (
        report.retained_rank == 17
        and report.positive_high_rank == 119
        and report.gauge_vertex_rank == 80
        and report.adapted_q2_bound_built
    )
    closes = q2_ok and report.adapted_margin_closes_at_current_j0
    return C5BHDecision(
        q2_constants_sharpened=bool(q2_ok),
        current_j0_truncated_chart_margin_closes=bool(closes),
        delayed_start_required_for_truncated_chart_margin=bool(q2_ok and not closes),
        full_compact_chart_theorem_proved=False,
        status=(
            "partial pass: adapted Q2 constant closes the truncated chart margin "
            "at current J0, so delayed UV start is not needed for the truncated "
            "Q2/Q3 estimate; compact remainder/locality and exceptional sectors "
            "remain theorem debts"
            if closes
            else (
                "partial pass with warning: adapted Q2 constant is built but "
                "current J0 still does not close; delayed-start route must be "
                "proved compatible"
                if q2_ok
                else "fail/redesign: adapted Q2 constant could not be built"
            )
        ),
        next_checkpoint=(
            "C5BI: compact chart-remainder and R_ge3 locality theorem contract"
        ),
        reason=(
            "C5BH sharpens the C5BG Q2 constant by preserving exact path "
            "cancellations, but the truncated margin still fails at the current "
            "J0.  A compatible delayed start and the compact Taylor/RN remainder "
            "and exceptional-sector estimates therefore remain open."
        ),
    )


def main() -> None:
    report, decision, q2_report, scenarios, locality = run_diagnostic()
    output = {
        "checkpoint": "C5BH",
        "title": "sharpen rooted chart constants or prove delayed-start UV margin",
        "report": asdict(report),
        "q2_sharpening_report": asdict(q2_report),
        "chart_margin_scenarios": [asdict(row) for row in scenarios],
        "remainder_locality_rows": [asdict(row) for row in locality],
        "decision": asdict(decision),
    }
    out_path = Path("outputs/c5bh_sharpen_chart_or_delayed_start.json")
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
