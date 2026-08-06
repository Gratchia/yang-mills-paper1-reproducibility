"""
C5BB: shifted good-sector one-shell expansion diagnostic.

This checkpoint starts the post-C5BA proof attempt.  It does not prove the
compact nonlinear Yang--Mills one-shell theorem.  It does three narrower jobs:

  1. freeze the finite-dimensional shifted K/H split data inherited from
     C5AL--C5AZ;
  2. place the C5AJ cumulant identity and C5BA shifted residual budgets into a
     single good-sector expansion ledger;
  3. state, in mechanically checkable form, which exact theorem clauses remain
     before the route can claim a shifted one-shell expansion.

The output is a JSON ledger and decision record used by the human-readable
C5BB report.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path


# Rooted common-stem constants frozen by the corrected C5AZ computation.
ROOTED_OP_NORM = 1.0189627218075927
ROOTED_COV_SQ = 0.21480467679414722
ROOTED_COV = 0.463470254486895
OLD_COV_SQ = 0.0315650978782

# Finite-block ranks from C5AL/C5AW/C5AZ.
RETAINED_SCALAR_RANK = 17
POSITIVE_HIGH_SCALAR_RANK = 119
GAUGE_VERTEX_SCALAR_RANK = 80
COLOR_DIM = 3

# Finite-block shifted package residuals from C5AZ.
ROOT_VERTEX = (1, 0, 0, 0)
ROOT_REFLECTION_FIXED = True
LOOP_UNMATCHED_REFLECTIONS = 0
STEM_UNMATCHED_REFLECTIONS = 0
ROOTED_PATH_UNMATCHED_REFLECTIONS = 0
MAX_ROOTED_GAUGE_COVARIANCE_RESIDUAL = 1.27373599508e-15
MAX_ROOTED_REFLECTION_RESIDUAL = 7.52989890787e-16
ROOTED_COMBINED_DENSITY_REFLECTION_RESIDUAL = 4.34925153573e-14
ROOTED_N2_DIVERGENCE = 0.0

# Normalized scale model.
J0 = 13082
MU = 16.0
PHI4_UNIT = 0.0280541227163


@dataclass(frozen=True)
class TangentSplitCheck:
    retained_scalar_rank: int
    positive_high_scalar_rank: int
    gauge_vertex_scalar_rank: int
    total_scalar_rank: int
    retained_color_dim: int
    positive_high_color_dim: int
    gauge_color_dim: int
    total_color_dim: int
    dimension_closure: bool
    root_vertex: tuple[int, int, int, int]
    root_reflection_fixed: bool
    loop_unmatched_reflections: int
    stem_unmatched_reflections: int
    rooted_path_unmatched_reflections: int
    finite_rooted_reflection_covariant: bool
    finite_rooted_gauge_covariant: bool
    max_rooted_gauge_covariance_residual: float
    max_rooted_reflection_residual: float


@dataclass(frozen=True)
class ExpansionClause:
    label: str
    statement: str
    finite_or_formal_evidence: str
    good_sector_status: str
    required_next_lemma: str
    failure_trigger: str


@dataclass(frozen=True)
class SlotCheck:
    slot: str
    formal_order: str
    required_treatment: str
    shifted_status: str
    exact_good_sector_status: str
    residual_allowed: bool
    failure_trigger: str


@dataclass(frozen=True)
class ResidualGate:
    name: str
    model_term: str
    unit_bound: float
    partial_10j0: float
    partial_100j0: float
    tail_after_100j0: float | None
    summable: bool
    gate: str
    meaning: str


@dataclass(frozen=True)
class CenteringLedger:
    density_source_degree: str
    even_centered_channel: str
    odd_channel_form: str
    reference_conditional_mean_zero: bool
    exact_conditional_mean_zero_proved: bool
    variance_unit_bound: float
    conservative_log_cumulant_unit_bound: float
    residual_model: str
    shifted_partial_100j0: float
    shifted_tail_after_100j0: float
    failure_trigger: str


@dataclass(frozen=True)
class C5BBDecision:
    finite_rooted_split_passes: bool
    reference_centering_passes: bool
    no_known_unmatched_o1_o2_survivor: bool
    exact_good_sector_expansion_proved: bool
    status: str
    next_checkpoint: str
    reason: str


def term_value(j: int, g_power: float, m_power: float, unit_bound: float, mu: float = MU) -> float:
    log_scale = mu * math.log(j + 1.0)
    return unit_bound * (j ** (-0.5 * g_power)) * (log_scale ** (0.5 * m_power))


def partial_sum(
    j0: int,
    j1: int,
    g_power: float,
    m_power: float,
    unit_bound: float,
    mu: float = MU,
) -> float:
    return sum(term_value(j, g_power, m_power, unit_bound, mu) for j in range(j0, j1 + 1))


def integral_tail(
    j_start: int,
    g_power: float,
    m_power: float,
    unit_bound: float,
    mu: float = MU,
) -> float | None:
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
    return unit_bound * (mu**n) * (float(j_start) ** (1.0 - alpha)) * total


def make_gate(
    name: str,
    model_term: str,
    g_power: float,
    m_power: float,
    unit_bound: float,
    gate: str,
    meaning: str,
) -> ResidualGate:
    p10 = partial_sum(J0, 10 * J0, g_power, m_power, unit_bound)
    p100 = partial_sum(J0, 100 * J0, g_power, m_power, unit_bound)
    tail = integral_tail(100 * J0 + 1, g_power, m_power, unit_bound)
    return ResidualGate(
        name=name,
        model_term=model_term,
        unit_bound=unit_bound,
        partial_10j0=p10,
        partial_100j0=p100,
        tail_after_100j0=tail,
        summable=tail is not None,
        gate=gate,
        meaning=meaning,
    )


def tangent_split_check() -> TangentSplitCheck:
    total_scalar = RETAINED_SCALAR_RANK + POSITIVE_HIGH_SCALAR_RANK + GAUGE_VERTEX_SCALAR_RANK
    retained_color = COLOR_DIM * RETAINED_SCALAR_RANK
    high_color = COLOR_DIM * POSITIVE_HIGH_SCALAR_RANK
    gauge_color = COLOR_DIM * GAUGE_VERTEX_SCALAR_RANK
    total_color = retained_color + high_color + gauge_color
    dimension_closure = total_scalar == 216 and total_color == 648
    finite_reflection = (
        ROOT_REFLECTION_FIXED
        and LOOP_UNMATCHED_REFLECTIONS == 0
        and STEM_UNMATCHED_REFLECTIONS == 0
        and ROOTED_PATH_UNMATCHED_REFLECTIONS == 0
        and MAX_ROOTED_REFLECTION_RESIDUAL < 1.0e-12
        and ROOTED_COMBINED_DENSITY_REFLECTION_RESIDUAL < 1.0e-12
    )
    finite_gauge = MAX_ROOTED_GAUGE_COVARIANCE_RESIDUAL < 1.0e-12
    return TangentSplitCheck(
        retained_scalar_rank=RETAINED_SCALAR_RANK,
        positive_high_scalar_rank=POSITIVE_HIGH_SCALAR_RANK,
        gauge_vertex_scalar_rank=GAUGE_VERTEX_SCALAR_RANK,
        total_scalar_rank=total_scalar,
        retained_color_dim=retained_color,
        positive_high_color_dim=high_color,
        gauge_color_dim=gauge_color,
        total_color_dim=total_color,
        dimension_closure=dimension_closure,
        root_vertex=ROOT_VERTEX,
        root_reflection_fixed=ROOT_REFLECTION_FIXED,
        loop_unmatched_reflections=LOOP_UNMATCHED_REFLECTIONS,
        stem_unmatched_reflections=STEM_UNMATCHED_REFLECTIONS,
        rooted_path_unmatched_reflections=ROOTED_PATH_UNMATCHED_REFLECTIONS,
        finite_rooted_reflection_covariant=finite_reflection,
        finite_rooted_gauge_covariant=finite_gauge,
        max_rooted_gauge_covariance_residual=MAX_ROOTED_GAUGE_COVARIANCE_RESIDUAL,
        max_rooted_reflection_residual=MAX_ROOTED_REFLECTION_RESIDUAL,
    )


def expansion_clauses() -> list[ExpansionClause]:
    return [
        ExpansionClause(
            label="BB1 rooted chart and K/H split",
            statement=(
                "On the good sector, the rooted common-stem holonomy-log variables and positive-high "
                "coordinates must give a controlled local chart modulo gauge."
            ),
            finite_or_formal_evidence=(
                "C5AL gives the tangent high/gauge split; C5AZ gives finite rooted gauge and "
                "reflection covariance."
            ),
            good_sector_status="finite tangent pass; nonlinear compact uniformity open",
            required_next_lemma="uniform inverse-function/disintegration lemma in rooted coordinates",
            failure_trigger="chart derivative loses invertibility or constants grow with the UV cutoff/good-sector radius",
        ),
        ExpansionClause(
            label="BB2 conditional RN expansion",
            statement=(
                "The conditional high measure at fixed shifted K must have an RN expansion "
                "g X1 + g^2 X2 + R_ge3 with a summable good-sector remainder."
            ),
            finite_or_formal_evidence="C5AJ fixes the cumulant algebra; C5T-C5X provide finite BCH/action inputs.",
            good_sector_status="formal coefficient algebra pass; exact compact shell expansion open",
            required_next_lemma="analytic RN expansion through O(g^2) with explicit shifted coordinate/density terms",
            failure_trigger="nonanalytic compact effects or uncontrolled Jacobian terms enter at O(g) or O(g^2)",
        ),
        ExpansionClause(
            label="BB3 O(g) placement",
            statement=(
                "The O(g) retained cubic must be exact retained coordinate/action data, and high-mode "
                "O(g) tadpoles must be centered."
            ),
            finite_or_formal_evidence="C5AD places retained S1; C5W centers the high-mode cubic in the Gaussian reference.",
            good_sector_status="finite/formal pass; exact shifted conditional centering open",
            required_next_lemma="prove no shifted conditional O(g) tadpole survives outside retained data",
            failure_trigger="an unretained O(g_j M_j^p) mean appears in the exact conditional shell",
        ),
        ExpansionClause(
            label="BB4 O(g^2) slot matching",
            statement=(
                "Every O(g^2) local output must match V_RP, V_comp, or J_shifted^{1/2}; none may remain residual."
            ),
            finite_or_formal_evidence="C5AI/C5BA show no known unmatched finite-block O(g^2) survivor.",
            good_sector_status="ledger pass; exact coefficient theorem open",
            required_next_lemma="compute exact shifted O(g^2) coefficient map and compare to normalized slots",
            failure_trigger="any nonlocal, gauge-variant, OS-incompatible, or unmatched O(g^2) output remains",
        ),
        ExpansionClause(
            label="BB5 centered odd density",
            statement=(
                "The shifted odd K/H density channel must have zero conditional mean so its first effect is a "
                "second cumulant."
            ),
            finite_or_formal_evidence="C5AV gives degree/oddness; C5AZ restores reflection placement; C5BA supplies shifted constant.",
            good_sector_status="reference Gaussian centering pass; exact conditional centering open",
            required_next_lemma="prove the shifted bilinear odd channel has zero mean under the exact conditional reference",
            failure_trigger="a nonzero O(g_j^2 M_j^2) density drift appears from the odd channel",
        ),
        ExpansionClause(
            label="BB6 rooted covariance envelope",
            statement=(
                "The centered odd channel must satisfy an L2 envelope controlled by C_cov,shifted^2, or a "
                "replacement constant with the same summability class."
            ),
            finite_or_formal_evidence=(
                f"Corrected C5AZ/C5BA give C_cov,rooted^2={ROOTED_COV_SQ:.12g}, and the O(g_j^4 M_j^4) budget is summable."
            ),
            good_sector_status="finite-block bound pass; uniform good-sector and quasilocality open",
            required_next_lemma="prove a deterministic shifted L2 bound stable under the exact shell disintegration",
            failure_trigger="the covariance constant is nonlocal, cutoff-growing, or couples to exceptional near-Cartan valleys",
        ),
        ExpansionClause(
            label="BB7 exceptional sector",
            statement=(
                "Chart failures, large fields, and near-Cartan coherent valleys must have summable cost or be retained."
            ),
            finite_or_formal_evidence="C5M gives a schedule; C5N gives a finite normal-form tail only in a restricted model.",
            good_sector_status="open high-risk theorem clause",
            required_next_lemma="prove local Wilson-action/FP/chart exceptional estimates for the shifted coordinate package",
            failure_trigger="exceptional probability or comparison-form defect is not summable over shells",
        ),
    ]


def slot_checks() -> list[SlotCheck]:
    return [
        SlotCheck(
            slot="shifted retained cubic S1",
            formal_order="O(g_j M_j^3)",
            required_treatment="exact retained coordinate/action data",
            shifted_status="carried from C5AD into shifted variables",
            exact_good_sector_status="not yet proved in full compact disintegration",
            residual_allowed=False,
            failure_trigger="unretained cubic residual or conditional O(g) tadpole",
        ),
        SlotCheck(
            slot="quadratic RP running action",
            formal_order="O(g_j^2 M_j^2)",
            required_treatment="reflection-symmetric Wilson plaquette-coupling update",
            shifted_status="compatible at tangent/package level",
            exact_good_sector_status="nonlinear positivity/RP placement still open",
            residual_allowed=False,
            failure_trigger="unmatched quadratic drift or negative/non-RP running term",
        ),
        SlotCheck(
            slot="quartic comparison potential Phi4",
            formal_order="O(g_j^2 M_j^4)",
            required_treatment="C5K comparison potential with controlled defect",
            shifted_status="unchanged by shifted coordinate contract",
            exact_good_sector_status="comparison defect theorem open",
            residual_allowed=False,
            failure_trigger="quartic term left as raw residual or non-summable comparison defect",
        ),
        SlotCheck(
            slot="shifted OS half-density rho2",
            formal_order="O(g_j^2 M_j^2)",
            required_treatment="exact shifted half-density/running density",
            shifted_status="finite reflection placement repaired by C5AY/C5AZ",
            exact_good_sector_status="quasilocal conditional density theorem open",
            residual_allowed=False,
            failure_trigger="nonlocal or uncentered density drift outside J_j^{1/2,shifted}",
        ),
        SlotCheck(
            slot="centered shifted odd density channel",
            formal_order="O(g_j^4 M_j^4) after centering",
            required_treatment="summable second-cumulant residual",
            shifted_status="finite L2 constant available and reflection-compatible",
            exact_good_sector_status="exact conditional centering and uniform covariance bound open",
            residual_allowed=True,
            failure_trigger="failure of centering converts it back to forbidden O(g_j^2 M_j^2)",
        ),
        SlotCheck(
            slot="post-retention shifted remainder",
            formal_order="O(g_j^{2+delta}P(M_j)) or better",
            required_treatment="C4 summable residual/error budget",
            shifted_status="allowed by C5BA only after all O(g)/O(g^2) slots are removed",
            exact_good_sector_status="not proved",
            residual_allowed=True,
            failure_trigger="any residual lacks scale gain, quasilocal decay, or summable exceptional control",
        ),
    ]


def residual_gates() -> list[ResidualGate]:
    return [
        make_gate(
            name="forbidden if odd centering fails",
            model_term="O(g_j^2 M_j^2)",
            g_power=2.0,
            m_power=2.0,
            unit_bound=ROOTED_COV_SQ,
            gate="FAIL",
            meaning="A nonzero mean of the shifted odd density channel is a nonsummable O(g^2) drift.",
        ),
        make_gate(
            name="allowed centered shifted odd variance",
            model_term="O(g_j^4 M_j^4)",
            g_power=4.0,
            m_power=4.0,
            unit_bound=ROOTED_COV_SQ,
            gate="PASS IF CENTERED",
            meaning="The second-cumulant class is summable once exact centering is established.",
        ),
        make_gate(
            name="allowed post-retention cubic-gain class",
            model_term="O(g_j^3 M_j^6)",
            g_power=3.0,
            m_power=6.0,
            unit_bound=PHI4_UNIT,
            gate="PASS WITH PROOF",
            meaning="Summable by scale power, though constants are not automatically small.",
        ),
        make_gate(
            name="allowed post-retention quartic-gain class",
            model_term="O(g_j^4 M_j^8)",
            g_power=4.0,
            m_power=8.0,
            unit_bound=PHI4_UNIT,
            gate="PASS WITH PROOF",
            meaning="Summable by scale power, but log powers and comparison constants matter.",
        ),
    ]


def centering_ledger(gates: list[ResidualGate]) -> CenteringLedger:
    centered_gate = next(gate for gate in gates if gate.name == "allowed centered shifted odd variance")
    return CenteringLedger(
        density_source_degree="quadratic in A=K+H at analytic O(g^2)",
        even_centered_channel="vanishes by degree",
        odd_channel_form="B_shifted(K,H)=K^T M_odd^shifted H",
        reference_conditional_mean_zero=True,
        exact_conditional_mean_zero_proved=False,
        variance_unit_bound=ROOTED_COV_SQ,
        conservative_log_cumulant_unit_bound=ROOTED_COV_SQ,
        residual_model="O(g_j^4 M_j^4) after exact centering",
        shifted_partial_100j0=centered_gate.partial_100j0,
        shifted_tail_after_100j0=float(centered_gate.tail_after_100j0 or 0.0),
        failure_trigger="If E_H B_shifted(K,H) is nonzero at O(1), it creates a forbidden O(g_j^2 M_j^2) density drift.",
    )


def decision(split: TangentSplitCheck, slots: list[SlotCheck], ledger: CenteringLedger) -> C5BBDecision:
    no_known_unmatched = all("failure" not in slot.shifted_status.lower() for slot in slots)
    finite_pass = (
        split.dimension_closure
        and split.finite_rooted_reflection_covariant
        and split.finite_rooted_gauge_covariant
        and ROOTED_N2_DIVERGENCE == 0.0
    )
    reference_centering = ledger.reference_conditional_mean_zero
    return C5BBDecision(
        finite_rooted_split_passes=finite_pass,
        reference_centering_passes=reference_centering,
        no_known_unmatched_o1_o2_survivor=no_known_unmatched,
        exact_good_sector_expansion_proved=False,
        status="partial pass: finite shifted algebra and reference centering pass; exact good-sector theorem remains open",
        next_checkpoint=(
            "C5BC: shifted disintegration/RN expansion lemma. Build the actual good-sector map "
            "(K,H)->links in rooted variables and prove/falsify uniform invertibility plus the O(g^2) RN expansion."
        ),
        reason=(
            "C5BB finds no new finite-block O(g) or O(g^2) survivor after the C5AZ shift. "
            "The centered odd channel is summable in the rooted covariance model, but only under exact centering. "
            "The unproved burden is now concentrated in the nonlinear compact good-sector disintegration and RN expansion."
        ),
    )


def main() -> None:
    split = tangent_split_check()
    clauses = expansion_clauses()
    slots = slot_checks()
    gates = residual_gates()
    ledger = centering_ledger(gates)
    result = {
        "checkpoint": "C5BB",
        "title": "shifted good-sector one-shell expansion diagnostic",
        "constants": {
            "rooted_op_norm": ROOTED_OP_NORM,
            "rooted_cov_sq": ROOTED_COV_SQ,
            "rooted_cov": ROOTED_COV,
            "old_cov_sq": OLD_COV_SQ,
            "rooted_to_old_cov_ratio": ROOTED_COV_SQ / OLD_COV_SQ,
            "j0": J0,
            "mu": MU,
        },
        "tangent_split_check": asdict(split),
        "expansion_clauses": [asdict(clause) for clause in clauses],
        "slot_checks": [asdict(slot) for slot in slots],
        "residual_gates": [asdict(gate) for gate in gates],
        "centering_ledger": asdict(ledger),
        "decision": asdict(decision(split, slots, ledger)),
    }

    assert split.dimension_closure
    assert split.finite_rooted_reflection_covariant
    assert split.finite_rooted_gauge_covariant
    assert ledger.reference_conditional_mean_zero
    assert not ledger.exact_conditional_mean_zero_proved
    assert not result["decision"]["exact_good_sector_expansion_proved"]

    out_path = Path("outputs/c5bb_shifted_good_sector_expansion.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
