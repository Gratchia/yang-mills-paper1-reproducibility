"""
C5BA: shifted normalized one-shell theorem contract.

This is not a proof of the Yang--Mills mass gap.  It is a bookkeeping and
consistency checkpoint for the current clean-start route:

  * freeze the C5AZ shifted/common-stem retained coordinate package;
  * translate the normalized one-shell package into those shifted variables;
  * separate forbidden O(g) / O(g^2) leftovers from summable centered or
    post-retention remainders;
  * expose the exact theorem obligations that remain before the next step.

The numerical constants recorded here are inherited from the completed
C5AY/C5AZ finite-block computations.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
from typing import Callable


# Frozen rooted common-stem constants from the corrected C5AZ computation.
ROOTED_OP_NORM = 1.0189627218075927
ROOTED_COV_SQ = 0.21480467679414722
ROOTED_COV = 0.463470254486895

# Previous unshifted centered-odd covariance proxy, kept only for comparison.
OLD_COV_SQ = 0.0315650978782

# Normalized scale model used in the C5AG/C5AH package.
J0 = 13082
MU = 16.0


@dataclass(frozen=True)
class PackageComponent:
    name: str
    shifted_status: str
    formal_size: str
    theorem_role: str
    may_remain_as_residual: bool
    open_debt: str


@dataclass(frozen=True)
class TheoremClause:
    label: str
    statement_needed: str
    current_evidence: str
    status: str
    failure_meaning: str


@dataclass(frozen=True)
class ResidualBudget:
    name: str
    model_term: str
    unit_bound: float
    partial_10j0: float
    partial_100j0: float
    tail_after_100j0: float | None
    classification: str
    interpretation: str


@dataclass(frozen=True)
class C5BADecision:
    shifted_coordinate_package_frozen: bool
    reflection_covariant_finite_block: bool
    normalized_contract_consistent: bool
    exact_one_shell_theorem_proved: bool
    next_checkpoint: str
    reason: str


def g(j: int, mu: float = MU) -> float:
    return 1.0 / (mu * math.log(j + 1.0))


def M(j: int, mu: float = MU) -> float:
    return mu * math.log(j + 1.0)


def term_value(j: int, g_power: float, m_power: float, unit_bound: float, mu: float = MU) -> float:
    """Model term j^{-g_power/2} (mu log j)^{m_power/2} times unit.

    This follows the C5AG/C5AH convention:

        g_j ~= j^{-1/2},     M_j^2 = mu log(j+1).

    The logarithmic damping carried elsewhere in the project is not folded into
    this diagnostic table; this table is deliberately the conservative
    normalized-shell budget used by C5AG/C5AH.
    """
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
    """Numerical tail estimate for summable power classes.

    Returns None for non-summable model exponents.  For the current checkpoint we
    only need a transparent upper diagnostic, not a sharp constant.
    """
    alpha = 0.5 * g_power
    log_power = 0.5 * m_power
    if alpha <= 1.0:
        return None

    if abs(log_power - round(log_power)) > 1.0e-12:
        # Current rows have integer log powers.  Returning None avoids inventing
        # a fragile special-function estimate for a diagnostic path we do not use.
        return None

    # Integral of x^{-alpha} (log x)^n from j_start to infinity, multiplied by
    # unit * mu^n.  This mirrors the exact diagnostic used in C5AG/C5AH.
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


def make_budget(
    name: str,
    model_term: str,
    g_power: float,
    m_power: float,
    unit_bound: float,
    classification: str,
    interpretation: str,
) -> ResidualBudget:
    p10 = partial_sum(J0, 10 * J0, g_power, m_power, unit_bound)
    p100 = partial_sum(J0, 100 * J0, g_power, m_power, unit_bound)
    tail = integral_tail(100 * J0 + 1, g_power, m_power, unit_bound)
    return ResidualBudget(
        name=name,
        model_term=model_term,
        unit_bound=unit_bound,
        partial_10j0=p10,
        partial_100j0=p100,
        tail_after_100j0=tail,
        classification=classification,
        interpretation=interpretation,
    )


def package_components() -> list[PackageComponent]:
    return [
        PackageComponent(
            name="rooted common-stem retained coordinate K_g^root",
            shifted_status="fixed by C5AY/C5AZ finite-block construction",
            formal_size="coordinate, not residual",
            theorem_role="retained 17-dimensional tangent quotient with reflection-covariant rooted paths",
            may_remain_as_residual=False,
            open_debt="prove exact one-shell disintegration and good-sector uniformity in the rooted chart",
        ),
        PackageComponent(
            name="retained cubic S1",
            shifted_status="must be expressed in shifted holonomy-log coordinates",
            formal_size="O(g_j M_j^3)",
            theorem_role="main odd retained interaction; cannot be discarded shell-by-shell",
            may_remain_as_residual=False,
            open_debt="show it is placed in the retained local coordinate/action package with correct reflection parity",
        ),
        PackageComponent(
            name="running quadratic/RP action",
            shifted_status="must be transported to the shifted coordinate basis",
            formal_size="O(g_j^2 M_j^2)",
            theorem_role="renormalized quadratic contribution, part of the one-shell comparison action",
            may_remain_as_residual=False,
            open_debt="prove reflection-positive placement without producing an unmatched boundary defect",
        ),
        PackageComponent(
            name="closed-loop quartic comparison potential Phi4",
            shifted_status="must be kept as a comparison-potential term, not a residual",
            formal_size="O(g_j^2 M_j^4)",
            theorem_role="absorbs the dangerous local quartic sector",
            may_remain_as_residual=False,
            open_debt="prove shifted-coordinate compatibility and quantify comparison-potential defect",
        ),
        PackageComponent(
            name="shifted OS half-density rho2",
            shifted_status="finite-block reflection residual killed by C5AY/C5AZ",
            formal_size="O(g_j^2 M_j^2)",
            theorem_role="Jacobian/conditional-density correction inside the retained package",
            may_remain_as_residual=False,
            open_debt="prove cutoff-stable conditional density control in the physical shell",
        ),
        PackageComponent(
            name="centered shifted odd K/H density channel",
            shifted_status="allowed only after exact conditional centering",
            formal_size="first nonzero effect modeled as O(g_j^4 M_j^4)",
            theorem_role="summable second-cumulant remainder after mean-zero cancellation",
            may_remain_as_residual=True,
            open_debt="prove conditional mean zero plus covariance envelope in shifted variables",
        ),
        PackageComponent(
            name="post-retention nonlinear remainder",
            shifted_status="allowed only with scale gain beyond O(g_j^2)",
            formal_size="O(g_j^{2+delta} P(M_j))",
            theorem_role="small summable error after retained coordinates/action are removed",
            may_remain_as_residual=True,
            open_debt="prove deterministic expansion and probabilistic good/exceptional-sector summability",
        ),
    ]


def theorem_clauses() -> list[TheoremClause]:
    return [
        TheoremClause(
            label="T1 shifted disintegration",
            statement_needed=(
                "The rooted common-stem variables define the retained quotient and admit a "
                "controlled conditional K/H split on each finite-ratio shell."
            ),
            current_evidence=(
                "C5AY gives rank-17 tangent quotient with exact reflection pairing; "
                "C5AZ gives rooted gauge covariance and finite-block reflection covariance."
            ),
            status="open theorem obligation",
            failure_meaning="The C5AY/C5AZ repair would remain a finite-block coordinate trick, not a shell theorem.",
        ),
        TheoremClause(
            label="T2 no unretained O(g) or O(g^2) survivors",
            statement_needed=(
                "All O(g_j M_j^3), O(g_j^2 M_j^2), and O(g_j^2 M_j^4) terms are either "
                "retained, renormalized, or placed in a comparison potential."
            ),
            current_evidence="C5AG/C5AH bookkeeping identifies exactly which terms are forbidden as residuals.",
            status="contractual requirement, not yet proven in shifted nonlinear chart",
            failure_meaning="A nonsummable shell series survives, so the route fails in its current form.",
        ),
        TheoremClause(
            label="T3 centered odd density channel",
            statement_needed=(
                "The odd K/H density contribution has zero conditional mean and its first surviving "
                "effect satisfies a rooted covariance bound at O(g_j^4 M_j^4)."
            ),
            current_evidence=(
                f"Corrected C5AZ rooted covariance scale C_cov^2={ROOTED_COV_SQ:.12g}; "
                "C5AV-C5AW supplied the deterministic identity template."
            ),
            status="open proof/falsification target",
            failure_meaning="A hidden O(g_j^2 M_j^2) density drift remains; C5AP/C5AQ/C5BA collapse.",
        ),
        TheoremClause(
            label="T4 quasilocal conditional density",
            statement_needed=(
                "The conditional density/Jacobian constants are stable under microscopic integration and "
                "depend quasilocally on the shell boundary."
            ),
            current_evidence="Only finite-block and formal shifted-coordinate evidence so far.",
            status="major open bridge",
            failure_meaning="The finite-block calculation cannot be promoted to constructive continuum control.",
        ),
        TheoremClause(
            label="T5 exceptional sector",
            statement_needed=(
                "Large-field, chart-boundary, and near-Cartan exceptional events have summable probability "
                "or are absorbed by retained variables/comparison potentials."
            ),
            current_evidence="Near-Cartan visibility has been diagnosed as physically necessary, not solved.",
            status="open and high-risk",
            failure_meaning="The route may miss the coherent abelian/Cartan valleys that motivated the project.",
        ),
        TheoremClause(
            label="T6 RP/reflection placement",
            statement_needed=(
                "The retained coordinate, density, and comparison-potential package is compatible with "
                "reflection positivity at the one-shell level."
            ),
            current_evidence="C5AY/C5AZ removed the based-log reflection-covariance obstruction at finite block.",
            status="partly repaired, theorem still open",
            failure_meaning="Even correct local estimates may not imply a physical Hilbert-space mass gap.",
        ),
    ]


def residual_budgets() -> list[ResidualBudget]:
    return [
        make_budget(
            name="forbidden unretained cubic",
            model_term="O(g_j M_j^3)",
            g_power=1.0,
            m_power=3.0,
            unit_bound=1.0,
            classification="FORBIDDEN",
            interpretation="This grows too strongly to leave outside the retained shell action.",
        ),
        make_budget(
            name="forbidden unmatched shifted quadratic density/action",
            model_term="O(g_j^2 M_j^2)",
            g_power=2.0,
            m_power=2.0,
            unit_bound=ROOTED_COV_SQ,
            classification="FORBIDDEN",
            interpretation="Even with the shifted constant, an unmatched O(g^2) shell drift is nonsummable.",
        ),
        make_budget(
            name="forbidden unmatched quartic comparison defect",
            model_term="O(g_j^2 M_j^4)",
            g_power=2.0,
            m_power=4.0,
            unit_bound=0.0280541227163,
            classification="FORBIDDEN",
            interpretation="Quartic local effects must be retained or absorbed in Phi4, not treated as error.",
        ),
        make_budget(
            name="allowed centered shifted odd second cumulant",
            model_term="O(g_j^4 M_j^4)",
            g_power=4.0,
            m_power=4.0,
            unit_bound=ROOTED_COV_SQ,
            classification="ALLOWED IF CENTERED",
            interpretation="Summable only after exact conditional mean-zero cancellation is proven.",
        ),
        make_budget(
            name="allowed post-retention cubic-gain remainder",
            model_term="O(g_j^3 M_j^6)",
            g_power=3.0,
            m_power=6.0,
            unit_bound=0.0280541227163,
            classification="ALLOWED WITH PROOF",
            interpretation="Summable scale gain, but needs deterministic expansion plus probability control.",
        ),
        make_budget(
            name="allowed post-retention quartic-gain remainder",
            model_term="O(g_j^4 M_j^8)",
            g_power=4.0,
            m_power=8.0,
            unit_bound=0.0280541227163,
            classification="ALLOWED WITH PROOF",
            interpretation="Summable in principle, but log powers make constants nontrivial.",
        ),
    ]


def decision() -> C5BADecision:
    return C5BADecision(
        shifted_coordinate_package_frozen=True,
        reflection_covariant_finite_block=True,
        normalized_contract_consistent=True,
        exact_one_shell_theorem_proved=False,
        next_checkpoint=(
            "C5BB: prove or falsify the shifted good-sector one-shell expansion, "
            "especially T1-T3: disintegration, no O(g)/O(g^2) survivors, and centered odd density covariance."
        ),
        reason=(
            "C5BA passes as a contract checkpoint because the finite-block shifted coordinate package removes "
            "the reflection-coordinate obstruction and the normalized budgets still separate forbidden from "
            "summable terms. It is not a theorem checkpoint: T1-T6 remain proof obligations."
        ),
    )


def old_vs_rooted_centered_odd() -> dict[str, float]:
    old_10 = partial_sum(J0, 10 * J0, 4.0, 4.0, OLD_COV_SQ)
    old_100 = partial_sum(J0, 100 * J0, 4.0, 4.0, OLD_COV_SQ)
    new_10 = partial_sum(J0, 10 * J0, 4.0, 4.0, ROOTED_COV_SQ)
    new_100 = partial_sum(J0, 100 * J0, 4.0, 4.0, ROOTED_COV_SQ)
    return {
        "old_cov_sq": OLD_COV_SQ,
        "rooted_cov_sq": ROOTED_COV_SQ,
        "ratio_rooted_to_old": ROOTED_COV_SQ / OLD_COV_SQ,
        "old_partial_10j0": old_10,
        "rooted_partial_10j0": new_10,
        "old_partial_100j0": old_100,
        "rooted_partial_100j0": new_100,
    }


def main() -> None:
    result = {
        "checkpoint": "C5BA",
        "title": "shifted normalized one-shell theorem contract",
        "constants": {
            "rooted_op_norm": ROOTED_OP_NORM,
            "rooted_cov_sq": ROOTED_COV_SQ,
            "rooted_cov": ROOTED_COV,
            "j0": J0,
            "mu": MU,
        },
        "package_components": [asdict(x) for x in package_components()],
        "theorem_clauses": [asdict(x) for x in theorem_clauses()],
        "residual_budgets": [asdict(x) for x in residual_budgets()],
        "old_vs_rooted_centered_odd": old_vs_rooted_centered_odd(),
        "decision": asdict(decision()),
    }
    out_path = Path("outputs/c5ba_shifted_normalized_shell_contract.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
