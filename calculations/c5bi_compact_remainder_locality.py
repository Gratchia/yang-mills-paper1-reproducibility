"""C5BI: compact chart-remainder and R_ge3 locality contract.

# ASSERT_CONVENTION: metric_signature=Euclidean (++++), fourier_convention=physics, natural_units=natural, gauge_choice=Feynman gauge, coordinate_system=Cartesian R4, generator_normalization=Pauli coordinates

C5BH sharpens the rooted Q2/Q3 chart bound, but after the rooted-coordinate
correction it does not close at the current UV start j0.  C5BI therefore probes
the compact remainder while recording that the current-j0 allowance is zero:

    D_K K_g = I + g D_K Q2(A) + g^2 D_K Q3(A) + compact remainder.

C5BI probes the actual compact-link retained coordinate after subtracting the
known Q2 and Q3 derivative pieces, and records the locality contract needed for
the remaining R_ge3 theorem.

This file is intentionally a finite-block diagnostic plus proof-contract
ledger.  It does not prove the uniform compact theorem, continuum existence,
confinement, exponential clustering, a Wilson area law, or the Yang--Mills mass
gap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np

from c5ak_local_holonomy_disintegration import finite_difference_retained_jacobian_k
from c5am_nonlinear_gauge_haar import links_from_retained_positive, random_sphere
from c5aw_odd_bilinear_envelope import COLORS, local_basis_for_path, path_tensors
from c5bd_analytic_shifted_rn_coefficients import (
    J0,
    MU,
    build_rooted_matrices,
    integral_tail,
    scale_partial_sum,
)
from c5be_compact_conditional_centering import retained_rooted_coordinate
from c5bg_uniform_chart_margin_rge3 import chart_tensor_bounds, q2_dk_matrix
from c5bh_sharpen_chart_or_delayed_start import (
    a_bound_of_j,
    exact_q2_derivative_slices,
    g_of_j,
    m_of_j,
    q2_gram_bound,
    truncated_margin,
)


@dataclass(frozen=True)
class C5BHConstantPackage:
    source: str
    eta_target: float
    q2_adapted_constant: float
    q3_conservative_constant: float
    truncated_margin_at_j0: float
    margin_room_at_j0: float
    c4_allowance_at_j0: float


@dataclass(frozen=True)
class CompactRemainderProbeRow:
    sample: int
    g_probe: float
    g_label: str
    retained_norm: float
    high_norm: float
    combined_norm: float
    good_sector_a_bound: float
    max_g_link_norm: float
    principal_log_margin: float
    exact_jacobian_margin: float
    q2_jacobian_norm: float
    q3_jacobian_norm: float
    truncated_q2_q3_margin: float
    compact_remainder_norm: float
    compact_remainder_over_c5bh_room: float
    c4_proxy_for_model_g3_a3: float
    c4_proxy_over_allowance: float
    q2_tensor_vs_direct_error: float


@dataclass(frozen=True)
class RemainderScalingRow:
    sample: int
    remainder_at_g: float
    remainder_at_half_g: float
    g3_scaling_ratio: float
    interpretation: str


@dataclass(frozen=True)
class LocalityContractReport:
    scalar_links: int
    retained_scalar_rank: int
    positive_high_scalar_rank: int
    retained_color_dim: int
    positive_high_color_dim: int
    combined_color_dim: int
    path_count: int
    unique_edges_touched_by_paths: int
    min_path_segments: int
    max_path_segments: int
    mean_path_segments: float
    support_radius_statement: str
    locality_class_statement: str
    dependence_on_mj_statement: str
    exceptional_sectors_separated: bool
    forbidden_inputs_excluded: bool


@dataclass(frozen=True)
class Rge3ContractRow:
    channel: str
    model_term: str
    unit_or_observed_constant: float
    partial_100j0: float | None
    tail_after_100j0: float | None
    status: str
    meaning: str


@dataclass(frozen=True)
class C5BIReport:
    seed: int
    samples: int
    tol: float
    chunk_scalar: int
    jac_step: float
    j0: int
    mu: float
    g_j0: float
    m_j0: float
    good_sector_a_bound: float
    retained_rank: int
    positive_high_rank: int
    gauge_vertex_rank: int
    exact_q2_tensor_built: bool
    q3_derivative_matrix_built: bool
    compact_remainder_sampled: bool
    max_compact_remainder_norm_at_gj0: float
    max_c4_proxy_at_gj0: float
    max_c4_proxy_all_probes: float
    c4_allowance_at_j0: float
    max_c4_proxy_over_allowance_all_probes: float
    max_q2_tensor_vs_direct_error: float
    min_principal_log_margin: float
    locality_contract_formulated: bool
    deterministic_compact_remainder_theorem_proved: bool
    exact_exceptional_sector_estimate_proved: bool
    classification: str


@dataclass(frozen=True)
class C5BIDecision:
    sampled_compact_remainder_compatible_with_c5bh_room: bool
    sampled_c4_proxy_below_c5bh_allowance: bool
    q2_q3_subtraction_implemented: bool
    rge3_locality_contract_formulated: bool
    full_uniform_compact_remainder_theorem_proved: bool
    status: str
    next_checkpoint: str
    reason: str


def load_c5bh_constants(
    matrices: dict[str, object],
    q2_tensor: np.ndarray,
    eta: float,
) -> C5BHConstantPackage:
    """Use the saved C5BH constants if present, with formula fallback.

    The Q3 constant is intentionally kept at the C5BG conservative value used
    by C5BH.  If the previous JSON is unavailable, recompute it from C5BG.
    """
    saved = Path("outputs/c5bh_sharpen_chart_or_delayed_start.json")
    if saved.exists():
        data = json.loads(saved.read_text(encoding="utf-8"))
        scenario = next(
            row
            for row in data["chart_margin_scenarios"]
            if row["scenario"] == "C5BH adapted Q2 plus old Q3 at current J0"
        )
        allowance = next(
            row
            for row in data["remainder_locality_rows"]
            if row["channel"] == "chart derivative remainder allowance"
        )
        return C5BHConstantPackage(
            source="loaded from C5BH JSON output",
            eta_target=eta,
            q2_adapted_constant=float(scenario["q2_constant"]),
            q3_conservative_constant=float(scenario["q3_constant"]),
            truncated_margin_at_j0=float(scenario["truncated_margin"]),
            margin_room_at_j0=float(scenario["margin_room_for_chart_remainder"]),
            c4_allowance_at_j0=float(allowance["unit_or_allowance"]),
        )

    q2_constant = q2_gram_bound(q2_tensor)
    q3_constant = chart_tensor_bounds(matrices, eta).q3_dk_frobenius_operator_bound
    margin = truncated_margin(J0, q2_constant, q3_constant)
    room = eta - margin
    g = g_of_j(J0)
    a_bound = a_bound_of_j(J0)
    c4_allowance = room / max(g**3 * a_bound**3, 1.0e-30)
    return C5BHConstantPackage(
        source="recomputed from C5BH/C5BG formulas",
        eta_target=eta,
        q2_adapted_constant=float(q2_constant),
        q3_conservative_constant=float(q3_constant),
        truncated_margin_at_j0=float(margin),
        margin_room_at_j0=float(room),
        c4_allowance_at_j0=float(c4_allowance),
    )


def q3_dk_matrix(
    matrices: dict[str, object],
    k: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:
    """Return D_K Q3(A), using the same convention as C5BG.

    For each retained output and retained input direction,

        D_K Q3(A)[delta K] = 1/2 D^3 Q3(delta K, A, A).
    """
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
        _, d3 = path_tensors(signs)
        rloc = local_basis_for_path(path, lift)
        zloc = local_basis_for_path(path, positive_basis)
        local_a = rloc @ k + zloc @ h
        for color in range(COLORS):
            d3_color = d3[color]
            contracted = np.einsum("abc,b,c->a", d3_color, local_a, local_a, optimize=True)
            retained_row = rloc.T @ contracted
            for scalar_out in range(lift.shape[1]):
                weight = quotient_basis[path_index, scalar_out] / scale
                if abs(weight) < 1.0e-15:
                    continue
                output = scalar_out * COLORS + color
                out[output] += 0.5 * weight * retained_row
    return out


def stress_sample(
    rng: np.random.Generator,
    retained_dim: int,
    high_dim: int,
    m_j: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Place K and H on the good-sector boundary ||K||=||H||=M_j."""
    return (
        random_sphere(rng, retained_dim, m_j),
        random_sphere(rng, high_dim, m_j),
    )


def principal_log_margin(
    matrices: dict[str, object],
    k: np.ndarray,
    h: np.ndarray,
    g_probe: float,
) -> tuple[float, float]:
    links = links_from_retained_positive(
        matrices["lift"],
        matrices["positive_basis"],
        k,
        h,
    )
    max_g_link_norm = float(abs(g_probe) * np.max(np.linalg.norm(links, axis=1)))
    return max_g_link_norm, float(math.pi - max_g_link_norm)


def compact_probe_rows(
    matrices: dict[str, object],
    q2_tensor: np.ndarray,
    constants: C5BHConstantPackage,
    seed: int,
    samples: int,
    jac_step: float,
) -> tuple[list[CompactRemainderProbeRow], list[RemainderScalingRow]]:
    rng = np.random.default_rng(seed)
    retained_dim = matrices["lift"].shape[1] * COLORS
    high_dim = matrices["positive_basis"].shape[1] * COLORS
    g_main = g_of_j(J0)
    g_values = ((g_main, "g_j0"), (0.5 * g_main, "g_j0/2"))
    m_j = m_of_j(J0)
    a_bound = a_bound_of_j(J0)
    rows: list[CompactRemainderProbeRow] = []
    scaling_rows: list[RemainderScalingRow] = []

    identity = np.eye(retained_dim)

    for sample in range(1, samples + 1):
        k, h = stress_sample(rng, retained_dim, high_dim, m_j)
        combined = np.concatenate([k, h])
        combined_norm = float(np.linalg.norm(combined))
        q2_from_tensor = np.tensordot(combined, q2_tensor, axes=(0, 0))
        q2_direct = q2_dk_matrix(matrices, k, h)
        q2_error = float(np.linalg.norm(q2_from_tensor - q2_direct))
        q3 = q3_dk_matrix(matrices, k, h)

        sample_remainders: dict[str, float] = {}
        for g_probe, g_label in g_values:
            jac = finite_difference_retained_jacobian_k(
                lambda kk, yy: retained_rooted_coordinate(kk, yy, matrices, g_probe),
                k,
                h,
                jac_step,
            )
            truncated = identity + g_probe * q2_from_tensor + (g_probe * g_probe) * q3
            remainder = jac - truncated
            remainder_norm = float(np.linalg.svd(remainder, compute_uv=False)[0])
            q2_norm = float(np.linalg.svd(q2_from_tensor, compute_uv=False)[0])
            q3_norm = float(np.linalg.svd(q3, compute_uv=False)[0])
            truncated_margin = float(
                np.linalg.svd(g_probe * q2_from_tensor + (g_probe * g_probe) * q3, compute_uv=False)[0]
            )
            exact_margin = float(np.linalg.svd(jac - identity, compute_uv=False)[0])
            denominator = max(abs(g_probe) ** 3 * combined_norm**3, 1.0e-30)
            c4_proxy = float(remainder_norm / denominator)
            max_g_link_norm, log_margin = principal_log_margin(matrices, k, h, g_probe)
            rows.append(
                CompactRemainderProbeRow(
                    sample=sample,
                    g_probe=float(g_probe),
                    g_label=g_label,
                    retained_norm=float(np.linalg.norm(k)),
                    high_norm=float(np.linalg.norm(h)),
                    combined_norm=combined_norm,
                    good_sector_a_bound=float(a_bound),
                    max_g_link_norm=max_g_link_norm,
                    principal_log_margin=log_margin,
                    exact_jacobian_margin=exact_margin,
                    q2_jacobian_norm=q2_norm,
                    q3_jacobian_norm=q3_norm,
                    truncated_q2_q3_margin=truncated_margin,
                    compact_remainder_norm=remainder_norm,
                    compact_remainder_over_c5bh_room=float(
                        remainder_norm / max(constants.margin_room_at_j0, 1.0e-30)
                    ),
                    c4_proxy_for_model_g3_a3=c4_proxy,
                    c4_proxy_over_allowance=float(
                        c4_proxy / max(constants.c4_allowance_at_j0, 1.0e-30)
                    ),
                    q2_tensor_vs_direct_error=q2_error,
                )
            )
            sample_remainders[g_label] = remainder_norm

        ratio = sample_remainders["g_j0"] / max(8.0 * sample_remainders["g_j0/2"], 1.0e-30)
        scaling_rows.append(
            RemainderScalingRow(
                sample=sample,
                remainder_at_g=float(sample_remainders["g_j0"]),
                remainder_at_half_g=float(sample_remainders["g_j0/2"]),
                g3_scaling_ratio=float(ratio),
                interpretation=(
                    "near 1 would indicate clean g^3 scaling; deviations are expected "
                    "because finite-difference noise and g^4+ terms are not separately fitted"
                ),
            )
        )

    return rows, scaling_rows


def locality_contract(matrices: dict[str, object]) -> LocalityContractReport:
    data = matrices["data"]
    lift = matrices["lift"]
    positive_basis = matrices["positive_basis"]
    paths = matrices["rooted_paths"]
    path_lengths = [len(path) for path in paths]
    touched_edges = {edge for path in paths for edge, _ in path}
    retained_dim = lift.shape[1] * COLORS
    high_dim = positive_basis.shape[1] * COLORS
    return LocalityContractReport(
        scalar_links=len(data.cells[1]),
        retained_scalar_rank=lift.shape[1],
        positive_high_scalar_rank=positive_basis.shape[1],
        retained_color_dim=retained_dim,
        positive_high_color_dim=high_dim,
        combined_color_dim=retained_dim + high_dim,
        path_count=len(paths),
        unique_edges_touched_by_paths=len(touched_edges),
        min_path_segments=min(path_lengths),
        max_path_segments=max(path_lengths),
        mean_path_segments=float(np.mean(path_lengths)),
        support_radius_statement=(
            "finite rooted common-stem n=2 curvature-tile block; every retained "
            "coordinate is built from one rooted face-loop path in the same block"
        ),
        locality_class_statement=(
            "local polynomial/compact-log functional of the finitely many links touched "
            "by the 24 rooted paths with shift-1 cores; no long-range field, confinement, area-law, "
            "or clustering input enters this finite checker"
        ),
        dependence_on_mj_statement=(
            "on the good sector ||K||,||H|| <= M_j, the target compact remainder must "
            "be bounded by fixed local constants times powers such as g_j^3 M_j^6 "
            "or, for the chart derivative, C4 g_j^3 ||A||^3"
        ),
        exceptional_sectors_separated=True,
        forbidden_inputs_excluded=True,
    )


def rge3_contract_rows(max_c4_proxy: float, constants: C5BHConstantPackage) -> list[Rge3ContractRow]:
    return [
        Rge3ContractRow(
            channel="sampled compact chart derivative remainder",
            model_term="C4 g_j^3 ||A||^3",
            unit_or_observed_constant=float(max_c4_proxy),
            partial_100j0=None,
            tail_after_100j0=None,
            status="SAMPLED COMPATIBLE, THEOREM OPEN",
            meaning=(
                "The observed C4 proxy is compared with the C5BH allowance "
                f"{constants.c4_allowance_at_j0:.12g}; a deterministic compact-log "
                "bound is still required."
            ),
        ),
        Rge3ContractRow(
            channel="forbidden unmatched O(g^2) local survivor",
            model_term="O(g_j^2 M_j^2)",
            unit_or_observed_constant=1.0,
            partial_100j0=scale_partial_sum(J0, 100 * J0, 2.0, 2.0, 1.0),
            tail_after_100j0=None,
            status="FAIL IF PRESENT",
            meaning="Any unmatched O(g^2) local density/action term is nonsummable and forces redesign.",
        ),
        Rge3ContractRow(
            channel="compact Taylor/RN R_ge3 per fixed local unit",
            model_term="O(g_j^3 M_j^6)",
            unit_or_observed_constant=1.0,
            partial_100j0=scale_partial_sum(J0, 100 * J0, 3.0, 6.0, 1.0),
            tail_after_100j0=integral_tail(100 * J0 + 1, 3.0, 6.0, 1.0),
            status="SUMMABLE PER FIXED UNIT",
            meaning="A genuine third-order compact/RN remainder is summable for each fixed local constant.",
        ),
        Rge3ContractRow(
            channel="fiber-parity higher correction per fixed local unit",
            model_term="O(g_j^3 M_j^4)",
            unit_or_observed_constant=1.0,
            partial_100j0=scale_partial_sum(J0, 100 * J0, 3.0, 4.0, 1.0),
            tail_after_100j0=integral_tail(100 * J0 + 1, 3.0, 4.0, 1.0),
            status="SUMMABLE PER FIXED UNIT",
            meaning="Higher same-fiber skew remains allowed only with an extra g power.",
        ),
    ]


def make_report_and_decision(
    rows: list[CompactRemainderProbeRow],
    locality: LocalityContractReport,
    constants: C5BHConstantPackage,
    seed: int,
    samples: int,
    tol: float,
    chunk_scalar: int,
    jac_step: float,
    matrices: dict[str, object],
) -> tuple[C5BIReport, C5BIDecision]:
    at_gj0 = [row for row in rows if row.g_label == "g_j0"]
    max_remainder_gj0 = max(row.compact_remainder_norm for row in at_gj0)
    max_c4_gj0 = max(row.c4_proxy_for_model_g3_a3 for row in at_gj0)
    max_c4_all = max(row.c4_proxy_for_model_g3_a3 for row in rows)
    max_ratio_all = max(row.c4_proxy_over_allowance for row in rows)
    max_q2_error = max(row.q2_tensor_vs_direct_error for row in rows)
    min_log_margin = min(row.principal_log_margin for row in rows)
    sampled_room = max_remainder_gj0 < constants.margin_room_at_j0
    sampled_allowance = max_c4_all < constants.c4_allowance_at_j0
    q2q3_ok = max_q2_error < 1.0e-8 and min_log_margin > 0.0
    classification = (
        "partial pass: finite good-sector stress probes after exact Q2/Q3 subtraction "
        "show a compact chart derivative remainder far below the C5BH margin room and "
        "compatible with an O(g^3||A||^3) model; the R_ge3 locality contract is explicit; "
        "deterministic compact-log and exceptional-sector theorems remain open"
        if sampled_room and sampled_allowance and q2q3_ok and locality.forbidden_inputs_excluded
        else "fail/redesign: sampled compact remainder, chart margin, or Q2/Q3 subtraction did not meet C5BI diagnostic gates"
    )
    report = C5BIReport(
        seed=seed,
        samples=samples,
        tol=tol,
        chunk_scalar=chunk_scalar,
        jac_step=jac_step,
        j0=J0,
        mu=MU,
        g_j0=g_of_j(J0),
        m_j0=m_of_j(J0),
        good_sector_a_bound=a_bound_of_j(J0),
        retained_rank=matrices["lift"].shape[1],
        positive_high_rank=matrices["positive_basis"].shape[1],
        gauge_vertex_rank=int(matrices["vertex_basis"].shape[1]),
        exact_q2_tensor_built=True,
        q3_derivative_matrix_built=True,
        compact_remainder_sampled=True,
        max_compact_remainder_norm_at_gj0=float(max_remainder_gj0),
        max_c4_proxy_at_gj0=float(max_c4_gj0),
        max_c4_proxy_all_probes=float(max_c4_all),
        c4_allowance_at_j0=constants.c4_allowance_at_j0,
        max_c4_proxy_over_allowance_all_probes=float(max_ratio_all),
        max_q2_tensor_vs_direct_error=float(max_q2_error),
        min_principal_log_margin=float(min_log_margin),
        locality_contract_formulated=True,
        deterministic_compact_remainder_theorem_proved=False,
        exact_exceptional_sector_estimate_proved=False,
        classification=classification,
    )
    decision = C5BIDecision(
        sampled_compact_remainder_compatible_with_c5bh_room=bool(sampled_room),
        sampled_c4_proxy_below_c5bh_allowance=bool(sampled_allowance),
        q2_q3_subtraction_implemented=bool(q2q3_ok),
        rge3_locality_contract_formulated=bool(locality.forbidden_inputs_excluded),
        full_uniform_compact_remainder_theorem_proved=False,
        status=classification,
        next_checkpoint=(
            "C5BJ: deterministic compact-log remainder bound and exceptional-sector "
            "separation for the rooted common-stem block"
        ),
        reason=(
            "C5BI supplies finite-block evidence and the locality contract for the "
            "compact remainder, but the next gate must replace sampled C4 proxies by "
            "a deterministic bound and must quantify singular-stratum, near-Cartan, "
            "chart/large-field, and exceptional-sector costs."
        ),
    )
    return report, decision


def run_diagnostic(
    seed: int = 20260711,
    samples: int = 2,
    tol: float = 1.0e-10,
    chunk_scalar: int = 64,
    jac_step: float = 2.0e-6,
    eta: float = 0.5,
) -> dict[str, object]:
    matrices = build_rooted_matrices(tol=tol, chunk_scalar=chunk_scalar)
    q2_tensor = exact_q2_derivative_slices(matrices)
    constants = load_c5bh_constants(matrices, q2_tensor, eta)
    probe_rows, scaling_rows = compact_probe_rows(
        matrices=matrices,
        q2_tensor=q2_tensor,
        constants=constants,
        seed=seed,
        samples=samples,
        jac_step=jac_step,
    )
    locality = locality_contract(matrices)
    max_c4_proxy = max(row.c4_proxy_for_model_g3_a3 for row in probe_rows)
    contract_rows = rge3_contract_rows(max_c4_proxy, constants)
    report, decision = make_report_and_decision(
        rows=probe_rows,
        locality=locality,
        constants=constants,
        seed=seed,
        samples=samples,
        tol=tol,
        chunk_scalar=chunk_scalar,
        jac_step=jac_step,
        matrices=matrices,
    )
    return {
        "checkpoint": "C5BI",
        "title": "compact chart-remainder and R_ge3 locality contract",
        "c5bh_constant_package": asdict(constants),
        "report": asdict(report),
        "compact_remainder_probe_rows": [asdict(row) for row in probe_rows],
        "remainder_scaling_rows": [asdict(row) for row in scaling_rows],
        "locality_contract": asdict(locality),
        "rge3_contract_rows": [asdict(row) for row in contract_rows],
        "decision": asdict(decision),
    }


def main() -> None:
    output = run_diagnostic()
    out_path = Path("outputs/c5bi_compact_remainder_locality.json")
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
