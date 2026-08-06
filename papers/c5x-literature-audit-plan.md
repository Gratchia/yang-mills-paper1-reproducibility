# C5X primary-source literature audit

**Audit date:** 2026-07-12  
**Status:** corrected minimum audit for a research draft; not an exhaustive submission review  
**Scope:** finite-block reflection obstruction and shifted common-stem repair for based `SU(2)` holonomy-log coordinates

## Audit rule

Access is recorded as `FULL TEXT`, `ABSTRACT/METADATA`, or `NOT CHECKED`. A source is used only at the level actually inspected. In particular, an abstract does not support a detailed theorem comparison.

## Claim boundary

The defensible claim is specific:

> For the fixed `n=2`, four-dimensional, `SU(2)` retained curvature-tile block defined in the manuscript, the original ordered based-log coordinate has an exact quadratic BCH reflection obstruction, while uniform shifts 1 and 5 give exact same/inverse path closure; the shift-1 package can be rooted at the reflection-fixed vertex by explicit compatible stems.

This is not a claim that common roots, stems, maximal trees, based holonomies, Bianchi connectors, trace cyclicity, lattice reflection positivity, or gauge-covariant averaging are new.

## Full-text audit

| source | access and portions inspected | result relevant here | wording consequence |
|---|---|---|---|
| K. Osterwalder and R. Schrader, *Axioms for Euclidean Green's functions*, CMP 31 (1973) 83--112, DOI `10.1007/BF01645738` | `FULL TEXT`; axioms/main theorems and positivity/reconstruction portions, pp. 87--98. [Primary PDF](https://projecteuclid.org/journals/communications-in-mathematical-physics/volume-31/issue-2/Axioms-for-Euclidean-Greens-functions/cmp/1103858969.pdf) | Positivity is one condition in a reconstruction framework, not a synonym for covariance under a reflection map. | Reserve “reflection positivity” for positivity of forms/measures. Call the finite result “reflection compatibility” or “reflection covariance.” |
| K. Osterwalder and R. Schrader, *Axioms for Euclidean Green's functions II*, CMP 42 (1975) 281--305, DOI `10.1007/BF01608978` | `FULL TEXT`; Introduction and reconstruction theorem statement, pp. 281--289. [Primary PDF](https://projecteuclid.org/journals/communications-in-mathematical-physics/volume-42/issue-3/Axioms-for-Euclidean-Greens-functions-II-with-an-Appendix-by/cmp/1103899050.pdf) | The second paper explicitly extends and corrects the earlier reconstruction conditions. | Do not infer continuum reconstruction from one finite coordinate identity. |
| G. G. Batrouni, *Plaquette Formulation and the Bianchi Identity for Lattice Gauge Theories*, NPB 208 (1982) 467--483, DOI `10.1016/0550-3213(82)90231-0`; corresponding LBL thesis chapter | `FULL TEXT`; thesis Chapter I, especially pp. 10--20 of the PDF: path gauge, plaquette change of variables, nonabelian Bianchi identity, connectors, and geometric interpretation. [Primary thesis PDF](https://escholarship.org/content/qt60w7w4t7/qt60w7w4t7.pdf) | Batrouni chooses an origin and paths to sites, and the nonabelian Bianchi relation requires ordered parallel-transport connectors. Plaquette compatibility constraints are standard and genuinely nonabelian. | The manuscript's Bianchi and stem language must acknowledge path-gauge/connector prior art. Its novelty cannot be “using paths from a common point.” |
| I. M. Burbano and C. W. Bauer, *Gauge Loop-String-Hadron Formulation on General Graphs and Applications to Fully Gauge Fixed Hamiltonian Lattice Gauge Theory*, arXiv:2409.13812v2 | `FULL TEXT`; Introduction and Sec. III.A--B, PDF pp. 2, 4--6; Appendix C consulted for graph-coarsening context. [Primary PDF](https://arxiv.org/pdf/2409.13812) | A maximal tree gives unique paths; each remaining physical link defines a closed trajectory based at one origin; all such loops transform under the single residual `SU(2)` at that origin. The paper also explicitly uses petal/stem language. | A common-root loop package and residual simultaneous conjugation are standard constructions. Our contribution is only the fixed-block reflection-compatible shift and stem assignment. |
| K. G. Wilson, *The Origins of Lattice Gauge Theory*, NPB Proc. Suppl. 140 (2005) 3--19, arXiv:hep-lat/0412043 | `FULL TEXT`; historical description of the 1974 lattice construction and Wilson-loop setting. [Primary PDF](https://arxiv.org/pdf/hep-lat/0412043) | Group-valued lattice links, plaquette products, and Wilson-loop observables are foundational background. | Do not present link holonomies, Wilson loops, or trace invariance as new. |

## Abstract and metadata audit

| source | access | supported use | limit |
|---|---|---|---|
| K. Osterwalder and E. Seiler, *Gauge field theories on a lattice*, Ann. Phys. 110 (1978) 440--471, DOI `10.1016/0003-4916(78)90039-8` | `ABSTRACT/METADATA`; [publisher record](https://www.sciencedirect.com/science/article/abs/pii/0003491678900398) | The abstract states physical positivity for lattice Schwinger functions and existence of a positive self-adjoint transfer matrix. | No detailed proposition from this paper is attributed without full-text verification. |
| T. Balaban, *Averaging operations for lattice gauge theories*, CMP 98 (1985) 17--51, DOI `10.1007/BF01211042` | `ABSTRACT/METADATA`; [publisher record](https://link.springer.com/article/10.1007/BF01211042) | Gauge-field and gauge-transformation averaging, with regularity/analyticity domains, is established constructive-RG territory. | This audit does not claim that Balaban used or did not use the manuscript's exact reflection-compatible path package. |
| T. Balaban, *The variational problem and background fields in renormalization group method for lattice gauge theories*, CMP 102 (1985) 277--309, DOI `10.1007/BF01229381` | `ABSTRACT/METADATA`; [repository record](https://deepblue.lib.umich.edu/items/e9af57c2-30bd-4dfc-85cd-3f01956f02fa) | The abstract states existence of an action minimizer with fixed averages and uniqueness up to gauge transformations. | It supports constructive-RG context, not the paper's finite path theorem. |
| T. Balaban, *Renormalization group approach to lattice gauge field theories I*, CMP 109 (1987) 249--301, DOI `10.1007/BF01215223` | `ABSTRACT/METADATA`; [primary bibliographic record](https://doi.org/10.1007/BF01215223) | Four-dimensional small-field effective actions, cluster expansion, and coupling renormalization are prior constructive-RG work. | No novelty comparison at theorem level is made until full text is checked. |
| K. G. Wilson, *Confinement of quarks*, PRD 10 (1974) 2445--2459, DOI `10.1103/PhysRevD.10.2445` | `ABSTRACT/METADATA` for the original article; the author's retrospective above was read in full. | Original lattice gauge/confinement context. | The present note makes no confinement claim. |

## Audit conclusions

1. **Common rooting is prior art.** Maximal-tree/path-gauge constructions already transport loop data to a common origin and leave simultaneous conjugation at that origin.
2. **Connectors are prior art.** Nonabelian plaquette/Bianchi formulations already require ordered parallel transports between local basepoints.
3. **Trace insensitivity is standard.** Cyclic conjugation is invisible to Wilson traces; the paper's issue arises because it uses oriented based Lie-algebra logs.
4. **The retained rank is paper-specific bookkeeping, not a new Bianchi principle.** The exact value 17 follows from the boundary cellular complex of this coarse four-cube and the chosen 24-face package.
5. **The potentially publishable observation is narrow.** It is the exact obstruction and exact shift/stem repair for this specified retained block, together with the distinction between shifted-core and genuinely rooted density packages.

## Novelty wording

Use:

> We isolate, in one explicitly defined finite retained block, a quadratic BCH obstruction caused by reflection-induced inverse-cyclic rebasing, and we identify the exact uniform cyclic shifts and a reflection-compatible common-stem realization that remove it.

Do not use:

- “a new common-stem method”;
- “a new rooted-loop formalism”;
- “a new Bianchi-compatible lattice gauge construction”;
- “a reflection-positivity theorem”;
- “a step proving the Yang--Mills mass gap.”

## Remaining submission-stage work

- Obtain and read the full Balaban averaging, variational, and 1987 RG papers before making a detailed constructive-RG novelty comparison.
- Search older maximal-tree, complete axial/path-gauge, and loop-coordinate literature beyond the minimum sources above.
- Convert the manuscript's audit table into conventional in-text citations and a bibliography.
- Ask a lattice-gauge/constructive-QFT expert specifically whether the exact fixed-block shift result is already implicit in a known path convention.
