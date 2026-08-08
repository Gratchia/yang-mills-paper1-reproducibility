# C5X SciPost Physics Core submission-readiness audit

**Date:** 2026-08-06  
**Target venue:** SciPost Physics Core  
**Status:** local candidate package created; not yet submission-ready; no external submission made.

## 1. Official venue requirements checked

Official sources consulted:

- SciPost Physics Core authoring guidelines: <https://scipost.org/SciPostPhysCore/authoring>
- SciPost Physics Core about/acceptance criteria page: <https://scipost.org/SciPostPhysCore/about>
- SciPost Journals terms and conditions: <https://scipost.org/journals/journals_terms_and_conditions>
- SciPost LaTeX submission templates project: <https://git.scipost.org/scipost/SciPost_LaTeX_Templates_Submission>

Direct page opening was blocked by SciPost's current Anubis protection in this environment, but web-indexed official snippets were available and sufficient for the preparation checklist.  Before final submission, the author should open the pages directly in a normal browser and confirm that no item has changed.

## 2. Current package location

Local candidate folder:

`papers/submission/c5x-scipost-core-candidate-2026-08-06/`

Created files include:

- `c5x-manuscript-scipost-candidate.md`
- `c5x-manuscript-scipost-candidate.tex`
- `c5x-references-scipost-candidate.bib`
- `c5x-reproducibility-package.md`
- `reproducibility-scripts/`
- `packet-manifest.json`
- `source-endorser-review-pdf-for-comparison.pdf`

The older endorser packet remains preserved and should not be overwritten.

## 3. Claim boundary

The manuscript should remain a finite-block structural paper:

> In the explicitly defined \(n=2\), four-dimensional, \(SU(2)\) retained block, the original based holonomy-log coordinate has an exact retained quadratic BCH reflection obstruction; the exact uniform cyclic repairs are shifts \(1\) and \(5\); and the shift-\(1\) common-stem rooted package gives the stated reflection-compatible gauge-covariant coordinate repair.

It should not claim:

- continuum Yang--Mills construction;
- Osterwalder--Schrader reconstruction;
- confinement;
- exponential clustering;
- spectral mass gap;
- Wilson area law;
- physical-scale nonconcentration;
- completion of the later C5BJ/C5BK/C5BL theorem route.

## 4. Post-draft inclusion decision

Later C5BK/C5BL/C5BN work should not be added as a technical section.  It is mainly route-selection and obstruction analysis, not a clean theorem that strengthens the C5X finite-block result.

Allowed limited use:

- one short outlook or limitation sentence saying that later multiscale use still requires compact coarea/Radon--Nikodym control and exceptional-sector estimates;
- no checkpoint history;
- no C5BL private-link failure details;
- no C1 exact-kernel material unless a separate future paper or supplement explicitly needs it.

## 5. SciPost-readiness checklist

| item | status | required action |
|---|---:|---|
| finite theorem boundary | pass | keep current narrow claim |
| abstract under about 200 words | near pass | review after final typesetting |
| title descriptive and near two-line target | near pass | author may shorten if desired |
| clean LaTeX source | partial | candidate source created; final compile still required |
| SciPost template | open | either migrate to SciPost template or keep 10pt general LaTeX if accepted |
| DOI-linked references | pass-practical | DOI audit recorded in `literature-doi-audit.md`; final browser/author check still prudent |
| representative literature | near pass | Balaban/path-gauge/maximal-tree boundary is stated; one final expert-facing novelty check still recommended |
| publication-voice/internal-language scan | pass | v5 source and latest PDF checked: no audit-file/checkpoint/handoff/ledger/rooted-package/path-package/old-latest workflow language remains in manuscript prose |
| reproducibility package | local pass | GitHub-ready staging folder created and both core and downstream wrappers passed locally; public repository/release still needed |
| author metadata | pass | author supplied Higher Colleges of Technology--CERT and work email `gmkrttchian1@hct.ac.ae`; personal email retained as alternate note |
| acknowledgements | pass | author requested deletion; candidate manuscript omits acknowledgements |
| funding statement | pass | no external funding |
| competing interests | pass | no competing interests |
| AI-use disclosure | pass | author confirmed current wording is acceptable |
| public code availability | local-ready | MIT-licensed GitHub-ready staging package exists and passed locally; repository-content zip `yang-mills-paper1-reproducibility-mit-ready-2026-08-06.zip` created; public push/release still needed |
| final PDF | blocked | compile and visually inspect final PDF |
| final submission approval | blocked | author must explicitly approve submission |

## 6. Immediate next actions

1. Clean the candidate LaTeX source until it compiles without Markdown artifacts.
2. Convert the source to the SciPost template if practical, or keep a 10pt general LaTeX version with hyperref and DOI links.
3. Push or otherwise publish the reproducibility package and record the exact URL/release.
4. Confirm submission ethics/status and preferred SciPost specialty/referee suggestions.
5. Run the publication-voice/internal-language scan from `scientific-paper-style-guide.md`.
6. Compile and visually review the final PDF using `pdf-build-instructions.md`.
7. Only after author approval, proceed to SciPost direct submission.
