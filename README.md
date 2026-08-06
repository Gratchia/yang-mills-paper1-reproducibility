# Yang--Mills finite-block reflection-obstruction reproducibility package

This repository/package accompanies the paper:

**A finite-block reflection obstruction and common-stem repair for based holonomy-log coordinates in `SU(2)` lattice Yang--Mills**

## Scope

This package reproduces finite-dimensional checks for one `n=2`, four-dimensional, `SU(2)` lattice block.  It does **not** claim or reproduce a continuum Yang--Mills construction, confinement, exponential clustering, or a mass gap.

## Requirements

- Python 3.12 or compatible Python 3
- NumPy
- Windows PowerShell for the provided wrapper, or manual Python execution of the listed scripts

Install NumPy if needed:

```bash
python -m pip install -r requirements.txt
```

## One-command reproduction

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File calculations/run_c5x_paper_reproducibility.ps1
```

Optional downstream correction-chain rerun:

```powershell
powershell -ExecutionPolicy Bypass -File calculations/run_c5x_paper_reproducibility.ps1 -IncludeDownstream
```

Both commands write to `outputs/c5x-reproducibility-rerun/`.  The optional downstream command is slower and reruns C5BA--C5BI correction-chain diagnostics; it does not execute C5BJ and does not advance the research route.

## Manifest

The staging manifest is `reproducibility-staging-manifest.json`.  The scientific rerun manifest produced by the wrapper is `outputs/c5x-reproducibility-rerun/manifest.json`.

## License

The reproducibility code in this package is released under the MIT License; see `LICENSE`.
