param(
    [switch]$IncludeDownstream
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
$OutDir = Join-Path $RootDir "outputs\c5x-reproducibility-rerun"
$BundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if ($env:PYTHON -and (Test-Path $env:PYTHON)) {
    $PythonExe = $env:PYTHON
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonExe = "python"
}
elseif (Test-Path $BundledPython) {
    $PythonExe = $BundledPython
}
else {
    throw "No Python executable found. Set `$env:PYTHON or install Python."
}

$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONHASHSEED = "0"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Invoke-PythonReport {
    param(
        [Parameter(Mandatory = $true)][string]$Script,
        [Parameter(Mandatory = $true)][string]$Report
    )
    & $PythonExe -B $Script *> $Report
    $NativeExitCode = $LASTEXITCODE
    if ($NativeExitCode -ne 0) {
        throw "Python script failed with exit code $NativeExitCode`: $Script"
    }
}

function Assert-ReportPattern {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Pattern,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not (Select-String -LiteralPath $Path -Pattern $Pattern -Quiet)) {
        throw "Expected decision assertion failed: $Label"
    }
}

Push-Location $RootDir
try {
    $ExactReport = Join-Path $OutDir "c5x_exact_finite_claim_certificates.json"
    $AxReport = Join-Path $OutDir "c5ax_reflection_retained_density.md"
    $AyReport = Join-Path $OutDir "c5ay_reflection_covariant_retained_coordinates.md"
    $AzReport = Join-Path $OutDir "c5az_shifted_coordinate_package_regression.md"

    Invoke-PythonReport "calculations\c5x_exact_finite_claim_certificates.py" $ExactReport
    Invoke-PythonReport "calculations\c5ax_reflection_retained_density.py" $AxReport
    Invoke-PythonReport "calculations\c5ay_reflection_covariant_retained_coordinates.py" $AyReport
    Invoke-PythonReport "calculations\c5az_shifted_coordinate_package_regression.py" $AzReport

    $Exact = Get-Content -LiteralPath $ExactReport -Raw | ConvertFrom-Json
    if (-not $Exact.certificate.all_exact_gates_pass) {
        throw "C5X exact finite certificate gate failed."
    }
    if ($Exact.certificate.rooted_path_word_sha256 -ne "1EB6E829D66DCDFA6E12E18CACA3C09A3EDFC3EA56E7EBE0D7ED1E05CA01616C") {
        throw "Rooted path-word hash changed."
    }

    Assert-ReportPattern $AxReport '\| False \| True \| False \|' "C5AX obstruction split"
    Assert-ReportPattern $AyReport '\| True \| False \| False \|' "C5AY shifted-core repair"
    Assert-ReportPattern $AzReport '\| True \| True \| False \| False \|' "C5AZ rooted correction decision"
    Assert-ReportPattern $AzReport 'rooted_operator_norm \| 1\.01896272181' "C5AZ rooted operator constant"
    Assert-ReportPattern $AzReport 'rooted_covariance_l2_squared \| 0\.214804676794' "C5AZ rooted covariance constant"

    $EntryScripts = @(
        "calculations\c5x_exact_finite_claim_certificates.py",
        "calculations\c5ax_reflection_retained_density.py",
        "calculations\c5ay_reflection_covariant_retained_coordinates.py",
        "calculations\c5az_shifted_coordinate_package_regression.py"
    )
    $IncludedFiles = @(
        "calculations\c5x_reproducibility_manifest.py",
        "calculations\run_c5x_paper_reproducibility.ps1",
        "papers\c5x-preliminary-manuscript.md",
        "papers\c5x-reproducibility-package.md",
        "papers\c5x-literature-audit-plan.md",
        "papers\c5x-verification-control-protocol.md",
        "outputs\c5x-reproducibility-rerun\c5x_exact_finite_claim_certificates.json",
        "outputs\c5x-reproducibility-rerun\c5ax_reflection_retained_density.md",
        "outputs\c5x-reproducibility-rerun\c5ay_reflection_covariant_retained_coordinates.md",
        "outputs\c5x-reproducibility-rerun\c5az_shifted_coordinate_package_regression.md"
    )

    if ($IncludeDownstream) {
        $DownstreamScripts = @(
            "calculations\c5ba_shifted_normalized_shell_contract.py",
            "calculations\c5bb_shifted_good_sector_expansion.py",
            "calculations\c5bc_shifted_disintegration_rn_expansion.py",
            "calculations\c5bd_analytic_shifted_rn_coefficients.py",
            "calculations\c5be_compact_conditional_centering.py",
            "calculations\c5bf_deterministic_fiber_parity_bound.py",
            "calculations\c5bg_uniform_chart_margin_rge3.py",
            "calculations\c5bh_sharpen_chart_or_delayed_start.py",
            "calculations\c5bi_compact_remainder_locality.py"
        )
        foreach ($Script in $DownstreamScripts) {
            $Stem = [System.IO.Path]::GetFileNameWithoutExtension($Script)
            $Report = Join-Path $OutDir ($Stem + ".json")
            Invoke-PythonReport $Script $Report
            $EntryScripts += $Script
            $IncludedFiles += "outputs\c5x-reproducibility-rerun\$Stem.json"
        }

        $Ba = Get-Content -LiteralPath (Join-Path $OutDir "c5ba_shifted_normalized_shell_contract.json") -Raw | ConvertFrom-Json
        $Bd = Get-Content -LiteralPath (Join-Path $OutDir "c5bd_analytic_shifted_rn_coefficients.json") -Raw | ConvertFrom-Json
        $Bh = Get-Content -LiteralPath (Join-Path $OutDir "c5bh_sharpen_chart_or_delayed_start.json") -Raw | ConvertFrom-Json
        $Bi = Get-Content -LiteralPath (Join-Path $OutDir "c5bi_compact_remainder_locality.json") -Raw | ConvertFrom-Json
        if ([Math]::Abs($Ba.constants.rooted_cov_sq - 0.21480467679414722) -gt 1.0e-12) {
            throw "C5BA did not use the rooted covariance constant."
        }
        if (-not $Bd.decision.analytic_rooted_source_formulas_implemented) {
            throw "C5BD rooted source formula gate failed."
        }
        if ($Bd.report.rooted_path_word_sha256 -ne $Exact.certificate.rooted_path_word_sha256) {
            throw "C5BD used a different rooted path package."
        }
        if ($Bh.decision.current_j0_truncated_chart_margin_closes) {
            throw "C5BH unexpectedly reports current-J0 closure after rooted correction."
        }
        if ($Bi.decision.sampled_compact_remainder_compatible_with_c5bh_room) {
            throw "C5BI unexpectedly reports compatibility with zero current-J0 room."
        }
        if ($Bi.decision.full_uniform_compact_remainder_theorem_proved) {
            throw "C5BI must not claim a uniform compact-remainder theorem."
        }
    }

    $ManifestArgs = @(
        "calculations\c5x_reproducibility_manifest.py",
        "--root", ".",
        "--output-json", "outputs\c5x-reproducibility-rerun\manifest.json",
        "--output-text", "outputs\c5x-reproducibility-rerun\sha256_manifest.txt"
    )
    foreach ($Entry in $EntryScripts) {
        $ManifestArgs += @("--entry", $Entry)
    }
    foreach ($Included in $IncludedFiles) {
        $ManifestArgs += @("--include", $Included)
    }
    & $PythonExe -B @ManifestArgs *> (Join-Path $OutDir "manifest_build.log")
    $ManifestExitCode = $LASTEXITCODE
    if ($ManifestExitCode -ne 0) {
        throw "Manifest construction failed with exit code $ManifestExitCode."
    }

    $RunSummary = [ordered]@{
        schema = "c5x-reproducibility-run-v2"
        completed_utc = [DateTime]::UtcNow.ToString("o")
        include_downstream = [bool]$IncludeDownstream
        exact_certificate_pass = $true
        c5ax_expected_obstruction_split = $true
        c5ay_shifted_core_repair_pass = $true
        c5az_rooted_recomputation_pass = $true
        previous_shifted_core_constants_preserved = $false
        c5bj_executed = $false
        project_advanced = $false
    }
    $RunSummary | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $OutDir "run_summary.json")
}
finally {
    Pop-Location
}

Write-Host "C5X reproducibility rerun completed."
Write-Host "Python used: $PythonExe"
Write-Host "Downstream correction chain included: $([bool]$IncludeDownstream)"
Write-Host "C5BJ executed: False"
Write-Host "Outputs written to: $OutDir"
