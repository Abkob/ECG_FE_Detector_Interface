[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$uploadDirectory = Join-Path $repositoryRoot 'output\latest_branch_uploads'
$latexmk = Get-Command latexmk -ErrorAction Stop

$documents = @(
    [pscustomobject]@{
        Name = 'Morphology branch'
        Tex = Join-Path $repositoryRoot 'feature_extraction\docs\latex_morphology_branch\morphology_branch_complete_technical_record.tex'
        Pdf = Join-Path $repositoryRoot 'feature_extraction\docs\latex_morphology_branch\morphology_branch_complete_technical_record.pdf'
    }
    [pscustomobject]@{
        Name = 'Architecture'
        Tex = Join-Path $repositoryRoot 'output\latex\Architecture.tex'
        Pdf = Join-Path $repositoryRoot 'output\latex\Architecture.pdf'
    }
    [pscustomobject]@{
        Name = 'HRV/RR branch'
        Tex = Join-Path $repositoryRoot 'output\pdf\RR_HRV_Cascade_RPeak_Audit.tex'
        Pdf = Join-Path $repositoryRoot 'output\pdf\RR_HRV_Cascade_RPeak_Audit.pdf'
    }
)

New-Item -ItemType Directory -Path $uploadDirectory -Force | Out-Null

foreach ($document in $documents) {
    if (-not (Test-Path -LiteralPath $document.Tex -PathType Leaf)) {
        throw "Missing LaTeX source for $($document.Name): $($document.Tex)"
    }

    & $latexmk.Source -norc -cd -pdf -interaction=nonstopmode -halt-on-error $document.Tex
    if ($LASTEXITCODE -ne 0) {
        throw "LaTeX compilation failed for $($document.Name)."
    }

    if (-not (Test-Path -LiteralPath $document.Pdf -PathType Leaf)) {
        throw "Compilation did not create the expected PDF for $($document.Name): $($document.Pdf)"
    }

    Copy-Item -LiteralPath $document.Tex -Destination $uploadDirectory -Force
    Copy-Item -LiteralPath $document.Pdf -Destination $uploadDirectory -Force
}

$result = Get-ChildItem -LiteralPath $uploadDirectory -File |
    Where-Object { $_.Extension -in '.tex', '.pdf' } |
    Sort-Object Name |
    Select-Object Name, Length, LastWriteTime

Write-Host "Upload folder refreshed: $uploadDirectory"
$result | Format-Table -AutoSize
