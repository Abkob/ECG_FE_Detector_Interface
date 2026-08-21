$ErrorActionPreference = 'Stop'

$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$archiveRoot = Join-Path $workspaceRoot 'archive\pre_cleanup_2026-08-19'
$sourceRoot = Join-Path $workspaceRoot 'source_materials'

function Assert-WorkspacePath {
    param([Parameter(Mandatory = $true)][string]$Path)
    $absolute = [System.IO.Path]::GetFullPath($Path)
    $prefix = $workspaceRoot.TrimEnd('\') + '\'
    if (-not $absolute.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to operate outside workspace: $absolute"
    }
    return $absolute
}

function Move-Safely {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$DestinationDirectory,
        [string]$DestinationName
    )

    $sourcePath = Assert-WorkspacePath $Source
    $destinationPath = Assert-WorkspacePath $DestinationDirectory
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        return
    }

    New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null
    if ([string]::IsNullOrWhiteSpace($DestinationName)) {
        $DestinationName = Split-Path $sourcePath -Leaf
    }
    $finalPath = Assert-WorkspacePath (Join-Path $destinationPath $DestinationName)
    if (Test-Path -LiteralPath $finalPath) {
        throw "Archive destination already exists: $finalPath"
    }
    $item = Get-Item -Force -LiteralPath $sourcePath
    if ($item.PSIsContainer) {
        # Directory.Move performs a same-volume rename and does not enumerate
        # nested hidden repositories or read-only files.
        [System.IO.Directory]::Move($sourcePath, $finalPath)
    }
    else {
        [System.IO.File]::Move($sourcePath, $finalPath)
    }
}

New-Item -ItemType Directory -Path $archiveRoot -Force | Out-Null
New-Item -ItemType Directory -Path $sourceRoot -Force | Out-Null

$inventoryPath = Join-Path $archiveRoot 'inventory_before_cleanup.csv'
if (-not (Test-Path -LiteralPath $inventoryPath)) {
    $inventory = Get-ChildItem -LiteralPath $workspaceRoot -Recurse -Force -File |
        Where-Object { $_.FullName -notmatch '[\\/]\.git[\\/]' -and $_.FullName -notlike "$archiveRoot*" } |
        ForEach-Object {
            $hash = ''
            if ($_.Extension -in '.pdf', '.tex') {
                $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            }
            [pscustomobject]@{
                relative_path = $_.FullName.Substring($workspaceRoot.Length + 1)
                extension = $_.Extension.ToLowerInvariant()
                size_bytes = $_.Length
                modified_utc = $_.LastWriteTimeUtc.ToString('o')
                sha256 = $hash
            }
        }
    $inventory | Export-Csv -LiteralPath $inventoryPath -NoTypeInformation -Encoding utf8
}

# Preserve the complete previous output tree before creating the new bundle.
Move-Safely -Source (Join-Path $workspaceRoot 'output') -DestinationDirectory $archiveRoot -DestinationName 'output_legacy'

$transientDirectories = @(
    '_docx_build',
    '_qa_signal_quality_review_v1',
    '_qa_signal_quality_review_v2',
    '_qa_signal_quality_review_v3',
    '_qa_signal_quality_review_v4',
    '_qa_signal_quality_review_v5',
    '_qa_signal_quality_review_v6',
    '_qa_signal_quality_review_v7',
    '_qa_signal_quality_review_v8',
    '.codex_doc_build',
    '.codex_tmp',
    '.codex-docx-work',
    '.codex-qa',
    '.docx_qa_20260723',
    '.docx_qa_20260723_v2',
    '.docx_qa_20260723_v3',
    '.docx_qa_20260723_v4',
    '.docx_review',
    '.pptx_build_combined',
    '.pptx_build_morphology',
    '.pytest_cache',
    '.agents'
    # Root tmp is handled separately below because a prior interrupted move may
    # have left a remainder alongside the archived partial tree.
)
foreach ($name in $transientDirectories) {
    Move-Safely -Source (Join-Path $workspaceRoot $name) -DestinationDirectory (Join-Path $archiveRoot 'transient')
}

$rootTmp = Join-Path $workspaceRoot 'tmp'
if (Test-Path -LiteralPath $rootTmp) {
    $tmpName = if (Test-Path -LiteralPath (Join-Path $archiveRoot 'transient\tmp')) { 'tmp_remainder' } else { 'tmp' }
    Move-Safely -Source $rootTmp -DestinationDirectory (Join-Path $archiveRoot 'transient') -DestinationName $tmpName
}

$featureTransient = @('.codex_pdf_preview', '.codex_pdf_preview_final', '.pytest_cache')
foreach ($name in $featureTransient) {
    Move-Safely -Source (Join-Path $workspaceRoot "feature_extraction\$name") -DestinationDirectory (Join-Path $archiveRoot 'feature_extraction_transient')
}

# Re-home source collections without changing the active code or raw datasets.
$referenceCollections = @('AnnotatedPapers', 'ECG_detector', 'EEGECG', 'Figures')
foreach ($name in $referenceCollections) {
    Move-Safely -Source (Join-Path $workspaceRoot $name) -DestinationDirectory (Join-Path $sourceRoot 'reference_collections')
}

$projectReports = Get-ChildItem -LiteralPath $workspaceRoot -Force -File |
    Where-Object { $_.Extension -in '.docx', '.pdf' -and $_.Name -notlike '~$*' }
foreach ($file in $projectReports) {
    Move-Safely -Source $file.FullName -DestinationDirectory (Join-Path $sourceRoot 'project_reports')
}

$rootBuilders = @('build_conduction_repolarization_review.py', 'build_signal_quality_review.py')
foreach ($name in $rootBuilders) {
    Move-Safely -Source (Join-Path $workspaceRoot $name) -DestinationDirectory (Join-Path $workspaceRoot 'tools\legacy_document_builders')
}

$lockFiles = Get-ChildItem -LiteralPath $workspaceRoot -Force -File |
    Where-Object { $_.Name -like '~$*' -or $_.Extension -eq '.lnk' }
foreach ($file in $lockFiles) {
    Move-Safely -Source $file.FullName -DestinationDirectory (Join-Path $archiveRoot 'root_misc')
}

Move-Safely -Source (Join-Path $workspaceRoot 'LaTeX_Upload_Bundle_2026-08-14') -DestinationDirectory (Join-Path $archiveRoot 'legacy_upload_bundle')
Move-Safely -Source (Join-Path $workspaceRoot 'LaTeX_Upload_Bundle_2026-08-14.zip') -DestinationDirectory (Join-Path $archiveRoot 'legacy_upload_bundle')

$outputDirectories = @(
    'output',
    'output\architecture',
    'output\branches\B1_peak_rr_hrv',
    'output\branches\B2_morphology',
    'output\branches\B3_conduction_repolarization',
    'output\branches\B4_signal_quality',
    'output\evidence\artifacts',
    'output\evidence\datasets',
    'output\evidence\case_studies',
    'output\evidence\literature',
    'output\evidence\unresolved',
    'output\evidence\quarantine',
    'output\evidence\indexes',
    'output\notebook_runs',
    'output\qa'
)
foreach ($relative in $outputDirectories) {
    $path = Assert-WorkspacePath (Join-Path $workspaceRoot $relative)
    New-Item -ItemType Directory -Path $path -Force | Out-Null
}

[pscustomobject]@{
    workspace = $workspaceRoot
    archive = $archiveRoot
    source_materials = $sourceRoot
    output = (Join-Path $workspaceRoot 'output')
    inventory = $inventoryPath
} | ConvertTo-Json -Depth 3
