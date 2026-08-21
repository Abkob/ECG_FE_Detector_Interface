[CmdletBinding()]
param(
    [string[]]$Records,
    [switch]$IncludeAccelerometer
)

$ErrorActionPreference = "Stop"
$ButqVersion = "1.0.0"
$ButqBaseUrl = "https://physionet.org/files/butqdb/$ButqVersion"
$ButqDestination = Join-Path $PSScriptRoot "Datasets\butqdb-$ButqVersion"
$ButqAvailableRecords = @(
    "100001", "100002", "103001", "103002", "103003", "104001",
    "105001", "111001", "113001", "114001", "115001", "118001",
    "121001", "122001", "123001", "124001", "125001", "126001"
)

function Show-DownloadMenu {
    Write-Host ""
    Write-Host "BUT QDB selective downloader" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "1. Download 105001 only (recommended smallest pilot; about 265 MB ECG)"
    Write-Host "2. Download 100001, 105001, and 111001 (all fully annotated; about 605 MB ECG)"
    Write-Host "3. Enter custom record IDs"
    Write-Host "Q. Quit"
    Write-Host ""

    $Selection = (Read-Host "Choose 1, 2, 3, or Q").Trim().ToUpperInvariant()
    switch ($Selection) {
        "1" { return @("105001") }
        "2" { return @("100001", "105001", "111001") }
        "3" {
            $Entered = Read-Host "Enter six-digit IDs separated by spaces or commas"
            return @($Entered -split "[,\s]+" | Where-Object { $_ })
        }
        "Q" { return @() }
        default { throw "Unknown menu selection '$Selection'." }
    }
}

function Test-RequestedRecords {
    param([string[]]$RequestedRecords)

    $Normalized = @(
        $RequestedRecords |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ } |
            Select-Object -Unique
    )
    if ($Normalized.Count -eq 0) {
        return @()
    }
    foreach ($Record in $Normalized) {
        if ($Record -notmatch "^\d{6}$") {
            throw "Invalid record ID '$Record'. Record IDs must contain exactly six digits."
        }
        if ($Record -notin $ButqAvailableRecords) {
            throw "Record '$Record' is not present in BUT QDB v$ButqVersion."
        }
    }
    return $Normalized
}

function Get-ManifestHashes {
    param([string]$ManifestPath)

    $Hashes = @{}
    foreach ($Line in Get-Content -LiteralPath $ManifestPath) {
        if ($Line -match "^([0-9a-fA-F]{64})\s+(.+)$") {
            $Hashes[$Matches[2].Replace("/", "\")] = $Matches[1].ToLowerInvariant()
        }
    }
    return $Hashes
}

function Test-ExpectedHash {
    param(
        [string]$Path,
        [string]$ExpectedHash
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    $ActualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    return $ActualHash -eq $ExpectedHash
}

function Invoke-ReliableCurlTransfer {
    param(
        [string]$Url,
        [string]$OutputPath,
        [string]$DisplayName,
        [switch]$Resume,
        [int]$MaximumAttempts = 10
    )

    for ($Attempt = 1; $Attempt -le $MaximumAttempts; $Attempt++) {
        $ExistingBytes = 0
        if (Test-Path -LiteralPath $OutputPath -PathType Leaf) {
            $ExistingBytes = (Get-Item -LiteralPath $OutputPath).Length
        }

        if ($Resume -and $ExistingBytes -gt 0) {
            Write-Host ("Transfer attempt {0}/{1}; resuming {2} from {3:N0} bytes..." -f $Attempt, $MaximumAttempts, $DisplayName, $ExistingBytes) -ForegroundColor Cyan
        }
        elseif ($Attempt -gt 1) {
            Write-Host ("Transfer attempt {0}/{1} for {2}..." -f $Attempt, $MaximumAttempts, $DisplayName) -ForegroundColor Cyan
        }

        $CurlArguments = @(
            "-L",
            "--fail",
            "--http1.1",
            "--connect-timeout", "30",
            "--retry", "3",
            "--retry-all-errors",
            "--retry-delay", "3",
            "--retry-max-time", "180"
        )
        if ($Resume) {
            $CurlArguments += @("-C", "-")
        }
        $CurlArguments += @("--output", $OutputPath, $Url)

        & curl.exe @CurlArguments
        $CurlExitCode = $LASTEXITCODE
        if ($CurlExitCode -eq 0) {
            return
        }

        if ($Attempt -lt $MaximumAttempts) {
            $PauseSeconds = [Math]::Min(30, 5 * $Attempt)
            Write-Warning "curl exit code $CurlExitCode while downloading '$DisplayName'. The partial file is safe; retrying in $PauseSeconds seconds."
            Start-Sleep -Seconds $PauseSeconds
        }
    }

    throw "curl failed for '$DisplayName' after $MaximumAttempts transfer attempts. The .part file was kept for resuming."
}

function Invoke-ButqDownload {
    param(
        [string]$RelativePath,
        [hashtable]$ManifestHashes,
        [switch]$SkipManifestVerification
    )

    $NormalizedRelativePath = $RelativePath.Replace("/", "\")
    $FinalPath = Join-Path $ButqDestination $NormalizedRelativePath
    $FinalFolder = Split-Path -Parent $FinalPath
    New-Item -ItemType Directory -Force -Path $FinalFolder | Out-Null

    $ExpectedHash = $null
    if (-not $SkipManifestVerification) {
        if (-not $ManifestHashes.ContainsKey($NormalizedRelativePath)) {
            throw "No SHA-256 entry exists for '$RelativePath'."
        }
        $ExpectedHash = $ManifestHashes[$NormalizedRelativePath]
        if (Test-ExpectedHash -Path $FinalPath -ExpectedHash $ExpectedHash) {
            Write-Host "Verified; already present: $RelativePath" -ForegroundColor DarkGreen
            return
        }
    }

    if (Test-Path -LiteralPath $FinalPath -PathType Leaf) {
        $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $BackupPath = "$FinalPath.corrupt_$Timestamp"
        Move-Item -LiteralPath $FinalPath -Destination $BackupPath
        Write-Warning "Existing file failed verification and was preserved as '$BackupPath'."
    }

    $PartialPath = "$FinalPath.part"
    $UrlPath = $RelativePath.Replace("\", "/")
    $Url = "$ButqBaseUrl/$UrlPath"
    Write-Host "Downloading: $RelativePath" -ForegroundColor Yellow
    Invoke-ReliableCurlTransfer -Url $Url -OutputPath $PartialPath -DisplayName $RelativePath -Resume

    if (-not $SkipManifestVerification) {
        if (-not (Test-ExpectedHash -Path $PartialPath -ExpectedHash $ExpectedHash)) {
            $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
            $BadPartialPath = "$PartialPath.bad_$Timestamp"
            Move-Item -LiteralPath $PartialPath -Destination $BadPartialPath
            Write-Warning "The resumed file failed SHA-256 and was preserved as '$BadPartialPath'. Retrying once from the beginning."
            Invoke-ReliableCurlTransfer -Url $Url -OutputPath $PartialPath -DisplayName "$RelativePath (clean retry)"
            if (-not (Test-ExpectedHash -Path $PartialPath -ExpectedHash $ExpectedHash)) {
                throw "SHA-256 verification still failed for '$RelativePath' after a clean retry."
            }
        }
    }
    Move-Item -LiteralPath $PartialPath -Destination $FinalPath -Force
    Write-Host "Saved and verified: $FinalPath" -ForegroundColor Green
}

try {
    if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
        throw "curl.exe is unavailable. It is normally included with Windows 10 and 11."
    }

    if (-not $Records -or $Records.Count -eq 0) {
        $Records = Show-DownloadMenu
        if ($Records.Count -eq 0) {
            Write-Host "No records selected. Nothing was downloaded."
            exit 0
        }
        $AccAnswer = (Read-Host "Also download 3-axis ACC data? This is optional and larger [y/N]").Trim()
        $IncludeAccelerometer = $AccAnswer -match "^[Yy]"
    }
    $Records = Test-RequestedRecords -RequestedRecords $Records
    if ($Records.Count -eq 0) {
        Write-Host "No records selected. Nothing was downloaded."
        exit 0
    }

    New-Item -ItemType Directory -Force -Path $ButqDestination | Out-Null
    $ManifestPath = Join-Path $ButqDestination "SHA256SUMS.txt"
    Write-Host "Downloading the official checksum manifest..." -ForegroundColor Yellow
    Invoke-ReliableCurlTransfer -Url "$ButqBaseUrl/SHA256SUMS.txt" -OutputPath $ManifestPath -DisplayName "SHA256SUMS.txt"
    $ManifestHashes = Get-ManifestHashes -ManifestPath $ManifestPath

    foreach ($MetadataFile in @("LICENSE.txt", "RECORDS", "ANNOTATORS", "subject-info.csv")) {
        Invoke-ButqDownload -RelativePath $MetadataFile -ManifestHashes $ManifestHashes
    }

    foreach ($Record in $Records) {
        $RequestedFiles = @(
            "$Record/${Record}_ECG.hea",
            "$Record/${Record}_ECG.dat",
            "$Record/${Record}_ANN.csv"
        )
        if ($IncludeAccelerometer) {
            $RequestedFiles += @(
                "$Record/${Record}_ACC.hea",
                "$Record/${Record}_ACC.dat"
            )
        }
        foreach ($RelativePath in $RequestedFiles) {
            Invoke-ButqDownload -RelativePath $RelativePath -ManifestHashes $ManifestHashes
        }
    }

    Write-Host ""
    Write-Host "Selected records downloaded and SHA-256 verified:" -ForegroundColor Cyan
    $Records | ForEach-Object { Write-Host "  - $_" }
    Write-Host "Destination: $ButqDestination"
    Write-Host "ACC included: $([bool]$IncludeAccelerometer)"
    exit 0
}
catch {
    Write-Error $_
    exit 1
}
