param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot 'target/package')
)

$ErrorActionPreference = 'Stop'

$repo = $PSScriptRoot
$binary = Join-Path $repo 'target/release/xai-grok-pager.exe'
$packageDir = Join-Path $OutputDirectory 'grok-windows-x64'
$archive = "$packageDir.zip"

if (-not $env:PROTOC) {
    $realProtoc = (Get-Command protoc.exe -ErrorAction Stop).Source
    $wrapper = Join-Path $env:TEMP 'pi-protoc-wrapper.exe'
    if (Test-Path $wrapper) {
        $env:REAL_PROTOC = $realProtoc
        $env:PROTOC = $wrapper
    }
    else {
        $env:PROTOC = $realProtoc
    }
}

$env:CARGO_HTTP_CHECK_REVOKE = 'false'
$env:AWS_LC_SYS_PREBUILT_NASM = '1'
if (-not $env:CARGO_BUILD_JOBS) {
    $env:CARGO_BUILD_JOBS = [Math]::Min([Environment]::ProcessorCount, 12).ToString()
}

Push-Location $repo
try {
    # link.exe cannot emit this binary's PDB on Windows (LNK1318 LIMIT),
    # so disable only the final executable's PDB without rebuilding dependencies.
    & cargo rustc -p xai-grok-pager-bin --release --bin xai-grok-pager -- -C 'link-arg=/DEBUG:NONE'
    if ($LASTEXITCODE -ne 0) { throw "cargo rustc failed with exit code $LASTEXITCODE" }

    Remove-Item $packageDir -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item $archive -Force -ErrorAction SilentlyContinue
    New-Item $packageDir -ItemType Directory -Force | Out-Null

    Copy-Item $binary (Join-Path $packageDir 'grok.exe')
    Copy-Item (Join-Path $repo 'LICENSE') $packageDir
    Copy-Item (Join-Path $repo 'THIRD-PARTY-NOTICES') $packageDir

    Compress-Archive -Path (Join-Path $packageDir '*') -DestinationPath $archive -CompressionLevel Optimal
    Remove-Item $packageDir -Recurse -Force

    Write-Host "Release package: $archive" -ForegroundColor Green
}
finally {
    Pop-Location
}
