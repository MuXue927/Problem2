# move_alnscode_root_docs.ps1
# Move any remaining .md/.markdown/.pdf in ALNSCode root into ALNSCode/docs and report results

$root = Get-Location
$alns = Join-Path $root 'ALNSCode'
$docs = Join-Path $alns 'docs'
if (-not (Test-Path $docs)) { New-Item -ItemType Directory -Path $docs | Out-Null }

Write-Output ("ALNSCode root: {0}" -f $alns)

$exts = @('.md', '.markdown', '.pdf')
$items = Get-ChildItem -Path $alns -File -Force -ErrorAction SilentlyContinue | Where-Object { $exts -contains $_.Extension }

if (-not $items -or $items.Count -eq 0) {
    Write-Output 'No md/markdown/pdf files found in ALNSCode root.'
    exit 0
}

Write-Output 'Found the following files in ALNSCode root:'
foreach ($f in $items) { Write-Output (" - {0}" -f $f.Name) }

foreach ($f in $items) {
    $src = $f.FullName
    $dst = Join-Path $docs $f.Name
    try {
        Move-Item -LiteralPath $src -Destination $dst -Force -ErrorAction Stop
        Write-Output ("Moved: {0} -> {1}" -f $f.Name, $dst)
    } catch {
        Write-Output ("Failed to move {0}: {1}" -f $src, $_.Exception.Message)
    }
}

Write-Output "\nRemaining items in ALNSCode root (post-move):"
Get-ChildItem -Path $alns -File -Force | Where-Object { $exts -contains $_.Extension } | ForEach-Object { Write-Output (" - {0}" -f $_.FullName) }

Write-Output "\nContents of ALNSCode/docs (post-move):"
Get-ChildItem -Path $docs -File -Force | ForEach-Object { Write-Output (" - {0}" -f $_.Name) }

# Stage and show git status
& git add -A
Write-Output "\nGit status (porcelain):"
& git status --porcelain -b

Write-Output '\nDone.'
