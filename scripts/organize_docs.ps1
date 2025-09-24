# organize_docs.ps1
# Robustly remove empty markdown subfolders, move remaining md/pdf files under ALNSCode/docs,
# and rename unclear filenames to more descriptive ones.

$root = Get-Location
$alnsDir = Join-Path $root 'ALNSCode'
$docsDir = Join-Path $alnsDir 'docs'
if (-not (Test-Path $docsDir)) { New-Item -ItemType Directory -Path $docsDir | Out-Null }

Write-Output "Root: $root"

# Remove empty subdirectories under ALNSCode\markdown
$mdRoot = Join-Path $alnsDir 'markdown'
if (Test-Path $mdRoot) {
    $subdirs = Get-ChildItem -Path $mdRoot -Directory -Force -ErrorAction SilentlyContinue
    foreach ($d in $subdirs) {
        $files = Get-ChildItem -Path $d.FullName -Recurse -Force -ErrorAction SilentlyContinue
        if (-not $files) {
            try { Remove-Item -LiteralPath $d.FullName -Recurse -Force; Write-Output ("Removed empty folder: {0}" -f $d.FullName) } catch { Write-Output ("Failed to remove {0}: {1}" -f $d.FullName, $_.Exception.Message) }
        } else { Write-Output ("Skipping non-empty folder: {0}" -f $d.FullName) }
    }
    # If markdown root is now empty, remove it too
    $remaining = Get-ChildItem -Path $mdRoot -Force -ErrorAction SilentlyContinue
    if (-not $remaining) {
        try { Remove-Item -LiteralPath $mdRoot -Recurse -Force; Write-Output ("Removed empty folder: {0}" -f $mdRoot) } catch { Write-Output ("Failed to remove {0}: {1}" -f $mdRoot, $_.Exception.Message) }
    } else { Write-Output "ALNSCode/markdown still contains items or non-empty; not removed." }
} else { Write-Output "No ALNSCode/markdown folder found." }

# Move md/markdown/pdf from ALNSCode root into docs (except those already in docs)
Get-ChildItem -Path $alnsDir -File -Include '*.md','*.markdown','*.pdf' -ErrorAction SilentlyContinue | ForEach-Object {
    $src = $_.FullName
    $dst = Join-Path $docsDir $_.Name
    if ($src -eq $dst) { return }
    try { Move-Item -LiteralPath $src -Destination $dst -Force; Write-Output ("Moved {0} -> ALNSCode/docs/" -f $_.Name) } catch { Write-Output ("Failed to move {0}: {1}" -f $src, $_.Exception.Message) }
}

# Some files in docs have unclear names; rename the most unclear ones to descriptive names only if they exist
$renameMap = @{
    '________.md' = 'ALNS_Performance_Review.md'
    '___________.md' = 'ALNS_Operators_and_ML_Repair.md'
    'ALNS_______o4mini.md' = 'ALNS_o4mini.md'
    'ALNSTracker_____.md' = 'ALNSTracker.md'
    'DemandConstraintAccept_____.md' = 'DemandConstraintAccept.md'
    '_____OutPutData_check_solution.md' = 'OutputData_and_check_solution.md'
}

foreach ($k in $renameMap.Keys) {
    $src = Join-Path $docsDir $k
        if (Test-Path $src) {
        $dest = Join-Path $docsDir $renameMap[$k]
        try { Rename-Item -LiteralPath $src -NewName $renameMap[$k] -Force; Write-Output ("Renamed {0} -> {1}" -f $k, $renameMap[$k]) } catch { Write-Output ("Failed to rename {0}: {1}" -f $k, $_.Exception.Message) }
    }
}

# Show final docs directory listing
Write-Output "Final ALNSCode/docs contents:"
Get-ChildItem -Path $docsDir -File -Force | ForEach-Object { Write-Output " - $($_.Name)" }

# Stage changes and show git status; do not auto-commit here (we will commit after README merge)
& git add -A
Write-Output "Staged changes for git. Current status:" 
& git status --porcelain -b

Write-Output "organize_docs.ps1 finished."