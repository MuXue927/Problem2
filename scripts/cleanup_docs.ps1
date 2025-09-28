# cleanup_docs.ps1
# Archives ALNSCode/markdown to ALNSCode/docs (sanitizes filenames)
# Moves ALNSCode/*.md/*.pdf to ALNSCode/docs
# Moves top-level .docx/.pdf/.xlsx to docs/
# Removes .log files and updates .gitignore

$root = Get-Location
$alnsDocs = Join-Path $root 'ALNSCode\docs'
$rootDocs = Join-Path $root 'docs'
if (-not (Test-Path $alnsDocs)) { New-Item -ItemType Directory -Path $alnsDocs | Out-Null }
if (-not (Test-Path $rootDocs)) { New-Item -ItemType Directory -Path $rootDocs | Out-Null }

function Sanitize-Name([string]$name) {
    # replace forbidden path chars and non-ascii with underscore
    $s = $name -replace '[\\/:*?"<>| ]', '_'
    $s = -join ($s.ToCharArray() | ForEach-Object { if ([int]$_ -lt 32 -or [int]$_ -gt 126) { '_' } else { $_ } })
    if ($s.Length -gt 120) {
        $ext = [IO.Path]::GetExtension($name)
        $s = $s.Substring(0, 120 - $ext.Length) + $ext
    }
    return $s
}

$moveCount = 0
if (Test-Path 'ALNSCode\markdown') {
    Get-ChildItem -Path 'ALNSCode\markdown' -Recurse -File | ForEach-Object {
        $orig = $_.FullName
        $safe = Sanitize-Name $_.Name
        $dst = Join-Path $alnsDocs $safe
        try { Move-Item -LiteralPath $orig -Destination $dst -Force; $moveCount++ } catch { Write-Output ("Failed to move {0}: {1}" -f $orig, $_) }
    }
}

$moveAlnsRoot = 0
Get-ChildItem -Path 'ALNSCode' -File -Include '*.md','*.markdown','*.pdf' -ErrorAction SilentlyContinue | ForEach-Object {
    $orig = $_.FullName
    $safe = Sanitize-Name $_.Name
    $dst = Join-Path $alnsDocs $safe
    try { Move-Item -LiteralPath $orig -Destination $dst -Force; $moveAlnsRoot++ } catch { Write-Output ("Failed to move {0}: {1}" -f $orig, $_) }
}

$moveRootDocs = 0
Get-ChildItem -Path . -File | Where-Object { $_.Extension -in '.docx','.pdf','.xlsx' } | ForEach-Object {
    $orig = $_.FullName
    $dst = Join-Path $rootDocs $_.Name
    try { Move-Item -LiteralPath $orig -Destination $dst -Force; $moveRootDocs++ } catch { Write-Output ("Failed to move {0}: {1}" -f $orig, $_) }
}

$logRem = 0
Get-ChildItem -Path . -Recurse -File -Include '*.log' -ErrorAction SilentlyContinue | ForEach-Object { try { Remove-Item -LiteralPath $_.FullName -Force; $logRem++ } catch { Write-Output ("Failed to remove log {0}: {1}" -f $_.FullName, $_) } }

$ignoreEntries = @(
  '# auto-generated ignores by cleanup script',
  'OutPut-ALNS/',
  'OutPut-CG/',
  'outputs/',
  'logs/',
  '*.log',
  '*.out',
  '*.html',
  '.idea/',
  '.vscode/',
  '*.iml',
  '.python-version',
  'uv.lock',
  'untracked_backup_*/'
)
$gi = @()
if (Test-Path '.gitignore') { $gi = Get-Content -Path '.gitignore' }
$toAdd = @()
foreach ($entry in $ignoreEntries) { if ($gi -notcontains $entry) { $toAdd += $entry } }
if ($toAdd.Count -gt 0) { Add-Content -Path .gitignore -Value "`n# cleanup additions`n" -Encoding UTF8; Add-Content -Path .gitignore -Value ($toAdd -join "`n") -Encoding UTF8 }

# Git add/commit
& git add -A
$staged = (& git diff --cached --name-only)
if ($staged) { & git commit -m "chore(docs): archive markdown/pdf/xlsx to docs, sanitize filenames, remove logs and update .gitignore" } else { Write-Output 'No staged changes to commit.' }

Write-Output "Summary: moved $moveCount files from ALNSCode/markdown, moved $moveAlnsRoot ALNSCode-root md/pdf, moved $moveRootDocs root docx/pdf/xlsx, removed $logRem .log files."
& git status --porcelain -b
