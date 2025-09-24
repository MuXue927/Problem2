# fix_docs_move_and_index.ps1
# Reliable script to move PDFs from ALNSCode/docs to ALNSCode/docs/pdfs and generate INDEX.md

$root = Get-Location
$docs = Join-Path $root 'ALNSCode\docs'
$pdfDir = Join-Path $docs 'pdfs'
if (-not (Test-Path $docs)) { Write-Output "Docs folder not found: $docs"; exit 1 }
if (-not (Test-Path $pdfDir)) { New-Item -ItemType Directory -Path $pdfDir | Out-Null }

Write-Output "Scanning $docs"

# Move PDFs (use -Filter for reliability)
$pdfs = Get-ChildItem -Path (Join-Path $docs '*') -Filter '*.pdf' -File -ErrorAction SilentlyContinue
foreach ($p in $pdfs) {
    $dst = Join-Path $pdfDir $p.Name
    try { Move-Item -LiteralPath $p.FullName -Destination $dst -Force; Write-Output ("Moved PDF: {0}" -f $p.Name) } catch { Write-Output ("Failed to move {0}: {1}" -f $p.FullName, $_.Exception.Message) }
}

# Find markdown files directly under docs (not in pdfs)
$mds = Get-ChildItem -Path (Join-Path $docs '*') -Filter '*.md' -File -ErrorAction SilentlyContinue | Where-Object { $_.DirectoryName -eq $docs }

$index = New-Object System.Collections.Generic.List[string]
$index.Add('# Project documentation index')
$index.Add('')
$index.Add('This index lists the documentation files under `ALNSCode/docs`.')
$index.Add('')
$index.Add('## Documents')
$index.Add('')

foreach ($f in $mds | Sort-Object Name) {
    $title = $f.BaseName
    try {
        $lines = Get-Content -Path $f.FullName -TotalCount 20 -ErrorAction Stop
        foreach ($l in $lines) {
            $t = $l.Trim()
            if ($t -match '^#\s*(.+)') { $title = $Matches[1]; break }
            if ($t -ne '') { $title = $t; break }
        }
    } catch { }
    $rel = './' + $f.Name
    $index.Add(("- [{0}]({1}) — {2}" -f $f.Name, $rel, $title))
}

$index.Add('')
$index.Add('## PDFs')
$index.Add('')
$pdfsAfter = Get-ChildItem -Path $pdfDir -Filter '*.pdf' -File -ErrorAction SilentlyContinue | Sort-Object Name
foreach ($p in $pdfsAfter) { $index.Add(("- [{0}]({1})" -f $p.Name, ('./pdfs/' + $p.Name))) }

$indexPath = Join-Path $docs 'INDEX.md'
$index | Out-File -FilePath $indexPath -Encoding UTF8 -Force
Write-Output "Wrote INDEX.md with $($mds.Count) markdown entries and $($pdfsAfter.Count) pdfs"

# Stage and commit changes
& git add -A
try {
    & git commit -m "chore(docs): move PDFs to docs/pdfs and generate docs/INDEX.md"
} catch {
    Write-Output 'Nothing to commit or commit failed'
}
& git status --porcelain -b

Write-Output 'Done.'
