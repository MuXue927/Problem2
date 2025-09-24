# generate_docs_index_and_separate_pdfs.ps1
# - Move PDF files from ALNSCode/docs into ALNSCode/docs/pdfs/
# - Generate ALNSCode/docs/INDEX.md listing markdown files (title + link) and PDFs

$root = Get-Location
$docs = Join-Path $root 'ALNSCode\docs'
$pdfDir = Join-Path $docs 'pdfs'
if (-not (Test-Path $docs)) { Write-Output "Docs folder not found: $docs"; exit 1 }
if (-not (Test-Path $pdfDir)) { New-Item -ItemType Directory -Path $pdfDir | Out-Null }

# Move PDFs into pdfs/
Get-ChildItem -Path $docs -File -Include '*.pdf' -ErrorAction SilentlyContinue | ForEach-Object {
    $src = $_.FullName
    $dst = Join-Path $pdfDir $_.Name
    try { Move-Item -LiteralPath $src -Destination $dst -Force; Write-Output ("Moved PDF: {0}" -f $_.Name) } catch { Write-Output ("Failed to move {0}: {1}" -f $_.FullName, $_.Exception.Message) }
}

# Gather markdown files and extract first heading
$mdFiles = Get-ChildItem -Path $docs -File -Include '*.md' -ErrorAction SilentlyContinue | Where-Object { $_.DirectoryName -eq $docs -and $_.Name -ne 'INDEX.md' }
$indexLines = @()
$indexLines += '# Project documentation index'
$indexLines += ''
$indexLines += 'This index lists the documentation files under `ALNSCode/docs`. Click a link to open the document.'
$indexLines += ''
$indexLines += '## Documents'
$indexLines += ''
foreach ($f in $mdFiles | Sort-Object Name) {
    $path = $f.FullName
    $title = $null
    try {
        $lines = Get-Content -Path $path -TotalCount 20 -ErrorAction Stop
        foreach ($l in $lines) {
            $trim = $l.Trim()
            if ($trim -like '#*') { # a markdown heading
                # remove leading # and trim
                $title = $trim -replace '^#+\s*',''
                break
            }
            if ($trim -ne '') { # fallback to first non-empty line
                if (-not $title) { $title = $trim; break }
            }
        }
    } catch { $title = $f.Name }
    if (-not $title) { $title = $f.BaseName }
    $rel = './' + $f.Name
    $indexLines += ("- [{0}]({1}) — {2}" -f $f.Name, $rel, $title)
}

$indexLines += ''
$indexLines += '## PDFs'
$indexLines += ''
# list pdfs in pdfs/
Get-ChildItem -Path $pdfDir -File -Include '*.pdf' -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object {
    $rel = './pdfs/' + $_.Name
    $indexLines += ("- [{0}]({1})" -f $_.Name, $rel)
}

$indexPath = Join-Path $docs 'INDEX.md'
$indexLines | Out-File -FilePath $indexPath -Encoding UTF8 -Force
Write-Output "Wrote INDEX.md with $($mdFiles.Count) markdown files and PDFs from $pdfDir"

# Stage results
& git add -A
Write-Output 'Staged changes. Current git status:'
& git status --porcelain -b

Write-Output 'generate_docs_index_and_separate_pdfs.ps1 done.'
