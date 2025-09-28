#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flatten Markdown nesting and optionally remove file paths from text.

Usage:
    python scripts/flatten_md.py input.md output.md [--max-depth N] [--convert-to-paragraph]
        [--remove-paths] [--stub-text TEXT] [--dry-run]

Features:
- Limits nested blockquotes and list indentation (default max depth = 4).
- Skips fenced code blocks (```/~~~), HTML comments/blocks and LaTeX \\begin/\\end environments.
- Optionally removes detected file paths (Windows, UNC, POSIX, relative) from non-code text.
- --stub-text lets you replace removed paths with placeholder text (default: remove).
- --dry-run prints stats and sample changes without writing output file.
"""
from __future__ import annotations
import re
import sys
from typing import List, Tuple

# Regular expressions
RE_FENCE = re.compile(r'^(?P<fence>```|~~~)')
RE_BLOCKQUOTE = re.compile(r'^(?P<prefix>(>\s*)+)(?P<rest>.*)$')
RE_LIST = re.compile(r'^(?P<indent>\s*)(?P<marker>(?:[-+*]|\d+\.)\s+)(?P<content>.*)$')
RE_LATEX_BEGIN = re.compile(r'\\begin\{[^}]+\}')
RE_LATEX_END = re.compile(r'\\end\{[^}]+\}')
RE_HTML_OPEN = re.compile(r'^\s*<([a-zA-Z]+)(\s|>|$)')
RE_HTML_COMMENT_OPEN = re.compile(r'^\s*<!--')
RE_HTML_COMMENT_CLOSE = re.compile(r'-->')

# Path detection regexes (ordered: Windows drive, UNC, POSIX/relative)
RE_WIN_DRIVE = re.compile(r'\b[A-Za-z]:\\[^\s`<>{}\]\)]*')
RE_UNC = re.compile(r'\\\\[^\s`<>{}\]\)]+')
RE_POSIX = re.compile(r'(?:(?:\./|\.\./|/)[^\s`<>{}\]\)]*)')

PATH_PATTERNS = [RE_WIN_DRIVE, RE_UNC, RE_POSIX]

DEFAULT_MAX_DEPTH = 4
INDENT_PER_LEVEL = 4  # canonical indent level


def _count_blockquote_depth(prefix: str) -> int:
    return prefix.count('>')


def _normalize_indent(s: str) -> int:
    s = s.replace('\t', ' ' * INDENT_PER_LEVEL)
    return len(s) - len(s.lstrip(' '))


def _remove_paths_from_text(s: str, stub_text: str = '') -> Tuple[str, int]:
    """
    Remove/replace path-like substrings from s using PATH_PATTERNS.
    Returns (new_string, number_of_replacements).
    """
    total = 0
    new = s
    for pat in PATH_PATTERNS:
        new, n = pat.subn(stub_text, new)
        total += n
    if total > 0:
        # collapse multiple spaces created by removals
        new = re.sub(r'\s{2,}', ' ', new)
        # strip space before punctuation if left awkwardly, e.g., " ("
        new = re.sub(r'\s+([,.;:)\]])', r'\1', new)
    return new, total


def flatten_lines(
    lines: List[str],
    max_depth: int = DEFAULT_MAX_DEPTH,
    convert_to_paragraph: bool = False,
    remove_paths: bool = False,
    stub_text: str = '',
    dry_run: bool = False,
) -> Tuple[List[str], int]:
    """
    Process lines and limit blockquote/list nesting to max_depth.
    If remove_paths is True, remove detected file paths in non-code regions.
    Returns (new_lines, total_path_replacements)
    """
    out: List[str] = []
    in_fence = False
    fence_marker = ''
    in_html_comment = False
    in_latex_env = False

    total_replacements = 0
    sample_changes: List[Tuple[int, str, str]] = []

    for idx, raw in enumerate(lines, start=1):
        line = raw.rstrip('\n')
        newline = '\n'

        # Fence detection
        m_fence = RE_FENCE.match(line.strip())
        if m_fence:
            fence = m_fence.group('fence')
            if not in_fence:
                in_fence = True
                fence_marker = fence
                out.append(line + newline)
                continue
            else:
                if fence == fence_marker:
                    in_fence = False
                    fence_marker = ''
                out.append(line + newline)
                continue

        if in_fence:
            out.append(line + newline)
            continue

        # HTML comment blocks
        if in_html_comment:
            out.append(line + newline)
            if RE_HTML_COMMENT_CLOSE.search(line):
                in_html_comment = False
            continue
        if RE_HTML_COMMENT_OPEN.match(line):
            in_html_comment = True
            out.append(line + newline)
            if RE_HTML_COMMENT_CLOSE.search(line):
                in_html_comment = False
            continue

        # LaTeX env detection
        if in_latex_env:
            out.append(line + newline)
            if RE_LATEX_END.search(line):
                in_latex_env = False
            continue
        if RE_LATEX_BEGIN.search(line):
            in_latex_env = True
            out.append(line + newline)
            continue

        original_line = line

        # Handle blockquote lines
        m_bq = RE_BLOCKQUOTE.match(line)
        if m_bq:
            prefix = m_bq.group('prefix') or ''
            rest = m_bq.group('rest') or ''
            bq_depth = _count_blockquote_depth(prefix)
            if bq_depth > max_depth:
                reduced_bq = ('> ' * max_depth).rstrip()
                m_list_after_bq = RE_LIST.match(rest)
                if m_list_after_bq:
                    indent = _normalize_indent(m_list_after_bq.group('indent'))
                    list_marker = m_list_after_bq.group('marker')
                    content = m_list_after_bq.group('content')
                    list_depth = indent // INDENT_PER_LEVEL
                    if list_depth > max_depth:
                        if convert_to_paragraph:
                            new_line = f"{reduced_bq} {content} (flattened)\n"
                        else:
                            new_indent = ' ' * (INDENT_PER_LEVEL * max_depth)
                            new_line = f"{reduced_bq} {new_indent}{list_marker}{content} (flattened)\n"
                    else:
                        new_line = f"{reduced_bq} {rest}\n"
                else:
                    new_line = f"{reduced_bq} {rest} (flattened quote)\n"
                line = new_line.rstrip('\n')
            else:
                m_list_after_bq = RE_LIST.match(rest)
                if m_list_after_bq:
                    indent = _normalize_indent(m_list_after_bq.group('indent'))
                    list_marker = m_list_after_bq.group('marker')
                    content = m_list_after_bq.group('content')
                    list_depth = indent // INDENT_PER_LEVEL
                    if list_depth > max_depth:
                        if convert_to_paragraph:
                            new_line = f"{prefix.rstrip()} {content} (flattened)\n"
                        else:
                            new_indent = ' ' * (INDENT_PER_LEVEL * max_depth)
                            new_line = f"{prefix}{new_indent}{list_marker}{content} (flattened)\n"
                        line = new_line.rstrip('\n')
                    else:
                        line = original_line
                else:
                    line = original_line
        else:
            # Non-blockquote: lists or normal lines
            m_list = RE_LIST.match(line)
            if m_list:
                indent = _normalize_indent(m_list.group('indent'))
                marker = m_list.group('marker')
                content = m_list.group('content')
                depth = indent // INDENT_PER_LEVEL
                if depth > max_depth:
                    if convert_to_paragraph:
                        line = f"{content} (flattened)"
                    else:
                        new_indent = ' ' * (INDENT_PER_LEVEL * max_depth)
                        line = f"{new_indent}{marker}{content} (flattened)"
                else:
                    line = original_line
            else:
                line = original_line

        # After structural flattening, optionally remove paths if this line is safe (not in code/latex/html)
        if remove_paths:
            new_line, n = _remove_paths_from_text(line, stub_text=stub_text)
            if n:
                total_replacements += n
                if len(sample_changes) < 20:
                    sample_changes.append((idx, line, new_line))
            line = new_line

        out.append(line + newline)

    if dry_run:
        # print summary to stdout for inspection
        print(f"Dry run: total path-like replacements: {total_replacements}", file=sys.stderr)
        if sample_changes:
            print("Sample changes (line, before -> after):", file=sys.stderr)
            for ln, before, after in sample_changes[:10]:
                print(f"  {ln}: {before!r} -> {after!r}", file=sys.stderr)

    return out, total_replacements


def _parse_args(argv: List[str]) -> dict:
    import argparse
    p = argparse.ArgumentParser(description='Flatten nested markdown and optionally remove paths.')
    p.add_argument('input', help='Input markdown file')
    p.add_argument('output', help='Output markdown file')
    p.add_argument('--max-depth', type=int, default=DEFAULT_MAX_DEPTH, help='Maximum nesting depth to allow (default 4)')
    p.add_argument('--convert-to-paragraph', action='store_true', help='Convert overflow list items to plain paragraphs instead of keeping markers')
    p.add_argument('--remove-paths', action='store_true', help='Remove detected file paths from non-code markdown text')
    p.add_argument('--stub-text', type=str, default='', help='Replacement text for removed paths (default: empty/remove)')
    p.add_argument('--dry-run', action='store_true', help='Do not write file; print replacement stats and samples to stderr')
    return vars(p.parse_args(argv[1:]))


def main(argv: List[str]) -> int:
    args = _parse_args(argv)
    input_file = args['input']
    output_file = args['output']
    max_depth = args['max_depth']
    convert = args['convert_to_paragraph']
    remove_paths = args['remove_paths']
    stub_text = args['stub_text']
    dry_run = args['dry_run']

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    flattened, total_replacements = flatten_lines(
        lines,
        max_depth=max_depth,
        convert_to_paragraph=convert,
        remove_paths=remove_paths,
        stub_text=stub_text,
        dry_run=dry_run,
    )

    if dry_run:
        # don't write output when dry-run
        return 0

    # write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(flattened)

    # print a brief report
    if remove_paths:
        print(f"Removed/replaced {total_replacements} path-like occurrences.", file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
