#!/usr/bin/env python3
"""Re-wrap a LaTeX source so its lines fit a maximum width without changing the PDF.

Lines are only broken where TeX reads the same tokens either way:
  * at a run of spaces (a line end reads as one space; leading spaces are skipped)
  * inside comments, continued with the same '%' marker
  * between two tokens with a trailing '%', when no space fits (after } ] , or before \\ { [)
No empty lines are created, so no paragraphs are added. Verbatim-like environments,
\\verb/\\url/\\href/\\index arguments, magic comments and lines that change catcodes are
never touched. --verify compiles both files and checks the PDFs are byte-identical.

usage: python shorten_tex.py main.tex [-o out.tex] [-w 80] [--verify]
"""
import argparse
import bisect
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CODE, COMMENT, PROTECTED = 0, 1, 2
SPACE, PERCENT, COMMENT_SPACE, COMMENT_HARD = range(4)

PROTECTED_ENVS = {
    'verbatim', 'verbatim*', 'Verbatim', 'Verbatim*', 'BVerbatim', 'LVerbatim', 'spverbatim',
    'alltt', 'lstlisting', 'minted', 'comment', 'filecontents', 'filecontents*', 'luacode',
    'luacode*', 'pycode', 'tcblisting', 'Highlighting', 'VerbatimOut', 'sageblock',
    'sagesilent', 'asy', 'gnuplot', 'dot2tex',
}
# commands whose argument is read with changed catcodes
PROTECTED_CMDS = {'verb', 'Verb', 'lstinline', 'mintinline', 'url', 'path', 'nolinkurl',
                  'href', 'index', 'glossary'}
DELIMITED_CMDS = {'verb', 'Verb', 'lstinline'}  # \verb|x|: the next char is the delimiter
BRACE_ONLY_CMDS = {'index', 'glossary'}
# commands that change how spaces or line ends are read; their lines are left alone
CATCODE_CMDS = {'catcode', 'obeylines', 'obeyspaces', 'obeycr', 'endlinechar'}
MAGIC_COMMENTS = ('!', '&', 'arara')  # % !TeX ..., %&format, % arara: ...
BEGIN_RE = re.compile(r'[ \t]*\{([^{}\n]*)\}')


def glued(s, i):
    """True if s[i] is part of the token before it (\\x or ^^x), so no break may precede it."""
    k = i
    while k > 0 and s[k - 1] == '\\':
        k -= 1
    return (i - k) % 2 == 1 or s[max(i - 2, 0):i] == '^^'


def end_of_line(text, i):
    eol = text.find('\n', i)
    return len(text) if eol < 0 else eol


def group_end(text, j):
    """Index past the } matching the { at j, or the end of that line if unmatched."""
    depth, k = 0, j
    while k < len(text):
        c = text[k]
        if c == '\\':
            k += 2
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return k + 1
        k += 1
    return end_of_line(text, j)


def protected_arg_end(text, j, name):
    """Index past the verbatim-read argument of \\name, which starts at j."""
    eol = end_of_line(text, j)
    if text.startswith('*', j):
        j += 1
    if name != 'verb' and text.startswith('[', j):  # [options]
        k = text.find(']', j)
        if not j < k < eol:
            return eol
        j = k + 1
    if name not in DELIMITED_CMDS:
        while j < eol and text[j] in ' \t':
            j += 1
    if name == 'mintinline' and text.startswith('{', j):  # {language} comes first
        j = group_end(text, j)
    if j >= eol:
        return eol
    if text[j] == '{' and name != 'verb':
        return group_end(text, j)
    if name in BRACE_ONLY_CMDS:
        return j
    k = text.find(text[j], j + 1)
    return k + 1 if j < k < eol else eol


def scan(text, envs, cmds):
    """Classify every char as CODE/COMMENT/PROTECTED; also return frozen and catcode line numbers."""
    n = len(text)
    kinds = bytearray(n)
    line_starts = [0] + [m.end() for m in re.finditer('\n', text)]
    line_of = lambda i: bisect.bisect_right(line_starts, i) - 1
    frozen, catcode_lines = set(), set()
    i = 0
    while i < n:
        c = text[i]
        if c == '%':
            j = end_of_line(text, i)
            kinds[i:j] = bytes([COMMENT]) * (j - i)
            i = j
            continue
        if c != '\\':
            i += 1
            continue
        j = i + 1
        while j < n and text[j].isascii() and text[j].isalpha():
            j += 1
        name = text[i + 1:j]
        if not name:
            i = j + 1  # control symbol: \%, \\, \{ ...
        elif name in cmds:
            end = protected_arg_end(text, j, name)
            if '%' in text[i:end]:  # can't tell if that % starts a comment: freeze rest of line
                end = end_of_line(text, end)
            kinds[i:end] = bytes([PROTECTED]) * (end - i)
            i = end
        elif name == 'begin' and (m := BEGIN_RE.match(text, j)) and m.group(1).strip() in envs:
            env = re.escape(m.group(1))
            e = re.compile(r'\\end[ \t]*\{' + env + r'\}').search(text, m.end())
            stop = e.end() if e else n
            frozen.update(range(line_of(i), line_of(stop) + 1))
            i = stop
        else:
            if name in CATCODE_CMDS:
                catcode_lines.add(line_of(i))
            i = j
    return kinds, frozen, catcode_lines


def wrap_line(line, kinds, width, wrap_comments, percent_breaks):
    """Split one source line into pieces of at most width chars where safe break points allow."""
    if len(line) <= width:
        return [line]
    line = line.rstrip(' ')  # TeX drops trailing spaces itself
    n = len(line)
    kinds = kinds[:n]
    indent = line[:n - len(line.lstrip(' \t'))]
    cstart = kinds.find(COMMENT)
    cstart = n if cstart < 0 else cstart
    m = cstart
    while m < n and line[m] == '%':
        m += 1
    t = m  # comment text starts here
    while t < n and line[t] in ' \t':
        t += 1
    marker = line[cstart:m] + (' ' if t > m else '')
    wrap_comment = wrap_comments and t < n and not line[t:].startswith(MAGIC_COMMENTS)

    breaks = []  # (end of this piece, start of the next, kind)
    for run in re.finditer(r'[ \t]+', line):
        a, b = run.span()
        if a == 0 or b == n:
            continue
        if kinds[a] == CODE and not glued(line, a):
            breaks.append((a, b, SPACE))
        elif kinds[a] == COMMENT and a > t and wrap_comment:
            breaks.append((a, b, COMMENT_SPACE))
    if percent_breaks:
        for p in range(len(indent) + 1, cstart):
            if ((line[p - 1] in '}],' or line[p] in '\\{[')
                    and line[p - 1] not in ' \t' and line[p] not in ' \t'
                    and not kinds[p - 1] == kinds[p] == PROTECTED and not glued(line, p)):
                breaks.append((p, p, PERCENT))
    breaks.sort()

    pieces, start, prefix = [], 0, ''
    while len(prefix) + n - start > width:
        room = width - len(prefix)
        later = [br for br in breaks if br[0] > start]
        fits = [br for br in later if br[0] - start + (br[2] == PERCENT) <= room]
        best = (max((br for br in fits if br[2] != PERCENT), default=None)
                or max(fits, default=None))
        hard = start + room
        if best is None and wrap_comment and max(start, t) < hard < n:
            best = (hard, hard, COMMENT_HARD)
        if best is None and later:
            best = later[0]  # nothing fits: shortest overlong piece
        if best is None:
            break
        a, b, kind = best
        pieces.append(prefix + line[start:a] + ('%' if kind == PERCENT else ''))
        prefix = indent + marker if kind in (COMMENT_SPACE, COMMENT_HARD) else indent
        start = b
    pieces.append(prefix + line[start:])
    return pieces


def shorten(text, width, wrap_comments=True, percent_breaks=True,
            envs=PROTECTED_ENVS, cmds=PROTECTED_CMDS):
    """Return the re-wrapped text, lines still over width and lines that change catcodes."""
    kinds, frozen, catcode_lines = scan(text, envs, cmds)
    out, too_long, pos = [], [], 0
    for num, line in enumerate(text.split('\n')):
        if num in frozen or num in catcode_lines:
            pieces = [line]
        else:
            pieces = wrap_line(line, kinds[pos:pos + len(line)], width, wrap_comments,
                               percent_breaks)
        pos += len(line) + 1
        for piece in pieces:
            out.append(piece)
            if len(piece) > width:
                reason = ('verbatim-like environment' if num in frozen else
                          'changes catcodes' if num in catcode_lines else 'no safe break point')
                too_long.append((len(out), num + 1, len(piece), reason))
    return '\n'.join(out), too_long, sorted(catcode_lines)


def verify(src, dst, engine):
    """Compile src and dst the same way and check the PDFs are byte-identical."""
    env = dict(os.environ, SOURCE_DATE_EPOCH='0', FORCE_SOURCE_DATE='1')  # reproducible PDFs
    out = Path(tempfile.mkdtemp(prefix='shorten_tex_'))
    pdfs = []
    for tex in (src, dst):
        for f in out.iterdir():
            shutil.rmtree(f) if f.is_dir() else f.unlink()
        for _ in range(2):  # same jobname and output dir, so even the PDF /ID matches
            subprocess.run([engine, '-interaction=nonstopmode', f'-output-directory={out}',
                            f'-jobname={src.stem}', str(tex.resolve())],
                           cwd=src.resolve().parent, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pdf = out / f'{src.stem}.pdf'
        if not pdf.exists():
            print(f'verify: {engine} made no PDF from {tex}, see {out / src.stem}.log')
            return False
        pdfs.append(pdf.read_bytes())
    if pdfs[0] == pdfs[1]:
        shutil.rmtree(out)
        print(f'verify: PDFs from {src} and {dst} are byte-identical')
        return True
    for name, data in zip(('original.pdf', 'shortened.pdf'), pdfs):
        (out / name).write_bytes(data)
    print(f'verify: PDFs DIFFER, both kept in {out}')
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('input', help='.tex file to re-wrap')
    ap.add_argument('-o', '--output', help='default: shortened-<input name> next to the input')
    ap.add_argument('-w', '--width', type=int, default=80, help='maximum line length (default 80)')
    ap.add_argument('--keep-comments', action='store_true', help='do not re-wrap comments')
    ap.add_argument('--no-percent-breaks', action='store_true',
                    help="only break at spaces, never between tokens with a trailing %%")
    ap.add_argument('--protect-env', action='append', default=[], metavar='ENV',
                    help='extra verbatim-like environment to leave untouched (repeatable)')
    ap.add_argument('--protect-cmd', action='append', default=[], metavar='CMD',
                    help='extra command, without \\, whose argument is verbatim (repeatable)')
    ap.add_argument('--verify', action='store_true',
                    help='compile input and output and check the PDFs are byte-identical')
    ap.add_argument('--engine', default='pdflatex', help='LaTeX engine for --verify')
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output) if args.output else src.with_name('shortened-' + src.name)
    if dst.resolve() == src.resolve():
        sys.exit('refusing to overwrite the input file')
    with open(src, encoding='utf-8', errors='surrogateescape', newline='') as f:
        text = f.read()
    newline = '\r\n' if '\r\n' in text else '\n'
    new, too_long, catcode_lines = shorten(
        text.replace('\r\n', '\n'), args.width, not args.keep_comments,
        not args.no_percent_breaks, PROTECTED_ENVS | set(args.protect_env),
        PROTECTED_CMDS | set(args.protect_cmd))
    with open(dst, 'w', encoding='utf-8', errors='surrogateescape', newline='') as f:
        f.write(new.replace('\n', newline))

    for num in catcode_lines:
        print(f'warning: line {num + 1} changes catcodes or line ends; '
              'the text it affects may not wrap safely, try --verify')
    print(f'wrote {dst}: {len(text.splitlines())} -> {len(new.splitlines())} lines')
    if too_long:
        print(f'{len(too_long)} lines still over {args.width} chars:')
        for out_num, in_num, length, reason in too_long:
            print(f'  line {out_num} (input line {in_num}): {length} chars, {reason}')
    if args.verify and not verify(src, dst, args.engine):
        sys.exit(1)


if __name__ == '__main__':
    main()
