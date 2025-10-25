#!/usr/bin/env python3
import sys
import os
from typing import Dict, List, Tuple


def read_file_lines(path: str) -> List[str]:
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read().splitlines()
    except Exception:
        return []


def parse_file_table(ptx_lines: List[str]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for line in ptx_lines:
        s = line.strip()
        if not s.startswith('.file'):
            continue
        # Example: .file 1 "/A/learn-cuda/01_vector_addition/tma_1d_kernel.cu"
        try:
            after = s[len('.file'):].strip()
            parts = after.split(None, 1)
            if len(parts) < 2:
                continue
            idx = int(parts[0])
            rest = parts[1].strip()
            if rest.startswith('"') and rest.endswith('"'):
                path = rest[1:-1]
            else:
                # Fallback: take token up to whitespace
                path = rest.split()[0]
            mapping[idx] = path
        except Exception:
            continue
    return mapping


def parse_loc(line: str) -> Tuple[int, int, int]:
    # .loc <file> <line> <col> [ , ...]
    s = line.strip()
    assert s.startswith('.loc')
    s = s[len('.loc'):].lstrip()
    # split by comma first to drop trailing qualifiers
    left = s.split(',', 1)[0].strip()
    parts = left.split()
    if len(parts) < 3:
        raise ValueError('Malformed .loc: ' + line)
    file_idx = int(parts[0])
    line_no = int(parts[1])
    col_no = int(parts[2])
    return file_idx, line_no, col_no


def inline_loc(ptx_lines: List[str], file_table: Dict[int, str]) -> List[str]:
    cache: Dict[str, List[str]] = {}
    out: List[str] = []
    for line in ptx_lines:
        stripped = line.lstrip('\t')  # do not consider leading tabs for directive detection
        if stripped.strip().startswith('.loc'):
            # Preserve original leading indentation
            leading = line[: len(line) - len(line.lstrip(' \t'))]
            leading = ""
            try:
                file_idx, line_no, col_no = parse_loc(stripped.strip())
            except Exception:
                out.append(line)
                continue

            src_path = file_table.get(file_idx, f'<unknown:{file_idx}>')
            if src_path not in cache:
                cache[src_path] = read_file_lines(src_path)

            src_lines = cache[src_path]
            src_line_text = ''
            if 1 <= line_no <= len(src_lines):
                src_line_text = src_lines[line_no - 1]
            else:
                src_line_text = '<line unavailable>'

            # Build three lines:
            # 1) <path>: <line>   <original .loc ...>
            # 2)     <source line>
            # 3)     <spaces up to col>^
            header = f"{src_path}: {line_no}   {stripped.strip()}"
            out.append(leading + header)

            indent = leading + ''
            out.append(indent + src_line_text)

            caret_col = max(1, col_no)
            caret_spaces = ' ' * (caret_col - 1)
            out.append(indent + caret_spaces + '^')
        else:
            out.append(line)
    return out


def main() -> None:
    if len(sys.argv) not in (2, 3):
        print('Usage: inline_loc.py <input.ptx> [output.ptx]', file=sys.stderr)
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) == 3 else None

    ptx_lines = read_file_lines(in_path)
    file_table = parse_file_table(ptx_lines)
    out_lines = inline_loc(ptx_lines, file_table)

    if out_path:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(out_lines) + '\n')
    else:
        sys.stdout.write('\n'.join(out_lines) + '\n')


if __name__ == '__main__':
    main()


