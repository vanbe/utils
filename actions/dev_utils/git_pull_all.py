#!/usr/bin/env python3
"""
git_pull_all.py — Pull every git repository found under a directory.

Walks the directory tree, finds all git repos (directories containing .git),
and runs `git pull` on each. Does not descend into repos (no submodule handling).

Usage:
  python git_pull_all.py <directory> [--force]

  --force  Discard all local changes and reset to the remote tracking branch
           (git fetch --all + git reset --hard origin/<branch>).
"""

import os
import sys
import subprocess
import time

SEP = '─' * 60
R   = '\033[0m'
B   = '\033[1m'
GRN = '\033[32m'
YEL = '\033[33m'
RED = '\033[31m'
BLU = '\033[34m'
CYN = '\033[36m'
GRY = '\033[90m'
DIM = '\033[2m'


def find_git_repos(root: str) -> list:
    repos = []
    for dirpath, dirnames, _ in os.walk(root, topdown=True):
        if '.git' in dirnames:
            repos.append(dirpath)
            dirnames.clear()   # don't descend further into a repo
    return sorted(repos)


def pull_repo(path: str, force: bool = False) -> tuple:
    """Returns (success: bool, summary: str, detail: str)."""
    try:
        if force:
            fetch = subprocess.run(
                ['git', 'fetch', '--all'],
                cwd=path, capture_output=True, text=True, timeout=120,
            )
            if fetch.returncode != 0:
                return False, (fetch.stderr or fetch.stdout).strip().splitlines()[0] or 'fetch failed', ''
            branch_result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=path, capture_output=True, text=True,
            )
            branch = branch_result.stdout.strip() or 'HEAD'
            result = subprocess.run(
                ['git', 'reset', '--hard', f'origin/{branch}'],
                cwd=path, capture_output=True, text=True, timeout=60,
            )
        else:
            result = subprocess.run(
                ['git', 'pull'],
                cwd=path, capture_output=True, text=True, timeout=120,
            )
        output = (result.stdout + result.stderr).strip()
        lines  = [l for l in output.splitlines() if l.strip()]
        summary = lines[0] if lines else '(no output)'
        detail  = '\n'.join(lines[1:]) if len(lines) > 1 else ''
        return result.returncode == 0, summary, detail
    except subprocess.TimeoutExpired:
        return False, 'timed out after 120s', ''
    except Exception as exc:
        return False, str(exc), ''


def main():
    args = sys.argv[1:]
    force = '--force' in args
    dirs  = [a for a in args if not a.startswith('--')]

    if not dirs:
        print(f'Usage: {sys.argv[0]} <directory> [--force]', file=sys.stderr)
        sys.exit(1)

    root = os.path.abspath(dirs[0])
    if not os.path.isdir(root):
        print(f'Error: not a directory: {root}', file=sys.stderr)
        sys.exit(1)

    mode_label = f'{RED}FORCE RESET (local changes will be lost){R}' if force else 'pull'
    print(SEP)
    print(f'  {B}Git Pull All{R}  {DIM}{root}{R}  [{mode_label}]')
    print(SEP)

    repos = find_git_repos(root)
    if not repos:
        print(f'\n  {YEL}No git repositories found under:{R}\n  {root}\n')
        print(SEP)
        sys.exit(0)

    print(f'  {len(repos)} repositor{"y" if len(repos) == 1 else "ies"} found\n')

    ok_count   = 0
    skip_count = 0
    fail_count = 0
    t_total    = time.time()

    for repo in repos:
        rel   = os.path.relpath(repo, root)
        name  = os.path.basename(repo)
        print(f'  {BLU}{B}{name}{R}  {GRY}{rel}{R}')

        t0 = time.time()
        success, summary, detail = pull_repo(repo, force=force)
        elapsed = time.time() - t0
        elapsed_str = f'{int(elapsed // 60)}m{int(elapsed % 60):02d}s' if elapsed >= 60 else f'{elapsed:.1f}s'

        already_up = 'already up' in summary.lower() or 'already up-to-date' in summary.lower()

        if not success:
            fail_count += 1
            marker = f'{RED}✗{R}'
        elif already_up:
            skip_count += 1
            marker = f'{GRY}–{R}'
        else:
            ok_count += 1
            marker = f'{GRN}✓{R}'

        print(f'  {marker}  {DIM}{elapsed_str}{R}  {summary}')
        if detail:
            for line in detail.splitlines():
                if line.strip():
                    print(f'       {DIM}{line}{R}')
        print()

    total = time.time() - t_total
    total_str = f'{int(total // 60)}m {int(total % 60):02d}s' if total >= 60 else f'{total:.1f}s'

    print(SEP)
    parts = []
    if ok_count:   parts.append(f'{GRN}✓ {ok_count} updated{R}')
    if skip_count: parts.append(f'{GRY}– {skip_count} up-to-date{R}')
    if fail_count: parts.append(f'{RED}✗ {fail_count} failed{R}')
    print('  ' + '   '.join(parts) + f'   {DIM}({total_str}){R}')
    print(SEP)

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == '__main__':
    main()
