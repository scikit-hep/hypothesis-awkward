"""Check that the tests catch one deliberate change to the code.

Usage, from the repository root:

    python -I .claude/skills/mutation-check/scripts/mutate.py FILE OLD NEW [PYTEST_ARG ...]

Replace the single occurrence of OLD in FILE with NEW, run the test suite,
restore FILE, and run the suite again. Exit 0 only if the changed code fails
at least one test and the restored code passes all tests. A change that stops
pytest from running the tests at all (e.g. a syntax error) is reported as an
invalid mutant.

Any PYTEST_ARG, such as a test module, is passed to both runs of pytest, which
otherwise run the whole suite. The tests run with the Python that runs this
script, so run it with the project's virtual environment.

Before changing FILE, the script saves the working tree, untracked files
included, as a git tree, without changing the branch, HEAD, or the index, and
prints its ID. After restoring FILE, it checks FILE against the tree. If a run is
killed before it restores FILE, restore it with the printed command.

Bytecode is neither written nor reused: a mutant with the same length as the
original, restored within the same second, would otherwise leave a cached
.pyc that Python still considers valid.
"""

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

_CACHE_ROOTS = ('src', 'tests')
_TESTS_FAILED = 1  # pytest's exit status when some tests failed


def _snapshot() -> str:
    """Save the working tree as a git tree and return its ID."""
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=os.path.join(tmp, 'index'))

        def git(*args: str) -> str:
            return subprocess.run(
                ['git', *args], env=env, capture_output=True, text=True, check=True
            ).stdout.strip()

        git('read-tree', 'HEAD')
        git('add', '-A')
        return git('write-tree')


def _snapshot_text(snapshot: str, path: pathlib.Path) -> str:
    """Return the content of `path` in the tree `snapshot`."""
    return subprocess.run(
        ['git', 'show', f'{snapshot}:{path.as_posix()}'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _clear_pycache() -> None:
    for root in _CACHE_ROOTS:
        for path in pathlib.Path(root).rglob('__pycache__'):
            shutil.rmtree(path)


def _run_pytest(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    return subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', '--no-cov', '-p', 'no:cacheprovider']
        + args,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _summary(output: str) -> str:
    return '\n'.join(
        line
        for line in output.splitlines()
        if line.startswith('FAILED') or ' passed' in line or ' failed' in line
    )


def _run_mutant(
    path: pathlib.Path, original: str, mutated: str, args: list[str]
) -> int:
    """Run the tests on the mutated code and return pytest's exit status."""
    try:
        path.write_text(mutated)
        _clear_pycache()
        result = _run_pytest(args)
    finally:
        path.write_text(original)
        _clear_pycache()
    print('--- mutant:')
    print(_summary(result.stdout))
    return result.returncode


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    path = pathlib.Path(argv[0])
    old, new = argv[1], argv[2]
    args = argv[3:]
    original = path.read_text()
    if original.count(old) != 1:
        print(f'OLD must occur exactly once in {path}')
        return 2

    snapshot = _snapshot()
    print(f'--- snapshot: {snapshot}')
    # Flush, so that the command is printed even if the run is killed.
    print(
        f'    if killed, restore with: git restore --source={snapshot} -- {path}',
        flush=True,
    )

    status = _run_mutant(path, original, original.replace(old, new), args)

    if path.read_text() != _snapshot_text(snapshot, path):
        print(f'FILE NOT RESTORED: {path} differs from the snapshot')
        return 1

    restored = _run_pytest(args)
    print('--- restored:')
    print(_summary(restored.stdout))

    if status == 0:
        print('MUTANT SURVIVED: no test failed')
        return 1
    if status != _TESTS_FAILED:
        print(f'INVALID MUTANT: pytest exited with {status}, e.g. on a syntax error')
        return 1
    if restored.returncode != 0:
        print('RESTORED CODE FAILS')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
