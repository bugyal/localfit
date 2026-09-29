"""Keep the README honest: every documented command must reproduce its output.

A README that drifts from the tool is worse than no README — the 0.1.0 build
shipped two defects that only surfaced by running its own examples verbatim
(a quant table claiming q6_k fits on an 8 GB card, and a stale anchor note).
This test makes that class of drift fail in CI instead of in the reader's hands.

Design notes:
- Runs the CLI in-process via ``cli.main`` so it needs no console script on PATH.
- The README wraps prose to column 80 while the tool prints one long line, so
  comparison is on whitespace-normalised text, with a bare ``...`` in the README
  acting as "some rows elided here".
- Deliberately compares CONTENT, not layout: reflowing a paragraph is fine,
  changing a number or dropping a fact is not.
"""
from __future__ import annotations

import io
import re
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from localfit import cli

README = Path(__file__).resolve().parents[1] / "README.md"


def _flat(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _blocks() -> list[tuple[str, str]]:
    """Return (command, expected output) for each `$ ...` line in the README."""
    text = README.read_text(encoding="utf-8")
    pairs: list[tuple[str, str]] = []
    for block in re.findall(r"```console\n(.*?)```", text, re.S):
        lines = block.rstrip("\n").split("\n")
        i = 0
        while i < len(lines):
            if not lines[i].startswith("$ "):
                i += 1
                continue
            cmd = lines[i][2:]
            exp: list[str] = []
            j = i + 1
            while j < len(lines) and not lines[j].startswith("$ "):
                exp.append(lines[j])
                j += 1
            i = j
            pairs.append((cmd, "\n".join(exp)))
    return pairs


def _runs_cleanly(cmd: str) -> bool:
    """Only `localfit <sub>` invocations are replayed; setup lines are skipped."""
    return cmd.startswith("localfit ")


def _run(cmd: str) -> str:
    argv = cmd.split()[1:]
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            cli.main(argv)
        except SystemExit:  # argparse on --help
            pass
    return out.getvalue() + err.getvalue()


def _contains(needle_flat: str, hay_flat: str) -> str | None:
    """Walk needle through hay, treating a bare '...' as 'skip to next anchor'.

    Returns the first missing fragment, or None when everything was found.
    """
    parts = [p.strip() for p in re.split(r"(?:^|\s)\.\.\.(?:\s|$)", needle_flat) if p.strip()]
    pos = 0
    for part in parts:
        idx = hay_flat.find(part, pos)
        if idx < 0:
            return part
        pos = idx + len(part)
    return None


@pytest.mark.parametrize("cmd,expected", _blocks())
def test_readme_example_matches_real_output(cmd: str, expected: str) -> None:
    if not _runs_cleanly(cmd):
        pytest.skip(f"not a localfit invocation: {cmd}")
    if not expected.strip():
        pytest.skip(f"command shown without expected output: {cmd}")
    missing = _contains(_flat(expected), _flat(_run(cmd)))
    assert missing is None, (
        f"README is stale for `{cmd}`.\n"
        f"This text no longer appears in the real output: {missing!r}\n"
        "Re-run the command and paste the true output (or elide with `...`)."
    )


@pytest.mark.parametrize(
    "host_dependent", ["probe"], ids=lambda c: f"localfit {c}"
)
def test_host_dependent_commands_are_not_replay_blocks(host_dependent: str) -> None:
    """Commands whose output describes *this machine* must never sit in a
    ```console block.

    Console blocks are replayed by ``test_readme_example_matches_real_output``
    on every host CI picks — a pasted probe example is one specific machine's
    hardware and can only pass on that machine (0.1.0 shipped its author's
    Apple M4 inside a console block and broke CI on Linux). Paste it as an
    ```output sample instead.
    """
    text = README.read_text(encoding="utf-8")
    for block in re.findall(r"```console\n(.*?)```", text, re.S):
        for line in block.splitlines():
            if line.startswith(f"$ localfit {host_dependent}"):
                pytest.fail(
                    f"`localfit {host_dependent}` is host-specific and must not "
                    "appear in a ```console block (it is replayed in CI on "
                    "machines whose hardware differs). Use an ```output sample "
                    "block instead."
                )


def test_readme_states_the_true_test_count() -> None:
    """The README quotes a pass/skip summary; it must match reality.

    Checked against pytest's own collection rather than a nested run, so the
    expectation cannot be skewed by this test executing inside it (a nested run
    skips this very test and reports one fewer pass). The 0.1.0 README quoted a
    stale '150 passed' after the suite grew.
    """
    text = README.read_text(encoding="utf-8")
    m = re.search(r"\$ \.venv/bin/python -m pytest tests/ -q\n(\d+) passed(?:, (\d+) skipped)?", text)
    assert m, "README no longer quotes a pytest result line; update this test"
    quoted = int(m.group(1)) + int(m.group(2) or 0)

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q",
         "-p", "no:cacheprovider"],
        cwd=README.parent,
        capture_output=True,
        text=True,
    )
    real = re.search(r"(\d+) tests? collected", proc.stdout)
    assert real, f"could not read a collected count from pytest: {proc.stdout[-400:]!r}"
    assert quoted == int(real.group(1)), (
        f"README says {quoted} tests exist, but pytest collects "
        f"{real.group(1)}. Update the README block."
    )
