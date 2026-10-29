# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Selected, reproducible first-order mutations, each run in an isolated copy.

Run with the same Python environment used for CI (including ruff):
    python tests/mutation_check.py before
    python tests/mutation_check.py after

``before`` runs the CI gate with only the original test files; ``after`` runs it
with every test file. The original sources are never edited, and a timeout is
reported separately rather than counted as a kill. This is a targeted
experiment, not an exhaustive mutation score.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_TESTS = {"test_hawkes.py", "test_multiexp.py"}
H, M, B = ("src/hawkesdiffusion/hawkes.py", "src/hawkesdiffusion/multiexp.py",
           "src/hawkesdiffusion/binance.py")

# (module, identifier, original, replacement). Each original must occur once.
MUTANTS = [
    (H, "state-includes-equal-time", "source_times[j] < t:", "source_times[j] <= t:"),
    (H, "state-no-decay", "acc *= np.exp(-beta * (t - prev_t))", "acc *= 1.0"),
    (H, "intensity-transposed", "lam = lam + alpha[i][j] * r", "lam = lam + alpha[j][i] * r"),
    (H, "ll-baseline-sign", "total -= mu[i] * horizon", "total += mu[i] * horizon"),
    (H, "ll-compensator-no-decay", "decayed = 1.0 - np.exp(-beta[i][j] * (horizon - tj))",
     "decayed = np.ones_like(tj)"),
    (H, "ll-compensator-no-norm", "total -= (alpha[i][j] / beta[i][j]) * float(np.sum(decayed))",
     "total -= alpha[i][j] * float(np.sum(decayed))"),
    (H, "ll-intensity-transposed", "lam = lam + alpha[i][j] * _recursive_state(ti,",
     "lam = lam + alpha[j][i] * _recursive_state(ti,"),
    (H, "silent-boundary", "if len(t) < min_events:", "if len(t) <= min_events:"),
    (H, "silent-mutates-input", "alpha = np.array(alpha, dtype=float, copy=True)",
     "alpha = np.asarray(alpha, dtype=float)"),
    (H, "branching-min-eigenvalue", "return float(np.max(np.abs(np.linalg.eigvals(branching_matrix(",
     "return float(np.min(np.abs(np.linalg.eigvals(branching_matrix("),
    (H, "half-life-no-ln2", "np.where(beta > 0, np.log(2.0) / beta, np.inf)",
     "np.where(beta > 0, 1.0 / beta, np.inf)"),
    (H, "grid-keeps-worst", 'got["log_likelihood"] > best["log_likelihood"]',
     'got["log_likelihood"] < best["log_likelihood"]'),
    (H, "residuals-no-norm", "total += (alpha[i][j] / beta[i][j]) * float(",
     "total += alpha[i][j] * float("),
    (H, "residuals-not-differenced", "out.append(np.diff(compensator))",
     "out.append(compensator[1:])"),
    (H, "residuals-cross-transposed", "tj = np.asarray(times_by_type[j])\n                tj = tj[tj < t]",
     "tj = np.asarray(times_by_type[i])\n                tj = tj[tj < t]"),
    (H, "simulate-no-thinning", "if rng.random() <= total_new / lam_bar:", "if True:"),
    (H, "simulate-type-always-first", "which = rng.choice(d, p=lam_new / total_new)", "which = 0"),
    (M, "geometric-no-ln2", "return np.log(2.0) / hl", "return 1.0 / hl"),
    (M, "design-compensator-sign", "np.exp(-b * (self.horizon - self.times[j]))",
     "np.exp(-b * (self.horizon + self.times[j]))"),
    (M, "multi-ll-no-norm", "total -= float(np.sum(a[i][j] * self.S[j] / self.betas))",
     "total -= float(np.sum(a[i][j] * self.S[j]))"),
    (M, "multi-branching-times-beta", "return np.sum(a / betas, axis=2)",
     "return np.sum(a * betas, axis=2)"),
    (M, "multi-silent-boundary", "silent = [len(t) < MIN_EVENTS for t in des.times]",
     "silent = [len(t) <= MIN_EVENTS for t in des.times]"),
    (M, "unpack-divides", "return mu, w * self.betas", "return mu, w / self.betas"),
    (B, "maker-flag-swapped", '(sells if r["m"] else buys).append(t)',
     '(buys if r["m"] else sells).append(t)'),
    (B, "milliseconds-wrong-scale", 't = (r["T"] - t0) / 1000.0', 't = (r["T"] - t0) / 100.0'),
    (B, "jitter-ten-milliseconds", "rng.uniform(0.0, 0.001, size=len(arr))",
     "rng.uniform(0.0, 0.01, size=len(arr))"),
    (B, "jitter-unsorted", "return np.sort(arr)", "return arr"),
    (B, "horizon-no-margin", "s[-1] if len(s) else 0.0)) + 1e-3", "s[-1] if len(s) else 0.0))"),
    (B, "ties-counted-wrong", 'int(len(rows) - len(set(r["T"] for r in rows)))',
     'int(len(set(r["T"] for r in rows)))'),
    (B, "trades-sorted-descending", 'key=lambda r: r["T"])', 'key=lambda r: -r["T"])'),
    (B, "trades-not-deduplicated", '{r["a"]: r for r in rows}.values()', "rows"),
    (B, "empty-tape-accepted", 'if not rows:\n        raise RuntimeError("no trades returned")',
     'if False:\n        raise RuntimeError("no trades returned")'),
    (B, "get-no-retry", "for attempt in range(retries):", "for attempt in range(1):"),
]


def gate_commands():
    python = sys.executable
    return [
        [python, "scripts/check_spdx.py"],
        [python, "-m", "ruff", "check", "."],
        [python, "-m", "unittest", "discover", "-f", "-s", "tests"],
    ]


def run_gate(workspace):
    env = {**os.environ, "PYTHONPATH": str(workspace / "src"),
           "PYTHONDONTWRITEBYTECODE": "1", "HAWKES_CACHE": str(workspace / "cache")}
    for command in gate_commands():
        label = "python " + " ".join(command[1:])
        try:
            result = subprocess.run(command, cwd=workspace, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    timeout=300)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "command": label}
        if result.returncode:
            # Test names only. subTest parameters are dropped: they are noise
            # here, and a parameter like key='...' looks like a credential to a
            # secret scanner.
            failures = [line.split(" (")[0] for line in result.stdout.splitlines()
                        if line.startswith(("FAIL:", "ERROR:"))]
            return {"status": "killed", "command": label, "failures": failures}
    return {"status": "survived"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("before", "after"))
    parser.add_argument("--only", nargs="+", help="rerun named mutants and merge their results")
    args = parser.parse_args()
    if args.only and set(args.only) - {m[1] for m in MUTANTS}:
        parser.error("unknown mutant identifier")
    report_path = ROOT / "tests" / "mutation_results.json"
    report = (json.loads(report_path.read_text(encoding="utf-8"))
              if report_path.exists() else {})
    with tempfile.TemporaryDirectory(prefix=".mutation-", dir=ROOT) as temp:
        workspace = Path(temp)
        for directory in ("src", "tests", "scripts"):
            shutil.copytree(ROOT / directory, workspace / directory,
                            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        # Top-level files too: some tests read committed results beside the code.
        for path in ROOT.iterdir():
            if path.is_file():
                shutil.copy2(path, workspace)
        if args.phase == "before":
            for path in (workspace / "tests").glob("test_*.py"):
                if path.name not in ORIGINAL_TESTS:
                    path.unlink()
        baseline = run_gate(workspace)
        if baseline["status"] != "survived":
            raise SystemExit(f"Unmutated gate failed: {baseline}")
        results = []
        for module, identifier, original, replacement in MUTANTS:
            if args.only and identifier not in args.only:
                continue
            path = workspace / module
            source = path.read_text(encoding="utf-8")
            if source.count(original) != 1:
                raise SystemExit(f"{identifier}: expected exactly one mutation site")
            line = source[:source.index(original)].count("\n") + 1
            try:
                mutated = source.replace(original, replacement, 1)
                compile(mutated, module, "exec")
                path.write_text(mutated, encoding="utf-8")
                result = run_gate(workspace)
            finally:
                path.write_text(source, encoding="utf-8")
            results.append({"module": module, "id": identifier, "line": line, **result})
            print(f"{args.phase}: {module}:{line} {identifier}: {result['status']}", flush=True)
    merged = {r["id"]: r for r in report.get(args.phase, [])} if args.only else {}
    merged.update({r["id"]: r for r in results})
    report[args.phase] = [merged[m[1]] for m in MUTANTS if m[1] in merged]
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for module in dict.fromkeys(m[0] for m in MUTANTS):
        rows = [r for r in report[args.phase] if r["module"] == module]
        killed = sum(r["status"] == "killed" for r in rows)
        print(f"{module}: {killed}/{len(rows)} killed")


if __name__ == "__main__":
    main()
