"""Render README.md from README.template.md and results.json."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))


def count_tests():
    r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                       cwd=HERE, capture_output=True, text=True, timeout=2400)
    m = re.search(r"Ran (\d+) tests", r.stderr or r.stdout)
    if not m or r.returncode != 0:
        sys.exit("test suite must pass before the README is regenerated")
    return int(m.group(1))


res = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8"))
mres = json.load(open(os.path.join(HERE, "results_multi.json"), encoding="utf-8"))
scan = res["timescale_scan"]
best_ll = res["log_likelihood"]

rows = []
for r in scan:
    mark = " **(best)**" if abs(r["log_likelihood"] - best_ll) < 1e-9 else ""
    rows.append(f"| {r['half_life_seconds']:g} s | {r['log_likelihood']:,.1f}"
                f"{mark} | {r['branching_ratio']:.4f} |")

kb, ksl = res["residual_tests"][0], res["residual_tests"][1]

# The multi-exponential fit whose fastest half-life matches the single
# exponential's: the like-for-like comparison the headline has to quote.
matched = next(x for x in mres["multi"]["stability_scan"]
               if abs(x["fastest_half_life"] - mres["matched_half_life"]) < 1e-12)

V = {
    "asof": res["asof"][:10],
    "symbol": res["symbol"],
    "n_buy": f"{res['n_buy']:,}",
    "n_sell": f"{res['n_sell']:,}",
    "n_total": f"{res['n_buy'] + res['n_sell']:,}",
    "horizon": f"{res['horizon_seconds']:,.0f}",
    "branching": f"{res['branching_ratio']:.4f}",
    "br_lo": f"{res['branching_ratio_range'][0]:.2f}",
    "br_hi": f"{res['branching_ratio_range'][1]:.2f}",
    "scan_rows": "\n".join(rows),
    "ks_buy": f"{kb['ks_stat']:.4f}",
    "p_buy": f"{kb['p_value']:.2e}",
    "ks_sell": f"{ksl['ks_stat']:.4f}",
    "p_sell": f"{ksl['p_value']:.2e}",
    "lr": f"{res['likelihood_ratio']:,.0f}",
    "s_ll": f"{mres['single']['log_likelihood']:,.1f}",
    "m_ll": f"{mres['multi']['log_likelihood']:,.1f}",
    "poisson_ll": f"{mres['poisson_log_likelihood']:,.1f}",
    "s_br": f"{mres['single']['branching_ratio']:.4f}",
    "m_br": f"{mres['multi']['branching_ratio']:.4f}",
    "s_br_lo": f"{mres['single']['branching_range'][0]:.3f}",
    "s_br_hi": f"{mres['single']['branching_range'][1]:.3f}",
    "m_br_lo": f"{mres['multi']['branching_range'][0]:.3f}",
    "m_br_hi": f"{mres['multi']['branching_range'][1]:.3f}",
    "s_ks": f"{mres['single']['ks'][0]['stat']:.4f}",
    "m_ks": f"{mres['multi']['ks'][0]['stat']:.4f}",
    "m_ks_p": f"{mres['multi']['ks'][0]['p']:.1e}",
    "m_ncomp": mres['multi']['n_components'],
    "improvement": f"{mres['improvement']:+,.1f}",
    "improvement_matched": f"{mres['improvement_matched']:+,.1f}",
    "improvement_fast": f"{mres['improvement'] - mres['improvement_matched']:+,.1f}",
    "matched_ms": f"{mres['matched_half_life'] * 1000:g}",
    "res_ms": f"{mres['resolution_half_life'] * 1000:g}",
    "mm_ll": f"{matched['log_likelihood']:,.1f}",
    "mm_br": f"{matched['branching_ratio']:.4f}",
    "n_tied": f"{mres['n_tied_timestamps']:,}",
    "res_hl": mres['resolution_half_life'],
    "n_tests": count_tests(),
}

tpl = open(os.path.join(HERE, "README.template.md"), encoding="utf-8").read()


def sub(m):
    k = m.group(1)
    if k not in V:
        raise KeyError(f"template needs '{k}' but it was not computed")
    return str(V[k])


out = re.sub(r"<<(\w+)>>", sub, tpl)
left = re.findall(r"<<[^>]*>>", out)
if left:
    sys.exit(f"unfilled placeholders: {left}")

open(os.path.join(HERE, "README.md"), "w", encoding="utf-8").write(out)
print(f"wrote README.md ({len(V)} values injected, {V['n_tests']} tests passing)")
