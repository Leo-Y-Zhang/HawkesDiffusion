"""Single-exponential versus multi-exponential on the same real trade flow."""
import json, sys, time
import numpy as np
sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from scipy import stats
from hawkesdiffusion import binance
from hawkesdiffusion.hawkes import (fit_beta_grid, branching_matrix,
                                    branching_ratio, log_likelihood,
                                    rescaled_residuals)
from hawkesdiffusion.multiexp import (MultiExpDesign, fit_multi_exp,
                                      geometric_betas, branching_matrix_multi,
                                      branching_ratio_multi,
                                      rescaled_residuals_multi)

b, s, meta = binance.event_streams("BTCUSDT", 4000)
T = meta["horizon_seconds"]; times = [b, s]
print(f"BTCUSDT {len(b)} buy / {len(s)} sell over {T:.0f}s\n")

t0 = time.time()
single = fit_beta_grid(times, T)
sr = rescaled_residuals(times, T, single["mu"], single["alpha"], single["beta"])
sk = [stats.kstest(x, "expon") for x in sr]
print(f"SINGLE EXPONENTIAL  ({time.time()-t0:.0f}s)")
print(f"  logLik {single['log_likelihood']:.1f}  branching "
      f"{branching_ratio(single['alpha'], single['beta']):.4f}  "
      f"half-life {single['half_life_seconds']:.3f}s")
print(f"  KS buy {sk[0].statistic:.4f} (p={sk[0].pvalue:.2e})  "
      f"sell {sk[1].statistic:.4f} (p={sk[1].pvalue:.2e})")

t0 = time.time()
betas = geometric_betas(10, 0.005, 60.0)
multi = fit_multi_exp(times, T, betas=betas)
des = multi["design"]
mr = rescaled_residuals_multi(des, multi["mu"], multi["a"])
mk = [stats.kstest(x, "expon") for x in mr]
nm = branching_matrix_multi(multi["a"], betas)
print(f"\nMULTI-EXPONENTIAL, {len(betas)} components  ({time.time()-t0:.0f}s)")
print(f"  logLik {multi['log_likelihood']:.1f}  branching "
      f"{branching_ratio_multi(multi['a'], betas):.4f}")
print(f"  KS buy {mk[0].statistic:.4f} (p={mk[0].pvalue:.2e})  "
      f"sell {mk[1].statistic:.4f} (p={mk[1].pvalue:.2e})")
print(f"  kernel norms:\n{np.round(nm,4)}")
print(f"  improvement: {multi['log_likelihood']-single['log_likelihood']:+.1f} "
      f"log-likelihood for {multi['a'].size - 4} extra parameters")

hl = np.log(2)/betas
w = multi["a"][0][0]
print("\n  buy<-buy weight by half-life:")
for h, x in zip(hl, w):
    if x > 1e-9: print(f"    {h:8.3f}s : {x:.4f}")

out = {"n_buy": len(b), "n_sell": len(s), "horizon": T,
  "single": {"logLik": single["log_likelihood"],
             "branching": branching_ratio(single["alpha"], single["beta"]),
             "half_life": single["half_life_seconds"],
             "ks": [{"stat": k.statistic, "p": k.pvalue} for k in sk]},
  "multi": {"logLik": multi["log_likelihood"], "n_components": len(betas),
            "branching": branching_ratio_multi(multi["a"], betas),
            "half_lives": list(map(float, hl)),
            "kernel_norms": nm.tolist(),
            "weights_buy_buy": list(map(float, w)),
            "ks": [{"stat": k.statistic, "p": k.pvalue} for k in mk]}}
json.dump(out, open("comparison.json", "w"), indent=1)
print("\nwrote comparison.json")
