"""Reproduce every number and figure in the README:  python run_all.py"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import bayes, experiment, funnel, simulate, style

Z90 = 1.645


def single_world():
    df = simulate.simulate_partner_funnel(seed=7)
    df.to_csv("results/partner_funnel_simulated.csv", index=False)
    st, rt = funnel.stage_table(df), funnel.route_table(df)
    st.to_csv("results/stage_table.csv", index=False)
    rt.to_csv("results/route_table.csv", index=False)

    y, n = df["drawdown"].values, df["started"].values
    fit = bayes.fit(y, n, n_chains=4, n_iter=6000, burn=2000)
    pm, lo, hi = bayes.summarise(fit["p"], 0.90)
    naive = y / n
    wl, wh = funnel.wilson(y, n, Z90)
    overall = y.sum() / n.sum()
    overall_true = (df["p_true"] * n).sum() / n.sum()
    out = df[["partner", "started"]].copy()
    out["naive"], out["posterior_mean"] = naive, pm
    out["cri90_low"], out["cri90_high"] = lo, hi
    out["wilson90_low"], out["wilson90_high"] = wl, wh
    out["p_true"] = df["p_true"]
    out["prob_below_portfolio"] = (fit["p"] < overall).mean(0)
    out["truly_below_portfolio"] = df["p_true"] < overall_true
    out.to_csv("results/partner_estimates.csv", index=False)
    return df, st, rt, fit, out, overall


def monte_carlo(n_worlds=50):
    se_naive, se_hier, small = [], [], []
    cov_w, cov_b = [], []
    for s in range(100, 100 + n_worlds):
        d = simulate.simulate_partner_funnel(seed=s)
        y, n = d["drawdown"].values, d["started"].values
        f = bayes.fit(y, n, n_chains=1, n_iter=3000, burn=1000)
        pm, lo, hi = bayes.summarise(f["p"], 0.90)
        wl, wh = funnel.wilson(y, n, Z90)
        t = d["p_true"].values
        se_naive.append((y / n - t) ** 2)
        se_hier.append((pm - t) ** 2)
        small.append(n < 500)
        cov_w.append((wl <= t) & (t <= wh))
        cov_b.append((lo <= t) & (t <= hi))
    se_naive, se_hier = np.concatenate(se_naive), np.concatenate(se_hier)
    small = np.concatenate(small)
    cov_w, cov_b = np.concatenate(cov_w), np.concatenate(cov_b)
    return {
        "n_worlds": n_worlds,
        "partner_estimates": int(len(se_naive)),
        "rmse_naive_pp": float(np.sqrt(se_naive.mean()) * 100),
        "rmse_hier_pp": float(np.sqrt(se_hier.mean()) * 100),
        "rmse_naive_small_pp": float(np.sqrt(se_naive[small].mean()) * 100),
        "rmse_hier_small_pp": float(np.sqrt(se_hier[small].mean()) * 100),
        "share_small": float(small.mean()),
        "coverage90_wilson": float(cov_w.mean()),
        "coverage90_hier": float(cov_b.mean()),
        "coverage90_wilson_small": float(cov_w[small].mean()),
        "coverage90_hier_small": float(cov_b[small].mean()),
    }


def experiments(p_a):
    res = {"baseline_submit_rate": float(p_a), "fixed_horizon_n_per_arm": {}, "sequential": []}
    for lift in (0.03, 0.01):
        res["fixed_horizon_n_per_arm"][f"+{int(lift * 100)}pp"] = experiment.fixed_horizon_n(p_a, lift)
    for thr in (0.95, 0.975, 0.99):
        for lift in (0.0, 0.01, 0.03):
            oc = experiment.operating_characteristics(p_a, lift, ship_threshold=thr, seed=int(thr * 1000) + int(lift * 100))
            res["sequential"].append({"ship_threshold": thr, "true_lift_pp": lift * 100, **oc})
    return res


def figures(df, st, fit, est, ex):
    style.apply()
    # 1. funnel
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.2, 1]})
    totals = df[funnel.STAGES].sum()
    names = list(totals.index)
    a1.barh(names, totals.values, color=style.BLUE, height=0.55)
    a1.invert_yaxis()
    for i, v in enumerate(totals.values):
        a1.text(v + totals.max() * 0.01, i, f"{v:,}", va="center", color=style.INK2, fontsize=9)
    a1.set_xlim(0, totals.max() * 1.15)
    a1.set_title("Applications reaching each stage")
    a1.grid(axis="y", visible=False)
    ypos = np.arange(len(st))
    a2.hlines(ypos, st["ci_low"], st["ci_high"], color=style.BLUE, linewidth=2)
    a2.plot(st["conversion"], ypos, "o", color=style.BLUE, markersize=7, markeredgecolor=style.SURFACE, markeredgewidth=1.5)
    for i, r in st.iterrows():
        a2.text(r["ci_high"] + 0.02, i, f"{r['conversion']:.1%} ({r['ci_low']:.1%}–{r['ci_high']:.1%})", va="center", color=style.INK2, fontsize=8.5)
    a2.set_yticks(ypos)
    a2.set_yticklabels([s.replace(" -> ", " → ") for s in st["step"]], fontsize=8.5)
    a2.invert_yaxis()
    a2.set_xlim(0, 1.45)
    a2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    a2.set_title("Step conversion (95% Wilson CI)")
    a2.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig("figures/funnel.png")
    plt.close(fig)

    # 2. shrinkage: error vs volume
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color=style.AXIS, linewidth=1)
    ax.plot(est["started"], (est["naive"] - est["p_true"]) * 100, "o", color=style.ORANGE, markersize=7, markeredgecolor=style.SURFACE, markeredgewidth=1.2, label="Raw partner rate")
    ax.plot(est["started"], (est["posterior_mean"] - est["p_true"]) * 100, "o", color=style.BLUE, markersize=7, markeredgecolor=style.SURFACE, markeredgewidth=1.2, label="Hierarchical posterior mean")
    ax.set_xscale("log")
    ax.set_xticks([100, 200, 500, 1000, 2000])
    ax.set_xticklabels(["100", "200", "500", "1,000", "2,000"])
    ax.set_xlabel("Applications started (log scale)")
    ax.set_ylabel("Error vs true conversion (pp)")
    ax.set_title("Partner conversion: estimation error vs volume (simulated truth known)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("figures/shrinkage.png")
    plt.close(fig)

    # 3. operating characteristics
    seq = pd.DataFrame(ex["sequential"])
    thr = sorted(seq["ship_threshold"].unique())
    null = seq[seq["true_lift_pp"] == 0].set_index("ship_threshold").loc[thr]
    eff = seq[seq["true_lift_pp"] == 3].set_index("ship_threshold").loc[thr]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    x = np.arange(len(thr))
    for ax, d, ttl in ((a1, null, "No true effect: false ship rate"), (a2, eff, "True +3pp lift: chance of shipping")):
        ax.bar(x, d["ship_rate"] * 100, color=style.BLUE, width=0.5)
        for i, (v, w) in enumerate(zip(d["ship_rate"], d["mean_weeks"])):
            ax.text(i, v * 100 + 1.5, f"{v:.1%}\n{w:.1f} wks avg", ha="center", fontsize=8.5, color=style.INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"P(lift>0) ≥ {t}" for t in thr], fontsize=8.5)
        ax.set_title(ttl)
        ax.grid(axis="x", visible=False)
    a1.axhline(5, color=style.MUTED, linestyle="--", linewidth=1)
    a1.text(-0.42, 6.2, "5%", color=style.MUTED, fontsize=8.5, ha="left")
    a1.set_ylabel("% of simulated experiments")
    a1.set_ylim(0, 112)
    fig.tight_layout()
    fig.savefig("figures/sequential_experiment.png")
    plt.close(fig)


def main():
    df, st, rt, fit, est, overall = single_world()
    print("\nSTAGE TABLE\n", st.round(3).to_string())
    print("\nROUTE TABLE\n", rt.round(3).to_string())
    print("\nMCMC diagnostics:", {k: round(v, 3) for k, v in fit["diag"].items()})
    print("\nPARTNER ESTIMATES\n", est.round(3).to_string())
    mc = monte_carlo()
    print("\nMONTE CARLO\n", json.dumps(mc, indent=1))
    ex = experiments(float(st.iloc[0]["conversion"]))
    print("\nEXPERIMENTS\n", json.dumps(ex, indent=1))
    figures(df, st, fit, est, ex)
    with open("results/results.json", "w") as f:
        json.dump(
            {
                "mcmc_diagnostics": fit["diag"],
                "overall_end_to_end_conversion": float(overall),
                "monte_carlo": mc,
                "experiments": ex,
                "stage_table": st.to_dict("records"),
                "route_table": rt.to_dict("records"),
            },
            f,
            indent=1,
        )


if __name__ == "__main__":
    main()
