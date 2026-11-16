"""Summarise results/labeled.jsonl (one row per planned trial) into tables and one figure.

Outcomes, over trials with a judge label (status == ok):
  correction    = wrong post -> final claim matches the true answer
  capitulation  = right post -> final claim matches the false answer
  hedge         = final claim hedged; reported separately, never merged into the other two
Intervals are 95% Wilson. Tone effects are paired by claim (exact McNemar test).
"""
import json
from math import sqrt
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest

RES = Path(__file__).resolve().parent.parent / "results"
df = pd.DataFrame([json.loads(l) for l in open(RES / "labeled.jsonl", encoding="utf-8")])
print(f"{len(df)} planned trials. By status (everything except 'ok' is excluded below):")
print(df.groupby(["model", "status"]).size().unstack(fill_value=0).to_string(), "\n")

# Prior belief: how often does the "wrong" post contradict the model's own bare answer?
pri = df.drop_duplicates(["model", "cid"]).groupby("model")["prior"].value_counts(dropna=False).unstack(fill_value=0)
print("Prior belief per claim (true = the 'wrong' post contradicts what the model itself answers):")
print(pri.to_string(), "\n")

df = df[df.status == "ok"].copy()
df["outcome"] = df.side.map({"wrong": "correction", "right": "capitulation"})
df["hit"] = df.label == df.side.map({"wrong": "true", "right": "false"})
df["hedged"] = df.label == "hedged"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


rows = []
for (m, r, e, o, t), g in df.groupby(["model", "role", "evidence", "outcome", "tone"]):
    k, n = int(g.hit.sum()), len(g)
    lo, hi = wilson(k, n)
    rows.append({"model": m, "role": r, "evidence": e, "outcome": o, "tone": t, "n": n, "rate": k / n,
                 "lo": lo, "hi": hi, "hedge_rate": g.hedged.mean(),
                 "decision_change_rate": (g.decision == "CHANGE").mean()})
summ = pd.DataFrame(rows)
summ.to_csv(RES / "summary.csv", index=False)
print(summ.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def mcnemar(g, index):
    w = g.pivot_table(index=index, columns="tone", values="hit", aggfunc="first").dropna()
    if not {"neutral", "shaming"} <= set(w.columns) or w.empty:
        return None
    b = int((w.shaming & ~w.neutral).sum())   # outcome only under shaming
    c = int((~w.shaming & w.neutral).sum())   # outcome only under neutral
    return {"pairs": len(w), "neutral": w.neutral.mean(), "shaming": w.shaming.mean(),
            "only_shaming": b, "only_neutral": c, "mcnemar_p": binomtest(b, b + c, 0.5).pvalue if b + c else 1.0}


ns = df[df.tone.isin(["neutral", "shaming"])].assign(hit=lambda d: d.hit.astype(bool))
tests = []
for (m, r, e, o), g in ns.groupby(["model", "role", "evidence", "outcome"]):
    t = mcnemar(g, "cid")
    if t:
        tests.append({"model": m, "role": r, "evidence": e, "outcome": o, **t})
for (m, o), g in ns.groupby(["model", "outcome"]):   # pooled over role x evidence; pairs are not independent
    t = mcnemar(g, ["cid", "role", "evidence"])
    if t:
        tests.append({"model": m, "role": "ALL", "evidence": "ALL", "outcome": o, **t})
tests = pd.DataFrame(tests)
tests.to_csv(RES / "paired_tests.csv", index=False)
print("\nPaired shaming vs neutral, by claim (exploratory; 'ALL' rows pool the four cells):")
print(tests.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# Figure: rows = models, columns = role x evidence; bars = outcome by tone
models, tones = sorted(df.model.unique()), ["none", "neutral", "shaming"]
cells = [(r, e) for r in ("low", "high") for e in ("none", "note")]
colors = {"correction": "#4C72B0", "capitulation": "#C44E52"}
fig, axes = plt.subplots(len(models), 4, figsize=(9, 2.9 * len(models)), squeeze=False, sharey=True)
for i, m in enumerate(models):
    for j, (r, e) in enumerate(cells):
        ax = axes[i][j]
        for oi, o in enumerate(colors):
            s = summ[(summ.model == m) & (summ.role == r) & (summ.evidence == e) & (summ.outcome == o)].set_index("tone").reindex(tones)
            x = [k + (oi - 0.5) * 0.36 for k in range(3)]
            ax.bar(x, s.rate, 0.34, color=colors[o], yerr=[(s.rate - s.lo).clip(lower=0), (s.hi - s.rate).clip(lower=0)], capsize=3,
                   label="correction" if o == "correction" else "capitulation")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["none", "neutral", "shaming"], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{m.split('-')[0]}: {r}-stake, {'note' if e == 'note' else 'no evidence'}", fontsize=9)
        if i == len(models) - 1:
            ax.set_xlabel("critique", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        if j == 0:
            ax.set_ylabel("share of trials")
        if i == 0 and j == 0:
            ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(RES / "figure1.png", dpi=200)
print("\nwrote results/summary.csv, results/paired_tests.csv, results/figure1.png")
