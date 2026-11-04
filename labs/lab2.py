import warnings; warnings.filterwarnings("ignore")
import os, contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    import pm4py                                    # silence the licence banner in captured output
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
NAVY, TEAL, ORANGE, RED, GREY = "#14325C", "#0F7C84", "#D9731A", "#B3322F", "#5F6B7A"
os.environ["TQDM_DISABLE"] = "1"

# %% s1
import pandas as pd, numpy as np
import pm4py

log = pm4py.read_xes("real/receipt.xes")                     # real log, 4TU.ResearchData
log = log.rename(columns={"case:concept:name": "case_id", "concept:name": "activity",
                          "time:timestamp": "ts", "org:resource": "resource"})
log["activity"] = log["activity"].str.replace(r"^(T\d+(-\d)?) .*", r"\1", regex=True)
log = log.sort_values(["case_id", "ts"])
print(f"{log.case_id.nunique():,} cases | {len(log):,} events | {log.activity.nunique()} activities"
      f" | {log.resource.nunique()} resources | {log.ts.min():%Y-%m-%d} to {log.ts.max():%Y-%m-%d}")

traces = log.groupby("case_id")["activity"].agg(" > ".join)
variants = traces.value_counts()
top = pd.DataFrame({"cases": variants, "share": (variants / variants.sum()).round(3),
                    "cum": (variants.cumsum() / variants.sum()).round(3)}).head(5)
top.index = [v if len(v) < 60 else v[:57] + "..." for v in top.index]
print(f"{len(variants)} distinct variants; top 5:")
print(top.to_string())

# %% s2
main = log[log.case_id.isin(traces[traces.isin(variants.index[:8])].index)]
dfg, start_acts, end_acts = pm4py.discover_dfg(main, case_id_key="case_id",
                                               activity_key="activity", timestamp_key="ts")
pm4py.save_vis_dfg(dfg, start_acts, end_acts, "figs/lab2_dfg.png", rankdir="LR")
for (a, b), n in sorted(dfg.items(), key=lambda kv: -kv[1])[:6]:
    print(f"{a:>24s} -> {b:<6s} {n:5d}")

# %% s3
case = log.groupby("case_id").agg(start=("ts", "min"), end=("ts", "max"),
                                  events=("ts", "size"), deadline=("case:deadline", "first"))
case["days"] = (case.end - case.start).dt.total_seconds() / 86400
print(case["days"].describe(percentiles=[.5, .75, .9, .95]).round(2).to_string())

log["next_act"] = log.groupby("case_id")["activity"].shift(-1)
log["wait_h"] = (log.groupby("case_id")["ts"].shift(-1) - log["ts"]).dt.total_seconds() / 3600
hand = (log.dropna(subset=["next_act"]).groupby(["activity", "next_act"])["wait_h"]
          .agg(n="size", median_h="median", mean_h="mean", total_days=lambda x: x.sum() / 24)
          .sort_values("total_days", ascending=False).round(1))
print(hand.head(6).to_string())

# %% s4
def before(t, a, b):                    # if b occurs, a must have occurred earlier
    acts = t.split(" > ")
    return (b not in acts) or (a in acts and acts.index(a) < acts.index(b))

checks = pd.DataFrame({
    "reaches_T05_send": traces.str.contains("T05"),
    "T02_before_T04": traces.apply(before, a="T02", b="T04"),
    "T06_before_T10": traces.apply(before, a="T06", b="T10"),
    "no_repeated_task": traces.apply(lambda t: len(set(t.split(" > "))) == len(t.split(" > "))),
})
checks["all_rules"] = checks.all(axis=1)
print((100 * (1 - checks.mean())).round(1).rename("% cases violating").to_string())

case["late"] = case.end > case.deadline
print(f"\ncases finishing after their deadline: {case.late.mean():.1%}")
print(pd.DataFrame({"cases": checks.groupby("all_rules").size(),
                    "late_rate": case.groupby(checks.all_rules)["late"].mean().round(3),
                    "median_days": case.groupby(checks.all_rules)["days"].median().round(2)}).to_string())

# %% s5
win = case[(case.start >= "2011-01-01") & (case.start < "2011-11-01")]
lam = len(win) / (pd.Timestamp("2011-11-01") - pd.Timestamp("2011-01-01")).days   # cases per day
W = win["days"].mean()                                                              # days per case
days = pd.date_range("2011-03-01", "2011-10-31", freq="D", tz="UTC") + pd.Timedelta(hours=12)
wip = [((case.start <= d) & (case.end > d)).sum() for d in days]
print(f"lambda = {lam:.2f} cases/day   W = {W:.2f} days   lambda*W = {lam*W:.1f}")
print(f"measured average WIP (Mar-Oct 2011) = {np.mean(wip):.1f} open cases")

# %% s6
case["slow"] = case["days"] > 7                                   # service standard: 7 days
wk = case.set_index("start").resample("W")["slow"].agg(["sum", "size"])
wk = wk[(wk.index >= "2010-11-01") & (wk.index < "2011-12-31") & (wk["size"] >= 5)]
wk["p"] = wk["sum"] / wk["size"]
pbar = wk["sum"].sum() / wk["size"].sum()
wk["ucl"] = pbar + 3 * np.sqrt(pbar * (1 - pbar) / wk["size"])
wk["lcl"] = (pbar - 3 * np.sqrt(pbar * (1 - pbar) / wk["size"])).clip(lower=0)
out = wk[(wk.p > wk.ucl) | (wk.p < wk.lcl)]
print(f"centre line p-bar = {pbar:.3f};  weeks plotted = {len(wk)};  signals = {len(out)}")
print(out[["sum", "size", "p", "ucl"]].round(3).to_string())

fig, ax = plt.subplots(figsize=(6.4, 2.5))
ax.plot(wk.index, wk.p, marker="o", ms=3, color=NAVY, lw=1, label="weekly share > 7 days")
ax.step(wk.index, wk.ucl, where="mid", color=RED, lw=1, label="UCL / LCL")
ax.step(wk.index, wk.lcl, where="mid", color=RED, lw=1)
ax.axhline(pbar, color=GREY, ls="--", lw=1, label=r"$\bar p$")
ax.scatter(out.index, out.p, color=ORANGE, zorder=3, s=30, label="special cause")
ax.set_ylabel("share of cases"); ax.legend(fontsize=7, ncol=4, loc="upper left", frameon=False)
ax.set_ylim(0, max(0.9, wk.ucl.max() + .05)); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figs/lab2_pchart.pdf")

# %% s7
import json
temp = pd.read_csv("real/machine_temperature_system_failure.csv", parse_dates=["timestamp"])
temp = temp.drop_duplicates("timestamp").set_index("timestamp")["value"].sort_index()
windows = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in
           json.load(open("real/combined_windows.json"))["realKnownCause/machine_temperature_system_failure.csv"]]
C_FALSE, C_MISS = 150_000, 8_000_000     # XAF: one inspection call-out; one missed failure

def evaluate(alarm, gap="6h"):
    """Group alarms into episodes; an episode overlapping a labelled window is a hit."""
    t = alarm[alarm].index
    if len(t) == 0:
        return {"detected": 0, "false_alarms": 0, "cost_mXAF": 4 * C_MISS / 1e6}
    new = np.r_[True, np.diff(t) > pd.Timedelta(gap)]
    starts, ends = t[new], t[np.r_[new[1:], True]]
    hit_win = [any((starts <= b) & (ends >= a)) for a, b in windows]
    false = sum(not any((s0 <= b) & (e0 >= a) for a, b in windows) for s0, e0 in zip(starts, ends))
    return {"detected": sum(hit_win), "false_alarms": false,
            "cost_mXAF": (false * C_FALSE + (4 - sum(hit_win)) * C_MISS) / 1e6}

# Detector A: adaptive robust z-score against the trailing 24 hours
med = temp.shift(1).rolling("24h", min_periods=200).median()
mad = (temp.shift(1) - med).abs().rolling("24h", min_periods=200).median()
z = ((temp - med) / (1.4826 * mad)).abs()
# Detector B: Shewhart-style chart on the 6-hour moving average, limits fixed in Phase I
ma = temp.rolling("6h").mean()
ref = ma[:"2013-12-09"]                              # Phase I: first week, assumed in control
mu, sd = ref.mean(), ref.std()

rows = []
for k in [2.5, 3, 3.5, 4]:
    rows.append({"detector": "A adaptive z", "k": k, **evaluate((z > 2 * k).rolling(6).sum() >= 4)})
    rows.append({"detector": "B fixed-limit MA", "k": k, **evaluate(ma < mu - k * sd)})
print(f"Phase I limits: mean {mu:.1f}, sd {sd:.1f}")
print(pd.DataFrame(rows).sort_values(["detector", "k"]).to_string(index=False))

# %% s8
from sklearn.ensemble import IsolationForest
feat = pd.DataFrame({"level": temp, "dev_24h": temp - med,
                     "std_2h": temp.rolling("2h").std(), "slope_1h": temp.diff(12)}).dropna()
train = feat[feat.index < "2013-12-09"]               # same Phase I week as "normal" data
iso = IsolationForest(n_estimators=300, random_state=0).fit(train)
score = pd.Series(-iso.score_samples(feat), index=feat.index)
rows = []
for q in [0.95, 0.99, 0.995, 0.999]:
    thr = score[score.index < "2013-12-09"].quantile(q)
    alarm = ((score > thr).rolling(6).sum() >= 4).reindex(temp.index, fill_value=False)
    rows.append({"train_quantile": q, **evaluate(alarm)})
print(pd.DataFrame(rows).to_string(index=False))

fig, ax = plt.subplots(figsize=(6.4, 2.4))
ax.plot(temp.index, temp, color=GREY, lw=.4, label="temperature (5 min)")
ax.plot(ma.index, ma, color=NAVY, lw=1, label="6 h moving average")
ax.axhline(mu - 3 * sd, color=RED, ls="--", lw=1, label="LCL (k = 3)")
for j, (a, b) in enumerate(windows):
    ax.axvspan(a, b, color=ORANGE, alpha=.25, lw=0, label="labelled anomaly" if j == 0 else None)
ax.set_ylabel("temperature"); ax.legend(fontsize=7, frameon=False, loc="lower left", ncol=2)
ax.grid(alpha=.3); fig.tight_layout(); fig.savefig("figs/lab2_nab.pdf")
