import numpy as np, pandas as pd
from scipy import stats

hs = pd.read_csv("real/hillstrom.csv")                    # 64,000 customers, randomised e-mail test
n = hs.segment.value_counts()
chi2, p_srm = stats.chisquare(n.values)                   # sample-ratio mismatch: expect 1/3 each
print(n.to_string(), f"\nSRM check: chi2 = {chi2:.2f}, p = {p_srm:.2f}")

ctrl = hs[hs.segment == "No E-Mail"]
rows = []
for arm in ["Mens E-Mail", "Womens E-Mail"]:
    t = hs[hs.segment == arm]
    for y in ["visit", "conversion", "spend"]:
        diff = t[y].mean() - ctrl[y].mean()
        se = np.sqrt(t[y].var() / len(t) + ctrl[y].var() / len(ctrl))
        rows.append({"arm": arm, "outcome": y, "control": ctrl[y].mean(), "treated": t[y].mean(),
                     "lift": diff, "ci_low": diff - 1.96 * se, "ci_high": diff + 1.96 * se})
res = pd.DataFrame(rows)
print(res.round(4).to_string(index=False))
