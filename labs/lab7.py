import warnings; warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
NAVY, TEAL, ORANGE, RED, GREY = "#14325C", "#0F7C84", "#D9731A", "#B3322F", "#5F6B7A"

# %% s1
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

# %% s2
margin, cost_per_email = 0.35, 0.10                       # gross margin on spend; $ per e-mail sent
for arm in ["Mens E-Mail", "Womens E-Mail"]:
    r = res[(res.arm == arm) & (res.outcome == "spend")].iloc[0]
    profit = margin * r.lift - cost_per_email
    lo, hi = margin * r.ci_low - cost_per_email, margin * r.ci_high - cost_per_email
    print(f"{arm:14s} incremental profit per e-mail: ${profit:.3f}  (95% CI {lo:.3f} to {hi:.3f})")

# follow-up test: customers per arm to detect +0.3 points of conversion with 80% power
p0, delta = ctrl.conversion.mean(), 0.003
za, zb = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
n_arm = 2 * (za + zb) ** 2 * p0 * (1 - p0) / delta ** 2
print(f"baseline conversion {p0:.4f}; customers needed per arm: {n_arm:,.0f}")

# %% s3
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression

emp = pd.read_csv("data/employees.csv")                   # Mboa staff; TRUE training effect = +3
naive = emp.groupby("trained").productivity.mean().diff().iloc[-1]
reg = smf.ols("productivity ~ trained + skill_score + tenure_years", data=emp).fit()
X = emp[["skill_score", "tenure_years"]]
ps = LogisticRegression(max_iter=1000).fit(X, emp.trained).predict_proba(X)[:, 1]
w = np.where(emp.trained == 1, 1 / ps, 1 / (1 - ps))
ipw = (np.average(emp.productivity[emp.trained == 1], weights=w[emp.trained == 1])
       - np.average(emp.productivity[emp.trained == 0], weights=w[emp.trained == 0]))
# doubly robust (AIPW): outcome model + propensity weights, trimmed to [0.05, 0.95]
T, Yv = emp.trained.values, emp.productivity.values
pt = np.clip(ps, 0.05, 0.95)
mu1 = smf.ols("productivity ~ skill_score + tenure_years", emp[emp.trained == 1]).fit().predict(emp)
mu0 = smf.ols("productivity ~ skill_score + tenure_years", emp[emp.trained == 0]).fit().predict(emp)
aipw = np.mean(mu1 - mu0 + T * (Yv - mu1) / pt - (1 - T) * (Yv - mu0) / (1 - pt))
print(f"naive difference      {naive:5.2f}")
print(f"regression adjustment {reg.params['trained']:5.2f}  (95% CI {reg.conf_int().loc['trained', 0]:.2f}"
      f" to {reg.conf_int().loc['trained', 1]:.2f})")
print(f"inverse-propensity    {ipw:5.2f}   <- extreme weights, max w = {w.max():.0f}")
print(f"doubly robust (AIPW)  {aipw:5.2f}")
print(f"propensity range: trained {ps[emp.trained == 1].min():.2f}-{ps[emp.trained == 1].max():.2f}, "
      f"untrained {ps[emp.trained == 0].min():.2f}-{ps[emp.trained == 0].max():.2f}")

# %% s4
import causaldata
od = causaldata.organ_donations.load_pandas().data        # US states, quarterly registration rates
od["ca"] = (od.State == "California").astype(int)
od["post"] = (od.Quarter_Num >= 4).astype(int)            # active-choice rule from Q3 2011
cell = od.groupby(["ca", "post"]).Rate.mean().unstack()
print(cell.round(4).to_string())
print(f"2x2 DiD = {(cell.loc[1, 1] - cell.loc[1, 0]) - (cell.loc[0, 1] - cell.loc[0, 0]):+.4f}")

od["treat"] = od.ca * od.post
twfe = smf.ols("Rate ~ treat + C(State) + C(Quarter_Num)", data=od).fit(
    cov_type="cluster", cov_kwds={"groups": od.State})
print(f"two-way fixed effects: {twfe.params['treat']:+.4f} (cluster SE {twfe.bse['treat']:.4f})")

ev = []
for q in [1, 2, 4, 5, 6]:                                 # event study, reference quarter = 3
    od[f"ca_q{q}"] = od.ca * (od.Quarter_Num == q)
f = "Rate ~ " + " + ".join(f"ca_q{q}" for q in [1, 2, 4, 5, 6]) + " + C(State) + C(Quarter_Num)"
es = smf.ols(f, data=od).fit(cov_type="cluster", cov_kwds={"groups": od.State})
coef = pd.DataFrame({"q": [1, 2, 3, 4, 5, 6],
                     "b": [es.params.get(f"ca_q{q}", 0.0) for q in range(1, 7)],
                     "se": [es.bse.get(f"ca_q{q}", 0.0) for q in range(1, 7)]})
fig, ax = plt.subplots(figsize=(5.6, 2.3))
ax.errorbar(coef.q, coef.b, yerr=1.96 * coef.se, fmt="o", color=NAVY, capsize=3)
ax.axhline(0, color=GREY, lw=1); ax.axvline(3.5, color=ORANGE, ls="--", lw=1)
ax.set_xticks(range(1, 7), ["Q4-10", "Q1-11", "Q2-11", "Q3-11", "Q4-11", "Q1-12"])
ax.set_ylabel("California vs others\n(rate difference)"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figs/lab7_event.pdf")
print(coef.round(4).to_string(index=False))

# %% s5
import json
proposal = json.loads("""{
 "store": "S1", "budget_xaf": 3000000,
 "lines": [
  {"product_id": "P01", "qty": 300, "unit": "units", "reason": "December peak, 2 weeks cover"},
  {"product_id": "P02", "qty": 40,  "unit": "cases", "reason": "cases of 12"},
  {"product_id": "P13", "qty": 200, "unit": "units", "reason": "new yoghurt line"},
  {"product_id": "P07", "qty": 900, "unit": "units", "reason": "bread for the week"},
  {"product_id": "P06", "qty": -20, "unit": "units", "reason": "return excess"}
 ]}""")                                                   # e.g. produced by an LLM assistant

prod = pd.read_csv("data/products.csv").set_index("product_id")
CASE = {"P02": 12}                                         # pack sizes from master data

def validate(plan, max_days_cover=21, daily_mu=None):
    issues, total = [], 0
    for ln in plan["lines"]:
        pid, q = ln["product_id"], ln["qty"]
        if pid not in prod.index:
            issues.append((pid, "unknown product: not in master data")); continue
        if ln["unit"] == "cases":
            q = q * CASE.get(pid, np.nan)
            if np.isnan(q):
                issues.append((pid, "unit 'cases' without a known pack size")); continue
        if q <= 0:
            issues.append((pid, "non-positive order quantity")); continue
        if prod.loc[pid, "shelf_life_days"] < 7 and q > 2 * daily_mu[pid]:
            issues.append((pid, f"perishable: {q:.0f} units > 2 days of demand ({2 * daily_mu[pid]:.0f})"))
        if q > max_days_cover * daily_mu[pid]:
            issues.append((pid, f"{q:.0f} units exceeds {max_days_cover} days of cover"))
        total += q * prod.loc[pid, "unit_cost"]
    if total > plan["budget_xaf"]:
        issues.append(("ALL", f"cost {total:,.0f} XAF exceeds budget {plan['budget_xaf']:,}"))
    return issues, total

sales = pd.read_csv("data/sales.csv")
s1 = sales[(sales.store_id.str.upper() == "S1") & (sales.units >= 0) & (sales.units < 500)]
daily_mu = s1[s1.date >= "2024-10-01"].groupby("product_id").units.mean()
issues, total = validate(proposal, daily_mu=daily_mu)
print(f"valid lines cost {total:,.0f} XAF; {len(issues)} issues found:")
for pid, msg in issues:
    print(f"  [{pid}] {msg}")
