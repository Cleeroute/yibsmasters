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
