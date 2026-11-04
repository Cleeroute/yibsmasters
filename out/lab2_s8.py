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
