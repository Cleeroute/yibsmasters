from statsforecast.utils import ConformalIntervals
best = "WindowAverage"
sfp = StatsForecast(models=[WindowAverage(8, prediction_intervals=ConformalIntervals(h=HW, n_windows=6))],
                    freq="W-MON", n_jobs=-1)
cvp = sfp.cross_validation(df=W, h=HW, n_windows=NW, step_size=HW, level=[80, 95])
for lv in [80, 95]:
    inside = (cvp.y >= cvp[f"WindowAverage-lo-{lv}"]) & (cvp.y <= cvp[f"WindowAverage-hi-{lv}"])
    print(f"nominal {lv}% interval -> empirical coverage {inside.mean():.1%}")

cvw.to_parquet("real/weekly_cv_forecasts.parquet", index=False)     # errors feed Lab 4
cv.to_parquet("data/daily_cv_forecasts.parquet", index=False)

sku = cvw.groupby("unique_id")["y"].sum().idxmax()
hist = W[(W.unique_id == sku) & (W.ds >= "2011-01-01")]
f = cvp[cvp.unique_id == sku]
fig, ax = plt.subplots(figsize=(6.4, 2.5))
ax.plot(hist.ds, hist.y, color=NAVY, lw=1.2, label=f"actual ({sku})")
ax.plot(f.ds, f.WindowAverage, color=ORANGE, lw=1.2, label="4-week-ahead forecast")
ax.fill_between(f.ds, f["WindowAverage-lo-80"], f["WindowAverage-hi-80"], color=ORANGE, alpha=.2, label="80% interval")
ax.set_ylabel("units per week"); ax.legend(fontsize=7, frameon=False, loc="upper left"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figs/lab3_sku.pdf")
