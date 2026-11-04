import numpy as np
abc = con.sql("""
WITH sku AS (
  SELECT StockCode, any_value(Description) AS description, sum(line_value) AS revenue
  FROM flagged WHERE dq_class = '0 valid sale' GROUP BY StockCode),
ranked AS (
  SELECT *, sum(revenue) OVER (ORDER BY revenue DESC) / sum(revenue) OVER () AS cum_share,
         row_number() OVER (ORDER BY revenue DESC) AS rnk
  FROM sku)
SELECT *, CASE WHEN cum_share <= 0.80 THEN 'A' WHEN cum_share <= 0.95 THEN 'B' ELSE 'C' END AS abc
FROM ranked ORDER BY rnk""").df()

weekly = con.sql("""
SELECT StockCode, date_trunc('week', InvoiceDate)::DATE AS week, sum(Quantity) AS units
FROM flagged
WHERE dq_class = '0 valid sale' AND InvoiceDate >= '2009-12-07' AND InvoiceDate < '2011-12-05'
GROUP BY ALL""").df()
grid = weekly.pivot_table(index="week", columns="StockCode", values="units", fill_value=0)
# measure variability only over each item's selling life (first to last week with a sale)
alive = (grid.cumsum() > 0) & (grid[::-1].cumsum()[::-1] > 0)
life = grid.where(alive)
cv = (life.std() / life.mean()).rename("cv")
weeks_alive = alive.sum().rename("weeks_alive")
abc = abc.merge(pd.concat([cv, weeks_alive], axis=1), left_on="StockCode", right_index=True, how="left")
abc["xyz"] = pd.cut(abc["cv"], [-np.inf, 0.5, 1.0, np.inf], labels=["X", "Y", "Z"])

print(abc.groupby("abc").agg(skus=("StockCode", "size"), revenue_share=("revenue", "sum"))
         .assign(revenue_share=lambda d: (d.revenue_share / d.revenue_share.sum()).round(3)).to_string())
print(pd.crosstab(abc["abc"], abc["xyz"], margins=True).to_string())

fig, ax = plt.subplots(figsize=(6.2, 2.6))
share_sku = abc["rnk"] / len(abc)
ax.plot(100 * share_sku, 100 * abc["cum_share"], color=NAVY, lw=2)
for lim, lab in [(0.80, "A | B"), (0.95, "B | C")]:
    x = 100 * share_sku[abc["cum_share"] <= lim].max()
    ax.axvline(x, color=ORANGE, ls="--", lw=1); ax.text(x + 1, 8, f"{lab}\n{x:.0f}% of SKUs", color=ORANGE, fontsize=8)
ax.set_xlabel("% of stock codes (ranked by revenue)"); ax.set_ylabel("% of revenue (cumulative)")
ax.set_ylim(0, 102); ax.set_xlim(0, 100); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figs/lab1_pareto.pdf")
