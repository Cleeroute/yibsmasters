import warnings; warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
NAVY, TEAL, ORANGE, RED, GREY = "#14325C", "#0F7C84", "#D9731A", "#B3322F", "#5F6B7A"

# %% s1
import os
import duckdb
import pandas as pd

RAW_XLSX = "real/online_retail_II.xlsx"           # 45 MB, two sheets (2009-10, 2010-11)
RAW_PQ = "real/online_retail_II.parquet"          # 7 MB after conversion

if not os.path.exists(RAW_PQ):                    # convert once: Excel is slow to read
    sheets = pd.read_excel(RAW_XLSX, sheet_name=None,
                           dtype={"Invoice": str, "StockCode": str, "Description": str})
    pd.concat(sheets.values(), ignore_index=True).to_parquet(RAW_PQ, index=False)

con = duckdb.connect()                            # in-memory database, no server
con.sql(f"CREATE OR REPLACE VIEW raw AS SELECT * FROM '{RAW_PQ}'")
print(con.sql("SELECT column_name, column_type FROM (DESCRIBE raw)").df().to_string(index=False))

# %% s2
profile = con.sql("""
    SELECT count(*)                                  AS rows,
           count(DISTINCT Invoice)                   AS invoices,
           count(DISTINCT StockCode)                 AS stock_codes,
           min(InvoiceDate)::DATE                    AS first_day,
           max(InvoiceDate)::DATE                    AS last_day,
           count(*) FILTER (WHERE Invoice LIKE 'C%')             AS cancel_lines,
           count(*) FILTER (WHERE Quantity < 0)                  AS negative_qty,
           count(*) FILTER (WHERE Price <= 0)                    AS zero_or_neg_price,
           count(*) FILTER (WHERE "Customer ID" IS NULL)         AS no_customer
    FROM raw""").df()
for k, v in profile.iloc[0].items():
    print(f"{k:18s} {v}")

# %% s3
con.sql(r"""
CREATE OR REPLACE VIEW flagged AS
SELECT *, Quantity * Price AS line_value,
  CASE
    WHEN Invoice LIKE 'C%'                                   THEN '1 cancellation'
    WHEN NOT regexp_matches(StockCode, '^\d{5}[A-Za-z]*$')   THEN '2 non-product code'
    WHEN Quantity <= 0                                       THEN '3 stock write-off'
    WHEN Price <= 0                                          THEN '4 zero price'
    ELSE '0 valid sale'
  END AS dq_class
FROM raw""")
audit = con.sql("""
    SELECT dq_class, count(*) AS lines,
           round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct_lines,
           round(sum(line_value) / 1e3, 1)                     AS value_k_gbp
    FROM flagged GROUP BY dq_class ORDER BY dq_class""").df()
print(audit.to_string(index=False))

dups = con.sql("""SELECT count(*) - count(DISTINCT (Invoice, StockCode, Quantity, InvoiceDate, Price))
                  AS repeated_lines FROM flagged WHERE dq_class = '0 valid sale'""").fetchall()[0][0]
print("identical repeated lines among valid sales:", dups)

# %% s4
kpi = con.sql("""
WITH m AS (
  SELECT date_trunc('month', InvoiceDate)::DATE AS month,
         sum(line_value) FILTER (WHERE dq_class = '0 valid sale')            AS gross_sales,
         -sum(line_value) FILTER (WHERE dq_class = '1 cancellation')         AS cancelled,
         count(DISTINCT Invoice) FILTER (WHERE dq_class = '0 valid sale')    AS orders
  FROM flagged
  WHERE InvoiceDate < '2011-12-01'                      -- drop the incomplete last month
  GROUP BY 1)
SELECT month,
       round(gross_sales / 1e3)                                   AS sales_k,
       orders,
       round(gross_sales / orders)                                AS avg_order_gbp,
       round(100 * cancelled / gross_sales, 1)                    AS cancel_pct,
       round(100 * (gross_sales / lag(gross_sales, 12) OVER (ORDER BY month) - 1), 1) AS yoy_pct
FROM m ORDER BY month""").df()
print(kpi.tail(8).to_string(index=False))

# %% s5
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

# %% s6
focus = abc.query("abc == 'A' and xyz in ['X', 'Y'] and weeks_alive >= 100")["StockCode"]
weekly_focus = (grid[focus].stack().rename("units").reset_index()
                .rename(columns={"StockCode": "unique_id", "week": "ds", "units": "y"}))
weekly_focus.to_parquet("real/weekly_demand_AXY.parquet", index=False)
abc.to_csv("real/sku_abc_xyz.csv", index=False)
print(weekly_focus.shape, "| series:", weekly_focus.unique_id.nunique(),
      "| weeks:", weekly_focus.ds.nunique(), weekly_focus.ds.min(), "to", weekly_focus.ds.max())
