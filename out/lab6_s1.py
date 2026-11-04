import duckdb, numpy as np, pandas as pd
from math import factorial, exp

orders = duckdb.sql("""
    SELECT Invoice, min(InvoiceDate) AS t, count(*) AS lines
    FROM 'real/online_retail_II.parquet'
    WHERE Invoice NOT LIKE 'C%' AND Quantity > 0 AND Price > 0
      AND InvoiceDate >= '2011-09-01' AND InvoiceDate < '2011-12-01'
    GROUP BY Invoice""").df()
wk = orders[orders.t.dt.dayofweek < 5]                                  # Monday-Friday
n_days = wk.t.dt.normalize().nunique()
lam = (wk.groupby(wk.t.dt.hour).size() / n_days).reindex(range(8, 18), fill_value=0)   # orders/hour
service = 4 + 0.6 * orders.lines.clip(upper=200)                         # pick & pack minutes
ES, cs2 = service.mean(), service.var() / service.mean() ** 2
print(f"{n_days} weekdays | mean service {ES:.1f} min | SCV of service c_s^2 = {cs2:.2f}")

def erlang_c(c, a):
    """Probability an arrival waits in M/M/c with offered load a = lambda * E[S]."""
    if a >= c:
        return 1.0
    top = a ** c / factorial(c) * c / (c - a)
    return top / (sum(a ** k / factorial(k) for k in range(c)) + top)

def service_level(c, lam_h, ES_min, t_min):
    a = lam_h * ES_min / 60
    return 1 - erlang_c(c, a) * exp(-(c - a) * t_min / ES_min)

T_SLA, TARGET = 30, 0.80                     # 80% of orders start being picked within 30 minutes
plan = {}
for h, l in lam.items():
    c = max(1, int(np.ceil(l * ES / 60)))
    while service_level(c, l, ES, T_SLA) < TARGET:
        c += 1
    plan[h] = c
tab = pd.DataFrame({"orders_per_h": lam.round(1), "load_a": (lam * ES / 60).round(2),
                    "pickers": pd.Series(plan)})
tab["util_%"] = (100 * tab.load_a / tab.pickers).round(0)
print(tab.T.to_string())
print("picker-hours per day:", tab.pickers.sum())
