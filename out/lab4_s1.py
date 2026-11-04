import numpy as np, pandas as pd
from scipy.stats import norm

sales = load_mboa_sales()                     # same cleaning function as Lab 3
prod = pd.read_csv("data/products.csv").set_index("product_id")
rice = sales.query("store_id == 'S1' and product_id == 'P01'").set_index("date")["units"].asfreq("D")
p = prod.loc["P01"]
D = rice["2023"].sum()                        # annual demand, units
K, h = p.order_cost_xaf, 0.25 * p.unit_cost   # order cost; holding = 25% of cost per year
L = int(p.lead_time_days)
Q = np.sqrt(2 * D * K / h)
print(f"annual demand D = {D:,.0f} units, K = {K:,} XAF, h = {h:,.0f} XAF/unit/yr")
print(f"EOQ = {Q:.0f} units  (~{365 * Q / D:.0f} days of demand), "
      f"annual order+holding cost = {np.sqrt(2 * D * K * h):,.0f} XAF")
for f in [0.5, 0.8, 1.25, 2.0]:                 # the flat bottom of the cost curve
    tc = D / (f * Q) * K + f * Q / 2 * h
    print(f"  order {f:>4.2f} x EOQ -> cost +{100 * (tc / np.sqrt(2 * D * K * h) - 1):.1f}%")
