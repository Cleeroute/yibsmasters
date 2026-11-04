import warnings; warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
NAVY, TEAL, ORANGE, RED, GREY = "#14325C", "#0F7C84", "#D9731A", "#B3322F", "#5F6B7A"

def load_mboa_sales(path="data/sales.csv"):
    import numpy as np, pandas as pd
    df = pd.read_csv(path, parse_dates=["date"]).drop_duplicates()
    df["store_id"] = df["store_id"].str.strip().str.upper()
    df = df[df.units.isna() | (df.units >= 0)].copy()
    g = df.groupby("product_id")["units"]
    q1, q3 = g.transform(lambda s: s.quantile(.25)), g.transform(lambda s: s.quantile(.75))
    df.loc[df.units > q3 + 3 * (q3 - q1), "units"] = np.nan
    df["units"] = df.groupby(["store_id", "product_id"])["units"].transform(lambda s: s.fillna(s.median()))
    return df

# %% s1
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

# %% s2
fc = rice.shift(1).rolling(28).mean()                      # daily forecast known at the start of day t
lt_demand = sum(rice.shift(-k) for k in range(L))          # demand during the next L days
lt_error = (lt_demand - L * fc)["2023-02":"2023-12"]       # honest lead-time forecast errors, 2023
z = norm.ppf(0.95)                                         # target cycle service level 95%
sd = {"lead-time demand (unconditional)": lt_demand["2023"].std(),
      "daily demand sd x sqrt(L)": rice["2023"].std() * np.sqrt(L),
      "lead-time forecast error": lt_error.std()}
for k, v in sd.items():
    print(f"{k:34s} sd = {v:5.1f}  -> safety stock {z * v:4.0f} units")

def simulate_sQ(demand, fc, ss, Q, L):
    """Daily continuous review (s, Q), backorders; reorder point s_t = L * forecast_t + ss."""
    on_hand, pipeline, back, short, inv, orders = Q + ss, [], 0.0, 0.0, [], 0
    for t, d in enumerate(demand):
        on_hand += sum(q for (arr, q) in pipeline if arr == t)
        pipeline = [(arr, q) for (arr, q) in pipeline if arr > t]
        need = back + d; served = min(on_hand, need); on_hand -= served
        short += max(0.0, d - max(0.0, served - back)); back = need - served
        if on_hand - back + sum(q for _, q in pipeline) <= L * fc[t] + ss:
            pipeline.append((t + L, Q)); orders += 1
        inv.append(on_hand)
    return orders, 100 * (1 - short / demand.sum()), np.mean(inv)

test, fc24 = rice["2024"].values, fc["2024"].values
rows = []
for q_name, q in [("EOQ", round(Q)), ("weekly", round(D / 52))]:
    for ss_name, ss in [("none", 0), ("error-based", z * sd["lead-time forecast error"])]:
        n, fill, inv = simulate_sQ(test, fc24, ss, q, L)
        rows.append({"Q": f"{q_name} ({q})", "safety_stock": ss_name, "orders": n,
                     "fill_rate_%": round(fill, 1), "avg_stock": round(inv),
                     "cost_kXAF": round((n * K + inv * h) / 1e3)})
print(pd.DataFrame(rows).to_string(index=False))

# %% s3
bread = sales.query("store_id == 'S1' and product_id == 'P07'").set_index("date")["units"].asfreq("D")
b = prod.loc["P07"]
price, cost, salvage = b.unit_price, b.unit_cost, 50          # day-old bread sold at 50 XAF
cu, co = price - cost, cost - salvage
cr = cu / (cu + co)
print(f"underage cu = {cu} XAF, overage co = {co} XAF, critical ratio = {cr:.3f}")

# forecast = trailing 28-day level x weekday index (known the evening before)
wd = bread["2023"].groupby(bread["2023"].index.dayofweek).mean()
idx = bread.index.dayofweek.map(wd / wd.mean()).values
fcb = (bread / idx).shift(1).rolling(28).mean() * idx
ratio = (bread / fcb)["2023-02":"2023-12"]                    # calibration: actual / forecast
adj = ratio.quantile(cr)
print(f"quantile of actual/forecast at the critical ratio: {adj:.3f}")

def profit(order, demand):
    sold = np.minimum(order, demand)
    return (price * sold + salvage * (order - sold) - cost * order).sum()

d24, f24 = bread["2024"], fcb["2024"]
plans = {"order = mean forecast": f24.round(),
         "order = forecast x quantile adj.": (f24 * adj).round(),
         "order = 2023 average (fixed)": pd.Series(round(bread["2023"].mean()), index=d24.index)}
rows = []
for name, q in plans.items():
    rows.append({"plan": name, "profit_kXAF": round(profit(q, d24) / 1e3),
                 "in_stock_days_%": round(100 * (q >= d24).mean(), 1),
                 "waste_loaves": int(np.maximum(q - d24, 0).sum())})
print(pd.DataFrame(rows).to_string(index=False))

# %% s4
cvw = pd.read_parquet("real/weekly_cv_forecasts.parquet")      # Lab 3 rolling-origin forecasts
cvw = cvw.sort_values(["unique_id", "ds"])
cut = sorted(cvw.cutoff.unique())
calib = cvw[cvw.cutoff.isin(cut[:4])]                            # Apr-Jul 2011: estimate errors
evalw = cvw[cvw.cutoff.isin(cut[4:])]                            # Aug-Nov 2011: run the policy

R_L = 4                                                          # review 1 week + lead time 3 weeks
sigma = (calib.y - calib.ADIDA).groupby(calib.unique_id).std()   # weekly error sd per SKU

def simulate_RS(demand, fc_week, ss, L=3):
    """Weekly review order-up-to with lead time L weeks and backorders."""
    on_hand, pipe, back, short, inv = fc_week[0] * R_L + ss, [0.0] * L, 0.0, 0.0, []
    for t, d in enumerate(demand):
        on_hand += pipe.pop(0)
        need = d + back; served = min(on_hand, need); on_hand -= served
        short += max(0.0, d - max(0.0, served - back)); back = need - served
        S = fc_week[t] * R_L + ss
        pipe.append(max(0.0, S - (on_hand - back + sum(pipe))))
        inv.append(on_hand)
    return short, np.mean(inv)

rows = []
for target in [0.90, 0.95, 0.99]:
    zt = norm.ppf(target); dem = short = stock = 0.0
    for sku, g in evalw.groupby("unique_id"):
        s_, inv_ = simulate_RS(g.y.values, g.ADIDA.values, zt * sigma[sku] * np.sqrt(R_L))
        dem += g.y.sum(); short += s_; stock += inv_
    rows.append({"target_CSL": target, "z": round(zt, 2), "fill_rate_%": round(100 * (1 - short / dem), 1),
                 "avg_units_on_hand": round(stock)})
print(pd.DataFrame(rows).to_string(index=False))
bias = 100 * (evalw.ADIDA - evalw.y).sum() / evalw.y.sum()
print(f"forecast bias in the evaluation period (Aug-Nov 2011): {bias:+.1f}%")

# %% s5
rng = np.random.default_rng(1)
T, L, m = 300, 2, 8                                   # weeks, lead time per stage, MA window
customer = np.maximum(0, rng.normal(100, 10, T))
stages = ["Retailer", "Wholesaler", "Distributor", "Factory"]

def chain(shared_info):
    incoming, orders = customer, {}
    for s in stages:
        signal = customer if shared_info else incoming   # what this stage forecasts from
        q = np.zeros(T)
        for t in range(m, T):
            mu_now = signal[t - m + 1:t + 1].mean(); mu_prev = signal[t - m:t].mean()
            # order-up-to level S_t = (L+1) * mean; order = demand + change in S
            q[t] = max(0, incoming[t] + (L + 1) * (mu_now - mu_prev))
        orders[s] = q[m * 5:]; incoming = q
    return orders

base = np.var(customer[m * 5:])
for label, shared in [("local forecasts", False), ("shared POS data", True)]:
    o = chain(shared)
    print(f"{label:16s}", "  ".join(f"{s}: {np.var(o[s]) / base:5.1f}x" for s in stages))
print(f"theory, one stage (Chen et al.): >= {1 + 2 * L / m + 2 * L**2 / m**2:.2f}x")

o = chain(False)
fig, ax = plt.subplots(figsize=(6.4, 2.4))
ax.plot(customer[m * 5:m * 5 + 80], color=GREY, lw=1.2, label="customer demand")
for s, c in zip(stages, [NAVY, TEAL, ORANGE, RED]):
    ax.plot(o[s][:80], color=c, lw=1, label=s)
ax.set_xlabel("week"); ax.set_ylabel("units ordered"); ax.legend(fontsize=7, ncol=5, frameon=False, loc="upper left")
ax.grid(alpha=.3); fig.tight_layout(); fig.savefig("figs/lab4_bullwhip.pdf")
