import warnings; warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
NAVY, TEAL, ORANGE, RED, GREY = "#14325C", "#0F7C84", "#D9731A", "#B3322F", "#5F6B7A"

# %% s1
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

# %% s2
import simpy

hours = list(range(8, 18))
day_rates = lam.values / 60                                  # per minute, one value per hour
lam_max = day_rates.max()

def one_day(staff_by_hour, seed):
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    pickers = simpy.Store(env)
    waits, arr_hour = [], []

    def shift_manager():                                     # change the number of pickers each hour
        on = 0
        for k, h in enumerate(hours):
            target = staff_by_hour[h]
            while on < target:
                pickers.put(object()); on += 1
            while on > target:
                yield pickers.get(); on -= 1               # waits until a picker is free
            yield env.timeout(max(0, 60 * (k + 1) - env.now))

    def order(env):
        arrive = env.now
        token = yield pickers.get()
        waits.append(env.now - arrive); arr_hour.append(hours[int(arrive // 60)])
        yield env.timeout(rng.choice(service.values))       # empirical service times
        yield pickers.put(token)

    def arrivals():                                          # non-stationary Poisson by thinning
        while True:
            yield env.timeout(rng.exponential(1 / lam_max))
            if env.now >= 60 * len(hours):
                return
            if rng.random() < day_rates[int(env.now // 60)] / lam_max:
                env.process(order(env))

    env.process(shift_manager()); env.process(arrivals())
    env.run(until=60 * len(hours) + 240)                     # let the last orders finish
    w = pd.Series(waits, index=arr_hour)
    return (w <= T_SLA).mean(), w.mean(), w.quantile(.9), (w <= T_SLA).groupby(level=0).mean()

def evaluate(staff, reps=200):
    runs = [one_day(staff, s) for s in range(reps)]          # same seeds for every plan (CRN)
    r = np.array([x[:3] for x in runs])
    m, se = r.mean(axis=0), r.std(axis=0, ddof=1) / np.sqrt(reps)
    by_hour = pd.concat([x[3] for x in runs], axis=1).mean(axis=1)
    return {"picker_h": sum(staff.values()), "SLA_met_%": f"{100*m[0]:.1f} ± {196*se[0]:.1f}",
            "mean_wait": round(m[1], 1), "p90_wait": round(m[2], 1)}, by_hour

T_SLA = 15                                                   # tighter promise: start within 15 min
plan15 = {}
for h, l in lam.items():
    c = max(1, int(np.ceil(l * ES / 60)))
    while service_level(c, l, ES, T_SLA) < TARGET:
        c += 1
    plan15[h] = c
flat = {h: round(sum(plan15.values()) / len(hours)) for h in hours}
summary, hourly = {}, {}
for name, staff in [("Erlang C plan", plan15), ("flat staffing", flat)]:
    summary[name], hourly[name] = evaluate(staff)
print(pd.DataFrame(summary).T.to_string())
pred = pd.Series({h: service_level(plan15[h], lam[h], ES, T_SLA) for h in hours})
print(pd.DataFrame({"pickers": pd.Series(plan15), "ErlangC_pred_%": (100 * pred).round(0),
                    "simulated_%": (100 * hourly["Erlang C plan"]).round(0)}).T.to_string())

fig, ax = plt.subplots(figsize=(6.2, 2.4))
ax.plot(hours, 100 * pred.values, "o--", color=GREY, label="Erlang C prediction (each hour alone)")
ax.plot(hours, 100 * hourly["Erlang C plan"].values, "o-", color=NAVY, label="simulation, Erlang C plan")
ax.plot(hours, 100 * hourly["flat staffing"].values, "s-", color=ORANGE, label="simulation, flat staffing")
ax.axhline(100 * TARGET, color=RED, lw=1, label="target 80%")
ax.set_xlabel("hour of arrival"); ax.set_ylabel("% started within 15 min"); ax.set_ylim(0, 102)
ax.legend(fontsize=7, frameon=False, loc="lower left"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figs/lab6_sla.pdf")

# %% s3
import pulp
prod = pd.read_csv("data/products.csv").set_index("product_id")
sku = ["P01", "P02", "P06", "P12"]                         # rice, oil, milk powder, sardines
mu = pd.Series([620, 980, 470, 1150], index=sku)            # December forecast, units (store S1)
cv = 0.40                                                   # forecast-error coefficient of variation
vol = pd.Series([8.0, 1.2, 2.5, 0.3], index=sku)            # litres of storage per unit
V = 7000                                                    # back-room capacity, litres
price, cost = prod.unit_price[sku], prod.unit_cost[sku]
salvage = 0.40 * cost                                       # January clearance value

rng = np.random.default_rng(7)
def draw(n):
    s = np.sqrt(np.log(1 + cv ** 2))                         # lognormal with mean mu, CV = cv
    return pd.DataFrame(rng.lognormal(np.log(mu) - s ** 2 / 2, s, (n, len(sku))), columns=sku)

def solve_saa(scen):
    """Two-stage model: order q now; sales s_k = min(q, d_k) in each scenario k."""
    K = len(scen)
    m = pulp.LpProblem("december_order", pulp.LpMaximize)
    q = {i: pulp.LpVariable(f"q_{i}", lowBound=0) for i in sku}
    s = {(i, k): pulp.LpVariable(f"s_{i}_{k}", lowBound=0) for i in sku for k in range(K)}
    m += pulp.lpSum((price[i] - salvage[i]) * s[i, k] for i in sku for k in range(K)) / K \
         - pulp.lpSum((cost[i] - salvage[i]) * q[i] for i in sku)
    m += pulp.lpSum(vol[i] * q[i] for i in sku) <= V, "storage"
    for i in sku:
        for k in range(K):
            m += s[i, k] <= q[i]
            m += s[i, k] <= scen[i].iloc[k]
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    return pd.Series({i: q[i].value() for i in sku})

def expected_profit(q, test):
    sold = np.minimum(q.values, test.values)
    return (price.values * sold + salvage.values * (q.values - sold) - cost.values * q.values).sum(axis=1).mean()

test = draw(20000)                                           # out-of-sample evaluation
q_ev = solve_saa(pd.DataFrame([mu]))                          # deterministic: demand = forecast
q_saa = solve_saa(draw(300))                                  # stochastic: 300 sampled scenarios
p_ev, p_saa = expected_profit(q_ev, test), expected_profit(q_saa, test)
def wait_and_see(test):
    """Perfect information: in each scenario stock exactly the demand, best margin per litre first."""
    order = ((price - cost) / vol).sort_values(ascending=False).index
    total = 0.0
    for _, d in test.iterrows():
        room, prof = V, 0.0
        for i in order:
            x = min(d[i], room / vol[i]); room -= x * vol[i]; prof += (price[i] - cost[i]) * x
        total += prof
    return total / len(test)
p_ws = wait_and_see(test.iloc[:5000])
print(pd.DataFrame({"forecast": mu, "plan_EV": q_ev.round(), "plan_SAA": q_saa.round(),
                    "litres_SAA": (vol * q_saa).round()}).to_string())
print(f"storage used: EV {(vol * q_ev).sum():,.0f} L, SAA {(vol * q_saa).sum():,.0f} L of {V:,} L")
print(f"expected profit: EV plan {p_ev/1e3:,.0f} k XAF | SAA plan {p_saa/1e3:,.0f} k XAF"
      f" | VSS = {(p_saa - p_ev)/1e3:,.0f} k XAF")
print(f"perfect-information profit {p_ws/1e3:,.0f} k XAF -> EVPI = {(p_ws - p_saa)/1e3:,.0f} k XAF")
