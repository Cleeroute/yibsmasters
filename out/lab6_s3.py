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
