import pulp, numpy as np, pandas as pd

items = ["bread", "beignets", "meat_pie"]
margin = {"bread": 60, "beignets": 25, "meat_pie": 180}             # XAF per unit
use = {  # resource use per unit:   flour (kg), oven (min), labour (min)
    "bread":    {"flour": 0.25, "oven": 0.20, "labour": 0.50},
    "beignets": {"flour": 0.05, "oven": 0.00, "labour": 0.20},
    "meat_pie": {"flour": 0.10, "oven": 0.30, "labour": 1.20},
}
cap = {"flour": 400, "oven": 600, "labour": 2400}                   # available per day
max_sales = {"bread": 1200, "beignets": 3000, "meat_pie": 900}     # forecast demand caps

m = pulp.LpProblem("kitchen_mix", pulp.LpMaximize)
x = {i: pulp.LpVariable(f"x_{i}", lowBound=0) for i in items}
m += pulp.lpSum(margin[i] * x[i] for i in items)
for r in cap:
    m += pulp.lpSum(use[i][r] * x[i] for i in items) <= cap[r], r
for i in items:
    m += x[i] <= max_sales[i], f"demand_{i}"
m.solve(pulp.PULP_CBC_CMD(msg=False))
print(pulp.LpStatus[m.status], "| daily margin =", f"{pulp.value(m.objective):,.0f} XAF")
print(pd.Series({i: x[i].value() for i in items}).round(1).to_string())
print(pd.DataFrame({"slack": {n: c.slack for n, c in m.constraints.items()},
                    "shadow_price": {n: c.pi for n, c in m.constraints.items()}}).round(2).to_string())
