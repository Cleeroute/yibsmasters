import warnings; warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
NAVY, TEAL, ORANGE, RED, GREY = "#14325C", "#0F7C84", "#D9731A", "#B3322F", "#5F6B7A"

# %% s1
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

# %% s2
from math import radians, sin, cos, asin, sqrt
cities = pd.DataFrame([            # approx. coordinates; population in thousands (rounded, illustrative)
    ("Douala", 4.051, 9.768, 3900), ("Yaounde", 3.848, 11.502, 4300), ("Bamenda", 5.963, 10.159, 600),
    ("Bafoussam", 5.478, 10.418, 450), ("Garoua", 9.302, 13.392, 800), ("Maroua", 10.596, 14.325, 450),
    ("Ngaoundere", 7.328, 13.585, 400), ("Bertoua", 4.577, 13.685, 350), ("Kumba", 4.636, 9.447, 300),
    ("Buea", 4.153, 9.241, 200), ("Limbe", 4.024, 9.215, 130), ("Nkongsamba", 4.955, 9.940, 130),
    ("Edea", 3.800, 10.133, 120), ("Ebolowa", 2.900, 11.150, 120), ("Kribi", 2.940, 9.910, 100),
], columns=["city", "lat", "lon", "pop_k"]).set_index("city")
cities["demand_t"] = (0.02 * cities.pop_k).round(1)                # tonnes per week

def road_km(a, b, circuity=1.3):
    la1, lo1, la2, lo2 = map(radians, [a.lat, a.lon, b.lat, b.lon])
    h = sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2
    return circuity * 2 * 6371 * asin(sqrt(h))

cand = {"Douala": 5.0, "Yaounde": 5.0, "Bafoussam": 3.0, "Ngaoundere": 3.0, "Garoua": 3.2, "Bertoua": 2.8}
CAP = 150                                                         # tonnes per week per DC
dist = {(f, c): road_km(cities.loc[f], cities.loc[c]) for f in cand for c in cities.index}

def design(rate=90, single_source=True):
    """Capacitated facility location; rate = outbound trucking cost in XAF per tonne-km."""
    mdl = pulp.LpProblem("network", pulp.LpMinimize)
    y = {f: pulp.LpVariable(f"open_{f}", cat="Binary") for f in cand}
    kind = "Binary" if single_source else "Continuous"
    a = {(f, c): pulp.LpVariable(f"a_{f}_{c}", 0, 1, cat=kind) for f in cand for c in cities.index}
    fixed = pulp.lpSum(1e6 * cand[f] * y[f] for f in cand)
    # inbound: full trucks from the port of Douala (cheaper per t-km); outbound: city deliveries
    freight = pulp.lpSum((0.6 * rate * dist["Douala", f] + rate * dist[f, c]) * cities.demand_t[c] * a[f, c]
                         for f in cand for c in cities.index)
    mdl += fixed + freight
    for c in cities.index:
        mdl += pulp.lpSum(a[f, c] for f in cand) == 1                     # every city served
    for f in cand:
        mdl += pulp.lpSum(cities.demand_t[c] * a[f, c] for c in cities.index) <= CAP * y[f]
        for c in cities.index:
            mdl += a[f, c] <= y[f]                                          # strong linking
    mdl.solve(pulp.PULP_CBC_CMD(msg=False))
    opened = [f for f in cand if y[f].value() > 0.5]
    serve = {c: max(cand, key=lambda f: a[f, c].value()) for c in cities.index}
    return opened, serve, pulp.value(fixed) / 1e6, pulp.value(freight) / 1e6

opened, serve, fx, fr = design()
print("open DCs:", opened, f"| fixed {fx:.1f} + freight {fr:.1f} = {fx + fr:.1f} M XAF/week")
for f, grp in pd.Series(serve).groupby(pd.Series(serve)):
    print(f"  {f:10s} serves {', '.join(grp.index)} ({cities.demand_t[grp.index].sum():.0f} t/week)")
for rate in [60, 120, 150]:
    o, _, fx2, fr2 = design(rate=rate)
    print(f"freight rate {rate:>3} XAF/t-km -> open {o}, total {fx2 + fr2:.1f} M XAF/week")

# %% s3
need = pd.Series([6, 6, 6, 7, 9, 11, 8], index=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
days = list(need.index)
# pattern p = start day p, work 5 consecutive days, then 2 days off
covers = {p: [days[(p + k) % 7] for k in range(5)] for p in range(7)}
r = pulp.LpProblem("roster", pulp.LpMinimize)
n = {p: pulp.LpVariable(f"start_{days[p]}", lowBound=0, cat="Integer") for p in range(7)}
r += pulp.lpSum(n.values())
for d in days:
    r += pulp.lpSum(n[p] for p in range(7) if d in covers[p]) >= need[d], f"cover_{d}"
r.solve(pulp.PULP_CBC_CMD(msg=False))
plan = pd.Series({f"start {days[p]}": int(n[p].value()) for p in range(7)})
staffed = pd.Series({d: sum(int(n[p].value()) for p in range(7) if d in covers[p]) for d in days})
print("cashiers hired:", int(plan.sum()), "| lower bound ceil(sum need / 5) =", int(np.ceil(need.sum() / 5)))
print(plan[plan > 0].to_string())
print(pd.DataFrame({"need": need, "staffed": staffed, "surplus": staffed - need}).T.to_string())

# %% s4
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
sites = pd.DataFrame([   # approximate locations in Douala; demand in crates
    ("Akwa (depot)", 4.050, 9.700, 0), ("Bonapriso", 4.025, 9.695, 6), ("Bonanjo", 4.043, 9.688, 4),
    ("Deido", 4.065, 9.710, 8), ("Bonaberi", 4.075, 9.660, 9), ("Makepe", 4.073, 9.760, 7),
    ("Logpom", 4.087, 9.779, 5), ("Bonamoussadi", 4.093, 9.742, 10), ("Kotto", 4.070, 9.770, 6),
    ("Ndokoti", 4.045, 9.738, 8), ("Bepanda", 4.061, 9.729, 7), ("Logbaba", 4.035, 9.782, 5),
    ("Yassa", 4.010, 9.810, 6), ("New Bell", 4.038, 9.712, 9),
], columns=["site", "lat", "lon", "crates"])
D = np.array([[round(1000 * road_km(a, b, circuity=1.4)) for _, b in sites.iterrows()]
              for _, a in sites.iterrows()])                              # metres
VEH, CAPV = 3, 35

mgr = pywrapcp.RoutingIndexManager(len(sites), VEH, 0)
routing = pywrapcp.RoutingModel(mgr)
cb = routing.RegisterTransitCallback(lambda i, j: int(D[mgr.IndexToNode(i), mgr.IndexToNode(j)]))
routing.SetArcCostEvaluatorOfAllVehicles(cb)
dem = routing.RegisterUnaryTransitCallback(lambda i: int(sites.crates[mgr.IndexToNode(i)]))
routing.AddDimensionWithVehicleCapacity(dem, 0, [CAPV] * VEH, True, "Load")
prm = pywrapcp.DefaultRoutingSearchParameters()
prm.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
prm.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
prm.time_limit.seconds = 5
sol = routing.SolveWithParameters(prm)

routes, total = [], 0
for v in range(VEH):
    i, path, km, load = routing.Start(v), [], 0, 0
    while not routing.IsEnd(i):
        node = mgr.IndexToNode(i); path.append(node); load += sites.crates[node]
        j = sol.Value(routing.NextVar(i)); km += D[node, mgr.IndexToNode(j)] / 1000; i = j
    path.append(0); routes.append(path); total += km
    print(f"van {v + 1}: {load:2d} crates, {km:4.1f} km  " + " > ".join(sites.site[k].split(" ")[0] for k in path))
print(f"total distance {total:.1f} km")

fig, ax = plt.subplots(figsize=(5.6, 3.4))
for path, c in zip(routes, [NAVY, TEAL, ORANGE]):
    ax.plot(sites.lon[path], sites.lat[path], "-o", color=c, ms=4, lw=1.4)
ax.scatter(sites.lon[0], sites.lat[0], marker="s", s=80, color=RED, zorder=3)
for _, s in sites.iterrows():
    ax.annotate(s.site.split(" ")[0], (s.lon, s.lat), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_aspect("equal"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figs/lab5_vrp.pdf")
