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
