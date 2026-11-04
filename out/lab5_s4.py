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
