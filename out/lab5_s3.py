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
