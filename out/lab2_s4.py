def before(t, a, b):                    # if b occurs, a must have occurred earlier
    acts = t.split(" > ")
    return (b not in acts) or (a in acts and acts.index(a) < acts.index(b))

checks = pd.DataFrame({
    "reaches_T05_send": traces.str.contains("T05"),
    "T02_before_T04": traces.apply(before, a="T02", b="T04"),
    "T06_before_T10": traces.apply(before, a="T06", b="T10"),
    "no_repeated_task": traces.apply(lambda t: len(set(t.split(" > "))) == len(t.split(" > "))),
})
checks["all_rules"] = checks.all(axis=1)
print((100 * (1 - checks.mean())).round(1).rename("% cases violating").to_string())

case["late"] = case.end > case.deadline
print(f"\ncases finishing after their deadline: {case.late.mean():.1%}")
print(pd.DataFrame({"cases": checks.groupby("all_rules").size(),
                    "late_rate": case.groupby(checks.all_rules)["late"].mean().round(3),
                    "median_days": case.groupby(checks.all_rules)["days"].median().round(2)}).to_string())
