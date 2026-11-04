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
