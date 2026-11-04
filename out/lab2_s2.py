main = log[log.case_id.isin(traces[traces.isin(variants.index[:8])].index)]
dfg, start_acts, end_acts = pm4py.discover_dfg(main, case_id_key="case_id",
                                               activity_key="activity", timestamp_key="ts")
pm4py.save_vis_dfg(dfg, start_acts, end_acts, "figs/lab2_dfg.png", rankdir="LR")
for (a, b), n in sorted(dfg.items(), key=lambda kv: -kv[1])[:6]:
    print(f"{a:>24s} -> {b:<6s} {n:5d}")
