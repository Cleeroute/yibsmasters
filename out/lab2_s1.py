import pandas as pd, numpy as np
import pm4py

log = pm4py.read_xes("real/receipt.xes")                     # real log, 4TU.ResearchData
log = log.rename(columns={"case:concept:name": "case_id", "concept:name": "activity",
                          "time:timestamp": "ts", "org:resource": "resource"})
log["activity"] = log["activity"].str.replace(r"^(T\d+(-\d)?) .*", r"\1", regex=True)
log = log.sort_values(["case_id", "ts"])
print(f"{log.case_id.nunique():,} cases | {len(log):,} events | {log.activity.nunique()} activities"
      f" | {log.resource.nunique()} resources | {log.ts.min():%Y-%m-%d} to {log.ts.max():%Y-%m-%d}")

traces = log.groupby("case_id")["activity"].agg(" > ".join)
variants = traces.value_counts()
top = pd.DataFrame({"cases": variants, "share": (variants / variants.sum()).round(3),
                    "cum": (variants.cumsum() / variants.sum()).round(3)}).head(5)
top.index = [v if len(v) < 60 else v[:57] + "..." for v in top.index]
print(f"{len(variants)} distinct variants; top 5:")
print(top.to_string())
