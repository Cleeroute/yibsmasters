case = log.groupby("case_id").agg(start=("ts", "min"), end=("ts", "max"),
                                  events=("ts", "size"), deadline=("case:deadline", "first"))
case["days"] = (case.end - case.start).dt.total_seconds() / 86400
print(case["days"].describe(percentiles=[.5, .75, .9, .95]).round(2).to_string())

log["next_act"] = log.groupby("case_id")["activity"].shift(-1)
log["wait_h"] = (log.groupby("case_id")["ts"].shift(-1) - log["ts"]).dt.total_seconds() / 3600
hand = (log.dropna(subset=["next_act"]).groupby(["activity", "next_act"])["wait_h"]
          .agg(n="size", median_h="median", mean_h="mean", total_days=lambda x: x.sum() / 24)
          .sort_values("total_days", ascending=False).round(1))
print(hand.head(6).to_string())
