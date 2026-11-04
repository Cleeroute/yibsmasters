import json
temp = pd.read_csv("real/machine_temperature_system_failure.csv", parse_dates=["timestamp"])
temp = temp.drop_duplicates("timestamp").set_index("timestamp")["value"].sort_index()
windows = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in
           json.load(open("real/combined_windows.json"))["realKnownCause/machine_temperature_system_failure.csv"]]
C_FALSE, C_MISS = 150_000, 8_000_000     # XAF: one inspection call-out; one missed failure

def evaluate(alarm, gap="6h"):
    """Group alarms into episodes; an episode overlapping a labelled window is a hit."""
    t = alarm[alarm].index
    if len(t) == 0:
        return {"detected": 0, "false_alarms": 0, "cost_mXAF": 4 * C_MISS / 1e6}
    new = np.r_[True, np.diff(t) > pd.Timedelta(gap)]
    starts, ends = t[new], t[np.r_[new[1:], True]]
    hit_win = [any((starts <= b) & (ends >= a)) for a, b in windows]
    false = sum(not any((s0 <= b) & (e0 >= a) for a, b in windows) for s0, e0 in zip(starts, ends))
    return {"detected": sum(hit_win), "false_alarms": false,
            "cost_mXAF": (false * C_FALSE + (4 - sum(hit_win)) * C_MISS) / 1e6}

# Detector A: adaptive robust z-score against the trailing 24 hours
med = temp.shift(1).rolling("24h", min_periods=200).median()
mad = (temp.shift(1) - med).abs().rolling("24h", min_periods=200).median()
z = ((temp - med) / (1.4826 * mad)).abs()
# Detector B: Shewhart-style chart on the 6-hour moving average, limits fixed in Phase I
ma = temp.rolling("6h").mean()
ref = ma[:"2013-12-09"]                              # Phase I: first week, assumed in control
mu, sd = ref.mean(), ref.std()

rows = []
for k in [2.5, 3, 3.5, 4]:
    rows.append({"detector": "A adaptive z", "k": k, **evaluate((z > 2 * k).rolling(6).sum() >= 4)})
    rows.append({"detector": "B fixed-limit MA", "k": k, **evaluate(ma < mu - k * sd)})
print(f"Phase I limits: mean {mu:.1f}, sd {sd:.1f}")
print(pd.DataFrame(rows).sort_values(["detector", "k"]).to_string(index=False))
