cvw = pd.read_parquet("real/weekly_cv_forecasts.parquet")      # Lab 3 rolling-origin forecasts
cvw = cvw.sort_values(["unique_id", "ds"])
cut = sorted(cvw.cutoff.unique())
calib = cvw[cvw.cutoff.isin(cut[:4])]                            # Apr-Jul 2011: estimate errors
evalw = cvw[cvw.cutoff.isin(cut[4:])]                            # Aug-Nov 2011: run the policy

R_L = 4                                                          # review 1 week + lead time 3 weeks
sigma = (calib.y - calib.ADIDA).groupby(calib.unique_id).std()   # weekly error sd per SKU

def simulate_RS(demand, fc_week, ss, L=3):
    """Weekly review order-up-to with lead time L weeks and backorders."""
    on_hand, pipe, back, short, inv = fc_week[0] * R_L + ss, [0.0] * L, 0.0, 0.0, []
    for t, d in enumerate(demand):
        on_hand += pipe.pop(0)
        need = d + back; served = min(on_hand, need); on_hand -= served
        short += max(0.0, d - max(0.0, served - back)); back = need - served
        S = fc_week[t] * R_L + ss
        pipe.append(max(0.0, S - (on_hand - back + sum(pipe))))
        inv.append(on_hand)
    return short, np.mean(inv)

rows = []
for target in [0.90, 0.95, 0.99]:
    zt = norm.ppf(target); dem = short = stock = 0.0
    for sku, g in evalw.groupby("unique_id"):
        s_, inv_ = simulate_RS(g.y.values, g.ADIDA.values, zt * sigma[sku] * np.sqrt(R_L))
        dem += g.y.sum(); short += s_; stock += inv_
    rows.append({"target_CSL": target, "z": round(zt, 2), "fill_rate_%": round(100 * (1 - short / dem), 1),
                 "avg_units_on_hand": round(stock)})
print(pd.DataFrame(rows).to_string(index=False))
bias = 100 * (evalw.ADIDA - evalw.y).sum() / evalw.y.sum()
print(f"forecast bias in the evaluation period (Aug-Nov 2011): {bias:+.1f}%")
