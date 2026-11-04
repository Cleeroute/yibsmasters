bread = sales.query("store_id == 'S1' and product_id == 'P07'").set_index("date")["units"].asfreq("D")
b = prod.loc["P07"]
price, cost, salvage = b.unit_price, b.unit_cost, 50          # day-old bread sold at 50 XAF
cu, co = price - cost, cost - salvage
cr = cu / (cu + co)
print(f"underage cu = {cu} XAF, overage co = {co} XAF, critical ratio = {cr:.3f}")

# forecast = trailing 28-day level x weekday index (known the evening before)
wd = bread["2023"].groupby(bread["2023"].index.dayofweek).mean()
idx = bread.index.dayofweek.map(wd / wd.mean()).values
fcb = (bread / idx).shift(1).rolling(28).mean() * idx
ratio = (bread / fcb)["2023-02":"2023-12"]                    # calibration: actual / forecast
adj = ratio.quantile(cr)
print(f"quantile of actual/forecast at the critical ratio: {adj:.3f}")

def profit(order, demand):
    sold = np.minimum(order, demand)
    return (price * sold + salvage * (order - sold) - cost * order).sum()

d24, f24 = bread["2024"], fcb["2024"]
plans = {"order = mean forecast": f24.round(),
         "order = forecast x quantile adj.": (f24 * adj).round(),
         "order = 2023 average (fixed)": pd.Series(round(bread["2023"].mean()), index=d24.index)}
rows = []
for name, q in plans.items():
    rows.append({"plan": name, "profit_kXAF": round(profit(q, d24) / 1e3),
                 "in_stock_days_%": round(100 * (q >= d24).mean(), 1),
                 "waste_loaves": int(np.maximum(q - d24, 0).sum())})
print(pd.DataFrame(rows).to_string(index=False))
