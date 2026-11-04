fc = rice.shift(1).rolling(28).mean()                      # daily forecast known at the start of day t
lt_demand = sum(rice.shift(-k) for k in range(L))          # demand during the next L days
lt_error = (lt_demand - L * fc)["2023-02":"2023-12"]       # honest lead-time forecast errors, 2023
z = norm.ppf(0.95)                                         # target cycle service level 95%
sd = {"lead-time demand (unconditional)": lt_demand["2023"].std(),
      "daily demand sd x sqrt(L)": rice["2023"].std() * np.sqrt(L),
      "lead-time forecast error": lt_error.std()}
for k, v in sd.items():
    print(f"{k:34s} sd = {v:5.1f}  -> safety stock {z * v:4.0f} units")

def simulate_sQ(demand, fc, ss, Q, L):
    """Daily continuous review (s, Q), backorders; reorder point s_t = L * forecast_t + ss."""
    on_hand, pipeline, back, short, inv, orders = Q + ss, [], 0.0, 0.0, [], 0
    for t, d in enumerate(demand):
        on_hand += sum(q for (arr, q) in pipeline if arr == t)
        pipeline = [(arr, q) for (arr, q) in pipeline if arr > t]
        need = back + d; served = min(on_hand, need); on_hand -= served
        short += max(0.0, d - max(0.0, served - back)); back = need - served
        if on_hand - back + sum(q for _, q in pipeline) <= L * fc[t] + ss:
            pipeline.append((t + L, Q)); orders += 1
        inv.append(on_hand)
    return orders, 100 * (1 - short / demand.sum()), np.mean(inv)

test, fc24 = rice["2024"].values, fc["2024"].values
rows = []
for q_name, q in [("EOQ", round(Q)), ("weekly", round(D / 52))]:
    for ss_name, ss in [("none", 0), ("error-based", z * sd["lead-time forecast error"])]:
        n, fill, inv = simulate_sQ(test, fc24, ss, q, L)
        rows.append({"Q": f"{q_name} ({q})", "safety_stock": ss_name, "orders": n,
                     "fill_rate_%": round(fill, 1), "avg_stock": round(inv),
                     "cost_kXAF": round((n * K + inv * h) / 1e3)})
print(pd.DataFrame(rows).to_string(index=False))
