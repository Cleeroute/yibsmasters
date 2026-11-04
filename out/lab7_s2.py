margin, cost_per_email = 0.35, 0.10                       # gross margin on spend; $ per e-mail sent
for arm in ["Mens E-Mail", "Womens E-Mail"]:
    r = res[(res.arm == arm) & (res.outcome == "spend")].iloc[0]
    profit = margin * r.lift - cost_per_email
    lo, hi = margin * r.ci_low - cost_per_email, margin * r.ci_high - cost_per_email
    print(f"{arm:14s} incremental profit per e-mail: ${profit:.3f}  (95% CI {lo:.3f} to {hi:.3f})")

# follow-up test: customers per arm to detect +0.3 points of conversion with 80% power
p0, delta = ctrl.conversion.mean(), 0.003
za, zb = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
n_arm = 2 * (za + zb) ** 2 * p0 * (1 - p0) / delta ** 2
print(f"baseline conversion {p0:.4f}; customers needed per arm: {n_arm:,.0f}")
