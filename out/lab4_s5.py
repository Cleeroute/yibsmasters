rng = np.random.default_rng(1)
T, L, m = 300, 2, 8                                   # weeks, lead time per stage, MA window
customer = np.maximum(0, rng.normal(100, 10, T))
stages = ["Retailer", "Wholesaler", "Distributor", "Factory"]

def chain(shared_info):
    incoming, orders = customer, {}
    for s in stages:
        signal = customer if shared_info else incoming   # what this stage forecasts from
        q = np.zeros(T)
        for t in range(m, T):
            mu_now = signal[t - m + 1:t + 1].mean(); mu_prev = signal[t - m:t].mean()
            # order-up-to level S_t = (L+1) * mean; order = demand + change in S
            q[t] = max(0, incoming[t] + (L + 1) * (mu_now - mu_prev))
        orders[s] = q[m * 5:]; incoming = q
    return orders

base = np.var(customer[m * 5:])
for label, shared in [("local forecasts", False), ("shared POS data", True)]:
    o = chain(shared)
    print(f"{label:16s}", "  ".join(f"{s}: {np.var(o[s]) / base:5.1f}x" for s in stages))
print(f"theory, one stage (Chen et al.): >= {1 + 2 * L / m + 2 * L**2 / m**2:.2f}x")

o = chain(False)
fig, ax = plt.subplots(figsize=(6.4, 2.4))
ax.plot(customer[m * 5:m * 5 + 80], color=GREY, lw=1.2, label="customer demand")
for s, c in zip(stages, [NAVY, TEAL, ORANGE, RED]):
    ax.plot(o[s][:80], color=c, lw=1, label=s)
ax.set_xlabel("week"); ax.set_ylabel("units ordered"); ax.legend(fontsize=7, ncol=5, frameon=False, loc="upper left")
ax.grid(alpha=.3); fig.tight_layout(); fig.savefig("figs/lab4_bullwhip.pdf")
