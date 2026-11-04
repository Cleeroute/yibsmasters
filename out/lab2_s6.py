case["slow"] = case["days"] > 7                                   # service standard: 7 days
wk = case.set_index("start").resample("W")["slow"].agg(["sum", "size"])
wk = wk[(wk.index >= "2010-11-01") & (wk.index < "2011-12-31") & (wk["size"] >= 5)]
wk["p"] = wk["sum"] / wk["size"]
pbar = wk["sum"].sum() / wk["size"].sum()
wk["ucl"] = pbar + 3 * np.sqrt(pbar * (1 - pbar) / wk["size"])
wk["lcl"] = (pbar - 3 * np.sqrt(pbar * (1 - pbar) / wk["size"])).clip(lower=0)
out = wk[(wk.p > wk.ucl) | (wk.p < wk.lcl)]
print(f"centre line p-bar = {pbar:.3f};  weeks plotted = {len(wk)};  signals = {len(out)}")
print(out[["sum", "size", "p", "ucl"]].round(3).to_string())

fig, ax = plt.subplots(figsize=(6.4, 2.5))
ax.plot(wk.index, wk.p, marker="o", ms=3, color=NAVY, lw=1, label="weekly share > 7 days")
ax.step(wk.index, wk.ucl, where="mid", color=RED, lw=1, label="UCL / LCL")
ax.step(wk.index, wk.lcl, where="mid", color=RED, lw=1)
ax.axhline(pbar, color=GREY, ls="--", lw=1, label=r"$\bar p$")
ax.scatter(out.index, out.p, color=ORANGE, zorder=3, s=30, label="special cause")
ax.set_ylabel("share of cases"); ax.legend(fontsize=7, ncol=4, loc="upper left", frameon=False)
ax.set_ylim(0, max(0.9, wk.ucl.max() + .05)); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figs/lab2_pchart.pdf")
