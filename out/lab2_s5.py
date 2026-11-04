win = case[(case.start >= "2011-01-01") & (case.start < "2011-11-01")]
lam = len(win) / (pd.Timestamp("2011-11-01") - pd.Timestamp("2011-01-01")).days   # cases per day
W = win["days"].mean()                                                              # days per case
days = pd.date_range("2011-03-01", "2011-10-31", freq="D", tz="UTC") + pd.Timedelta(hours=12)
wip = [((case.start <= d) & (case.end > d)).sum() for d in days]
print(f"lambda = {lam:.2f} cases/day   W = {W:.2f} days   lambda*W = {lam*W:.1f}")
print(f"measured average WIP (Mar-Oct 2011) = {np.mean(wip):.1f} open cases")
