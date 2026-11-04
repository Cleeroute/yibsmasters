import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression

emp = pd.read_csv("data/employees.csv")                   # Mboa staff; TRUE training effect = +3
naive = emp.groupby("trained").productivity.mean().diff().iloc[-1]
reg = smf.ols("productivity ~ trained + skill_score + tenure_years", data=emp).fit()
X = emp[["skill_score", "tenure_years"]]
ps = LogisticRegression(max_iter=1000).fit(X, emp.trained).predict_proba(X)[:, 1]
w = np.where(emp.trained == 1, 1 / ps, 1 / (1 - ps))
ipw = (np.average(emp.productivity[emp.trained == 1], weights=w[emp.trained == 1])
       - np.average(emp.productivity[emp.trained == 0], weights=w[emp.trained == 0]))
# doubly robust (AIPW): outcome model + propensity weights, trimmed to [0.05, 0.95]
T, Yv = emp.trained.values, emp.productivity.values
pt = np.clip(ps, 0.05, 0.95)
mu1 = smf.ols("productivity ~ skill_score + tenure_years", emp[emp.trained == 1]).fit().predict(emp)
mu0 = smf.ols("productivity ~ skill_score + tenure_years", emp[emp.trained == 0]).fit().predict(emp)
aipw = np.mean(mu1 - mu0 + T * (Yv - mu1) / pt - (1 - T) * (Yv - mu0) / (1 - pt))
print(f"naive difference      {naive:5.2f}")
print(f"regression adjustment {reg.params['trained']:5.2f}  (95% CI {reg.conf_int().loc['trained', 0]:.2f}"
      f" to {reg.conf_int().loc['trained', 1]:.2f})")
print(f"inverse-propensity    {ipw:5.2f}   <- extreme weights, max w = {w.max():.0f}")
print(f"doubly robust (AIPW)  {aipw:5.2f}")
print(f"propensity range: trained {ps[emp.trained == 1].min():.2f}-{ps[emp.trained == 1].max():.2f}, "
      f"untrained {ps[emp.trained == 0].min():.2f}-{ps[emp.trained == 0].max():.2f}")
