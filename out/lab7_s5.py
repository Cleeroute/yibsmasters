import json
proposal = json.loads("""{
 "store": "S1", "budget_xaf": 3000000,
 "lines": [
  {"product_id": "P01", "qty": 300, "unit": "units", "reason": "December peak, 2 weeks cover"},
  {"product_id": "P02", "qty": 40,  "unit": "cases", "reason": "cases of 12"},
  {"product_id": "P13", "qty": 200, "unit": "units", "reason": "new yoghurt line"},
  {"product_id": "P07", "qty": 900, "unit": "units", "reason": "bread for the week"},
  {"product_id": "P06", "qty": -20, "unit": "units", "reason": "return excess"}
 ]}""")                                                   # e.g. produced by an LLM assistant

prod = pd.read_csv("data/products.csv").set_index("product_id")
CASE = {"P02": 12}                                         # pack sizes from master data

def validate(plan, max_days_cover=21, daily_mu=None):
    issues, total = [], 0
    for ln in plan["lines"]:
        pid, q = ln["product_id"], ln["qty"]
        if pid not in prod.index:
            issues.append((pid, "unknown product: not in master data")); continue
        if ln["unit"] == "cases":
            q = q * CASE.get(pid, np.nan)
            if np.isnan(q):
                issues.append((pid, "unit 'cases' without a known pack size")); continue
        if q <= 0:
            issues.append((pid, "non-positive order quantity")); continue
        if prod.loc[pid, "shelf_life_days"] < 7 and q > 2 * daily_mu[pid]:
            issues.append((pid, f"perishable: {q:.0f} units > 2 days of demand ({2 * daily_mu[pid]:.0f})"))
        if q > max_days_cover * daily_mu[pid]:
            issues.append((pid, f"{q:.0f} units exceeds {max_days_cover} days of cover"))
        total += q * prod.loc[pid, "unit_cost"]
    if total > plan["budget_xaf"]:
        issues.append(("ALL", f"cost {total:,.0f} XAF exceeds budget {plan['budget_xaf']:,}"))
    return issues, total

sales = pd.read_csv("data/sales.csv")
s1 = sales[(sales.store_id.str.upper() == "S1") & (sales.units >= 0) & (sales.units < 500)]
daily_mu = s1[s1.date >= "2024-10-01"].groupby("product_id").units.mean()
issues, total = validate(proposal, daily_mu=daily_mu)
print(f"valid lines cost {total:,.0f} XAF; {len(issues)} issues found:")
for pid, msg in issues:
    print(f"  [{pid}] {msg}")
