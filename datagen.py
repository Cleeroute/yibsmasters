"""
datagen.py -- Synthetic data for BUSINESS PROCESS ANALYTICS (YIBS, MTech 2), 2nd edition.
Creates the Mboa Fresh Market case files. Fully synthetic, reproducible (seed = 42).
Run once:  python datagen.py        (writes CSV files into ./data)

The real public datasets used alongside this case are fetched by get_data.py.
Planted truths you will try to recover later: loyalty-app uplift +8% (Douala, from 1 July 2024)
and training effect +3 productivity points (Week 7).
"""
import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=DeprecationWarning)

SEED = 42
rng = np.random.default_rng(SEED)
OUT = "data"
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ master tables
stores = pd.DataFrame({
    "store_id": ["S1", "S2", "S3", "S4", "S5", "S6"],
    "store_name": ["Yaounde Centre", "Yaounde Mvan", "Yaounde Bastos",
                   "Douala Akwa", "Douala Bonamoussadi", "Bafoussam Marche A"],
    "city": ["Yaounde"] * 3 + ["Douala"] * 2 + ["Bafoussam"],
    "size_m2": [900, 650, 720, 1000, 800, 540],
})
products = pd.DataFrame({
    "product_id": [f"P{i:02d}" for i in range(1, 13)],
    "product_name": ["Rice 5kg", "Cooking oil 1L", "Sugar 1kg", "Tomato paste", "Water 1.5L",
                     "Milk powder", "Bread loaf", "Eggs tray", "Soap bar", "Detergent 1kg",
                     "Spaghetti 500g", "Sardines tin"],
    "category": ["Staples", "Staples", "Staples", "Canned", "Beverages", "Dairy", "Fresh",
                 "Fresh", "Household", "Household", "Staples", "Canned"],
    "unit_cost": [2800, 1100, 700, 350, 250, 1900, 200, 1600, 150, 900, 450, 400],
})
products["unit_price"] = (products["unit_cost"] * rng.uniform(1.15, 1.35, 12)).round(-1).astype(int)
# replenishment master data (deterministic, used from Week 4)
products["lead_time_days"] = [7, 7, 7, 10, 3, 14, 1, 2, 10, 10, 7, 10]
products["order_cost_xaf"] = [15000] * 6 + [3000, 5000] + [15000] * 4
products["shelf_life_days"] = [365, 270, 365, 540, 180, 365, 1, 21, 720, 720, 540, 720]
base_demand = np.array([14, 30, 25, 40, 55, 12, 60, 18, 35, 16, 28, 33], dtype=float)
store_factor = np.array([1.25, 0.90, 1.00, 1.35, 1.05, 0.70])

# ------------------------------------------------------------------ daily sales
dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
t = np.arange(len(dates))
dow = dates.dayofweek.values
month = dates.month.values
weekly = np.array([0.90, 0.88, 0.92, 0.97, 1.15, 1.30, 1.10])[dow]          # Mon..Sun
yearly = 1 + 0.12 * np.sin(2 * np.pi * (month - 9) / 12) + np.where(month == 12, 0.18, 0)
trend = 1 + 0.0004 * t
rows = []
for si, s in enumerate(stores.store_id):
    for pi, p in enumerate(products.product_id):
        promo = (rng.random(len(dates)) < 0.08).astype(int)                  # 8% of days on promotion
        price = products.unit_price[pi] * (1 - 0.10 * promo)
        mu = base_demand[pi] * store_factor[si] * weekly * yearly * trend * (1 + 0.35 * promo)
        if s in ("S4", "S5"):                                                # loyalty app, Douala: +8%
            mu = mu * np.where(dates >= "2024-07-01", 1.08, 1.0)
        lam = rng.gamma(shape=12, scale=mu / 12)                              # overdispersion
        units = rng.poisson(lam)
        rows.append(pd.DataFrame({"date": dates, "store_id": s, "product_id": p,
                                  "units": units, "price": price.round(0), "promo": promo}))
sales = pd.concat(rows, ignore_index=True)
sales["month"] = sales["date"].dt.strftime("%Y-%m")

# -- realistic data-quality problems (for the cleaning exercises)
n = len(sales)
sales.loc[rng.choice(n, int(0.005 * n), replace=False), "units"] = np.nan          # missing units
sales.loc[rng.choice(n, 25, replace=False), "units"] *= 40                          # keying errors
sales.loc[rng.choice(n, 30, replace=False), "units"] = -1                           # returns as -1
sales.loc[rng.choice(n, 400, replace=False), "store_id"] = sales["store_id"].str.lower()
sales = pd.concat([sales, sales.sample(40, random_state=1)], ignore_index=True)     # duplicates
sales = sales.sample(frac=1, random_state=7).sort_values(["date", "store_id", "product_id"]).reset_index(drop=True)
sales["date"] = sales["date"].dt.strftime("%Y-%m-%d")

# ------------------------------------------------------------------ process-mining event log
acts = ["Application Submitted", "Documents Check", "Fee Payment",
        "Registration Approval", "Course Enrolment", "Student Card Issued"]
res_pool = {"Documents Check": ["Clerk_A", "Clerk_B"], "Fee Payment": ["Bursar_1", "Bursar_2"],
            "Registration Approval": ["Registrar"], "Course Enrolment": ["Dept_Officer_1", "Dept_Officer_2"],
            "Student Card Issued": ["ID_Office"], "Application Submitted": ["Portal"]}
events = []
n_cases = 1500
starts = pd.Timestamp("2024-09-02 08:00") + pd.to_timedelta(
    np.sort(rng.uniform(0, 60 * 24 * 60, n_cases)), unit="m")
for c in range(n_cases):
    ts = starts[c]
    case = f"C{c+1:04d}"
    seq = ["Application Submitted", "Documents Check"]
    if rng.random() < 0.22:                                                   # rework
        seq += ["Documents Check"]
    seq += ["Fee Payment", "Registration Approval", "Course Enrolment"]
    if rng.random() < 0.90:
        seq += ["Student Card Issued"]
    if rng.random() < 0.04:                                                   # deviation
        i, j = seq.index("Fee Payment"), seq.index("Course Enrolment")
        seq[i], seq[j] = seq[j], seq[i]
    for k, a in enumerate(seq):
        if k > 0:
            wait_mean = {"Documents Check": 90, "Fee Payment": 240, "Registration Approval": 600,
                         "Course Enrolment": 120, "Student Card Issued": 300}[a]
            if a == "Fee Payment" and ts.dayofweek == 0:
                wait_mean *= 1.8                                              # Monday congestion
            ts = ts + pd.Timedelta(minutes=float(rng.exponential(wait_mean)))
        events.append((case, a, ts, rng.choice(res_pool[a])))
        ts = ts + pd.Timedelta(minutes=float(rng.uniform(3, 15)))
event_log = pd.DataFrame(events, columns=["case_id", "activity", "timestamp", "resource"])
event_log = event_log.sort_values(["case_id", "timestamp"]).reset_index(drop=True)

# ------------------------------------------------------------------ clinic arrivals (queueing exercises)
hour_rate = np.array([1, 1, 1, 1, 1, 2, 4, 8, 11, 11, 10, 9, 7, 8, 9, 9, 8, 6, 4, 3, 2, 2, 1, 1], dtype=float)
arr = []
for d in range(30):
    day = pd.Timestamp("2024-10-01") + pd.Timedelta(days=d)
    for h in range(7, 19):
        k = rng.poisson(hour_rate[h])
        arr += list(day + pd.Timedelta(hours=h) + pd.to_timedelta(rng.uniform(0, 60, k), unit="m"))
clinic = pd.DataFrame({"arrival_time": sorted(arr)})
clinic["service_min"] = rng.gamma(shape=4, scale=5.5, size=len(clinic)).round(1)  # mean ~22 min
clinic["patient_id"] = [f"PT{i+1:05d}" for i in range(len(clinic))]

# ------------------------------------------------------------------ employees (causal inference)
ne = 600
skill = rng.normal(50, 10, ne)
p_train = 1 / (1 + np.exp(-(skill - 50) / 6))            # stronger staff are more likely to be trained
train = (rng.random(ne) < p_train).astype(int)
tenure = rng.integers(1, 15, ne)
prod = 20 + 0.8 * skill + 0.5 * tenure + 3.0 * train + rng.normal(0, 4, ne)   # TRUE effect = +3
employees = pd.DataFrame({"employee_id": [f"E{i+1:03d}" for i in range(ne)], "skill_score": skill.round(1),
                          "tenure_years": tenure, "trained": train, "productivity": prod.round(2)})

# ------------------------------------------------------------------ save
for name, df in [("stores", stores), ("products", products), ("sales", sales),
                 ("event_log", event_log), ("clinic", clinic), ("employees", employees)]:
    df.to_csv(f"{OUT}/{name}.csv", index=False)
    print(f"{name:10s} {df.shape[0]:>7,d} rows x {df.shape[1]} cols")
