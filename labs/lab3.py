import warnings; warnings.filterwarnings("ignore")
import os; os.environ["NIXTLA_ID_AS_COL"] = "1"
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
NAVY, TEAL, ORANGE, RED, GREY = "#14325C", "#0F7C84", "#D9731A", "#B3322F", "#5F6B7A"

# %% s1
import numpy as np, pandas as pd

def load_mboa_sales(path="data/sales.csv"):
    """Week 1 cleaning rules, packaged as one reusable function."""
    df = pd.read_csv(path, parse_dates=["date"]).drop_duplicates()
    df["store_id"] = df["store_id"].str.strip().str.upper()
    df = df[df.units.isna() | (df.units >= 0)].copy()
    g = df.groupby("product_id")["units"]
    q1, q3 = g.transform(lambda s: s.quantile(.25)), g.transform(lambda s: s.quantile(.75))
    df.loc[df.units > q3 + 3 * (q3 - q1), "units"] = np.nan          # keying errors -> missing
    df["units"] = df.groupby(["store_id", "product_id"])["units"].transform(lambda s: s.fillna(s.median()))
    return df

sales = load_mboa_sales()
Y = (sales.assign(unique_id=sales.store_id + "_" + sales.product_id)
          .rename(columns={"date": "ds", "units": "y"})[["unique_id", "ds", "y", "promo"]]
          .sort_values(["unique_id", "ds"]))
print(Y.shape, "|", Y.unique_id.nunique(), "daily series |", Y.ds.min().date(), "to", Y.ds.max().date())

# %% s2
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, WindowAverage, AutoETS

def scores(cv, models, y="y", season_scale=None):
    """WAPE and bias across all series; MASE uses each series' in-sample seasonal-naive MAE."""
    out = {}
    for m in models:
        e = cv[m] - cv[y]
        out[m] = {"WAPE_%": 100 * e.abs().sum() / cv[y].abs().sum(),
                  "bias_%": 100 * e.sum() / cv[y].sum()}
        if season_scale is not None:
            mae = e.abs().groupby(cv["unique_id"]).mean()
            out[m]["MASE"] = (mae / season_scale).mean()
    return pd.DataFrame(out).T.round(2).sort_values("WAPE_%")

H = 14                                                     # horizon = lead time + review period
sf = StatsForecast(models=[Naive(), SeasonalNaive(7), WindowAverage(28),
                           AutoETS(season_length=7)], freq="D", n_jobs=-1)
cv = sf.cross_validation(df=Y[["unique_id", "ds", "y"]], h=H, n_windows=6, step_size=H)
train = Y[Y.ds < cv.cutoff.min() + pd.Timedelta(days=1)]
snaive_mae = train.groupby("unique_id")["y"].apply(lambda s: s.diff(7).abs().mean())
print(scores(cv, ["Naive", "SeasonalNaive", "WindowAverage", "AutoETS"], season_scale=snaive_mae))

# %% s3
W = pd.read_parquet("real/weekly_demand_AXY.parquet")          # built in Lab 1
W = W.sort_values(["unique_id", "ds"])
from statsforecast.models import HistoricAverage, CrostonOptimized, ADIDA, SimpleExponentialSmoothingOptimized as SESOpt

HW, NW = 4, 8                                                   # 4-week horizon, 8 rolling origins
sfw = StatsForecast(models=[Naive(), SeasonalNaive(52), WindowAverage(8), SESOpt(),
                            CrostonOptimized(), ADIDA()], freq="W-MON", n_jobs=-1)
cvw = sfw.cross_validation(df=W, h=HW, n_windows=NW, step_size=HW)
print("test weeks:", cvw.ds.min().date(), "to", cvw.ds.max().date())
w_train = W[W.ds <= cvw.cutoff.min()]
scale_w = w_train.groupby("unique_id")["y"].apply(lambda s: s.diff(1).abs().mean())
stat_models = ["Naive", "SeasonalNaive", "WindowAverage", "SESOpt", "CrostonOptimized", "ADIDA"]
print(scores(cvw, stat_models, season_scale=scale_w))

# %% s4
import lightgbm as lgb

def make_features(df):
    """Only information available HW weeks before the target week (no leakage)."""
    df = df.sort_values(["unique_id", "ds"]).copy()
    g = df.groupby("unique_id")["y"]
    for L in [HW, HW + 1, HW + 2, HW + 4, 52]:
        df[f"lag{L}"] = g.shift(L)
    df["mean8"] = g.transform(lambda s: s.shift(HW).rolling(8).mean())
    df["mean26"] = g.transform(lambda s: s.shift(HW).rolling(26).mean())
    df["woy"] = df.ds.dt.isocalendar().week.astype(int)
    df["item"] = df.unique_id.astype("category")
    return df

F = make_features(W)
feats = [c for c in F.columns if c not in ("unique_id", "ds", "y")]
preds = []
for cutoff in sorted(cvw.cutoff.unique()):
    tr = F[(F.ds <= cutoff)].dropna(subset=["lag52"])
    te = F[(F.ds > cutoff) & (F.ds <= cutoff + pd.Timedelta(weeks=HW))]
    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
                              min_child_samples=20, objective="tweedie", verbose=-1, random_state=0)
    model.fit(tr[feats], tr["y"])
    preds.append(te[["unique_id", "ds"]].assign(LightGBM=model.predict(te[feats]), cutoff=cutoff))
cvw = cvw.merge(pd.concat(preds), on=["unique_id", "ds", "cutoff"], how="left")
print(scores(cvw, stat_models + ["LightGBM"], season_scale=scale_w).head(4))

# %% s5
from statsforecast.utils import ConformalIntervals
best = "WindowAverage"
sfp = StatsForecast(models=[WindowAverage(8, prediction_intervals=ConformalIntervals(h=HW, n_windows=6))],
                    freq="W-MON", n_jobs=-1)
cvp = sfp.cross_validation(df=W, h=HW, n_windows=NW, step_size=HW, level=[80, 95])
for lv in [80, 95]:
    inside = (cvp.y >= cvp[f"WindowAverage-lo-{lv}"]) & (cvp.y <= cvp[f"WindowAverage-hi-{lv}"])
    print(f"nominal {lv}% interval -> empirical coverage {inside.mean():.1%}")

cvw.to_parquet("real/weekly_cv_forecasts.parquet", index=False)     # errors feed Lab 4
cv.to_parquet("data/daily_cv_forecasts.parquet", index=False)

sku = cvw.groupby("unique_id")["y"].sum().idxmax()
hist = W[(W.unique_id == sku) & (W.ds >= "2011-01-01")]
f = cvp[cvp.unique_id == sku]
fig, ax = plt.subplots(figsize=(6.4, 2.5))
ax.plot(hist.ds, hist.y, color=NAVY, lw=1.2, label=f"actual ({sku})")
ax.plot(f.ds, f.WindowAverage, color=ORANGE, lw=1.2, label="4-week-ahead forecast")
ax.fill_between(f.ds, f["WindowAverage-lo-80"], f["WindowAverage-hi-80"], color=ORANGE, alpha=.2, label="80% interval")
ax.set_ylabel("units per week"); ax.legend(fontsize=7, frameon=False, loc="upper left"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figs/lab3_sku.pdf")
