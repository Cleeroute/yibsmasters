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
