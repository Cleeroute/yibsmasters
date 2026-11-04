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
