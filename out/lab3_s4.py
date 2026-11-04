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
