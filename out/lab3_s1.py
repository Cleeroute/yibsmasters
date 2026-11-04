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
