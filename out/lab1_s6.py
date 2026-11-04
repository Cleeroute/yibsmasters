focus = abc.query("abc == 'A' and xyz in ['X', 'Y'] and weeks_alive >= 100")["StockCode"]
weekly_focus = (grid[focus].stack().rename("units").reset_index()
                .rename(columns={"StockCode": "unique_id", "week": "ds", "units": "y"}))
weekly_focus.to_parquet("real/weekly_demand_AXY.parquet", index=False)
abc.to_csv("real/sku_abc_xyz.csv", index=False)
print(weekly_focus.shape, "| series:", weekly_focus.unique_id.nunique(),
      "| weeks:", weekly_focus.ds.nunique(), weekly_focus.ds.min(), "to", weekly_focus.ds.max())
