import os
import duckdb
import pandas as pd

RAW_XLSX = "real/online_retail_II.xlsx"           # 45 MB, two sheets (2009-10, 2010-11)
RAW_PQ = "real/online_retail_II.parquet"          # 7 MB after conversion

if not os.path.exists(RAW_PQ):                    # convert once: Excel is slow to read
    sheets = pd.read_excel(RAW_XLSX, sheet_name=None,
                           dtype={"Invoice": str, "StockCode": str, "Description": str})
    pd.concat(sheets.values(), ignore_index=True).to_parquet(RAW_PQ, index=False)

con = duckdb.connect()                            # in-memory database, no server
con.sql(f"CREATE OR REPLACE VIEW raw AS SELECT * FROM '{RAW_PQ}'")
print(con.sql("SELECT column_name, column_type FROM (DESCRIBE raw)").df().to_string(index=False))
