profile = con.sql("""
    SELECT count(*)                                  AS rows,
           count(DISTINCT Invoice)                   AS invoices,
           count(DISTINCT StockCode)                 AS stock_codes,
           min(InvoiceDate)::DATE                    AS first_day,
           max(InvoiceDate)::DATE                    AS last_day,
           count(*) FILTER (WHERE Invoice LIKE 'C%')             AS cancel_lines,
           count(*) FILTER (WHERE Quantity < 0)                  AS negative_qty,
           count(*) FILTER (WHERE Price <= 0)                    AS zero_or_neg_price,
           count(*) FILTER (WHERE "Customer ID" IS NULL)         AS no_customer
    FROM raw""").df()
for k, v in profile.iloc[0].items():
    print(f"{k:18s} {v}")
