kpi = con.sql("""
WITH m AS (
  SELECT date_trunc('month', InvoiceDate)::DATE AS month,
         sum(line_value) FILTER (WHERE dq_class = '0 valid sale')            AS gross_sales,
         -sum(line_value) FILTER (WHERE dq_class = '1 cancellation')         AS cancelled,
         count(DISTINCT Invoice) FILTER (WHERE dq_class = '0 valid sale')    AS orders
  FROM flagged
  WHERE InvoiceDate < '2011-12-01'                      -- drop the incomplete last month
  GROUP BY 1)
SELECT month,
       round(gross_sales / 1e3)                                   AS sales_k,
       orders,
       round(gross_sales / orders)                                AS avg_order_gbp,
       round(100 * cancelled / gross_sales, 1)                    AS cancel_pct,
       round(100 * (gross_sales / lag(gross_sales, 12) OVER (ORDER BY month) - 1), 1) AS yoy_pct
FROM m ORDER BY month""").df()
print(kpi.tail(8).to_string(index=False))
