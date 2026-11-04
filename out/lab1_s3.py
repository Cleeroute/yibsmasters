con.sql(r"""
CREATE OR REPLACE VIEW flagged AS
SELECT *, Quantity * Price AS line_value,
  CASE
    WHEN Invoice LIKE 'C%'                                   THEN '1 cancellation'
    WHEN NOT regexp_matches(StockCode, '^\d{5}[A-Za-z]*$')   THEN '2 non-product code'
    WHEN Quantity <= 0                                       THEN '3 stock write-off'
    WHEN Price <= 0                                          THEN '4 zero price'
    ELSE '0 valid sale'
  END AS dq_class
FROM raw""")
audit = con.sql("""
    SELECT dq_class, count(*) AS lines,
           round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct_lines,
           round(sum(line_value) / 1e3, 1)                     AS value_k_gbp
    FROM flagged GROUP BY dq_class ORDER BY dq_class""").df()
print(audit.to_string(index=False))

dups = con.sql("""SELECT count(*) - count(DISTINCT (Invoice, StockCode, Quantity, InvoiceDate, Price))
                  AS repeated_lines FROM flagged WHERE dq_class = '0 valid sale'""").fetchall()[0][0]
print("identical repeated lines among valid sales:", dups)
