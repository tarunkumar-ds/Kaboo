# Sales Analysis Report: example_sales.csv

Generated: 2026-10-09T23:18:42

## Workflow summary
1. Loaded and validated the CSV.
2. Profiled missing values and numeric fields.
3. Compared sales across detected region/product columns where available.
4. Flagged possible sales outliers using the 1.5×IQR rule.
5. Saved this report and the anomaly rows locally.

## Dataset overview
- Rows: 10
- Columns (9): order_id, date, product, category, quantity, unit_price, sales, region, customer_rating
- Missing-value cells: 1
- Columns with missing values: {"customer_rating": 1}

## Sales findings
- Sales column: `sales`
- Total sales: 271500.0
- Average sales per row: 27150.0
- Top region: Chennai (118000.0)
- Top product/category: Laptop (220000.0)

## Regional sales
| Region | Sales |
|---|---:|
| Chennai | 118000.0 |
| Mumbai | 111500.0 |
| Hyderabad | 36000.0 |
| Bengaluru | 6000.0 |

## Product/category sales
| Product/category | Sales |
|---|---:|
| Laptop | 220000.0 |
| Monitor | 36000.0 |
| Mouse | 6000.0 |
| Headphones | 5000.0 |
| Keyboard | 4500.0 |

## Numeric profile
| Column | Sum | Mean | Min | Max |
|---|---:|---:|---:|---:|
| order_id | 10055.0 | 1005.5 | 1001.0 | 1010.0 |
| quantity | 20.0 | 2.0 | 1.0 | 5.0 |
| unit_price | 196000.0 | 19600.0 | 750.0 | 55000.0 |
| sales | 271500.0 | 27150.0 | 1500.0 | 110000.0 |
| customer_rating | 39.0 | 4.3333 | 3.0 | 5.0 |

## Potential anomalies
- Rows flagged by 1.5×IQR: 0
- This is a statistical flag for review, not proof of an error.
