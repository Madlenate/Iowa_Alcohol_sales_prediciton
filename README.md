# Iowa Liquor Sales

Class of Iowa wholesale liquor purchase transactions, 2012–2026, combined into a
single queryable table (`sales`, ~36.4 million rows).

## What's in the repo

| Path | Description |
|---|---|
| `parquet/by_year/sale_year=YYYY/data_0.parquet` | The dataset, one Parquet file per year (Hive-partitioned, zstd). ~544 MB total. |
| `query_db.py` | Query helper. Works off the Parquet files directly — no database file needed. |
| `Data_Base_creation.py` | Rebuilds `iowa_liquor.duckdb` from the raw CSVs (CSVs not included). |
| `export_parquet.py` | Regenerates `parquet/` from `iowa_liquor.duckdb`. |
| `Project_1_analysis.ipynb` | Analysis notebook. |
| `requirements.txt` | Python dependencies. |

The raw CSVs (`Iowa_data_set/`, ~7 GB) and the built database (`iowa_liquor.duckdb`,
~1.3 GB) are git-ignored. You don't need them — everything reads from `parquet/`.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Command line:

```bash
python query_db.py "SELECT sale_year, SUM(sales_dollars) AS total FROM sales GROUP BY 1 ORDER BY 1"
python query_db.py                     # interactive SQL prompt
python query_db.py --to out.csv "SELECT * FROM sales WHERE store_city = 'AMES'"
```

Notebook / Python:

```python
from query_db import q

q("SELECT category_name, SUM(sales_dollars) AS sales FROM sales GROUP BY 1 ORDER BY sales DESC LIMIT 10")
```

Plain DuckDB, if you'd rather not use the helper:

```python
import duckdb
con = duckdb.connect()
con.execute("""
    CREATE VIEW sales AS
    SELECT * FROM read_parquet('parquet/by_year/**/*.parquet', hive_partitioning = true)
""")
```

## Columns

`invoice_id, ordered_on, store_no, store_name, store_address, store_city,
store_zip_code, county_fips_code, county_name, category_code, category_name,
vendor_number, vendor_name, item_no, im_desc, pack, bottle_volume_ml,
state_bottle_cost, state_bottle_retail, sales_bottles, sales_dollars,
sales_liters, sales_gallons, sale_year, source_file`

Numeric columns were cast with `try_cast`, so malformed source values are `NULL`
rather than errors. `state_bottle_cost` / `state_bottle_retail` are `NULL` for
2012–2018 (absent in those source files).
