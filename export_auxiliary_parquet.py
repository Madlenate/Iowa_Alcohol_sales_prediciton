"""Export the small auxiliary DuckDB tables to Parquet for version control / sharing.

`sales` already gets exported by export_parquet.py, but the smaller tables
(unemployment, homelessness, county_population, alcohol_policy) were never
exported anywhere -- they only ever existed inside iowa_liquor.duckdb, which is
git-ignored. That's fine locally, but it means any deployment (e.g. Streamlit
Community Cloud) that only has the git repo -- not the local .duckdb file --
can query `sales` via the Parquet fallback in query_db.py but gets a "table
does not exist" error the moment it touches any of these tables.

These tables are tiny (tens to low thousands of rows), so each gets a single
Parquet file, no partitioning needed.

    python export_auxiliary_parquet.py
"""

from pathlib import Path

import duckdb

PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = str(PROJECT_DIR / "iowa_liquor.duckdb")
OUT_DIR = PROJECT_DIR / "parquet" / "auxiliary"

TABLES = ["unemployment", "homelessness", "county_population", "alcohol_policy"]

if not Path(DB_PATH).exists():
    raise SystemExit(f"{DB_PATH} not found - run Data_Base_creation.py first.")

OUT_DIR.mkdir(parents=True, exist_ok=True)
con = duckdb.connect(DB_PATH, read_only=True)

for table in TABLES:
    out_path = str(OUT_DIR / f"{table}.parquet")
    con.execute(
        f"COPY (SELECT * FROM {table}) TO ? (FORMAT parquet, COMPRESSION zstd)",
        [out_path],
    )
    rows = con.execute(f"SELECT COUNT(*) FROM read_parquet(?)", [out_path]).fetchone()[0]
    size_kb = Path(out_path).stat().st_size / 1e3
    print(f"  {size_kb:8.1f} KB  {rows:6,} rows  -> {Path(out_path).relative_to(PROJECT_DIR)}")

con.close()
