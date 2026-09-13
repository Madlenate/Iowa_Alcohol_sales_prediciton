"""Export the DuckDB `sales` table to Parquet for version control / sharing.

Writes one Parquet file per year under parquet/by_year/ (Hive-partitioned,
zstd-compressed). Each file stays well under GitHub's 100 MB per-file limit.

    python export_parquet.py
"""

from pathlib import Path

import duckdb

PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = str(PROJECT_DIR / "iowa_liquor.duckdb")
OUT_DIR = str(PROJECT_DIR / "parquet" / "by_year")

if not Path(DB_PATH).exists():
    raise SystemExit(f"{DB_PATH} not found - run Data_Base_creation.py first.")

con = duckdb.connect(DB_PATH, read_only=True)

con.execute(
    """
    COPY (SELECT * FROM sales)
    TO ?
    (FORMAT parquet,
     PARTITION_BY (sale_year),
     COMPRESSION zstd,
     OVERWRITE_OR_IGNORE);
    """,
    [OUT_DIR],
)

rows = con.execute(
    "SELECT COUNT(*) FROM read_parquet(?, hive_partitioning = true)",
    [OUT_DIR + "/**/*.parquet"],
).fetchone()[0]
con.close()

files = sorted(Path(OUT_DIR).glob("**/*.parquet"))
total_mb = sum(f.stat().st_size for f in files) / 1e6
print(f"Wrote {len(files)} files, {total_mb:.0f} MB total, {rows:,} rows -> {OUT_DIR}")
for f in files:
    print(f"  {f.stat().st_size / 1e6:6.1f} MB  {f.relative_to(PROJECT_DIR)}")
