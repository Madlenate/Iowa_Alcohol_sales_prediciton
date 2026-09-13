"""Query the Iowa liquor sales DuckDB database.

Usage:
    python query_db.py                          # open an interactive SQL prompt
    python query_db.py "SELECT * FROM sales LIMIT 5"   # run one query and print it
    python query_db.py -f my_query.sql          # run the SQL in a file
    python query_db.py --to out.csv "SELECT ..."  # also save the result to CSV

Inside the interactive prompt:
    - type any SQL, end with ; or just press Enter
    - \\t          list tables
    - \\d sales    describe a table
    - \\q          quit
"""

import sys
from pathlib import Path

import duckdb

PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "iowa_liquor.duckdb"
PARQUET_GLOB = str(PROJECT_DIR / "parquet" / "by_year" / "**" / "*.parquet")

_con = None


def connect(read_only=True):
    """Return a cached connection exposing the `sales` table.

    Uses iowa_liquor.duckdb if it exists; otherwise (e.g. a fresh git clone)
    falls back to an in-memory connection with `sales` as a view over the
    Parquet files in parquet/by_year/.
    """
    global _con
    if _con is None:
        if DB_PATH.exists():
            _con = duckdb.connect(str(DB_PATH), read_only=read_only)
        elif list(PROJECT_DIR.glob("parquet/by_year/**/*.parquet")):
            _con = duckdb.connect()
            path = PARQUET_GLOB.replace("\\", "/").replace("'", "''")
            _con.execute(
                f"CREATE VIEW sales AS "
                f"SELECT * FROM read_parquet('{path}', hive_partitioning = true)"
            )
        else:
            raise SystemExit(
                "No data found. Expected iowa_liquor.duckdb (run Data_Base_creation.py) "
                "or parquet/by_year/*.parquet."
            )
    return _con


def q(sql):
    """Run SQL and return the result as a pandas DataFrame.

    Notebook usage:
        from query_db import q
        q("SELECT sale_year, SUM(sales_dollars) AS total FROM sales GROUP BY 1 ORDER BY 1")
    """
    return connect().execute(sql).df()


def run(con, sql):
    """Execute one statement and return a DataFrame (or None if it produced no result)."""
    rel = con.execute(sql)
    try:
        return rel.df()
    except Exception:
        return None


def show(df, to_csv=None):
    if df is None:
        print("(no rows returned)")
        return
    import pandas as pd

    with pd.option_context("display.max_rows", 100, "display.max_columns", None, "display.width", 200):
        print(df)
    print(f"\n[{len(df)} rows x {len(df.columns)} cols]")
    if to_csv:
        df.to_csv(to_csv, index=False)
        print(f"saved -> {to_csv}")


def interactive(con):
    src = DB_PATH.name if DB_PATH.exists() else "parquet/by_year/"
    print(f"Connected ({src})")
    print("Enter SQL (end with ; or blank line).  \\t tables   \\d <t> describe   \\q quit")
    buf = []
    while True:
        try:
            line = input("sql> " if not buf else "...> ")
        except EOFError:
            break
        s = line.strip()
        if not buf and s in ("\\q", "quit", "exit"):
            break
        if not buf and s == "\\t":
            show(run(con, "SHOW TABLES"))
            continue
        if not buf and s.startswith("\\d"):
            parts = s.split()
            tbl = parts[1] if len(parts) > 1 else "sales"
            show(run(con, f"DESCRIBE {tbl}"))
            continue
        buf.append(line)
        if s.endswith(";") or s == "":
            sql = "\n".join(buf).strip().rstrip(";")
            buf = []
            if not sql:
                continue
            try:
                show(run(con, sql))
            except Exception as e:
                print(f"ERROR: {e}")
    print("bye")


def main():
    args = sys.argv[1:]
    to_csv = None
    if "--to" in args:
        i = args.index("--to")
        to_csv = args[i + 1]
        del args[i:i + 2]

    con = connect()

    if not args:
        interactive(con)
    elif args[0] in ("-f", "--file"):
        sql = Path(args[1]).read_text()
        show(run(con, sql), to_csv)
    else:
        sql = " ".join(args)
        show(run(con, sql), to_csv)

    con.close()


if __name__ == "__main__":
    main()
