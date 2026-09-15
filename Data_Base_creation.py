"""Build a DuckDB database from the Iowa liquor sales CSVs.

Reads every file in Iowa_data_set/ that matches the yearly-part naming pattern,
combines them into a single `sales` table, and stores it in iowa_liquor.duckdb.
"""

from pathlib import Path

import duckdb

# Resolve paths relative to this script, so it runs from any working directory.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_GLOB = str(PROJECT_DIR / "Iowa_data_set" / "iowa_liquor_sales_*_rows_part_*.csv")
UNRATE_PATH = str(PROJECT_DIR / "External_datasets" / "UNRATE.csv")
PLACES_PATH = str(PROJECT_DIR / "External_datasets" / "cdc_places_iowa_multi_year.csv")
HOMELESSNESS_PATH = str(PROJECT_DIR / "External_datasets" / "iowa_homelessness_pit_2007_2025.csv")
POPULATION_PATH = str(PROJECT_DIR / "External_datasets" / "iowa_county_population_2012_2025.csv")
POLICY_PATH = str(PROJECT_DIR / "External_datasets" / "iowa_alcohol_policy_panel_2012_2025.csv")
DB_PATH = str(PROJECT_DIR / "iowa_liquor.duckdb")

n_files = len(list((PROJECT_DIR / "Iowa_data_set").glob("iowa_liquor_sales_*_rows_part_*.csv")))
print(f"Found {n_files} CSV files")
print(f"Writing database to {DB_PATH}")

con = duckdb.connect(DB_PATH)

con.execute(
    """
    CREATE OR REPLACE TABLE sales AS
    SELECT
        invoice_id,
        try_cast(ordered_on AS DATE) AS ordered_on,
        try_cast(store_no AS INTEGER) AS store_no,
        store_name,
        store_address,
        store_city,
        store_zip_code,
        try_cast(county_fips_code AS INTEGER) AS county_fips_code,
        county_name,
        category_code,
        category_name,
        vendor_number,
        vendor_name,
        item_no,
        im_desc,
        try_cast(pack AS INTEGER) AS pack,
        try_cast(bottle_volume_ml AS INTEGER) AS bottle_volume_ml,
        try_cast(state_bottle_cost AS DECIMAL(12,2)) AS state_bottle_cost,
        try_cast(state_bottle_retail AS DECIMAL(12,2)) AS state_bottle_retail,
        try_cast(sales_bottles AS INTEGER) AS sales_bottles,
        try_cast(sales_dollars AS DECIMAL(14,2)) AS sales_dollars,
        try_cast(sales_liters AS DOUBLE) AS sales_liters,
        try_cast(sales_gallons AS DOUBLE) AS sales_gallons,
        year(try_cast(ordered_on AS DATE)) AS sale_year,
        filename AS source_file
    FROM read_csv(
        ?,
        header = true,
        all_varchar   = true,
        union_by_name = true,
        filename      = true
    );
    """,
    [DATA_GLOB],
)


# UNRATE.csv is the U.S. national monthly unemployment rate (FRED series UNRATE),
# not Iowa-specific -- useful as a macro proxy, but coarser than a county-level series.
con.execute(
    """
    CREATE OR REPLACE TABLE unemployment AS
    SELECT
        try_cast(observation_date AS DATE) AS observation_date,
        year(try_cast(observation_date AS DATE)) AS obs_year,
        month(try_cast(observation_date AS DATE)) AS obs_month,
        try_cast(UNRATE AS DOUBLE) AS unemployment_rate
    FROM read_csv(
        ?,
        header = true
    );
    """,
    [UNRATE_PATH],
)

print("\n-- unemployment: row count / date range --")
print(con.execute(
    "SELECT COUNT(*) AS rows, MIN(observation_date) AS first_month, MAX(observation_date) AS last_month FROM unemployment"
).df())

print("\n-- unemployment schema --")
print(con.execute("DESCRIBE unemployment").df())

# CDC PLACES: Local Data for Better Health, county-level, Iowa only, combined across the
# 2020-2025 releases (fetched from CDC's Socrata API and reshaped -- see
# scratchpad/fetch_places_multi_year.py). CDC changed the API schema partway through these
# releases: 2020 and 2024 are "wide" format (no explicit survey year, so data_year is NULL
# for those rows -- release_year is the only time signal); 2021, 2022, 2023, 2025 are "long"
# format with an explicit data_year (the actual BRFSS survey year, which can span two years
# within one release). Includes "Binge drinking among adults" and "Depression among adults".
# county_fips_code is the 5-digit county FIPS code, matching sales.county_fips_code.
# Multiple releases means this now supports a real (if uneven) trend over time, not just a
# single-year snapshot.
con.execute(
    """
    CREATE OR REPLACE TABLE county_health AS
    SELECT
        try_cast(county_fips_code AS INTEGER) AS county_fips_code,
        county_name,
        try_cast(data_year AS INTEGER) AS data_year,
        category,
        measure,
        short_measure,
        value_type,
        try_cast(value AS DOUBLE) AS value,
        value_unit,
        try_cast(total_population AS BIGINT) AS total_population,
        try_cast(release_year AS INTEGER) AS release_year
    FROM read_csv(
        ?,
        header = true,
        all_varchar = true
    );
    """,
    [PLACES_PATH],
)

print("\n-- county_health: row count / counties / measures / releases --")
print(con.execute(
    "SELECT COUNT(*) AS rows, COUNT(DISTINCT county_fips_code) AS counties, COUNT(DISTINCT measure) AS measures, COUNT(DISTINCT release_year) AS releases FROM county_health"
).df())

print("\n-- county_health schema --")
print(con.execute("DESCRIBE county_health").df())

# HUD Point-in-Time homelessness count, Iowa statewide, one row per year (2007-2025).
# Annual snapshot (a single night's count each January) -- joins to yearly aggregates
# on `year`, not to the county-month grain (state-level only, no county breakdown).
con.execute(
    """
    CREATE OR REPLACE TABLE homelessness AS
    SELECT
        try_cast("Year" AS INTEGER) AS pit_year,
        "State" AS state,
        try_cast("Number of CoCs" AS INTEGER) AS n_cocs,
        try_cast("Overall Homeless" AS INTEGER) AS overall_homeless,
        try_cast("Sheltered ES Homeless" AS INTEGER) AS sheltered_es_homeless,
        try_cast("Sheltered TH Homeless" AS INTEGER) AS sheltered_th_homeless,
        try_cast("Sheltered SH Homeless" AS DOUBLE) AS sheltered_sh_homeless,
        try_cast("Sheltered Total Homeless" AS INTEGER) AS sheltered_total_homeless,
        try_cast("Unsheltered Homeless" AS INTEGER) AS unsheltered_homeless,
        try_cast("Overall Homeless Individuals" AS INTEGER) AS overall_homeless_individuals,
        try_cast("Sheltered Total Homeless Individuals" AS INTEGER) AS sheltered_total_homeless_individuals,
        try_cast("Unsheltered Homeless Individuals" AS INTEGER) AS unsheltered_homeless_individuals,
        try_cast("Overall Homeless People in Families" AS INTEGER) AS overall_homeless_people_in_families,
        try_cast("Sheltered Total Homeless People in Families" AS INTEGER) AS sheltered_total_homeless_people_in_families,
        try_cast("Unsheltered Homeless People in Families" AS INTEGER) AS unsheltered_homeless_people_in_families,
        try_cast("Overall Chronically Homeless" AS DOUBLE) AS overall_chronically_homeless,
        try_cast("Sheltered Total Chronically Homeless" AS DOUBLE) AS sheltered_total_chronically_homeless,
        try_cast("Unsheltered Chronically Homeless" AS DOUBLE) AS unsheltered_chronically_homeless,
        try_cast("Overall Homeless Veterans" AS INTEGER) AS overall_homeless_veterans,
        try_cast("Sheltered Total Homeless Veterans" AS INTEGER) AS sheltered_total_homeless_veterans,
        try_cast("Unsheltered Homeless Veterans" AS INTEGER) AS unsheltered_homeless_veterans
    FROM read_csv(
        ?,
        header = true,
        all_varchar = true
    );
    """,
    [HOMELESSNESS_PATH],
)

print("\n-- homelessness: row count / year range --")
print(con.execute(
    "SELECT COUNT(*) AS rows, MIN(pit_year) AS first_year, MAX(pit_year) AS last_year FROM homelessness"
).df())

print("\n-- homelessness schema --")
print(con.execute("DESCRIBE homelessness").df())

# Census Bureau county population estimates (Vintage 2019 for 2012-2019, Vintage 2025 for
# 2020-2025 -- static files at www2.census.gov, no API key needed unlike the /pep/population
# API endpoint). One row per county per year. county_name is uppercased with " County"
# stripped to match sales.county_name -- see scratchpad/fetch_iowa_population.py. Covers all
# 99 real Iowa counties; sales.county_name has a 100th value, "EL PASO", which is not an Iowa
# county and won't match anything here (a data-quality artifact in the source sales CSVs).
con.execute(
    """
    CREATE OR REPLACE TABLE county_population AS
    SELECT
        county_name,
        try_cast(year AS INTEGER) AS pop_year,
        try_cast(population AS BIGINT) AS population
    FROM read_csv(
        ?,
        header = true,
        all_varchar = true
    );
    """,
    [POPULATION_PATH],
)

print("\n-- county_population: row count / counties / year range --")
print(con.execute(
    "SELECT COUNT(*) AS n, COUNT(DISTINCT county_name) AS counties, MIN(pop_year) AS first_year, MAX(pop_year) AS last_year FROM county_population"
).df())

print("\n-- county_population schema --")
print(con.execute("DESCRIBE county_population").df())

# Iowa alcohol regulatory panel, statewide, one row per year (2012-2025). Documents real
# policy changes: wholesale_spirits_system shifts from "State-run" (2012-2018) to
# "Mixed/Overlapping" (2019-2025) -- private spirits wholesale licensing became possible
# starting 2019 (post_2019_spirits_deregulation flips to 1), with a further ABV-band
# expansion in 2023 (post_2023_spirits_expansion flips to 1). This is a candidate root
# cause for the store-count growth found elsewhere in this notebook, rather than just an
# unexplained trend -- worth testing as a policy-period dummy in the regression.
con.execute(
    """
    CREATE OR REPLACE TABLE alcohol_policy AS
    SELECT
        try_cast("Year" AS INTEGER) AS policy_year,
        retail_beer_system,
        retail_wine_system,
        retail_spirits_system,
        try_cast(wholesale_beer_staterun_threshold_abv AS DOUBLE) AS wholesale_beer_staterun_threshold_abv,
        try_cast(wholesale_wine_staterun_threshold_abv AS DOUBLE) AS wholesale_wine_staterun_threshold_abv,
        wholesale_spirits_system,
        try_cast(wholesale_spirits_private_license_exists AS INTEGER) AS wholesale_spirits_private_license_exists,
        wholesale_spirits_license_abv_band,
        try_cast(post_2019_spirits_deregulation AS INTEGER) AS post_2019_spirits_deregulation,
        try_cast(post_2023_spirits_expansion AS INTEGER) AS post_2023_spirits_expansion
    FROM read_csv(
        ?,
        header = true,
        all_varchar = true
    );
    """,
    [POLICY_PATH],
)

print("\n-- alcohol_policy: row count / year range --")
print(con.execute(
    "SELECT COUNT(*) AS n, MIN(policy_year) AS first_year, MAX(policy_year) AS last_year FROM alcohol_policy"
).df())

print("\n-- alcohol_policy schema --")
print(con.execute("DESCRIBE alcohol_policy").df())

print("\n-- row count / date range --")
print(con.execute(
    "SELECT COUNT(*) AS rows, MIN(ordered_on) AS first_day, MAX(ordered_on) AS last_day FROM sales"
).df())

print("\n-- rows per year --")
print(con.execute(
    "SELECT sale_year, COUNT(*) AS rows FROM sales GROUP BY 1 ORDER BY 1"
).df())

print("\n-- schema --")
print(con.execute("DESCRIBE sales").df())

con.close()
print("\nDone.")
