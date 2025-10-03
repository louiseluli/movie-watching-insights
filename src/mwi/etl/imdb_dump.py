from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import duckdb
import polars as pl

from ..config import get_config
from ..log import Timer, logger, print_kv_table, progress_bar, stamp_utc


# --------- IMDb schemas (explicit, stable dtypes) ---------

# IMDb uses \N for null. Dates/years are mostly strings/ints.
# We load as close to the official schema as practical.
SCHEMAS: Dict[str, Dict[str, pl.DataType]] = {
    # https://developer.imdb.com/non-commercial-datasets/
    "title.basics": {
        "tconst": pl.Utf8,
        "titleType": pl.Utf8,
        "primaryTitle": pl.Utf8,
        "originalTitle": pl.Utf8,
        "isAdult": pl.Int8,
        "startYear": pl.Utf8,   # may contain \N or yyyy
        "endYear": pl.Utf8,     # may contain \N or yyyy
        "runtimeMinutes": pl.Utf8,  # keep as string then coerce to Int64
        "genres": pl.Utf8,          # CSV string of up to 3
    },
    "title.ratings": {
        "tconst": pl.Utf8,
        "averageRating": pl.Float64,
        "numVotes": pl.Int64,
    },
    "title.akas": {
        "titleId": pl.Utf8,
        "ordering": pl.Int32,
        "title": pl.Utf8,
        "region": pl.Utf8,
        "language": pl.Utf8,
        "types": pl.Utf8,
        "attributes": pl.Utf8,
        "isOriginalTitle": pl.Int8,
    },
    "title.crew": {
        "tconst": pl.Utf8,
        "directors": pl.Utf8,  # comma-separated nconst
        "writers": pl.Utf8,    # comma-separated nconst
    },
    "title.principals": {
        "tconst": pl.Utf8,
        "ordering": pl.Int32,
        "nconst": pl.Utf8,
        "category": pl.Utf8,
        "job": pl.Utf8,
        "characters": pl.Utf8,
    },
    "title.episode": {
        "tconst": pl.Utf8,
        "parentTconst": pl.Utf8,
        "seasonNumber": pl.Utf8,   # keep string → coerce later
        "episodeNumber": pl.Utf8,  # keep string → coerce later
    },
    "name.basics": {
        "nconst": pl.Utf8,
        "primaryName": pl.Utf8,
        "birthYear": pl.Utf8,
        "deathYear": pl.Utf8,
        "primaryProfession": pl.Utf8,  # CSV string (top-3)
        "knownForTitles": pl.Utf8,     # CSV of tconst
    },
}


def _imdb_path_for(logical_name: str, dump_dir: Path) -> Path:
    return dump_dir / f"{logical_name}.tsv.gz"


def _output_parquet_path(logical_name: str, out_dir: Path) -> Path:
    return out_dir / f"{logical_name}.parquet"


def _clean_frame(logical_name: str, df: pl.DataFrame) -> pl.DataFrame:
    r"""
    Canonicalize per-table:
      - NOTE: we already pass null_values=["\\N"] to read_csv, so '\\N' is treated as None.
      - Coerce numeric-like fields.
      - Add list columns where helpful.
    """

    if logical_name == "title.basics":
        df = df.with_columns(
            [
                pl.col("isAdult").cast(pl.Int8).alias("isAdult"),
                pl.col("startYear").cast(pl.Int32, strict=False),
                pl.col("endYear").cast(pl.Int32, strict=False),
                pl.col("runtimeMinutes").cast(pl.Int64, strict=False),
                pl.when(pl.col("genres").is_null())
                .then(pl.lit(None))
                .otherwise(pl.col("genres").str.split(","))
                .alias("genres_list"),
            ]
        )

    elif logical_name == "title.episode":
        df = df.with_columns(
            [
                pl.col("seasonNumber").cast(pl.Int32, strict=False),
                pl.col("episodeNumber").cast(pl.Int32, strict=False),
            ]
        )

    elif logical_name == "name.basics":
        df = df.with_columns(
            [
                pl.col("birthYear").cast(pl.Int32, strict=False),
                pl.col("deathYear").cast(pl.Int32, strict=False),
                pl.when(pl.col("primaryProfession").is_null())
                .then(pl.lit(None))
                .otherwise(pl.col("primaryProfession").str.split(","))
                .alias("primaryProfession_list"),
                pl.when(pl.col("knownForTitles").is_null())
                .then(pl.lit(None))
                .otherwise(pl.col("knownForTitles").str.split(","))
                .alias("knownForTitles_list"),
            ]
        )

    # other tables: no extra normalization needed for now
    return df



def _read_imdb_tsv(path: Path, schema: Dict[str, pl.DataType]) -> pl.DataFrame:
    """
    Fast TSV reader with polars.
    Key fix: disable quoting for TSVs (quote_char=None) so content like:
      "Rome brûle" (Portrait de Shirley Clarke)
    is treated as literal text, not as CSV quotes.
    """
    df = pl.read_csv(
        path,
        separator="\t",
        has_header=True,
        null_values=[r"\N"],
        schema_overrides=schema,   # renamed from deprecated `dtypes`
        quote_char=None,           # <-- important for TSVs
        low_memory=True,
        try_parse_dates=False,
        encoding="utf8",           # use "utf8-lossy" if you ever hit bad bytes
        # If you ever encounter malformed lines in the wild, you can add:
        # truncate_ragged_lines=True,
        # ignore_errors=True,   # (prefer to keep False; only enable if necessary)
    )
    return df



def _write_parquet(df: pl.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)


def _write_duckdb(df: pl.DataFrame, table_name: str, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        # Create/replace via Arrow to keep types consistent
        arrow_tbl = df.to_arrow()
        con.register("tmp_tbl", arrow_tbl)
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM tmp_tbl;")
        con.unregister("tmp_tbl")
    finally:
        con.close()


def load_all_imdb(
    dump_dir: Path,
    out_parquet_dir: Path,
    duckdb_path: Optional[Path] = None,
) -> Dict[str, int]:
    """
    Load known IMDb TSVs that exist in dump_dir → write Parquet to out_parquet_dir.
    Optionally mirror to DuckDB (one table per logical name).
    Returns a dict of {logical_name: row_count}.
    """
    expected = list(SCHEMAS.keys())
    print_kv_table("IMDb Dump → Targets", {k: _imdb_path_for(k, dump_dir) for k in expected})

    found = {k: _imdb_path_for(k, dump_dir) for k in expected if _imdb_path_for(k, dump_dir).exists()}
    missing = sorted(set(expected) - set(found.keys()))
    if missing:
        logger.warning("Missing IMDb TSVs (skipping): {}", ", ".join(missing))
    else:
        logger.success("All expected IMDb TSVs present.")

    counts: Dict[str, int] = {}
    db_path = duckdb_path

    with progress_bar() as prog:
        task = prog.add_task("Reading & writing IMDb TSVs", total=len(found))

        for logical_name, tsv_path in found.items():
            schema = SCHEMAS[logical_name]
            with Timer(f"imdb_read::{logical_name}"):
                df = _read_imdb_tsv(tsv_path, schema)
                df = _clean_frame(logical_name, df)
                counts[logical_name] = df.height
                logger.info("{:s}: {:,} rows", logical_name, df.height)

            with Timer(f"imdb_parquet::{logical_name}"):
                out_path = _output_parquet_path(logical_name, out_parquet_dir)
                _write_parquet(df, out_path)
                logger.info("Wrote Parquet → {}", out_path)

            if db_path is not None:
                with Timer(f"imdb_duckdb::{logical_name}"):
                    _write_duckdb(df, table_name=logical_name.replace(".", "_"), db_path=db_path)
                    logger.info("Upserted DuckDB table {}", logical_name.replace(".", "_"))

            prog.update(task, advance=1)

    # Write a small manifest with row counts and locations
    manifest = {
        "timestamp_utc": stamp_utc(),
        "dump_dir": str(dump_dir),
        "parquet_dir": str(out_parquet_dir),
        "duckdb": str(db_path) if db_path else None,
        "tables": counts,
    }
    manifest_path = out_parquet_dir / "_imdb_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Manifest written → {}", manifest_path)

    return counts


def main():
    """
    Allow running as:
        python -m mwi.etl.imdb_dump
    We’ll also wire the CLI command to this in the next step.
    """
    cfg = get_config()

    parquet_dir = Path(cfg.processed_dir) / "imdb"
    duckdb_file = Path(cfg.cache_dir) / "imdb.duckdb"

    print_kv_table(
        "Runtime",
        {
            "dump_dir": cfg.imdb_dump_dir,
            "parquet_dir": parquet_dir,
            "duckdb_file": duckdb_file,
        },
    )

    with Timer("etl::imdb_dump_all"):
        counts = load_all_imdb(
            dump_dir=cfg.imdb_dump_dir,
            out_parquet_dir=parquet_dir,
            duckdb_path=duckdb_file,
        )
    logger.success("IMDb ingest complete: {}", counts)


if __name__ == "__main__":
    main()
