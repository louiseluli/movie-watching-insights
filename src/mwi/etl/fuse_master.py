from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import polars as pl

from ..config import get_config
from ..log import Timer, logger, print_kv_table, progress_bar, stamp_utc


WATCHLIST_REQUIRED_COLS = {
    "const", "title", "original_title", "title_type", "imdb_rating",
    "runtime_mins", "year", "genres", "num_votes", "release_date",
    "directors", "your_rating", "date_rated", "created", "modified", "position", "url", "description"
}


def _load_watchlist_csv(path: Path) -> pl.DataFrame:
    """
    Load the user's IMDb export CSV (watchlist). We treat this as 'all watched'.
    Normalize column names and parse dates.
    """
    df = pl.read_csv(
        path,
        infer_schema_length=5000,
        null_values=["", "NaN", "nan"],
        try_parse_dates=False,
        encoding="utf8",
    )

    # Normalize column names to snake_case used earlier
    # If your CSV already matches, harmless.
    rename_map = {
        "Const": "const",
        "Created": "created",
        "Modified": "modified",
        "Description": "description",
        "Title": "title",
        "Original Title": "original_title",
        "URL": "url",
        "Title Type": "title_type",
        "IMDb Rating": "imdb_rating",
        "Runtime (mins)": "runtime_mins",
        "Year": "year",
        "Genres": "genres",
        "Num Votes": "num_votes",
        "Release Date": "release_date",
        "Directors": "directors",
        "Your Rating": "your_rating",
        "Date Rated": "date_rated",
        "Position": "position",
    }
    df = df.rename({k: v for k, v in rename_map.items() if k in df.columns})

    # Quick check for expected columns (non-fatal; we log any missing)
    missing = sorted(list(WATCHLIST_REQUIRED_COLS - set(df.columns)))
    if missing:
        logger.warning("Watchlist CSV missing columns: {}", ", ".join(missing))

    # Type fixes and date parsing
    df = df.with_columns(
        [
            pl.col("const").cast(pl.Utf8),
            pl.col("title").cast(pl.Utf8),
            pl.col("original_title").cast(pl.Utf8),
            pl.col("title_type").cast(pl.Utf8),
            pl.col("imdb_rating").cast(pl.Float64, strict=False),
            pl.col("runtime_mins").cast(pl.Int64, strict=False),
            pl.col("year").cast(pl.Int32, strict=False),
            pl.col("genres").cast(pl.Utf8),
            pl.col("num_votes").cast(pl.Int64, strict=False),
            pl.col("release_date").str.strptime(pl.Date, strict=False),
            pl.col("directors").cast(pl.Utf8),
            pl.col("your_rating").cast(pl.Float64, strict=False),
            pl.col("date_rated").str.strptime(pl.Date, strict=False),
            pl.col("created").str.strptime(pl.Date, strict=False),
            pl.col("modified").str.strptime(pl.Date, strict=False),
            pl.lit(True).alias("watched"),
        ]
    )

    # Compute a "date_watched" proxy: prefer date_rated; fall back to created
    df = df.with_columns(
        [
            pl.when(pl.col("date_rated").is_not_null())
            .then(pl.col("date_rated"))
            .otherwise(pl.col("created"))
            .alias("date_watched")
        ]
    )

    # For convenience: genres list column from CSV if present
    df = df.with_columns(
        [
            pl.when(pl.col("genres").is_null() | (pl.col("genres") == ""))
            .then(pl.lit(None))
            .otherwise(pl.col("genres").str.split(",").list.eval(pl.element().str.strip_chars()))
            .alias("genres_list_from_csv")
        ]
    )

    return df


def _read_parquet_or_empty(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.exists() else pl.DataFrame()


def _explode_csv_to_list(col: str) -> pl.Expr:
    """Split a comma-separated string column into a list (null-safe)."""
    return (
        pl.when(pl.col(col).is_null() | (pl.col(col) == ""))
        .then(pl.lit(None))
        .otherwise(pl.col(col).str.split(",").list.eval(pl.element().str.strip_chars()))
    )


def _map_nconst_to_names(nconst_series: pl.Series, name_map: Dict[str, str]) -> List[str] | None:
    if nconst_series is None:
        return None
    vals = [name_map.get(x) for x in nconst_series if x]
    vals = [v for v in vals if v]
    return vals or None


def fuse_watchlist_with_imdb() -> Path:
    """
    Fuse user's watchlist CSV with IMDb parquet dumps to produce a canonical master table.
    Returns the path to the written master parquet.
    """
    cfg = get_config()
    parquet_dir = Path(cfg.processed_dir) / "imdb"
    master_dir = Path(cfg.processed_dir)
    master_dir.mkdir(parents=True, exist_ok=True)

    wl_path = Path(cfg.imdb_watchlist_csv) if cfg.imdb_watchlist_csv else None
    if wl_path is None or not wl_path.exists():
        raise FileNotFoundError("MWI_IMDB_WATCHLIST_CSV not set or file not found. Set it in your .env.")

    print_kv_table("Fuse Inputs", {
        "watchlist_csv": wl_path,
        "imdb_parquet_dir": parquet_dir,
    })

    with Timer("load_watchlist_csv"):
        wl = _load_watchlist_csv(wl_path)
        logger.info("watchlist: {:,} rows", wl.height)

    # Load IMDb Parquets
    basics_pq = parquet_dir / "title.basics.parquet"
    ratings_pq = parquet_dir / "title.ratings.parquet"
    crew_pq = parquet_dir / "title.crew.parquet"
    principals_pq = parquet_dir / "title.principals.parquet"
    names_pq = parquet_dir / "name.basics.parquet"

    with Timer("load_imdb_parquets"):
        basics = _read_parquet_or_empty(basics_pq)
        ratings = _read_parquet_or_empty(ratings_pq)
        crew = _read_parquet_or_empty(crew_pq)
        principals = _read_parquet_or_empty(principals_pq)
        names = _read_parquet_or_empty(names_pq)

        logger.info(
            "loaded basics={} ratings={} crew={} principals={} names={}",
            basics.height, ratings.height, crew.height, principals.height, names.height
        )

    # Minimal projections for efficiency
    basics_sel = basics.select(
        [
            pl.col("tconst"),
            pl.col("titleType"),
            pl.col("primaryTitle"),
            pl.col("originalTitle"),
            pl.col("isAdult"),
            pl.col("startYear"),
            pl.col("endYear"),
            pl.col("runtimeMinutes"),
            pl.col("genres"),
            pl.col("genres_list"),
        ]
    )
    ratings_sel = ratings.select(["tconst", "averageRating", "numVotes"])
    crew_sel = crew.select(["tconst", "directors", "writers"])
    names_sel = names.select(["nconst", "primaryName"])

    # Precompute a name map for quick lookups (small enough; else we’d join)
    name_map = dict(zip(names_sel["nconst"].to_list(), names_sel["primaryName"].to_list()))

    # Expand crew directors/writers into names
    def _split_to_list(col: str) -> pl.Expr:
        return _explode_csv_to_list(col)

    with Timer("crew_name_mapping"):
        crew_named = crew_sel.with_columns(
            [
                _split_to_list("directors").alias("director_ids"),
                _split_to_list("writers").alias("writer_ids"),
            ]
        ).with_columns(
            [
                pl.col("director_ids").map_elements(
                    lambda L: _map_nconst_to_names(L, name_map), return_dtype=pl.List(pl.Utf8)
                ).alias("directors_names"),
                pl.col("writer_ids").map_elements(
                    lambda L: _map_nconst_to_names(L, name_map), return_dtype=pl.List(pl.Utf8)
                ).alias("writers_names"),
            ]
        ).select(["tconst", "director_ids", "directors_names", "writer_ids", "writers_names"])

    # Principals: get top billed cast names (actors/actresses), keep top 5 by 'ordering'
    with Timer("principals_top_cast"):
        principals_top = principals.filter(
            pl.col("category").is_in(["actor", "actress"])
        ).with_columns(
            [
                pl.col("ordering").cast(pl.Int32, strict=False),
            ]
        ).sort(["tconst", "ordering"]).groupby("tconst").agg(
            [
                pl.col("nconst").head(5).alias("top_cast_ids"),
            ]
        ).with_columns(
            [
                pl.col("top_cast_ids").map_elements(
                    lambda L: _map_nconst_to_names(L, name_map), return_dtype=pl.List(pl.Utf8)
                ).alias("top_cast_names")
            ]
        )

    # Join chain: watchlist → basics → ratings → crew → principals
    with Timer("join_chain"):
        master = wl.rename({"const": "tconst"}).join(
            basics_sel, on="tconst", how="left"
        ).join(
            ratings_sel, on="tconst", how="left"
        ).join(
            crew_named, on="tconst", how="left"
        ).join(
            principals_top, on="tconst", how="left"
        )

    # Canonical columns / tidy-up
    with Timer("finalize_master"):
        # decide titles: prefer basics.primaryTitle / originalTitle where present
        master = master.with_columns(
            [
                pl.coalesce([pl.col("title"), pl.col("primaryTitle")]).alias("title_final"),
                pl.coalesce([pl.col("original_title"), pl.col("originalTitle")]).alias("original_title_final"),
                # prefer ratings from IMDb ratings table
                pl.coalesce([pl.col("averageRating"), pl.col("imdb_rating")]).alias("imdb_rating_final"),
                pl.coalesce([pl.col("numVotes"), pl.col("num_votes")]).alias("imdb_votes_final"),
                # ensure a single runtime
                pl.coalesce([pl.col("runtimeMinutes"), pl.col("runtime_mins")]).alias("runtime_minutes_final"),
                # genres final: prefer list from basics; else CSV split from watchlist
                pl.coalesce([pl.col("genres_list"), pl.col("genres_list_from_csv")]).alias("genres_final"),
            ]
        )

        # placeholders for enrichment you asked for
        master = master.with_columns(
            [
                pl.lit(None, dtype=pl.List(pl.Utf8)).alias("keywords_tmdb"),
                pl.lit(None, dtype=pl.List(pl.Struct(
                    {"topic_id": pl.Int64, "name": pl.Utf8, "yes": pl.Int64, "no": pl.Int64}
                ))).alias("ddd_topics"),
                pl.lit(None, dtype=pl.Boolean).alias("ddd_dog_dies"),
                pl.lit(None, dtype=pl.Utf8).alias("omdb_plot"),
                pl.lit(None, dtype=pl.List(pl.Utf8)).alias("omdb_languages"),
                pl.lit(None, dtype=pl.List(pl.Utf8)).alias("omdb_countries"),
                pl.lit(None, dtype=pl.List(pl.Utf8)).alias("tmdb_keywords_raw"),
                pl.lit(None, dtype=pl.Utf8).alias("cinemorgue_notes"),
            ]
        )

        # select & order columns for clarity
        master = master.select(
            [
                # identity
                "tconst", "url",
                # titles
                "title_final", "original_title_final",
                "title_type", "titleType",
                # years & runtime
                "year", "startYear", "endYear", "runtime_minutes_final",
                # genres
                "genres_final",
                # people
                "directors", "director_ids", "directors_names",
                "writer_ids", "writers_names",
                "top_cast_ids", "top_cast_names",
                # ratings (public)
                "imdb_rating_final", "imdb_votes_final",
                # your personal fields
                "your_rating", "date_rated",
                # state/dates
                "watched", "date_watched", "created", "modified", "position",
                # enrichment placeholders
                "keywords_tmdb", "tmdb_keywords_raw",
                "ddd_topics", "ddd_dog_dies",
                "omdb_plot", "omdb_languages", "omdb_countries",
                "cinemorgue_notes",
                # raw carry-overs (optional for debugging)
                "title", "primaryTitle", "original_title", "originalTitle",
                "runtime_mins", "runtimeMinutes", "genres", "genres_list_from_csv",
                "averageRating", "numVotes",
                "release_date", "isAdult",
            ]
        )

    # Write outputs
    out_parquet = Path(cfg.processed_dir) / "movies_master.parquet"
    out_preview = Path(cfg.processed_dir) / f"movies_master_preview_{stamp_utc()}.csv"

    with Timer("write_master"):
        master.write_parquet(out_parquet)
        # also write a small preview CSV (head 500)
        master.head(500).write_csv(out_preview)
        logger.info("Master rows: {:,}", master.height)
        logger.info("Wrote master parquet → {}", out_parquet)
        logger.info("Preview CSV (first 500 rows) → {}", out_preview)

    return out_parquet


def main():
    cfg = get_config()
    print_kv_table("Fuse Config", {
        "watchlist_csv": cfg.imdb_watchlist_csv or "",
        "processed_dir": cfg.processed_dir,
    })

    with Timer("etl::fuse_master"):
        out = fuse_watchlist_with_imdb()
    logger.success("Fuse complete → {}", out)


if __name__ == "__main__":
    main()
