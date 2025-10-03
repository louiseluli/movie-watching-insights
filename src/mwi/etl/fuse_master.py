from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import polars as pl

from ..config import get_config
from ..log import Timer, logger, print_kv_table, stamp_utc


def _paths() -> Tuple[Path, Path]:
    """Return (parquet_dir_imdb, master_out_dir)."""
    cfg = get_config()
    parquet_dir = Path(cfg.processed_dir) / "imdb"
    master_dir = Path(cfg.processed_dir)
    return parquet_dir, master_dir


def _load_watchlist_csv(path: Path) -> pl.DataFrame:
    """
    Load the user's IMDb export (watchlist CSV). Your file is Title-Case (Const, Position, ...).
    We normalize headers to snake_case so the rest of the pipeline uses stable names.
    """
    # 1) Read as-is (Polars will infer sensible dtypes; we'll coerce later)
    df = pl.read_csv(
        path,
        infer_schema_length=5000,
        try_parse_dates=True,
        null_values=["", "NaN", "nan", "NULL"],
    )

    # 2) Build a robust header normalizer (case-insensitive, strip, snake_case)
    def norm(col: str) -> str:
        c = col.strip()
        # common unicode apostrophes etc.
        c = c.replace("’", "'")
        c = c.lower()
        # replace non-alnum with underscores
        import re
        c = re.sub(r"[^a-z0-9]+", "_", c).strip("_")
        # specific fixes from IMDb export
        if c in {"imdb_rating", "i_m_d_b_rating"}:
            return "imdb_rating"
        if c in {"runtime_mins", "runtime_mins_"}:
            return "runtime_mins"
        if c in {"release_date"}:
            return "release_date"
        if c in {"original_title"}:
            return "original_title"
        if c in {"title_type"}:
            return "title_type"
        if c in {"num_votes"}:
            return "num_votes"
        if c in {"date_rated"}:
            return "date_rated"
        return c

    # 3) Special header synonyms map (exact IMDb column names → canonical)
    header_map = {
        "position": "position",
        "const": "const",
        "created": "created",
        "modified": "modified",
        "title": "title",
        "original_title": "original_title",
        "url": "url",
        "title_type": "title_type",
        "imdb_rating": "imdb_rating",
        "runtime_mins": "runtime_mins",      # "Runtime (mins)"
        "year": "year",
        "genres": "genres",
        "num_votes": "num_votes",
        "release_date": "release_date",
        "directors": "directors",
        "your_rating": "your_rating",
        "date_rated": "date_rated",
        "description": "description",
    }

    # 4) Apply normalization + mapping
    raw_cols = df.columns
    normalized = [norm(c) for c in raw_cols]
    canonical = [header_map.get(c, c) for c in normalized]
    if len(set(canonical)) != len(canonical):
        # avoid collisions silently; if it ever happens, warn
        logger.warning("Header collision after normalization: {}", canonical)
    df = df.rename(dict(zip(raw_cols, canonical)))

    # 5) Validate presence of key column
    if "const" not in df.columns:
        # help message with preview of columns to diagnose quickly
        raise ValueError(
            "Watchlist CSV must contain an IMDb id column (Const). "
            f"Found columns: {', '.join(df.columns)}"
        )

    # 6) Canonical types + helper lists, null-safe
    df = df.with_columns(
        [
            pl.col("const").alias("tconst"),
            pl.col("year").cast(pl.Int32, strict=False) if "year" in df.columns else pl.lit(None).cast(pl.Int32).alias("year"),
            pl.col("runtime_mins").cast(pl.Int32, strict=False) if "runtime_mins" in df.columns else pl.lit(None).cast(pl.Int32).alias("runtime_mins"),
            pl.when(pl.col("genres").is_null() | ~pl.col("genres").is_utf8())
            .then(pl.lit(None))
            .otherwise(pl.col("genres").str.split(",").arr.eval(pl.element().str.strip_chars()))
            .alias("genres_list"),
            pl.when(pl.col("directors").is_null() | ~pl.col("directors").is_utf8())
            .then(pl.lit(None))
            .otherwise(pl.col("directors").str.split(",").arr.eval(pl.element().str.strip_chars()))
            .alias("directors_list_wl"),
        ]
    )

    # 7) Ensure unique by tconst
    df = df.unique(subset=["tconst"], keep="first")
    return df



def _read_imdb_parquets(parquet_dir: Path) -> Dict[str, pl.LazyFrame]:
    """
    Load required IMDb datasets as LazyFrames (fast + memory friendly).
    We use only the columns we need for fusing.
    """
    basics = pl.scan_parquet(parquet_dir / "title.basics.parquet").select(
        "tconst", "titleType", "primaryTitle", "originalTitle", "isAdult", "startYear",
        "endYear", "runtimeMinutes", "genres", "genres_list",
    )

    ratings = pl.scan_parquet(parquet_dir / "title.ratings.parquet").select(
        "tconst", "averageRating", "numVotes"
    )

    crew = (
        pl.scan_parquet(parquet_dir / "title.crew.parquet")
        .select("tconst", "directors", "writers")
        .with_columns(
            pl.when(pl.col("directors").is_null())
            .then(pl.lit(None))
            .otherwise(
                pl.col("directors")
                .cast(pl.Utf8, strict=False)
                .str.split(",")
                .arr.eval(pl.element().str.strip_chars())
            )
            .alias("directors_list")
        )
    )

    names = pl.scan_parquet(parquet_dir / "name.basics.parquet").select(
        "nconst", "primaryName"
    )

    return {"basics": basics, "ratings": ratings, "crew": crew, "names": names}


def _resolve_first_director_name(crew_df: pl.LazyFrame, names_df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Adds `first_director_nconst` and `first_director_name` to the crew LF.
    Entirely expression-based (no map_elements), null-safe.
    """
    crew_aug = crew_df.with_columns(
        [
            pl.col("directors_list").list.first().alias("first_director_nconst")
        ]
    )

    # Join to names on the first director nconst
    crew_named = crew_aug.join(
        names_df,
        left_on="first_director_nconst",
        right_on="nconst",
        how="left",
    ).select(
        "tconst",
        "directors",
        "directors_list",
        "first_director_nconst",
        pl.col("primaryName").alias("first_director_name"),
    )
    return crew_named


def fuse_watchlist_with_imdb() -> Path:
    """
    Fuse user's watchlist (canonical 'watched') with IMDb Parquets to a single
    canonical dataset: data/processed/movies_master.parquet.

    - Keeps IMDb’s official ratings (averageRating/numVotes)
    - Keeps your watchlist fields (no self-ratings required)
    - Resolves first director name
    - Prepares placeholder columns for future enrichment:
        tmdb_id, tmdb_keywords (list), omdb_data (struct/json), ddd_topics (list)
    """
    cfg = get_config()
    parquet_dir, master_dir = _paths()
    master_dir.mkdir(parents=True, exist_ok=True)

    wl_path = Path(cfg.imdb_watchlist_csv) if cfg.imdb_watchlist_csv else None
    if wl_path is None or not wl_path.exists():
        raise FileNotFoundError("MWI_IMDB_WATCHLIST_CSV not set or file not found. Set it in your .env.")

    print_kv_table("Fuse Inputs", {"watchlist_csv": wl_path, "imdb_parquet_dir": parquet_dir})

    # 1) Load watchlist
    with Timer("load_watchlist_csv"):
        wl = _load_watchlist_csv(wl_path)
        logger.info("watchlist: {:,} rows", wl.height)

    # 2) Load IMDb Parquets as LazyFrames (needed subsets only)
    with Timer("load_imdb_parquets"):
        lfs = _read_imdb_parquets(parquet_dir)
        basics, ratings, crew, names = lfs["basics"], lfs["ratings"], lfs["crew"], lfs["names"]
        logger.info(
            "loaded basics=? ratings=? crew=? names=? (lazy); will filter to watchlist tconsts at join"
        )

    # Turn WL to Lazy for joining
    wl_lf = wl.lazy().select(
        "tconst",
        "position",
        "title",
        "original_title",
        "url",
        "title_type",
        "imdb_rating",
        "runtime_mins",
        "year",
        "genres",
        "genres_list",
        "num_votes",
        "release_date",
        "directors",
        "directors_list_wl",
        "your_rating",
        "date_rated",
        "created",
        "modified",
        "description",
    )

    # 3) Join watchlist → basics
    fused = wl_lf.join(
        basics,
        left_on="tconst",
        right_on="tconst",
        how="left",
    )

    # 4) Join ratings
    fused = fused.join(
        ratings,
        on="tconst",
        how="left",
        suffix="_imdb",
    )

    # 5) Crew + first director name
    crew_named = _resolve_first_director_name(crew, names)
    fused = fused.join(
        crew_named,
        on="tconst",
        how="left",
    )

    # 6) Final tidy-up + enrichment placeholders (null by design)
    fused = fused.with_columns(
        [
            # Canonical numeric types
            pl.col("startYear").cast(pl.Int32, strict=False),
            pl.col("endYear").cast(pl.Int32, strict=False),
            pl.col("runtimeMinutes").cast(pl.Int32, strict=False),

            # Prefer titles from watchlist if available, else IMDb primary/original
            pl.coalesce([pl.col("title"), pl.col("primaryTitle")]).alias("title_final"),
            pl.coalesce([pl.col("original_title"), pl.col("originalTitle")]).alias("original_title_final"),

            # A unified list of genres (prefer the richer watchlist if present)
            pl.coalesce([pl.col("genres_list"), pl.col("genres_list_right")]).alias("genres_list_final"),

            # Director list: prefer crew’s directors_list; fallback to WL's parsed list
            pl.coalesce([pl.col("directors_list"), pl.col("directors_list_wl")]).alias("directors_list_final"),

            # Helpful counts
            pl.col("directors_list").list.len().alias("director_count").fill_null(0),

            # Placeholders for enrichment we’ll fill later
            pl.lit(None).cast(pl.Utf8).alias("tmdb_id"),
            pl.lit(None).cast(pl.List(pl.Utf8)).alias("tmdb_keywords"),
            pl.lit(None).cast(pl.Utf8).alias("omdb_data"),  # JSON string later
            pl.lit(None).cast(pl.List(pl.Utf8)).alias("ddd_topics"),
        ]
    )

    # 7) Select and rename to a clean, compact schema
    out = fused.select(
        # Keys
        "tconst",
        "title_final",
        "original_title_final",
        pl.col("titleType").alias("imdb_titleType"),
        pl.col("isAdult").alias("imdb_isAdult"),
        "startYear",
        "endYear",
        pl.col("runtimeMinutes").alias("runtime_mins_imdb"),
        # Genres
        "genres",
        "genres_list_final",
        # Ratings
        pl.col("averageRating").alias("imdb_average_rating"),
        pl.col("numVotes").alias("imdb_num_votes"),
        # First director
        "first_director_nconst",
        "first_director_name",
        "directors_list_final",
        "director_count",
        # From your watchlist (provenance columns)
        pl.col("position").alias("wl_position"),
        pl.col("url").alias("wl_url"),
        pl.col("title_type").alias("wl_title_type"),
        pl.col("imdb_rating").alias("wl_imdb_rating_snapshot"),
        pl.col("num_votes").alias("wl_num_votes_snapshot"),
        pl.col("runtime_mins").alias("wl_runtime_mins_snapshot"),
        pl.col("year").alias("wl_year_snapshot"),
        "release_date",
        "your_rating",
        "date_rated",
        "created",
        "modified",
        "description",
        # Enrichment placeholders
        "tmdb_id",
        "tmdb_keywords",
        "omdb_data",
        "ddd_topics",
    ).collect(streaming=True)

    # 8) Write outputs
    parquet_out = Path(get_config().processed_dir) / "movies_master.parquet"
    out.write_parquet(parquet_out)
    logger.success("Wrote master parquet → {}", parquet_out)

    # Small manifest to help downstream
    manifest = {
        "rows": out.height,
        "timestamp_utc": stamp_utc(),
        "source_watchlist": str(wl_path),
        "imdb_parquet_dir": str(parquet_dir),
        "output": str(parquet_out),
        "notes": "Placeholders tmdb_id/tmdb_keywords/omdb_data/ddd_topics are empty by design; enrichment step fills them with caching & rate limits.",
    }
    (parquet_out.parent / "_master_manifest.txt").write_text("\n".join(f"{k}: {v}" for k, v in manifest.items()))
    logger.info("Master manifest written → {}", parquet_out.parent / "_master_manifest.txt")

    return parquet_out


def main():
    cfg = get_config()
    with Timer("etl::fuse_master"):
        out = fuse_watchlist_with_imdb()
    logger.success("Fuse complete → {}", out)


if __name__ == "__main__":
    main()
