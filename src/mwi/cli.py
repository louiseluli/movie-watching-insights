from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from .config import get_config, load_config
from .log import logger, print_kv_table, setup_logging, Timer

app = typer.Typer(add_completion=False, help="Movie Watching Insights CLI")

# Sub-apps
etl_app = typer.Typer(help="ETL: ingest and enrich data")
features_app = typer.Typer(help="Feature engineering")
viz_app = typer.Typer(help="Charts and figure exports")
ui_app = typer.Typer(help="Run interactive dashboards (Dash/Streamlit)")

app.add_typer(etl_app, name="etl")
app.add_typer(features_app, name="features")
app.add_typer(viz_app, name="viz")
app.add_typer(ui_app, name="app")


# ---------- app-level options ----------

@app.callback()
def main(
    ctx: typer.Context,
    env_file: Optional[Path] = typer.Option(
        ".env",
        exists=False,
        dir_okay=False,
        readable=True,
        help="Path to a .env file with secrets and config (optional).",
    ),
    log_level: Optional[str] = typer.Option(
        None,
        help="Override logging level for this run (DEBUG/INFO/WARNING/ERROR).",
    ),
):
    """
    Initializes config and logging for the whole CLI process.
    """
    # Load config; allow overriding the default .env path
    cfg = load_config(env_file=env_file, quiet=True)

    # If user provided --log-level, override; else use cfg.log_level
    effective_level = (log_level or cfg.log_level or "INFO").upper()
    setup_logging(level=effective_level)

    print_kv_table("Runtime", {
        "env_file": str(env_file) if env_file else "",
        "log_level": effective_level,
    })

    # Stash cfg on context for subcommands
    ctx.obj = cfg


# ---------- top-level convenience ----------

@app.command("version")
def version_cmd():
    """Print MWI version and exit."""
    from . import __version__
    typer.echo(f"MWI version {__version__}")


@app.command("doctor")
def doctor():
    """
    Quick environment check: paths, keys, and expected IMDb TSV file presence.
    """
    cfg = get_config()
    cfg.summarize()
    missing_tsv = [k for k, p in cfg.imdb_tsv_paths().items() if not p.exists()]
    if missing_tsv:
        logger.warning("Missing IMDb TSVs: {}", ", ".join(missing_tsv))
    else:
        logger.success("All expected IMDb TSVs are present.")
    typer.echo("Doctor check complete.")


# ---------- ETL commands (stubs for now) ----------

@etl_app.command("imdb-dump")
def etl_imdb_dump():
    """
    Load IMDb non-commercial TSVs into Parquet and a DuckDB cache.
    Usage:
        python -m mwi.cli etl imdb-dump
    """
    from .etl.imdb_dump import load_all_imdb  # local import to speed CLI startup

    cfg = get_config()
    parquet_dir = Path(cfg.processed_dir) / "imdb"
    duckdb_file = Path(cfg.cache_dir) / "imdb.duckdb"

    print_kv_table("IMDb ETL Outputs", {
        "parquet_dir": parquet_dir,
        "duckdb_file": duckdb_file,
    })

    with Timer("etl::imdb_dump_all"):
        counts = load_all_imdb(
            dump_dir=cfg.imdb_dump_dir,
            out_parquet_dir=parquet_dir,
            duckdb_path=duckdb_file,
        )
    logger.success("IMDb ingest complete: {}", counts)



@etl_app.command("fuse-master")
def etl_fuse_master():
    """
    Merge your watchlist CSV (treated as 'watched') with IMDb Parquet dumps
    into a canonical master table ready for enrichment.
    """
    from .etl.fuse_master import fuse_watchlist_with_imdb

    with Timer("etl::fuse_master"):
        out = fuse_watchlist_with_imdb()
    logger.success("Fuse complete → {}", out)



# ---------- Features (stub) ----------

@features_app.command("build")
def features_build():
    """Compute derived fields (decade, season, rolling trends, networks...)."""
    with Timer("features_build"):
        logger.info("Feature engineering will be implemented after ETL steps.")


# ---------- Visualization (stub) ----------

@viz_app.command("export-figures")
def export_figures(output_dir: Path = typer.Option(Path("reports/figures"), help="Where to save images")):
    """Generate and export a curated set of figures (matplotlib / seaborn / plotly)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with Timer("viz_export_figures"):
        logger.info("Chart exports will be implemented after features.")


# ---------- UI (stub) ----------

@ui_app.command("dash")
def run_dash(port: int = 8050):
    """Run the Plotly Dash app."""
    cfg = get_config()
    with Timer("app_dash"):
        logger.info("Dash app will be implemented after viz. Target port={}", port)
        logger.info("Preferred default from config: {}", cfg.dashboard_default)


@ui_app.command("streamlit")
def run_streamlit(port: int = 8501):
    """Run the Streamlit app."""
    cfg = get_config()
    with Timer("app_streamlit"):
        logger.info("Streamlit app will be implemented after viz. Target port={}", port)
        logger.info("Preferred default from config: {}", cfg.dashboard_default)


if __name__ == "__main__":
    # Allow: python src/mwi/cli.py (if src on PYTHONPATH) or python -m mwi.cli
    app()
