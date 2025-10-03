# Movie Watching Insights (MWI)

An end-to-end, reproducible, **interactive analytics project** that reveals
my movie-watching tendencies across years — with **rich enrichment** from
IMDb Non-Commercial datasets, OMDb, TMDb, DoesTheDogDie, and Cinemorgue.
It’s built to **auto-update** as I watch new movies.

---

## 🔥 What this project does

- **ETL + Enrichment pipeline**
  - Ingest IMDb export(s) (watchlist/ratings/history).
  - Join with **IMDb Non-Commercial TSVs** (basics, crew, ratings, principals, akas).
  - Fill gaps via **OMDb** and **TMDb**.
  - Pull content-warning topics via **DoesTheDogDie** (DDD).
  - (Optional) scrape **Cinemorgue** metadata (e.g., character death notes).
- **Analytics**
  - Yearly trends (count watched, genres, runtimes, languages, countries).
  - Director/actor streaks & phases.
  - Discovery vs. rewatch behavior.
  - Rating drift (mine vs IMDb), runtime phases, seasonality (month/day-of-week).
  - Networks & communities (director–actor graphs).
- **Interactive UI**
  - **Dashboards** (Plotly Dash / Streamlit) with filters, cross-highlighting,
    bookmarks, and shareable views.
  - **Notebooks** for deep dives (matplotlib + seaborn + plotly).
- **Reproducibility & Speed**
  - DuckDB/SQLite-backed cache, idempotent ETL steps, **timers** and **rich**
    progress bars, **loguru** logging.
  - Incremental updates as new movies are added.

---

## 🧱 Tech stack

- **Python 3.11+**
- **pandas**, **duckdb**, **polars** (fast joins on TSVs), **pyarrow**
- **matplotlib**, **seaborn**, **plotly**, **plotly-express**
- **dash** or **streamlit** (choose at run-time)
- **typer** (CLI), **loguru** (logging), **rich** (progress/timing)
- **requests** / **httpx** with **tenacity** (retries)
- **SQLModel** or **SQLAlchemy** for small metadata tables (optional)

---

## 📁 Folder structure

movie-watching-insights/
├─ README.md
├─ .gitignore
├─ .env # your real secrets (NOT committed)
├─ .env.example # template (SAFE to commit)
├─ pyproject.toml # project config (coming next)
├─ requirements.txt # pinned deps (coming next)
├─ src/
│ └─ mwi/
│ ├─ init.py
│ ├─ config.py # config loader (env vars, paths)
│ ├─ log.py # loguru setup, timers
│ ├─ cli.py # Typer CLI entrypoint
│ ├─ etl/
│ │ ├─ imdb_dump.py # IMDb TSV ingestion (DuckDB/Polars)
│ │ ├─ enrich_omdb.py # OMDb enrichment
│ │ ├─ enrich_tmdb.py # TMDb enrichment
│ │ ├─ enrich_ddd.py # DoesTheDogDie enrichment
│ │ ├─ enrich_cinemorgue.py# optional: scraping helpers
│ │ └─ fuse_master.py # merge all sources into one model
│ ├─ features/
│ │ ├─ build_features.py # derived fields (decade, seasonality, etc.)
│ │ └─ networks.py # actor/director graphs
│ ├─ viz/
│ │ ├─ charts_matplotlib.py
│ │ ├─ charts_seaborn.py
│ │ └─ charts_plotly.py
│ └─ app/
│ ├─ dashboard_dash.py # Plotly Dash app
│ └─ dashboard_streamlit.py
├─ data/
│ ├─ raw/
│ │ ├─ imdb/ # the non-commercial TSVs (gzipped)
│ │ └─ external/ # OMDb/TMDb cached responses (JSON)
│ ├─ processed/ # parquet/csv results
│ └─ cache/ # duckdb.db, sqlite.db, temp join tables
├─ notebooks/
│ ├─ 01_explore.ipynb
│ ├─ 02_trends.ipynb
│ └─ 03_networks.ipynb
├─ dashboards/
│ └─ assets/ # CSS, logos
├─ reports/
│ └─ figures/ # exported PNG/SVGs
├─ tests/
│ └─ test_etl.py
└─ .github/workflows/
└─ ci.yaml # lint/test (optional)

---

## 🔐 Secrets & Config

**Never commit real keys.** Use `.env` locally and commit only `.env.example`.

Environment variables (what the code expects):
Core paths (customize to your machine)
MWI_DATA_DIR=./data
MWI_IMDB_DUMP_DIR=./data/raw/imdb
MWI_CACHE_DIR=./data/cache
MWI_PROCESSED_DIR=./data/processed
IMDb exported CSV (your watchlist/ratings export from IMDb web UI)
MWI_IMDB_WATCHLIST_CSV=./data/raw/imdb_watchlist.csv
External APIs (put real keys ONLY in .env, keep example safe)
OMDB_API_KEY=
TMDB_API_KEY=
TMDB_API_READ_ACCESS_TOKEN=
DDD_API_KEY=
Optional: seedbox indexing (do NOT store real creds here)
SEEDBOX_HOST=
SEEDBOX_USER=
SEEDBOX_PASS=
SEEDBOX_PATHS=torrents/rtorrent,torrents/deluge,torrents/qbittorrent

---

## 🧬 Data sources

- **IMDb Non-Commercial datasets (TSV GZ)**
  - `title.basics`, `title.ratings`, `title.akas`, `title.crew`, `title.principals`, etc.
- **Your IMDb exports** (CSV): watchlist / ratings / history.
- **OMDb API**: missing fields, poster, plot, languages, countries, awards…
- **TMDb API**: alternative titles, images, collections, keywords.
- **DoesTheDogDie API**: content-warning topics (e.g., “a dog dies”).
- **Cinemorgue (optional)**: scrape character-death notes.

---

## 🚀 Typical workflow

1. **Put data in place**

   - Download IMDb dumps (TSVs) into `data/raw/imdb/`.
   - Save your IMDb export CSV to `data/raw/imdb_watchlist.csv`.
   - Create a local `.env` with your real keys and paths.

2. **Install & run ETL**

   - `python -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt`
   - `python -m mwi.cli etl imdb-dump`
   - `python -m mwi.cli etl fuse-master` (joins all sources)
   - `python -m mwi.cli features build`

3. **Explore**

   - `python -m mwi.cli viz export-figures`
   - Open `dashboards`:
     - `python -m mwi.cli app dash` (or) `python -m mwi.cli app streamlit`

4. **Incremental updates**
   - Re-run `etl` commands anytime you add new movies; the pipeline is idempotent and caches by IDs.

---

## 📝 Roadmap (what we’ll add next, file-by-file)

1. **`.gitignore` & `.env.example`** (security first).
2. **`requirements.txt`** with pinned versions.
3. **`pyproject.toml`** and basic package `src/mwi/...`.
4. **`src/mwi/log.py`** (loguru + timing helpers).
5. **`src/mwi/config.py`** (env-driven config loader).
6. **`src/mwi/cli.py`** (Typer CLI scaffolding).
7. **`src/mwi/etl/imdb_dump.py`** (DuckDB/Polars ingestion of TSVs).
8. **`src/mwi/etl/fuse_master.py`** (joins CSV+TSVs into canonical table).
9. **`src/mwi/etl/enrich_*.py`** (OMDb, TMDb, DDD; cached requests).
10. **`src/mwi/features/*.py`** (decades, seasonality, network prep).
11. **`src/mwi/viz/*.py`** (matplotlib/seaborn/plotly chart factory).
12. **Dash app** & **Streamlit app**.
13. **Notebooks** (curated stories with the richest plots).
14. **CI hooks** (optional) and data quality checks.

---

## ⚖️ Licensing & Non-Commercial Use

- Respect IMDb’s **Non-Commercial** license for TSVs.
- Follow each API’s Terms of Use; cache responsibly; no bulk redistribution of raw data.

---

## 🧑🏽‍💻 About

Built for a power user who loves classic cinema and wants **serious, explainable, beautiful analytics** — with professional engineering practices (small commits, comments, timers, and tests).
