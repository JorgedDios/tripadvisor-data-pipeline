# TripAdvisor Rating Prediction — End-to-End Data Pipeline

**Can we predict the average rating a new European restaurant will get on TripAdvisor *before* it opens?**

An end-to-end distributed data architecture over ~1M European restaurants: orchestrated ETL, streaming ingestion, and distributed machine learning. Individually designed, built and defended as the year-long project for *Sistemas Distribuidos de Procesamiento de Datos II* (Data Science & Engineering BSc, Universidad Rey Juan Carlos, 2025/26).

**Stack:** Apache Airflow · Apache Kafka · Apache Spark (Structured Streaming + MLlib) · Polars · PyArrow · Docker

---

## The problem

Opening a restaurant is a high-uncertainty investment. An investment group planning a new venue — a given cuisine, price range and city — would like to estimate whether it will score well, *before* committing capital. The venue does not exist yet, so it has no reviews and no rating history.

That constraint drives every design decision in this project: the model may only use **structural attributes a restaurant has on day zero**, never anything derived from operating history.

---

## Architecture

```
                 tripadvisor_european_restaurants.csv  (~1M rows)
                                  │
   ┌──────────────────────────────▼──────────────────────────────┐
   │  PHASE 1 — ETL orchestration   (Apache Airflow DAG, Polars) │
   │                                                              │
   │   extract_and_validate ──► transform_and_partition ──► load_to_kafka
   │           │                         │                       │
   │           ▼                         ▼                       │
   │   dlq/corrupted_         processed_lake/Spain/              │
   │   records.parquet        (by country, PyArrow)      watermark.json
   └──────────────────────────────┬──────────────────────────────┘
                                  │  JSON events
                    ┌─────────────▼─────────────┐
                    │  Apache Kafka (Docker)     │
                    │  topic: tripadvisor_restaurants
                    └─────────────┬─────────────┘
                                  │
   ┌──────────────────────────────▼──────────────────────────────┐
   │  PHASE 2 — Spark Structured Streaming                        │
   │   StructType schema ──► from_json ──► unbounded DataFrame    │
   │     • Append mode:   gluten-free & rating ≥ 4.5 → salida1_txt/
   │     • Complete mode: rolling count by country → console      │
   └──────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────┐
   │  PHASE 3 — Spark MLlib (batch)                               │
   │   Linear Regression  vs  Random Forest Regressor             │
   │   → RMSE, R², feature importances                            │
   └──────────────────────────────────────────────────────────────┘
```

---

## Results

Trained on **779,745 validated records**, 80/20 train/test split with a fixed seed for reproducibility. The metrics below come from the original run over the full dataset.

| Model (Spark MLlib) | RMSE | R² |
|---|---|---|
| Linear Regression (baseline) | 0.6869 | 0.0668 |
| Random Forest (50 trees, depth 5) | **0.6824** | **0.0790** |

### What actually drives a restaurant's rating

Random Forest feature importances:

| Feature | Importance | Reading |
|---|---|---|
| `is_claimed` | **34.36%** | Owners who claim and actively manage their profile gain a measurable rating advantage. The single strongest structural factor. |
| `city_avg_rating` | **25.33%** | Opening in a city that already rates well pulls the new venue up with it. |
| `is_veg_friendly` | 18.28% | — |
| `is_gluten_free` | 10.60% | Dietary niches together account for ~29% of model importance. |
| `total_reviews_count` | 11.43% | Volume stabilises the average; it does not raise it. |
| `price_level_num` | **0.00%** | Price was discarded entirely as a predictor — diners rate the quality/expectation ratio, not the price tag. |

### Honest read of these numbers

An RMSE of 0.68 stars on a 5-point scale looks acceptable until you look at the distribution: TripAdvisor ratings cluster heavily between 3.5 and 5.0 (global mean 4.03, left-skewed). In that density, a 0.68 error is **large** — it blurs the line between an average restaurant and an excellent one. An R² of 0.08 says the same thing plainly.

The diagnosis is not a modelling failure, it is a feature-space limit. The model knows a restaurant's *structure* but nothing about its *substance*: food quality, service, atmosphere, waiting times. None of that is encoded in these variables.

**The way past this ceiling is NLP over the free-text reviews**, extracting sentiment as additional features — not more trees or deeper ones.

---

## Engineering decisions worth reading

### Data leakage prevention
The raw dataset includes fields collected *after* a restaurant is operating: counts of "excellent"/"terrible" votes, and sub-scores for service and atmosphere. Using them to predict the overall rating would be circular — and a brand-new restaurant would never have them. All of them were dropped. The model only sees attributes that exist at planning time.

### Dead Letter Queue
Corrupt records (missing coordinates, malformed rows) are never silently dropped — that destroys the traceability of the error. The DAG intercepts them and routes them to `dlq/corrupted_records.parquet`. The main flow never collapses on an unexpected exception, and the rejected rows stay available to monitor, alert on, or fix at source.

### Idempotency via offset watermarking
Re-running the DAG blindly would re-inject a million rows into Kafka and corrupt every downstream consumer. The source has no update timestamps, so the pipeline persists its own checkpoint: `watermark.json` holds the number of source CSV rows already consumed — rows published to Kafka plus rows diverted to the DLQ, which count as processed and are never retried — which is also the 0-based index of the next row to read. On each run the first task reads it and takes the next slice of the CSV, `[watermark, watermark + max_records_per_run)` (or everything left if the cap is `0`); the DLQ split, transformation and publishing all operate on exactly that slice. The watermark only moves forward, to the end of the slice, after Kafka has acknowledged every message of the batch; if any delivery fails the task errors out and the next run retries the same slice. If there are no new rows, the load is skipped with *"Carga en Kafka omitida. No hubo datos nuevos"*. Consecutive runs therefore walk through the file without gaps or overlaps, and the DAG allows only one active run so two runs can never read the same watermark. The guarantee is at-least-once rather than strict exactly-once: a run that fails *after* some messages were delivered will republish its slice on the next run.

### Partitioned data lake
After preprocessing, PyArrow writes the output as a lake partitioned by country, one directory per value (`processed_lake/Spain/`, `processed_lake/Italy/`, …). Downstream jobs can then open only the countries they need instead of scanning the continent. Each batch writes its own files (`lote-<first source row>-<n>.parquet`), so the lake accumulates batches across runs, a retried batch overwrites its own files instead of duplicating them, and `load_to_kafka` publishes only the files of the current batch.

### Competitive context with window functions
A `GROUP BY` would collapse individual restaurants. Using Polars window functions (`.over('city')`), the city average is computed in parallel and joined back onto each row, producing `rating_diff_city`. This matters: a 4.0 is a success in a town averaging 3.5 and a failure in a city averaging 4.5.

### Append vs Complete output modes
Both streaming modes are implemented deliberately, because Spark's choice is forced by the query shape:
- **Append** — a row-by-row filter (gluten-free, rating ≥ 4.5) with no aggregation. Spark requires append; results are written incrementally and immutably.
- **Complete** — a `.count()` aggregation by country. Totals mutate with every event, so Spark requires the full result to be rewritten on each micro-batch.

---

## Resolved issue: `country` arriving as `NULL`

**Symptom.** In the Complete-mode aggregation, every event was grouped under a single `NULL` country, which made the per-country count meaningless.

**Root cause.** It came from the Phase 1 storage layout, not from the streaming layer. Partitioning the lake by `country` lifts that column out of the Parquet files and encodes it only in the directory names (`processed_lake/Spain/…`). The `load_to_kafka` task then read the files with a flat glob (`pl.read_parquet("…/**/*.parquet")`), which does not reconstruct the partition key, so `country` never made it into the JSON payload and the strict `StructType` schema in the consumer initialised it as null.

**Fix.** `load_to_kafka` now reads the lake with `pyarrow.dataset`, declaring the same `country` partitioning used when writing it, so the column is rebuilt from the directory names before publishing. The task fails loudly if `country` is ever missing again instead of silently publishing nulls.

---

## Project structure

```
.
├── dags/                 # Airflow DAG: extract → transform → load
├── spark_streaming/      # Structured Streaming consumer
├── machine_learning/     # Spark MLlib training and evaluation
├── notebooks/            # Exploratory data analysis
├── utils/                # Shared helpers
├── config.toml           # Paths, Kafka settings, cleaning rules
├── docker-compose.yml    # Local Kafka broker
└── pyproject.toml
```

Configuration is kept out of the code: paths, Kafka endpoint and topic, streaming output directories, critical columns and numeric columns all live in `config.toml`, and every script reads it through `utils/configuracion.py`.

---

## Running it

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- JDK 11 with `JAVA_HOME` set (required by Spark)

Environment variables (documented in [`.env.example`](.env.example)):

| Variable | Required | Purpose |
|---|---|---|
| `JAVA_HOME` | Yes, for Spark | JDK used by the streaming and ML scripts. It is never overwritten by the code; the scripts warn if it is missing. |
| `SDPD2_HOME` | No | Project root. Defaults to the repository root. All `[paths]` entries in `config.toml` are relative and resolved against `$SDPD2_HOME/<data_dir>` (`data/` by default). |
| `AIRFLOW_HOME` | Yes, for Airflow | Airflow working directory (the repository root in the steps below). |

### Dataset
Download `tripadvisor_european_restaurants.csv` from [Kaggle](https://www.kaggle.com/datasets/stefanoleone992/tripadvisor-european-restaurants) and place it in `data/` (or change `[paths] raw_csv` in `config.toml`). The file is not committed — it is ~1M rows.

### Upgrading from an earlier version
Before the first run with this version, delete `data/processed_lake/` and `data/watermark.json`. A lake built by the previous code holds `part-0.parquet` files; mixed with the new per-batch `lote-*.parquet` files, `entrenar_modelos.py` (which reads every Parquet file in the lake) would count those rows twice. The first run then rebuilds the lake from row 0.

### 1. Environment
```bash
git clone https://github.com/JorgedDios/tripadvisor-data-pipeline.git
cd tripadvisor-data-pipeline
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install apache-airflow polars pyarrow confluent-kafka tomli pyspark==3.5.0
```

### 2. Phase 1 — Kafka and the ETL DAG
```bash
docker-compose up -d
export AIRFLOW_HOME=$(pwd)
airflow standalone
```
Open `http://localhost:8080`, log in with the credentials printed in the terminal, enable the `tripadvisor_etl_pipeline` DAG and trigger it. It builds the data lake and starts publishing JSON events to the `tripadvisor_restaurants` topic. By default a run processes every pending row of the CSV (`[kafka] max_records_per_run = 0` in `config.toml`). For a quick demo on a single-node local broker you can set a cap, e.g. `5000`: each run then processes the next 5,000 rows, so it publishes at most 5,000 records, and triggering the DAG again continues from where the previous run stopped. Keep the cap for demos only: the window functions run per batch, so with a cap `city_avg_rating` (and `rating_diff_city`) is the city average within each slice rather than over the full dataset, and that is the model's second most important feature (~25% of the Random Forest importance).

### 3. Phase 2 — Streaming consumer
In a second terminal, with the venv active and Kafka running:
```bash
python spark_streaming/consumidor_tripadvisor.py
```
Spark consumes the topic, prints the per-country count per micro-batch, and writes high-rated gluten-free restaurants to `salida1_txt/`.

### 4. Phase 3 — Model training
```bash
python machine_learning/entrenar_modelos.py --modelo ambos   # or: lr | rf
```
Outputs RMSE and R² per model, the linear regression coefficients and the Random Forest feature importance ranking, plus a side-by-side comparison when both are trained.

---

## Next steps

- **NLP on review text** — sentiment features are the only credible route past the current R² ceiling.
- **Streaming inference** — the trained Random Forest is saved to a model registry; the streaming job loads it with `.load()` and scores incoming venues with `.transform()`, writing predictions to PostgreSQL or back to a Kafka topic instead of the console.
---

## Author

**Jorge de Dios Orellana** — [LinkedIn](https://www.linkedin.com/in/jorge-de-dios-orellana/) · [GitHub](https://github.com/JorgedDios)

Final-year Data Science and Engineering student, Universidad Rey Juan Carlos.
Every phase of this project — architecture, ETL orchestration, streaming layer, modelling and documentation — was designed and implemented individually.

## License

MIT — see [LICENSE](LICENSE).
