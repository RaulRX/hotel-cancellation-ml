# Hotel Cancellation ML API

REST API built with FastAPI that exposes the training, inference and evaluation
flow of the hotel booking cancellation classifier.

## Architecture

```
api/
├── main.py              # FastAPI app factory + router wiring
├── core/
│   ├── logging.py       # Structured logging setup
│   └── errors.py        # Exception → HTTP response mapping
├── routers/
│   ├── health.py        # GET /health, /ready
│   ├── models.py        # GET /models, POST /models/reload
│   ├── train.py         # POST /train (async), GET /train/jobs/{id}, GET /train/jobs
│   ├── predict.py       # POST /predict
│   └── evaluate.py      # POST /evaluate
├── schemas/             # Pydantic request/response models
└── services/
    ├── model_service.py # Lazy-loaded singleton wrapping the persisted Pipeline
    └── job_store.py     # In-memory training job registry
```

Design decisions:

- **Single sklearn Pipeline as the artifact.** Preprocessing and the final
  estimator are persisted together so `/predict` and `/evaluate` cannot drift
  from the training-time transformations.
- **Async training via `BackgroundTasks`.** `/train` returns `202 Accepted`
  immediately with a `job_id`; the client polls `/train/jobs/{id}`. This avoids
  blocking the HTTP connection while several models train.
- **Singleton `ModelService`.** Caches the loaded `Pipeline` in memory and is
  invalidated automatically after a successful training run, so the next
  `/predict` picks up the new model without restarting the process.
- **Dependency injection via `Depends`.** Routers do not import the singletons
  directly; this makes the services swappable (e.g., a real DB-backed
  `JobStore`) without touching the endpoints.
- **Centralised error handling.** Domain exceptions (`DatasetNotFoundError`,
  missing model) map to consistent JSON bodies via `register_exception_handlers`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
pip install -r requirements.txt
```

## Run

```bash
uvicorn api.main:app --reload
```

Open the interactive docs at <http://localhost:8000/docs>.

## Endpoints

| Method | Path                    | Description                                      |
|--------|-------------------------|--------------------------------------------------|
| GET    | `/health`               | Liveness probe.                                  |
| GET    | `/ready`                | Readiness: checks the model artifact exists.     |
| GET    | `/models`               | Metadata about the currently loaded model.       |
| POST   | `/models/reload`        | Force a reload of the artifact from disk.        |
| POST   | `/train`                | Launch a training run (async, returns job_id).   |
| GET    | `/train/jobs/{job_id}`  | Status + leaderboard of a training job.          |
| GET    | `/train/jobs`           | List of all training jobs in the current process.|
| POST   | `/predict`              | Score a batch of booking records.                |
| POST   | `/evaluate`             | Evaluate the persisted model on a holdout split. |

### Example: train

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"primary_metric": "f1", "random_state": 42}'
```

Response (`202 Accepted`):

```json
{
  "job_id": "f8c1...-...-...",
  "status": "pending",
  "status_url": "/train/jobs/f8c1...-...-..."
}
```

Poll for status:

```bash
curl http://localhost:8000/train/jobs/<job_id>
```

### Example: predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "records": [
          {"hotel": "Resort Hotel", "lead_time": 342, "arrival_date_year": 2015,
           "arrival_date_month": "July", "arrival_date_week_number": 27,
           "arrival_date_day_of_month": 1, "stays_in_weekend_nights": 0,
           "stays_in_week_nights": 0, "adults": 2, "children": 0, "babies": 0,
           "meal": "BB", "country": "PRT", "market_segment": "Direct",
           "distribution_channel": "Direct", "is_repeated_guest": 0,
           "previous_cancellations": 0, "previous_bookings_not_canceled": 0,
           "reserved_room_type": "C", "assigned_room_type": "C",
           "booking_changes": 3, "deposit_type": "No Deposit",
           "agent": null, "company": null, "days_in_waiting_list": 0,
           "customer_type": "Transient", "adr": 0,
           "required_car_parking_spaces": 0, "total_of_special_requests": 0}
        ]
      }'
```

### Example: evaluate

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"test_size": 0.2, "random_state": 42}'
```

## Environment variables

| Variable             | Default                       | Purpose                                  |
|----------------------|-------------------------------|------------------------------------------|
| `LOG_LEVEL`          | `INFO`                        | Root logger level.                       |
| `DATASET_CSV`        | `data/raw/dataset.csv`        | Override the dataset path.               |
| `DATA_RAW_DIR`       | `data/raw`                    | Raw data directory.                      |
| `DATA_PROCESSED_DIR` | `data/processed`              | Processed data directory.                |
| `MODELS_DIR`         | `models`                      | Where `best_model.pkl` is persisted.     |
| `OUTPUTS_DIR`        | `outputs`                     | Plots, reports, metrics dumps.           |

## Error model

All errors return:

```json
{"error": "<machine_code>", "detail": "<human readable message>"}
```

Common codes: `dataset_not_found`, `model_not_trained`, `invalid_input`,
`validation_error`, `internal_error`.
