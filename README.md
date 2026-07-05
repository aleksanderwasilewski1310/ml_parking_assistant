# 🚗 Smart Parking Assistant API

[![Python](https://img.shields.io/badge/python-%3E%3D3.13-blue?style=flat-square&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-%3E%3D0.139-blueviolet?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PySpark](https://img.shields.io/badge/PySpark-%3E%3D4.1.2-orange?style=flat-square&logo=apache-spark)](https://spark.apache.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-%3E%3D2.13.4-009688?style=flat-square)](https://pydantic.dev/)
[![Uvicorn](https://img.shields.io/badge/Uvicorn-%3E%3D0.49.0-444444?style=flat-square)](https://www.uvicorn.org/)

A production-ready REST API for parking occupancy forecasting using FastAPI and PySpark. This repository demonstrates model training, inference, and lifecycle management for a smart parking assistant service.

---

## Overview

VW Smart Parking Assistant API provides:

- Real-time parking occupancy prediction for road segments
- A PySpark-based training pipeline using historical telemetry and weather data
- Adaptive alternative road segment recommendations for active and standard driver profiles
- Health and readiness monitoring endpoints

---

## Architecture

```mermaid
flowchart LR
  A[CSV Source Files] -->|Ingest| B[read_data.py]
  B -->|Feature Merge| C[Unified Spark DataFrame]
  C -->|Train| D[train_model.py]
  D -->|Persist| E[In-memory Model State]
  E -->|Serve| F[api.py]
  F -->|Expose| G[FastAPI Endpoints]
  G --> H[Client Applications]
```

```mermaid
flowchart TD
  subgraph API
    API1[POST /predict]
    API2[GET /health]
  end
  subgraph ML
    Trainer[ParkingModelTrainer]
    Spark[PySpark Pipeline]
    Profiles[Historical Profiles]
  end
  Client --> API1
  Client --> API2
  API1 --> Spark
  Spark --> Profiles
  Spark --> Trainer
  Trainer --> API1
```

---

## Project Structure

- `api.py` — FastAPI application, lifecycle management, prediction endpoints, and model retraining orchestration
- `train_model.py` — Model training, feature assembly, prediction logic, and feature importance extraction
- `read_data.py` — Data ingestion, cleaning, filtering, and dataset consolidation
- `groundtruth.csv` — Parking telemetry ground-truth dataset
- `road_features.csv` — Road segment metadata and static attributes
- `weather_features.csv` — Weather observations merged into the training dataset
- `compose.yaml` — Container composition for local orchestration
- `Dockerfile` — Docker image definition for containerized deployment
- `requirements.txt` — Python dependency manifest
- `pyproject.toml` — Packaging metadata and tool configuration

---

## Installation

### Prerequisites

- Python 3.13+
- Java 11 or 17
- `pip` installed

### Python environment

```bash
python -m venv .venv
source .venv/Scripts/activate    # Windows
# or
source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

### Optional: Docker

Build and run the container locally:

```bash
docker build -t vw-parking-assistant .
docker run --rm -p 5000:5000 vw-parking-assistant
```

---

## Running Locally
Save input files in the working directory

Start the API server with Uvicorn:

```bash
uvicorn api:APP --host 0.0.0.0 --port 5000
```

The service will initialize the Spark session, perform the first training pass, and begin serving requests.

---

## API Endpoints

### `GET /health`

Returns service health and model readiness.

Response model:

- `status`: `ok`
- `model_ready`: boolean
- `last_trained_at_utc`: timestamp or null

### `POST /predict`

Request body:

- `road_segment_id`: `string`
- `timestamp`: `datetime` (ISO-8601)
- `driver_profile`: `active | standard | inactive`

Response model:

- `road_segment_id`
- `timestamp`
- `driver_profile`
- `occupancy_probability`
- `top_alternatives` (optional list)

Example request:

```json
{
  "road_segment_id": "21976",
  "timestamp": "2026-07-03T08:30:00",
  "driver_profile": "active"
}
```

---

## Data Pipeline

1. `read_data.py` loads and cleans the CSV files
2. Datasets are joined using `road_segment_id` and `timestamp`
3. `train_model.py` prepares features, computes target-encoded historical ratios, and trains a RandomForest pipeline
4. `api.py` caches the trained model and serves inference requests
5. A background retraining thread refreshes the model hourly

---

## Notes
- The API expects complete data and drops rows containing null values during ingestion.
- Historical occupancy ratios are computed from the training set and filled with `0.5` when data is missing during inference.
- The initial startup may take time because Spark initialization and model training occur before the server becomes ready.

---
