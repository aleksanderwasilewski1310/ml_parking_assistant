"""api.py — FastAPI REST API for the VW Smart Parking Assistant.

Key improvements over legacy implementations:
  - Full Pydantic V2 alignment (using modern 'examples' list mapping).
  - FastAPI lifespan state coordinator manages pristine initialization and cleanup.
  - Windows & Virtual Environment environment bugfix forced for PySpark workers.
  - Replaced deprecated timezone-naive datetime constructs with explicit UTC calls.
  - Auto-generated interactive docs available out-of-the-box at /docs (Swagger UI).
"""

# pylint: disable=import-error
import os
import sys
import logging
import threading
from contextlib import asynccontextmanager  # pylint: disable=no-name-in-module
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.errors import AnalysisException
import pyspark.sql.functions as F

from read_data import create_dataframe
from train_model import ParkingModelTrainer
# pylint: enable=import-error

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("VW-Parking-API")

# --- Constants & Configuration ---
DATA_PATHS: Dict[str, str] = {
    "groundtruth": "groundtruth.csv",
    "road_features": "road_features.csv",
    "weather_features": "weather_features.csv",
}

RETRAIN_INTERVAL_SECONDS: int = 3600


# --- Thread-Safe Global Environment State ---
class EngineState:
    def __init__(self):
        self.trainer: Optional[object] = None
        self.historical_profiles: Optional[object] = None
        self.last_trained_at: Optional[datetime] = None


MODEL_LOCK = threading.Lock()
STATE = EngineState()


# --- Data Validation Schemas (Pydantic V2) ---
# pylint: disable=too-few-public-methods
class DriverProfile(str, Enum):
    """Supported telemetry routing depth criteria profiles."""

    ACTIVE = "active"
    STANDARD = "standard"
    INACTIVE = "inactive"


class PredictionRequest(BaseModel):
    """Inbound telemetry request blueprint with structural JSON validation."""

    road_segment_id: str = Field(
        ...,
        description="Unique database identifier of the target road segment.",
        examples=["21976"],
    )
    timestamp: datetime = Field(
        ...,
        description="ISO-8601 compliant datetime stamp for the prediction instance.",
        examples=["2026-07-03T08:30:00"],
    )
    driver_profile: DriverProfile = Field(
        default=DriverProfile.INACTIVE,
        description="Driver persona parameter dictating routing search criteria depth.",
    )


class AlternativeSegment(BaseModel):
    """A granular alternative parking node prediction matrix output."""

    road_segment_id: str
    occupancy_probability: float


class PredictionResponse(BaseModel):
    """Outbound prediction response contract transmitted back to client nodes."""

    road_segment_id: str
    timestamp: datetime
    driver_profile: DriverProfile
    occupancy_probability: float = Field(
        ...,
        description="Calculated probability vector that target segment is occupied (0.0 to 1.0).",
    )
    top_alternatives: Optional[List[AlternativeSegment]] = Field(
        default=None,
        description="Ranked list of contextually optimized alternatives.",
    )


class HealthResponse(BaseModel):
    """Monitoring schema mapping the infrastructure vital components."""

    status: str
    model_ready: bool
    last_trained_at_utc: Optional[str]


# --- Model Pipeline Processing Core ---
def _load_and_train() -> None:
    """Reads transactional data, constructs features, and fits ML pipelines."""

    LOGGER.info("Initiating model training loop and temporal matrix generation...")

    try:
        df_data: DataFrame = create_dataframe(DATA_PATHS, LOGGER)

        trainer = ParkingModelTrainer()
        base_df = trainer.prepare_features(df_data).withColumn(
            "is_occupied", F.when(F.col("available") == 0, 1).otherwise(0)
        )

        raw_train, _ = base_df.randomSplit([0.8, 0.2], seed=42)

        historical_profiles: DataFrame = raw_train.groupBy(
            "road_segment_id", "hour", "day_of_week"
        ).agg(F.avg("is_occupied").alias("historical_occupancy_ratio"))
        historical_profiles.cache()

        train_data = raw_train.join(
            historical_profiles,
            on=["road_segment_id", "hour", "day_of_week"],
            how="left",
        )
        trainer.build_and_train_pipeline(train_data, LOGGER)
        trainer.get_feature_importance(LOGGER)

        with MODEL_LOCK:
            STATE.trainer = trainer
            STATE.historical_profiles = historical_profiles
            STATE.last_trained_at = datetime.now(timezone.utc)

            is_empty = STATE.historical_profiles.isEmpty() if STATE.historical_profiles else True
            LOGGER.info(
                "Thread-bound context updated safely. Trainer object: %s | "
                "Profiles loaded successfully: %s | Timestamp: %s",
                type(STATE.trainer).__name__,
                "NO (Empty)" if is_empty else "YES (Active)",
                STATE.last_trained_at.isoformat(),
            )

    except AnalysisException as spark_exc:
        LOGGER.error("Spark engine calculation failed: %s", spark_exc, exc_info=True)
    except OSError as io_exc:
        LOGGER.error("File system reading or caching breakdown: %s", io_exc, exc_info=True)
    except Exception as exc:
        LOGGER.error("Unexpected failure in pipeline execution: %s", exc, exc_info=True)


def _schedule_retraining() -> None:
    """Cyclical worker task loop checking routines over standard intervals."""
    LOGGER.info(
        "Asynchronous batch daemon online checking pattern: %ss",
        RETRAIN_INTERVAL_SECONDS,
    )
    stop_event = threading.Event()
    while not stop_event.wait(timeout=RETRAIN_INTERVAL_SECONDS):
        LOGGER.info("Timed interval interrupt detected. Triggering background pipeline update...")
        _load_and_train()


# --- FastAPI Lifecycle Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Coordinates explicit infrastructure boots and deterministic system teardowns."""
    LOGGER.info("Warming local operating kernel context. Building PySpark Session context...")

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    app.state.spark = (
        SparkSession.builder.appName("VW_SmartParking_API")
        .config("spark.driver.memory", "4g")
        .config(
            "spark.driver.extraJavaOptions",
            "--add-opens=java.base/javax.security.auth=ALL-UNNAMED "
            "--add-opens=java.base/java.lang=ALL-UNNAMED",
        )
        .config(
            "spark.executor.extraJavaOptions",
            "--add-opens=java.base/javax.security.auth=ALL-UNNAMED "
            "--add-opens=java.base/java.lang=ALL-UNNAMED",
        )
        .getOrCreate()
    )

    LOGGER.info("Pre-warming model analytics arrays prior to interface exposing...")
    _load_and_train()

    if STATE.trainer is None:
        LOGGER.critical("Initial training sequence collapsed. Secure boot failed.")
        raise RuntimeError("Prerequisite component building crashed during startup layer.")

    retrainer_thread = threading.Thread(
        target=_schedule_retraining,
        name="ModelRetrainer",
        daemon=True,
    )
    retrainer_thread.start()
    LOGGER.info("Subprocess daemon 'ModelRetrainer' safely assigned to execution queue.")

    yield

    LOGGER.info("Intercepted teardown invocation. Disengaging cluster engine links...")
    if STATE.historical_profiles is not None:
        STATE.historical_profiles.unpersist()
    if app.state.spark:
        app.state.spark.stop()
    LOGGER.info("API cluster termination processes successfully closed down.")


APP = FastAPI(
    title="VW Smart Parking Assistant API",
    description="Predicts parking availability for a given road segment and timestamp.",
    version="2.0.0",
    lifespan=lifespan,
)


# --- REST Endpoints Layer ---
def _prepare_inference_dataframe(hist_profiles: DataFrame, time_stamp: datetime) -> DataFrame:
    """Constructs the full feature matrix required by the ML model for inference."""
    all_segments_df = hist_profiles.select("road_segment_id").distinct()

    inference_df = (
        all_segments_df.withColumn("timestamp", F.lit(time_stamp.isoformat()).cast("timestamp"))
        .withColumn("hour", F.lit(time_stamp.hour))
        .withColumn("day_of_week", F.lit(time_stamp.isoweekday()))
        .withColumn("month", F.lit(time_stamp.month))
    )

    static_features = {
        "tempC": 21,
        "windspeedKmph": 12,
        "precipMM": 0.0,
        "commercial": 1.0,
        "residential": 1.0,
        "schools": 0.0,
        "shopping": 0.0,
        "office": 0.0,
        "supermarket": 0.0,
        "restaurant": 0.0,
        "eventsites": 0.0,
        "transportation": 0.0,
        "off_street_capa": 50.0,
        "num_off_street_parking": 1.0,
    }

    for col_name, value in static_features.items():
        inference_df = inference_df.withColumn(col_name, F.lit(value))

    return inference_df.join(
        hist_profiles, on=["road_segment_id", "hour", "day_of_week"], how="left"
    ).na.fill(value=0.5, subset=["historical_occupancy_ratio"])


@APP.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check monitoring node interface.",
    tags=["Monitoring"],
)
def health() -> HealthResponse:
    """Verifies infrastructure integrity state and model deployment metrics."""
    with MODEL_LOCK:
        ready = STATE.trainer is not None
        trained = STATE.last_trained_at.isoformat() if STATE.last_trained_at else None

    if not ready:
        raise HTTPException(
            status_code=503,
            detail="Operational matrices are initializing inside the engine.",
        )

    return HealthResponse(
        status="ok",
        model_ready=ready,
        last_trained_at_utc=trained,
    )


@APP.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict parking occupancy rates with adaptive routing alternatives.",
    tags=["Predictions"],
)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Generates structured target predictions supplemented with driver profile metrics."""
    LOGGER.info(
        "Prediction demand | Node: %s | Time: %s | Profile: %s",
        request.road_segment_id,
        request.timestamp.isoformat(),
        request.driver_profile,
    )

    with MODEL_LOCK:
        trainer = STATE.trainer
        hist_profiles = STATE.historical_profiles

    if trainer is None or hist_profiles is None:
        raise HTTPException(status_code=503, detail="Engine arrays are undergoing refresh loops.")

    try:
        inference_df = _prepare_inference_dataframe(hist_profiles, request.timestamp)
        results_df = trainer.predict_probabilities(inference_df, LOGGER)

        target_rows = results_df.filter(
            F.col("road_segment_id") == request.road_segment_id
        ).collect()

        if not target_rows:
            raise HTTPException(
                status_code=404,
                detail=f"Specified location entity '{request.road_segment_id}' is unknown.",
            )

        target_probability = round(float(target_rows[0]["occupancy_probability"]), 4)
        top_alternatives: Optional[List[AlternativeSegment]] = None

        if request.driver_profile in (DriverProfile.ACTIVE, DriverProfile.STANDARD):
            top_n = 10 if request.driver_profile == DriverProfile.ACTIVE else 5

            alt_rows = (
                results_df.filter(F.col("road_segment_id") != request.road_segment_id)
                .orderBy(F.col("occupancy_probability").asc())
                .limit(top_n)
                .select("road_segment_id", "occupancy_probability")
                .collect()
            )

            top_alternatives = [
                AlternativeSegment(
                    road_segment_id=str(row["road_segment_id"]),
                    occupancy_probability=round(float(row["occupancy_probability"]), 4),
                )
                for row in alt_rows
            ]

        return PredictionResponse(
            road_segment_id=request.road_segment_id,
            timestamp=request.timestamp,
            driver_profile=request.driver_profile,
            occupancy_probability=target_probability,
            top_alternatives=top_alternatives,
        )

    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.error("Inference execution sequence broke down: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail="Core server encountered transformation issues."
        )


if __name__ == "__main__":
    uvicorn.run(
        "api:APP",
        host="0.0.0.0",
        port=5000,
        reload=False,
        log_level="info",
    )
