"""
api.py — FastAPI REST API for the VW Smart Parking Assistant.

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
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
import pyspark.sql.functions as F

from read_data import create_dataframe
from train_model import ParkingModelTrainer
# pylint: enable=import-error

# ---------------------------------------------------------------------------
# LOGGING SYSTEM CONFIGURATION
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("VW-Parking-API")

# Global reference wrapper for the SparkSession engine
SPARK: Optional[SparkSession] = None

# ---------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# ---------------------------------------------------------------------------
DATA_PATHS: dict = {
    "groundtruth": "groundtruth.csv",
    "road_features": "road_features.csv",
    "weather_features": "weather_features.csv",
}

# In-memory synchronization timer configuration (1 hour cycle)
RETRAIN_INTERVAL_SECONDS: int = 3600

# ---------------------------------------------------------------------------
# DATA VALIDATION SCHEMAS (Pydantic V2 Compliant)
# ---------------------------------------------------------------------------


class DriverProfile(str, Enum):
    """Supported telemetry routing depth criteria profiles."""

    active = "active"
    standard = "standard"
    inactive = "inactive"


class PredictionRequest(BaseModel):
    """Inbound telemetry request blueprint with structural JSON validation."""

    road_segment_id: str = Field(
        ...,
        description="Unique database identifier of the target road segment.",
        examples=["SEG_001"],
    )
    timestamp: datetime = Field(
        ...,
        description="ISO-8601 compliant datetime stamp for the prediction instance.",
        examples=["2026-07-03T15:30:00"],
    )
    driver_profile: DriverProfile = Field(
        default=DriverProfile.inactive,
        description="Driver persona parameter dictating routing search criteria depth.",
    )


# pylint: disable=too-few-public-methods
class AlternativeSegment(BaseModel):
    """A granular alternative parking node prediction matrix output."""

    road_segment_id: str
    occupancy_probability: float


# pylint: disable=too-few-public-methods
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
        description="""Ranked list of contextually optimized alternatives
          (active: top 10, standard: top 5).""",
    )


# pylint: disable=too-few-public-methods
class HealthResponse(BaseModel):
    """Monitoring schema mapping the infrastructure vital components."""

    status: str
    model_ready: bool
    last_trained_at_utc: Optional[str]


# ---------------------------------------------------------------------------
# THREAD-SAFE GLOBAL IN-MEMORY ENVIRONMENT STATE
# ---------------------------------------------------------------------------
_model_lock = threading.Lock()
_TRAINER: Optional[ParkingModelTrainer] = None
_HISTORICAL_PROFILES: Optional[DataFrame] = None
_LAST_TRAINED_AT: Optional[datetime] = None

# ---------------------------------------------------------------------------
# MODEL PIPELINE PROCESSING CORE
# ---------------------------------------------------------------------------


def _load_and_train() -> None:
    """Reads transactional analytics data, constructs features, evaluates

    target encodings, fits pipelines, and executes hot-swaps under atomic guards.
    """
    global _TRAINER, _HISTORICAL_PROFILES, _LAST_TRAINED_AT

    LOGGER.info("Initiating model training loop and temporal matrix generation...")

    try:
        # 1. Join file system data inputs into an analytical Spark DataFrame
        df: DataFrame = create_dataframe(DATA_PATHS, LOGGER)

        # 2. Extract analytical engineering signatures and construct target classification vectors
        trainer = ParkingModelTrainer()
        base_df = trainer.prepare_features(df).withColumn(
            "is_occupied", F.when(F.col("available") == 0, 1).otherwise(0)
        )

        # 3. Rigid division segment to prevent data leak vectors during structural encoding
        raw_train, _ = base_df.randomSplit([0.8, 0.2], seed=42)

        # 4. Compute target ratio arrays (Topological Entity + Hour Coordinates + Weekday)
        historical_profiles: DataFrame = raw_train.groupBy(
            "road_segment_id", "hour", "day_of_week"
        ).agg(F.avg("is_occupied").alias("historical_occupancy_ratio"))
        historical_profiles.cache()

        # 5. Enrich dataset parameters and calibrate active ML Estimator classes
        train_data = raw_train.join(
            historical_profiles,
            on=["road_segment_id", "hour", "day_of_week"],
            how="left",
        )
        trainer.build_and_train_pipeline(train_data, LOGGER)
        trainer.get_feature_importance(LOGGER)

        # 6. Secure critical resource execution via single thread-bound context block
        with _model_lock:
            _TRAINER = trainer
            _HISTORICAL_PROFILES = historical_profiles
            _LAST_TRAINED_AT = datetime.now(timezone.utc)

        LOGGER.info(
            f"Model update successfully executed at {_LAST_TRAINED_AT.isoformat()}"
        )

    except Exception as exc:
        LOGGER.error(
            f"Execution failed on background context assembly pipeline: {exc}",
            exc_info=True,
        )


def _schedule_retraining() -> None:
    """Cyclical worker task block looping context routines over standard intervals."""
    LOGGER.info(
        f"Asynchronous batch daemon online — standard checking pattern: {RETRAIN_INTERVAL_SECONDS}s"
    )
    stop_event = threading.Event()
    while not stop_event.wait(timeout=RETRAIN_INTERVAL_SECONDS):
        LOGGER.info(
            "Timed interval interrupt detected. Triggering background pipeline update loop..."
        )
        _load_and_train()


# ---------------------------------------------------------------------------
# FASTAPI LIFECYCLE MANAGEMENT (LIFESPAN COORD)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Coordinates explicit infrastructure boots and deterministic system teardowns.

    Enforces python execution path parity for secondary PySpark context allocation workers.
    """
    # -- STARTUP SEQUENCE ----------------------------------------------------
    global SPARK
    LOGGER.info(
        "Warming local operating kernel context. Building PySpark Session context..."
    )

    # WINDOWS ENVIRONMENT STABILITY CURE: Enforce workers to bind to virtual env paths explicitly
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    SPARK = (
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

    if _TRAINER is None:
        LOGGER.critical(
            "Initial training sequence collapsed. Service framework cannot boot securely."
        )
        raise RuntimeError(
            "Prerequisite component building crashed during startup layer."
        )

    retrainer_thread = threading.Thread(
        target=_schedule_retraining,
        name="ModelRetrainer",
        daemon=True,
    )
    retrainer_thread.start()
    LOGGER.info(
        "Subprocess daemon 'ModelRetrainer' safely assigned to execution queue."
    )

    yield  # Handover execution control back to web engine context loop

    # -- SHUTDOWN SEQUENCE ---------------------------------------------------
    LOGGER.info("Intercepted teardown invocation. Disengaging cluster engine links...")
    global _HISTORICAL_PROFILES
    if _HISTORICAL_PROFILES is not None:
        _HISTORICAL_PROFILES.unpersist()
    if SPARK:
        SPARK.stop()
    LOGGER.info("API cluster termination processes successfully closed down.")


# Instantiating high-performance engine blueprint
app = FastAPI(
    title="VW Smart Parking Assistant API",
    description=(
        "Predicts parking availability for a given road segment and timestamp. "
        "Supports three driver profiles: active, standard, and inactive."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# REST ENDPOINTS LAYER
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check monitoring node interface.",
    tags=["Monitoring"],
)
def health() -> HealthResponse:
    """Verifies infrastructure integrity state and model deployment metrics."""
    with _model_lock:
        ready = _TRAINER is not None
        trained = _LAST_TRAINED_AT.isoformat() if _LAST_TRAINED_AT else None

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


# pylint: disable=too-many-local-variables
@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict parking occupancy rates with adaptive routing alternatives.",
    tags=["Predictions"],
)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Generates structured target predictions supplemented with driver profile metrics.

    Evaluates localized node arrays to output optimized vacancy indexes.
    """
    # 1. Temporal extraction execution over validated payload attributes
    time_stamp = request.timestamp
    hour = time_stamp.hour
    day_of_week = time_stamp.isoweekday()  # Monday=1 ... Sunday=7 layout
    month = time_stamp.month

    LOGGER.info(
        f"""Prediction demand | Target Node: {request.road_segment_id} |
          Temporal marker: {time_stamp.isoformat()} """
        f"""| Routing depth criteria: {request.driver_profile} |
          Context metrics: H:{hour} DOW:{day_of_week} M:{month}"""
    )

    # 2. Extract stable runtime reference variables using lock managers
    with _model_lock:
        trainer = _TRAINER
        hist_profiles = _HISTORICAL_PROFILES

    if trainer is None or hist_profiles is None:
        raise HTTPException(
            status_code=503, detail="Engine arrays are undergoing refresh loops."
        )

    try:
        # 3. Assemble full dimensional inference tracking maps
        all_segments_df: DataFrame = hist_profiles.select("road_segment_id").distinct()

        inference_df: DataFrame = (
            all_segments_df.withColumn(
                "timestamp", F.lit(time_stamp.isoformat()).cast("timestamp")
            )
            .withColumn("hour", F.lit(hour))
            .withColumn("day_of_week", F.lit(day_of_week))
            .withColumn("month", F.lit(month))
        )

        inference_df = (
            inference_df.withColumn("tempC", F.lit(21))
            .withColumn("windspeedKmph", F.lit(12))
            .withColumn("precipMM", F.lit(0.0))
            .withColumn("commercial", F.lit(1.0))
            .withColumn("residential", F.lit(1.0))
            .withColumn("schools", F.lit(0.0))
            .withColumn("shopping", F.lit(0.0))
            .withColumn("office", F.lit(0.0))
            .withColumn("supermarket", F.lit(0.0))
            .withColumn("restaurant", F.lit(0.0))
            .withColumn("eventsites", F.lit(0.0))
            .withColumn("transportation", F.lit(0.0))
            .withColumn("off_street_capa", F.lit(50.0))
            .withColumn("num_off_street_parking", F.lit(1.0))
        )

        # Enrich vector maps against historical records using target values
        inference_df = inference_df.join(
            hist_profiles, on=["road_segment_id", "hour", "day_of_week"], how="left"
        ).na.fill(value=0.5, subset=["historical_occupancy_ratio"])

        # 4. Process dataframe matrices via trained estimator pot
        results_df: DataFrame = trainer.predict_probabilities(inference_df, LOGGER)

        # 5. Extract prediction rows mapping specified client requirements
        target_rows = results_df.filter(
            F.col("road_segment_id") == request.road_segment_id
        ).collect()

        if not target_rows:
            raise HTTPException(
                status_code=404,
                detail=f"""Specified location entity '{request.road_segment_id}'
                  is unknown inside matrix models.""",
            )

        target_probability = round(float(target_rows[0]["occupancy_probability"]), 4)

        # 6. Compile sorted replacement pathways according to depth profiles
        top_alternatives: Optional[List[AlternativeSegment]] = None

        if request.driver_profile in (DriverProfile.active, DriverProfile.standard):
            top_n = 10 if request.driver_profile == DriverProfile.active else 5

            # Rank candidate entities based on relative vacancy
            #  (lowest saturation maps optimal status)
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

        # 7. Deliver validated, serialized data records back across transport networks
        return PredictionResponse(
            road_segment_id=request.road_segment_id,
            timestamp=request.timestamp,
            driver_profile=request.driver_profile,
            occupancy_probability=target_probability,
            top_alternatives=top_alternatives,
        )

    except HTTPException:
        raise  # Pass explicitly recognized validation and domain breaks unmolested
    except Exception as exc:
        LOGGER.error(f"Inference execution sequence broke down: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Core server encountered transformation issues."
        )


# ---------------------------------------------------------------------------
# CORE APPLICATION LAUNCH PAD
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=5000,
        reload=False,
        log_level="info",
    )
