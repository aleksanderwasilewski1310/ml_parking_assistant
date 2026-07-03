"""Module for reading, cleaning, and merging data for the VW Smart Parking project."""

import logging
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.errors import AnalysisException

# Java modular architecture access flags required for PySpark arrow / memory compliance
java_opens_flags = (
    "--add-opens=java.base/javax.security.auth=ALL-UNNAMED "
    "--add-opens=java.base/java.lang=ALL-UNNAMED"
)

# Initialize the Spark Session with memory and Java compliance configurations
spark = (
    SparkSession.builder.appName("VW_SmartParking")
    .config("spark.driver.memory", "4g")
    .config("spark.driver.extraJavaOptions", java_opens_flags)
    .config("spark.executor.extraJavaOptions", java_opens_flags)
    .getOrCreate()
)


class ReadFile:
    """A data extraction and staging abstraction layer for source data files.

    This class encapsulates the pipeline's ingestion phase, utilizing an active
    Spark session to read raw text-delimited data, resolve schema structural constraints,
    and perform fundamental row-level and column-level data cleansing before merging operations.
    """

    def __init__(self, path, logger):
        """Initializes the ReadFile class, loads data, and performs initial cleaning.

        Args:
            path (str): Path to the input CSV file.
            logger (logging.getLogger): Logger
        """
        # Load CSV data, infer schema,
        # and automatically convert 'null' strings to system Null values
        self.path = path
        try:
            logger.info(f"Attempting to load file: {path}")
            # Load CSV data
            self.data = spark.read.csv(
                path, header=True, inferSchema=True, nullValue="null"
            )

        except AnalysisException as error:
            logger.error(
                f"Spark failed to process or find the file at '{path}'. Error details: {error}"
            )
            raise  # Rerun the exception to halt the pipeline execution immediately
        except Exception as error:
            logger.error(
                f"An unexpected error occurred while reading '{path}': {error}"
            )
            raise

    def clean_data(self, logger):
        """
        Cleans data
        Args:
            logger (logging.getLogger): Logger
        """
        # Drop the ambiguous '_c0' index column immediately
        #  to prevent conflicts during drops and joins
        self.data = self.data.drop("_c0")

        # Drop rows where any column contains a null value
        # (required for Random Forest)
        self.data = self.data.dropna(how="any")

        # Eliminate exact duplicate rows from the dataset
        self.data = self.data.dropDuplicates()

        logger.info(
            f"File {self.path} successfully loaded and cleaned. Rows remaining: {self.data.count()}"
        )


def create_dataframe(data_paths: dict, logger: logging.getLogger) -> DataFrame:
    """Combines telemetry, road features, and weather datasets into a single PySpark DataFrame.

    Args:
        data_paths (dict): A dictionary containing keys like 'groundtruth', 'road', 'weather'
                           mapped to their respective file paths
        logger (logging.getLogger): Logger instance for tracking the process.

    Returns:
        DataFrame: The final cleansed and merged PySpark DataFrame ready for ML training.
    """
    # Load and preprocess files dynamically using the dictionary paths
    groundtruth_data = ReadFile(data_paths["groundtruth"], logger)
    road_features_data = ReadFile(data_paths["road_features"], logger)
    weather_data = ReadFile(data_paths["weather_features"], logger)
    groundtruth_data.clean_data(logger)
    road_features_data.clean_data(logger)
    weather_data.clean_data(logger)

    # First join: Merge telemetry with static road features using 'road_segment_id'
    groundtruth_data.data = groundtruth_data.data.join(
        road_features_data.data, on="road_segment_id", how="left"
    )
    # Clean up any potential nulls introduced by the left join mismatch
    groundtruth_data.data = groundtruth_data.data.dropna(how="any")

    # Second join: Enrich the dataset with weather conditions using a composite key
    groundtruth_data.data = groundtruth_data.data.join(
        weather_data.data, on=["road_segment_id", "timestamp"], how="left"
    )
    # Final null check to guarantee absolute data completeness for the model
    groundtruth_data.data = groundtruth_data.data.dropna(how="any")

    logger.info(f"Final combined dataset row count: {groundtruth_data.data.count()}")
    return groundtruth_data.data
