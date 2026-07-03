"""Handles training, evaluation, and prediction using PySpark RandomForest

with Historical Target Encoding for the VW Smart Parking dataset.
"""

# pylint: disable=import-error
import logging
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame
# pylint: enable=import-error


class ParkingModelTrainer:
    """A trainer wrapper managing the PySpark ML Random Forest pipeline lifecycle.

    This class handles feature engineering, orchestration of the Spark pipeline
    (VectorAssembler and RandomForestClassifier), metrics evaluation, model saving,
    and extraction of real-time prediction probabilities for parking segment occupancy.

    Attributes:
        model (PipelineModel, optional): The fitted PySpark Pipeline object containing
            the transformation stages and the trained classifier. Defaults to None.
        is_trained (bool): Operational flag indicating whether the model pipeline
            has been successfully fit on training data. Defaults to False.
    """

    def __init__(self):
        self.model = None
        self.is_trained = False

    @staticmethod
    def prepare_features(df_data: DataFrame) -> DataFrame:
        """Extracts temporal features from the timestamp column."""
        return (
            df_data.withColumn("hour", F.hour(F.col("timestamp")))
            .withColumn("day_of_week", F.dayofweek(F.col("timestamp")))
            .withColumn("month", F.month(F.col("timestamp")))
        )

    def build_and_train_pipeline(self, train_df: DataFrame, logger: logging.Logger) -> None:
        """Builds the MLlib Pipeline with Random Forest and trains the model."""
        logger.info("Building the Random Forest ML pipeline with historical target encoding...")

        # 1. Assemble independent variables into a single feature vector.
        # We drop raw segment IDs and use the continuous 'historical_occupancy_ratio'
        # combined with dynamic time and weather features.
        feature_columns = [
            "historical_occupancy_ratio",
            "hour",
            "day_of_week",
            "month",
            "tempC",
            "windspeedKmph",
            "precipMM",
            "commercial",
            "residential",
            "transportation",
            "schools",
            "eventsites",
            "restaurant",
            "shopping",
            "office",
            "supermarket",
            "num_off_street_parking",
            "off_street_capa",
        ]
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

        # 2. Define the Random Forest Classifier
        # numTrees: 100 for ensemble stability and generalization
        # maxDepth: 10 allows trees to split on both historical averages and weather tweaks
        rf_classifier = RandomForestClassifier(
            featuresCol="features",
            labelCol="is_occupied",
            probabilityCol="probability",
            numTrees=100,
            maxDepth=10,
            seed=42,
        )

        # 3. Chain stages into the Pipeline (Assembler -> Classifier)
        pipeline = Pipeline(stages=[assembler, rf_classifier])

        logger.info("Fitting the Random Forest Pipeline on training data...")
        self.model = pipeline.fit(train_df)
        self.is_trained = True
        logger.info("Random Forest model training completed successfully.")

    def evaluate_model(self, test_df: DataFrame) -> float:
        """Evaluates the Random Forest model on test data using ROC AUC metric."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation.")

        predictions = self.model.transform(test_df)

        evaluator = BinaryClassificationEvaluator(
            rawPredictionCol="probability",
            labelCol="is_occupied",
            metricName="areaUnderROC",
        )

        auc = evaluator.evaluate(predictions)
        return auc

    def predict_probabilities(self, inference_df: DataFrame, logger: logging.Logger) -> DataFrame:
        """Predicts occupancy probabilities using native Spark transformations."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making inference predictions.")

        logger.info("Calculating parking occupancy probabilities with Random Forest...")

        predictions = self.model.transform(inference_df)

        results_df = (
            predictions.withColumn("prob_array", vector_to_array(F.col("probability")))
            .withColumn("occupancy_probability", F.col("prob_array")[1])
            .select("road_segment_id", "timestamp", "occupancy_probability")
        )

        return results_df

    def save_model(self, path: str, logger: logging.Logger) -> None:
        """Saves the trained pipeline model to a specified local directory."""
        if not self.is_trained:
            raise ValueError("No model trained yet to save.")

        logger.info(f"Saving the latest model artifact to: {path}")
        self.model.write().overwrite().save(path)

    def get_feature_importance(self, logger=None) -> dict:
        """Extracts and maps feature importances from the trained Random Forest pipeline."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model has not been trained yet.")

        try:
            assembler_stage = None
            rf_stage = None

            for stage in self.model.stages:
                stage_name = stage.__class__.__name__
                if "VectorAssembler" in stage_name:
                    assembler_stage = stage
                elif "RandomForestClassificationModel" in stage_name:
                    rf_stage = stage

            if not assembler_stage or not rf_stage:
                raise ValueError("Could not find required pipeline stages.")

            feature_names = assembler_stage.getInputCols()
            importances = rf_stage.featureImportances.toArray()

            feature_importance_dict = {
                name: float(imp) for name, imp in zip(feature_names, importances)
            }
            sorted_importance = dict(
                sorted(
                    feature_importance_dict.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )

            if logger:
                logger.info("--- FEATURE IMPORTANCE RANKING ---")
                for feature, importance in sorted_importance.items():
                    logger.info(f"Feature: {feature:<28} Importance: {importance:.4f}")
                logger.info("----------------------------------")

            return sorted_importance

        except Exception as error:
            if logger:
                logger.error(f"Failed to extract feature importance: {str(error)}")
            raise error


def train_and_predict_pipeline(processed_df: DataFrame, logger: logging.Logger) -> DataFrame:
    """Orchestration function with advanced historical target encoding
      (Segment + Hour + Day of Week)

    and strict separation to prevent data leakage.
    """
    logger.info("Starting the ML pipeline preprocessing...")

    trainer = ParkingModelTrainer()

    # 1. Basic feature extraction
    base_features_df = trainer.prepare_features(processed_df).withColumn(
        "is_occupied", F.when(F.col("available") == 0, 1).otherwise(0)
    )

    # 2. STRICT SPLIT FIRST: Divide into train and test sets to maintain data integrity
    logger.info("Splitting data into train and test sets (80/20)...")
    raw_train, raw_test = base_features_df.randomSplit([0.8, 0.2], seed=42)

    # 3. TARGET ENCODING ON TRAIN ONLY:
    # Grouping by road_segment_id, hour, AND day_of_week captures accurate patterns
    # like weekday rush hours versus quiet Sunday afternoons.
    logger.info(
        "Computing granular historical occupancy profiles (Segment + Hour + Day of Week)..."
    )
    historical_profiles = raw_train.groupBy("road_segment_id", "hour", "day_of_week").agg(
        F.avg("is_occupied").alias("historical_occupancy_ratio")
    )
    historical_profiles.cache()

    # 4. JOIN profiles back using the expanded composite key
    # If a specific day/hour breakdown is missing in test data, we fall back to a neutral 0.5 ratio.
    join_keys = ["road_segment_id", "hour", "day_of_week"]

    train_data = raw_train.join(historical_profiles, on=join_keys, how="left")

    test_data = raw_test.join(historical_profiles, on=join_keys, how="left").na.fill(
        value=0.5, subset=["historical_occupancy_ratio"]
    )

    # 5. Build and train the Random Forest pipeline
    trainer.build_and_train_pipeline(train_data, logger)

    # 6. Print the updated feature importance distribution
    trainer.get_feature_importance(logger)

    # 7. Print auc metric
    auc = trainer.evaluate_model(test_data)
    logger.info(f"Random Forest Evaluation - Test Set ROC AUC: {auc:.4f}")

    # 8. Generate forecasts
    forecast_results = trainer.predict_probabilities(test_data, logger)

    return forecast_results
