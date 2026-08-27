import json
import logging
import os
from urllib.parse import unquote_plus

import boto3

from nfl_predictor.nfl_ai_scores import run_predictions
from nfl_predictor.utils.constants import (
    DEFAULT_LOG_LEVEL,
    DEFAULT_OUTPUT_BUCKET,
    INPUT_FILE_NAME,
    OUTPUT_FILE_NAME,
)

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL))

# S3 client
s3 = boto3.client("s3")

# Environment variables with defaults from constants
OUTPUT_BUCKET = os.getenv("OUTPUT_BUCKET", DEFAULT_OUTPUT_BUCKET)

# Paths in Lambda (only /tmp is writable)
TMP_DIR = "/tmp"  # nosec - Lambda environment constraint
INPUT_LOCAL_PATH = os.path.join(TMP_DIR, INPUT_FILE_NAME)
OUTPUT_LOCAL_PATH = os.path.join(TMP_DIR, OUTPUT_FILE_NAME)


def handler(event, context):
    """
    AWS Lambda handler for NFL prediction pipeline.
    Triggered by S3 uploads, processes matchup data, and returns predictions.
    """
    try:
        # Extract event details
        record = event["Records"][0]
        input_key = unquote_plus(record["s3"]["object"]["key"])
        input_bucket = record["s3"]["bucket"]["name"]
        logger.info(
            "Triggered by S3 upload - bucket: %s, key: %s", input_bucket, input_key
        )

        # Download input file to /tmp
        s3.download_file(input_bucket, input_key, INPUT_LOCAL_PATH)
        logger.info("Downloaded input CSV to: %s", INPUT_LOCAL_PATH)

        # Run the prediction pipeline
        results = run_predictions(matchups_path=INPUT_LOCAL_PATH)

        # Ensure we have valid results before upload
        if results is not None and not results.empty:
            logger.info("Generated %s predictions", len(results))

            # Write and upload prediction results.
            results.to_csv(OUTPUT_LOCAL_PATH, index=False)
            s3.upload_file(
                OUTPUT_LOCAL_PATH,
                OUTPUT_BUCKET,
                OUTPUT_FILE_NAME,
                ExtraArgs={"ContentType": "text/csv"},
            )
            logger.info(
                "Uploaded predictions to s3://%s/%s", OUTPUT_BUCKET, OUTPUT_FILE_NAME
            )

            response_body = {
                "status": "success",
                "message": f"Successfully processed {len(results)} predictions",
                "output_location": f"s3://{OUTPUT_BUCKET}/{OUTPUT_FILE_NAME}",
                "sample_predictions": json.loads(
                    results.head(5).to_json(orient="records")
                ),
            }

            return response_body
        else:
            logger.warning("Prediction script returned no results — skipping S3 upload")
            return {
                "status": "no_predictions",
                "message": "No predictions generated - check input data and logs",
                "predictions_count": 0,
            }

    except Exception:
        logger.exception("Unhandled error during Lambda execution")
        raise
