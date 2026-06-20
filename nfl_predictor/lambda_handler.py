import json
import logging
import os
from urllib.parse import unquote_plus

import boto3

from nfl_predictor.nfl_ai_scores import run_predictions
from nfl_predictor.utils.constants import (
    DEFAULT_INPUT_BUCKET,
    DEFAULT_OUTPUT_BUCKET,
    INPUT_FILE_NAME,
    OUTPUT_FILE_NAME,
)

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# S3 client
s3 = boto3.client("s3")

# Environment variables with defaults from constants
INPUT_BUCKET = os.getenv("INPUT_BUCKET", DEFAULT_INPUT_BUCKET)
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
            f"Triggered by S3 upload - bucket: {input_bucket}, key: {input_key}"
        )

        # Download input file to /tmp
        s3.download_file(input_bucket, input_key, INPUT_LOCAL_PATH)
        logger.info(f"Downloaded input CSV to: {INPUT_LOCAL_PATH}")

        # Run the prediction pipeline
        results = run_predictions()

        # Ensure we have valid results before upload
        if results is not None and not results.empty:
            logger.info(f"Generated {len(results)} predictions")

            # Write prediction results to /tmp
            results.to_csv(OUTPUT_LOCAL_PATH, index=False)

            # Upload the CSV to the output S3 bucket
            s3.upload_file(
                OUTPUT_LOCAL_PATH,
                OUTPUT_BUCKET,
                OUTPUT_FILE_NAME,
                ExtraArgs={"ContentType": "text/csv"},
            )
            logger.info(
                f"Uploaded predictions to s3://{OUTPUT_BUCKET}/{OUTPUT_FILE_NAME}"
            )

            return {
                "statusCode": 200,
                "body": json.dumps(
                    {
                        "message": f"Successfully processed {len(results)} predictions",
                        "output_location": f"s3://{OUTPUT_BUCKET}/{OUTPUT_FILE_NAME}",
                        "sample_predictions": results.head(5).to_dict(orient="records"),
                    }
                ),
            }
        else:
            logger.warning("Prediction script returned no results — skipping S3 upload")
            return {
                "statusCode": 204,
                "body": json.dumps(
                    {
                        "message": (
                            "No predictions generated - check input data and logs"
                        ),
                        "predictions_count": 0,
                    }
                ),
            }

    except Exception as e:
        logger.exception("Unhandled error during Lambda execution")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
