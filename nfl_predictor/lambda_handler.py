import json
import logging
from nfl_ai_scores import run_predictions

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def handler(event, context):
    try:
        # Parse JSON input from API Gateway POST body
        payload = json.loads(event.get("body", "{}"))
        
        # Call prediction logic with passed-in params
        result = run_predictions(**payload)

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"  # CORS for frontend
            },
            "body": json.dumps(result)
        }

    except Exception as e:
        logger.exception("Unhandled error during Lambda execution")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
