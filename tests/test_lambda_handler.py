"""
Test suite for AWS Lambda handler functionality.
"""

import importlib
import json
import os
from typing import Any
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from nfl_predictor.lambda_handler import INPUT_LOCAL_PATH, handler


class TestLambdaHandler:
    """Test AWS Lambda handler functionality."""

    @pytest.fixture
    def sample_s3_event(self) -> dict[str, Any]:
        """Sample S3 event that triggers the Lambda function."""
        return {
            "Records": [
                {
                    "s3": {
                        "bucket": {"name": "test-input-bucket"},
                        "object": {"key": "upcoming_matchups.csv"},
                    }
                }
            ]
        }

    @pytest.fixture
    def sample_context(self) -> Mock:
        """Mock Lambda context object."""
        context = Mock()
        context.function_name = "test-function"
        context.function_version = "1"
        context.invoked_function_arn = (
            "arn:aws:lambda:us-east-1:123456789012:function:test-function"
        )
        context.memory_limit_in_mb = 128
        context.remaining_time_in_millis = lambda: 30000
        return context

    @pytest.fixture
    def sample_predictions(self) -> pd.DataFrame:
        """Sample prediction results."""
        return pd.DataFrame(
            [
                {
                    "Week": 18,
                    "Home Team": "Philadelphia Eagles",
                    "Home Score": 28,
                    "Away Team": "Kansas City Chiefs",
                    "Away Score": 24,
                    "Result": "Philadelphia Eagles win by 4",
                    "Over/Under": 52,
                }
            ]
        )

    @patch("nfl_predictor.lambda_handler.s3")
    @patch("nfl_predictor.lambda_handler.run_predictions")
    def test_successful_prediction_execution(
        self,
        mock_run_predictions,
        mock_s3,
        sample_s3_event,
        sample_context,
        sample_predictions,
    ) -> None:
        """Test successful execution of prediction pipeline."""
        # Mock successful prediction results
        mock_run_predictions.return_value = sample_predictions

        # Mock S3 operations
        mock_s3.download_file = Mock()
        mock_s3.upload_file = Mock()

        # Execute handler
        response = handler(sample_s3_event, sample_context)

        # Verify response
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert "Successfully processed 1 predictions" in body["message"]
        assert "sample_predictions" in body

        # Verify S3 operations were called
        mock_s3.download_file.assert_called_once()
        mock_s3.upload_file.assert_called_once()
        mock_run_predictions.assert_called_once_with(matchups_path=INPUT_LOCAL_PATH)

    @patch("nfl_predictor.lambda_handler.s3")
    @patch("nfl_predictor.lambda_handler.run_predictions")
    def test_no_predictions_generated(
        self, mock_run_predictions, mock_s3, sample_s3_event, sample_context
    ) -> None:
        """Test handling when no predictions are generated."""
        # Mock empty prediction results
        mock_run_predictions.return_value = pd.DataFrame()

        # Mock S3 download
        mock_s3.download_file = Mock()

        # Execute handler
        response = handler(sample_s3_event, sample_context)

        # Verify response - 204 No Content is the appropriate HTTP status
        assert response["statusCode"] == 204
        body = json.loads(response["body"])
        assert "No predictions generated" in body["message"]
        assert body["predictions_count"] == 0

        # Verify no upload was attempted
        mock_s3.upload_file.assert_not_called()

    @patch("nfl_predictor.lambda_handler.s3")
    @patch("nfl_predictor.lambda_handler.run_predictions")
    def test_prediction_exception_handling(
        self, mock_run_predictions, mock_s3, sample_s3_event, sample_context
    ) -> None:
        """Test exception handling during prediction execution."""
        # Mock prediction error
        mock_run_predictions.side_effect = Exception("Prediction failed")

        # Mock S3 download
        mock_s3.download_file = Mock()

        # Execute handler
        response = handler(sample_s3_event, sample_context)

        # Verify error response
        assert response["statusCode"] == 500
        body = json.loads(response["body"])
        assert "error" in body

    @patch("nfl_predictor.lambda_handler.s3")
    def test_s3_download_error(self, mock_s3, sample_s3_event, sample_context) -> None:
        """Test handling of S3 download errors."""
        # Mock S3 download error
        mock_s3.download_file.side_effect = Exception("S3 download failed")

        # Execute handler
        response = handler(sample_s3_event, sample_context)

        # Verify error response
        assert response["statusCode"] == 500
        body = json.loads(response["body"])
        assert "error" in body

    @patch("nfl_predictor.lambda_handler.s3")
    @patch("nfl_predictor.lambda_handler.run_predictions")
    def test_s3_upload_error(
        self,
        mock_run_predictions,
        mock_s3,
        sample_s3_event,
        sample_context,
        sample_predictions,
    ) -> None:
        """Test handling of S3 upload errors."""
        # Mock successful predictions
        mock_run_predictions.return_value = sample_predictions

        # Mock S3 operations - download succeeds, upload fails
        mock_s3.download_file = Mock()
        mock_s3.upload_file.side_effect = Exception("S3 upload failed")

        # Execute handler
        response = handler(sample_s3_event, sample_context)

        # Verify error response
        assert response["statusCode"] == 500
        body = json.loads(response["body"])
        assert "error" in body

    def test_malformed_event_structure(self, sample_context) -> None:
        """Test handling of malformed S3 event."""
        malformed_event = {"Records": [{"malformed": "event"}]}

        # Execute handler
        response = handler(malformed_event, sample_context)

        # Verify error response
        assert response["statusCode"] == 500
        body = json.loads(response["body"])
        assert "error" in body

    @patch("nfl_predictor.lambda_handler.s3")
    @patch("nfl_predictor.lambda_handler.run_predictions")
    def test_url_encoded_key_handling(
        self, mock_run_predictions, mock_s3, sample_context, sample_predictions
    ) -> None:
        """Test proper handling of URL-encoded S3 object keys."""
        # Event with URL-encoded key
        encoded_event = {
            "Records": [
                {
                    "s3": {
                        "bucket": {"name": "test-bucket"},
                        "object": {"key": "folder%2Ffile%20with%20spaces.csv"},
                    }
                }
            ]
        }

        mock_run_predictions.return_value = sample_predictions
        mock_s3.download_file = Mock()
        mock_s3.upload_file = Mock()

        # Execute handler
        response = handler(encoded_event, sample_context)

        # Verify successful execution
        assert response["statusCode"] == 200

        # Verify S3 download was called with decoded key
        call_args = mock_s3.download_file.call_args[0]
        assert call_args[1] == "folder/file with spaces.csv"  # Should be decoded

    def test_environment_variable_override(
        self,
        sample_s3_event,
        sample_context,
        sample_predictions,
    ) -> None:
        """Test that output bucket environment variable overrides the default."""
        import nfl_predictor.lambda_handler as lambda_handler_module

        with patch.dict(
            os.environ,
            {"OUTPUT_BUCKET": "custom-output"},
            clear=False,
        ):
            reloaded_module = importlib.reload(lambda_handler_module)
            with (
                patch.object(reloaded_module, "s3") as mock_s3,
                patch.object(
                    reloaded_module, "run_predictions"
                ) as mock_run_predictions,
            ):
                mock_run_predictions.return_value = sample_predictions
                mock_s3.download_file = Mock()
                mock_s3.upload_file = Mock()

                # Execute handler
                reloaded_module.handler(sample_s3_event, sample_context)

                # Verify custom bucket from environment was used for upload
                upload_call_args = mock_s3.upload_file.call_args[0]
                assert upload_call_args[1] == "custom-output"

        # Restore module state for subsequent tests
        importlib.reload(lambda_handler_module)


class TestLambdaHandlerIntegration:
    """Integration tests for Lambda handler with constants."""

    @patch("nfl_predictor.lambda_handler.s3")
    @patch("nfl_predictor.lambda_handler.run_predictions")
    def test_constants_integration(self, mock_run_predictions, mock_s3) -> None:
        """Test that handler properly uses constants for file names and buckets."""
        from nfl_predictor.utils.constants import INPUT_FILE_NAME, OUTPUT_FILE_NAME

        # Create test event and context locally
        test_event = {
            "Records": [
                {
                    "s3": {
                        "bucket": {"name": "test-input-bucket"},
                        "object": {"key": INPUT_FILE_NAME},
                    }
                }
            ]
        }
        test_context = Mock()

        # Create a proper DataFrame for mock_run_predictions
        test_predictions = pd.DataFrame(
            [{"home_team": "TEST1", "away_team": "TEST2", "predicted_winner": "TEST1"}]
        )

        mock_run_predictions.return_value = test_predictions
        mock_s3.download_file = Mock()
        mock_s3.upload_file = Mock()

        # Execute handler
        handler(test_event, test_context)

        # Verify download uses correct file name from constants
        download_call_args = mock_s3.download_file.call_args[0]
        assert download_call_args[2].endswith(INPUT_FILE_NAME)

        # Verify upload uses correct file name from constants
        upload_call_args = mock_s3.upload_file.call_args[0]
        assert upload_call_args[2] == OUTPUT_FILE_NAME
