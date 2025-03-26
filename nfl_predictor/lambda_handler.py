def handler(event, context):
    # Defer import until handler is invoked (guarantees patched env is used)
    from nfl_ai_scores import run_predictions
    return run_predictions()
