import os
import pandas as pd
import logging

from nfl_predictor.utils.constants import STANDINGS_FILE_NAME, FLAGGED_OUTPUT_FILE_NAME
from nfl_predictor.utils.helpers import running_in_lambda

logger = logging.getLogger(__name__)

def run_upsets_agent(df_results: pd.DataFrame, path: str) -> pd.DataFrame:
    """
    Flags:
    - Games with <4 point difference as '⚠️ Close Call'
    - Predicted winners with fewer wins than their opponent as '🚨 Potential Upset'
    """
    standings_path = os.path.join(path, STANDINGS_FILE_NAME)

    if not os.path.exists(standings_path):
        logger.warning("Upsets Agent skipped: standings file not found.")
        return df_results

    df_standings = pd.read_csv(standings_path)

    # Merge win counts
    df_results = df_results.merge(df_standings[["Team", "W"]], left_on="Home Team", right_on="Team", how="left") \
                           .rename(columns={"W": "Home Wins"}).drop(columns=["Team"])
    df_results = df_results.merge(df_standings[["Team", "W"]], left_on="Away Team", right_on="Team", how="left") \
                           .rename(columns={"W": "Away Wins"}).drop(columns=["Team"])

    # Determine winner and point difference
    df_results["Predicted Winner"] = df_results["Result"].apply(
        lambda r: r.split(" win")[0] if "win" in r else "Tie"
    )

    # Apply flags
    def flag_row(row):
        flags = []

        if abs(row["Home Score"] - row["Away Score"]) < 4:
            flags.append("⚠️ Close Call")

        if row["Predicted Winner"] == row["Home Team"] and row["Home Wins"] < row["Away Wins"]:
            flags.append(f"🚨 Potential Upset: {row['Home Team']}")
        elif row["Predicted Winner"] == row["Away Team"] and row["Away Wins"] < row["Home Wins"]:
            flags.append(f"🚨 Potential Upset: {row['Away Team']}")

        return " + ".join(flags) if flags else ""

    df_results["Upset Flag"] = df_results.apply(flag_row, axis=1)

    # Summary counts
    close_calls = df_results["Upset Flag"].str.contains("Close Call").sum()
    upsets = df_results["Upset Flag"].str.contains("Potential Upset").sum()
    logger.info(f"Upsets Agent summary: {close_calls} close calls, {upsets} potential upsets flagged.")

    # Drop helper columns
    df_results.drop(columns=["Home Wins", "Away Wins", "Predicted Winner"], inplace=True)

    # Output
    if not running_in_lambda():
        output_path = os.path.join(path, FLAGGED_OUTPUT_FILE_NAME)
        df_results.to_csv(output_path, index=False)
        logger.info(f"Upsets Agent complete: flagged results saved to {output_path}")
    else:
        logger.info("Upsets Detection Results:\n%s", df_results.to_string(index=False))

    return df_results
