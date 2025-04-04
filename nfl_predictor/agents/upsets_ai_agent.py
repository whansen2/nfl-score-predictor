import os
import pandas as pd
import logging

from nfl_predictor.utils.constants import STANDINGS_FILE_NAME
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

    # Start with a copy to avoid mutating original
    df_flagged = df_results.copy()

    # Merge win counts
    df_flagged = df_flagged.merge(
        df_standings[["Team", "W"]],
        left_on="Home Team",
        right_on="Team",
        how="left"
    ).rename(columns={"W": "Home Wins"}).drop(columns=["Team"])

    df_flagged = df_flagged.merge(
        df_standings[["Team", "W"]],
        left_on="Away Team",
        right_on="Team",
        how="left"
    ).rename(columns={"W": "Away Wins"}).drop(columns=["Team"])

    # Determine winner
    df_flagged["Predicted Winner"] = df_flagged["Result"].apply(
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

    df_flagged["Upset Flag"] = df_flagged.apply(flag_row, axis=1)

    # Summary logs
    close_calls = df_flagged["Upset Flag"].str.contains("Close Call").sum()
    upsets = df_flagged["Upset Flag"].str.contains("Potential Upset").sum()
    logger.info(f"Upsets Agent summary: {close_calls} close calls, {upsets} potential upsets flagged.")

    # Clean up temporary columns
    df_flagged.drop(columns=["Home Wins", "Away Wins", "Predicted Winner"], inplace=True)

    return df_flagged
