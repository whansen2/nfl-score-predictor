import os
import sys
import io
import re
import contextlib
from pathlib import Path
import pandas as pd
import typer
from llama_cpp import Llama
from nfl_predictor.utils.constants import (
    OUTPUT_FILE_NAME,
    FLAGGED_OUTPUT_FILE_NAME,
)

# === Constants ===
MODEL_DIR = Path(__file__).parent / "models"
MODEL_NAME = "stablelm-zephyr-3b.Q4_K_M.gguf"
MODEL_PATH = MODEL_DIR / MODEL_NAME

app = typer.Typer(help="🧠 Blitz: Local LLM Agent for NFL predictions")

# === Load the StableLM-Zephyr model ===
def load_model():
    if not MODEL_PATH.exists():
        typer.secho(f"\n🚨 Missing model file: {MODEL_NAME}", fg=typer.colors.RED)
        typer.echo("Download it from:\n  https://huggingface.co/TheBloke/stablelm-zephyr-3b-GGUF")
        typer.echo(f"Then place it in:  llm/models/\n")
        sys.exit(1)

    with contextlib.redirect_stderr(io.StringIO()):
        return Llama(
            model_path=str(MODEL_PATH),
            n_ctx=2048,
            n_threads=4,
            n_batch=32,
            verbose=False
        )

# === Global LLM instance ===
LLM = load_model()

# === Blitz prompt wrapper (Zephyr format) ===
def ask_blitz(user_prompt: str) -> str:
    formatted = f"<|user|>\nYou are Blitz, an expert NFL analyst.\n\n{user_prompt}<|endoftext|>\n<|assistant|>"
    try:
        result = LLM(
            prompt=formatted,
            max_tokens=256,
            temperature=0.2,
            stream=False,
            stop=["<|user|>", "<|endoftext|>"]
        )
        return result["choices"][0]["text"].strip()
    except Exception as e:
        return f"[Error] LLM failed: {str(e)}"
    
# === Helper function for chat ===
def load_prediction_context(weeks_to_include=4):
    path = Path(__file__).parent.parent / "nfl_predictor" / "data" / OUTPUT_FILE_NAME
    if not path.exists():
        return [], []

    df = pd.read_csv(path)

    # Custom reverse-priority week order
    CUSTOM_ORDER = ["SuperBowl", "ConfChamp", "Division", "WildCard"] + [str(i) for i in range(18, 0, -1)]
    df["Week"] = df["Week"].astype(str)
    df["WeekOrder"] = df["Week"].apply(lambda x: CUSTOM_ORDER.index(x) if x in CUSTOM_ORDER else -1)
    df = df[df["WeekOrder"] >= 0].sort_values("WeekOrder")

    # Get latest N unique weeks
    recent_weeks = df["Week"].drop_duplicates().values[:weeks_to_include]
    df_recent = df[df["Week"].isin(recent_weeks)].copy()

    # Create match summary per row
    df_recent.loc[:, "summary"] = df_recent.apply(
        lambda row: f"Week {row['Week']}: {row['Away Team']} ({row['Away Score']}) at {row['Home Team']} ({row['Home Score']}) — {row['Result']} [O/U: {row['Over/Under']}]",
        axis=1
    )

    # Return both summaries and the available sorted weeks
    return df_recent["summary"].tolist(), list(recent_weeks)

# === CLI Commands ===
@app.command()
def chat():
    """Interactive chat with Blitz about matchups, teams, or stats."""
    typer.secho("💬 Chat with Blitz! Type 'exit' to quit.\n", fg=typer.colors.CYAN)

    # Load recent prediction summaries + sorted week labels
    context_lines, available_weeks = load_prediction_context(weeks_to_include=4)
    prediction_context = "\n".join(context_lines)
    weeks_summary = ", ".join(available_weeks)

    # Define system prompt
    system_prompt = (
        "You are Blitz, a smart but grounded NFL fan. "
        "You're chatting with another football fan. "
        "You ONLY know the prediction data shown below. "
        "You do NOT know how the predictions were made — so don't explain it. "
        "You don't have access to player stats, rosters, or algorithms. "
        "Never guess or make things up. "
        "Only answer based on what's provided. "
        "If the user greets you or says 'Hey', respond with a greeting and wait for a follow-up. "
        "Do NOT list prediction data unless asked directly. "
        "Use the exact week label from the prediction data (e.g., 'SuperBowl', 'ConfChamp'). Never make up or guess week numbers.\n\n"
        f"Prediction data includes these weeks (most recent first): {weeks_summary}\n\n"
        f"Prediction data:\n{prediction_context}\n"
    )

    MAX_TURNS = 4  # Keep the last max_turns exchanges (user+assistant)
    turns = []  # Stores tuples like (user_input, model_response)

    while True:
        try:
            user_input = typer.prompt("You")
            if user_input.strip().lower() in {"exit", "quit", "bye"}:
                typer.echo("👋 Later!")
                break

            # Build chat history from system prompt + recent turns
            chat_history = f"<|user|>\n{system_prompt}\n"
            for user, assistant in turns[-MAX_TURNS:]:
                chat_history += f"<|user|>\n{user}\n<|assistant|>{assistant}\n"

            # Add the latest user input
            chat_history += f"<|user|>\n{user_input.strip()}\n<|assistant|>"

            # Generate Blitz's response
            response = LLM(
                prompt=chat_history,
                max_tokens=300,
                temperature=0.3,
                stop=["<|user|>", "<|endoftext|>"]
            )["choices"][0]["text"].strip()

            # Save the turn
            turns.append((user_input.strip(), response))

            # Trim turns to maintain a sliding window of memory
            if len(turns) > MAX_TURNS:
                turns = turns[-MAX_TURNS:]

            # Output Blitz's response
            typer.echo(response)

        except KeyboardInterrupt:
            typer.echo("\n👋 Exiting Blitz chat.")
            break

@app.command()
def predict():
    """Show score predictions from the latest run."""
    path = Path(__file__).parent.parent / "nfl_predictor" / "data" / OUTPUT_FILE_NAME
    if path.exists():
        df = pd.read_csv(path)
        typer.echo(df)
    else:
        typer.secho("No prediction file found. Run `make run` first.", fg=typer.colors.RED)

@app.command()
def flag():
    """Show close calls or upset flags from the latest run."""
    path = Path(__file__).parent.parent / "nfl_predictor" / "data" / FLAGGED_OUTPUT_FILE_NAME
    if path.exists():
        df = pd.read_csv(path)
        df = df[df["Upset Flag"].notna() & (df["Upset Flag"] != "")]
        typer.echo(df)
    else:
        typer.secho("No flagged file found. Enable upsets agent or run `make run`.", fg=typer.colors.RED)

@app.command()
def recap():
    """Summarize the week's top flagged games (upsets or close calls)."""
    flagged_path = Path(__file__).parent.parent / "nfl_predictor" / "data" / FLAGGED_OUTPUT_FILE_NAME

    if not flagged_path.exists():
        typer.secho("No flagged prediction file found. Run `make run` or enable flagged data.", fg=typer.colors.RED)
        raise typer.Exit()

    flagged_df = pd.read_csv(flagged_path)

    # Sort logic for week ordering
    WEEK_ORDER = [str(i) for i in range(1, 19)] + ["WildCard", "Division", "ConfChamp", "SuperBowl"]

    def week_sort_key(week_val):
        return WEEK_ORDER.index(str(week_val)) if str(week_val) in WEEK_ORDER else -1

    # Ask user for week
    week_input = input("Which week? (e.g., '1', 'WildCard', 'SuperBowl' or leave blank for latest): ").strip()
    if not week_input:
        unique_weeks = flagged_df["Week"].dropna().unique()
        sorted_weeks = sorted(unique_weeks, key=week_sort_key)
        week_input = sorted_weeks[-1] if sorted_weeks else None
        typer.echo(f"(Using most recent week: {week_input})")

    # Filter to that week
    df_week = flagged_df[flagged_df["Week"].astype(str).str.lower() == str(week_input).lower()]

    if df_week.empty:
        typer.secho(f"No flagged results found for Week '{week_input}'", fg=typer.colors.YELLOW)
        raise typer.Exit()

    # Clean flag function to strip emojis and keep clarity
    def clean_flag(flag):
        if pd.isna(flag) or not str(flag).strip():
            return "None"
        # Remove known emojis
        return re.sub(r"[⚠️🚨️]", "", str(flag)).strip()

    # Loop through and summarize all games
    games = df_week[[
        "Home Team", "Home Score", "Away Team", "Away Score",
        "Result", "Over/Under", "Upset Flag"
    ]]

    for _, row in games.iterrows():
        matchup = f"{row['Away Team']} ({row['Away Score']}) at {row['Home Team']} ({row['Home Score']})"
        predicted_margin = "tie" if "tie" in row["Result"].lower() else row["Result"]
        total_line = row["Over/Under"]
        flag_line = clean_flag(row["Upset Flag"])

        prompt = f"""
        Game Info:
        - Week: {week_input}
        - Home Team: {row['Home Team']} ({row['Home Score']})
        - Away Team: {row['Away Team']} ({row['Away Score']})
        - Predicted Result: {predicted_margin}
        - Over/Under: {total_line}
        - Flagged: {flag_line}

        You are Blitz, an expert NFL analyst.

        Write exactly 2-3 concise bullet points using ONLY the data above.
        - Each bullet must start with "- "
        - First bullet: Begin with "The predicted result is..." and include the score or margin.
        - Second bullet: Begin with "The over/under is..." and mention the total.
        - If Flagged is "None", say: "This game is not flagged."
        - If Flagged has content, say: "This game is flagged as {flag_line}."
        Do NOT explain, summarize, or invent anything else.
        """

        typer.echo(f"\n🏈 {matchup}")
        response = ask_blitz(prompt.strip())
        typer.echo(response if response else "[No response]")

# === Entry point ===
if __name__ == "__main__":
    app()
