import os
import sys
import io
import contextlib
from pathlib import Path
import pandas as pd
import typer
from llama_cpp import Llama
from nfl_predictor.utils.constants import (
    OUTPUT_FILE_NAME,
    FLAGGED_OUTPUT_FILE_NAME,
)

# Constants
MODEL_DIR = Path(__file__).parent / "models"
MODEL_NAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_PATH = MODEL_DIR / MODEL_NAME

app = typer.Typer(help="🧠 Blitz: Local LLM Agent for NFL predictions")

# Load model (Mistral-7B Instruct v0.2)
def load_model():
    if not MODEL_PATH.exists():
        typer.secho(f"\n🚨 Missing model file: {MODEL_NAME}", fg=typer.colors.RED)
        typer.echo("Download it from:\n  https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
        typer.echo(f"Then place it in:  llm/models/\n")
        sys.exit(1)

    with contextlib.redirect_stderr(io.StringIO()):
        return Llama(
            model_path=str(MODEL_PATH),
            n_ctx=4096,
            n_threads=4,
            n_batch=64,
            verbose=False
        )

# Global load once
LLM = load_model()

SYSTEM_PROMPT = """You are Blitz, an expert NFL analyst.
You only use the data provided.
Never guess player names or make up stats.
Respond like a smart, sharp sports analyst."""

def ask_blitz(user_prompt: str) -> str:
    full_prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\n{user_prompt.strip()} [/INST]"
    result = LLM(
        prompt=full_prompt,
        max_tokens=512,
        temperature=0.2,
        stop=["</s>"]
    )
    return result["choices"][0]["text"].strip()

@app.command()
def chat():
    """Interactive chat with Blitz about matchups, teams, or stats."""
    typer.secho("💬 Chat with Blitz! Type 'exit' to quit.\n", fg=typer.colors.CYAN)

    while True:
        try:
            user_input = typer.prompt("You")
            if user_input.strip().lower() in {"exit", "quit"}:
                break
            typer.echo(ask_blitz(user_input))
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

    # Define a custom sort order for weeks
    WEEK_ORDER = [str(i) for i in range(1, 19)] + ["WildCard", "Division", "ConfChamp", "SuperBowl"]

    def week_sort_key(week_val):
        return WEEK_ORDER.index(str(week_val)) if str(week_val) in WEEK_ORDER else -1

    # Prompt user for week
    week_input = input("Which week? (e.g., '1', 'WildCard', 'SuperBowl' or leave blank for latest): ").strip()
    if not week_input:
        unique_weeks = flagged_df["Week"].dropna().unique()
        sorted_weeks = sorted(unique_weeks, key=week_sort_key)
        week_input = sorted_weeks[-1] if sorted_weeks else None
        typer.echo(f"(Using most recent week: {week_input})")

    df_week = flagged_df[flagged_df["Week"].astype(str).str.lower() == str(week_input).lower()]

    if df_week.empty:
        typer.secho(f"No flagged results found for Week '{week_input}'", fg=typer.colors.YELLOW)
        raise typer.Exit()

    games = df_week[["Home Team", "Home Score", "Away Team", "Away Score", "Result", "Upset Flag"]].head(4)

    for _, row in games.iterrows():
        matchup = f"{row['Home Team']} ({row['Home Score']}) vs {row['Away Team']} ({row['Away Score']})"
        result = row["Result"]
        flags = row["Upset Flag"] or "None"

        prompt = f"""
        Week {week_input} – Game Recap
        
        Matchup: {matchup}
        Result: {result}
        Flagged: {flags}
        
        Based on this info, give 2-3 sharp bullet points like a sports analyst.
        • Mention the score or margin.
        • Point out why this might have been close or an upset.
        • Never guess player names or make up stats — use only what's shown.
        """

        typer.echo("\n" + "=" * 50)
        typer.echo(f"🏈 {matchup}")
        typer.echo("--- BLITZ RESPONSE ---")
        response = ask_blitz(prompt.strip())
        typer.echo(response if response else "[No response]")

# Entry
if __name__ == "__main__":
    app()
