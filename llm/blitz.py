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
def load_prediction_context(limit=25):
    path = Path(__file__).parent.parent / "nfl_predictor" / "data" / OUTPUT_FILE_NAME
    if not path.exists():
        return []

    df = pd.read_csv(path)
    df["summary"] = df.apply(
        lambda row: f"Week {row['Week']}: {row['Away Team']} ({row['Away Score']}) at {row['Home Team']} ({row['Home Score']}) — {row['Result']} [O/U: {row['Over/Under']}]",
        axis=1
    )
    return df["summary"].tolist()[-limit:]

# === CLI Commands ===
@app.command()
def chat():
    """Interactive chat with Blitz about matchups, teams, or stats."""
    typer.secho("💬 Chat with Blitz! Type 'exit' to quit.\n", fg=typer.colors.CYAN)

    # Load context from prediction output
    context_lines = load_prediction_context(limit=25)
    prediction_context = "\n".join(context_lines)

    # Initial system prompt + recent data
    system_intro = (
        "You are Blitz, a smart and approachable NFL analyst. "
        "You are chatting with a fan using recent predicted NFL matchups. "
        "Respond naturally, casually, and with sharp insight. Be concise. "
        "Use the data below if relevant, but don't repeat it unless asked.\n"
        "Recent predicted matchups:\n"
        f"{prediction_context}"
    )

    context = f"<|user|>\n{system_intro}\n<|endoftext|>\n"

    while True:
        try:
            user_input = typer.prompt("You")
            if user_input.strip().lower() in {"exit", "quit"}:
                break

            full_prompt = f"{context}<|user|>\n{user_input.strip()}\n<|endoftext|>\n<|assistant|>"
            response = LLM(
                prompt=full_prompt,
                max_tokens=256,
                temperature=0.6,
                stop=["<|user|>", "<|endoftext|>"]
            )["choices"][0]["text"].strip()

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

    # Prompt user for week, default to latest
    week_input = input("Which week? (e.g., '1', 'WildCard', 'SuperBowl' or leave blank for latest): ").strip()
    if not week_input:
        unique_weeks = flagged_df["Week"].dropna().unique()
        sorted_weeks = sorted(unique_weeks, key=week_sort_key)
        week_input = sorted_weeks[-1] if sorted_weeks else None
        typer.echo(f"(Using most recent week: {week_input})")

    # Filter by week
    df_week = flagged_df[flagged_df["Week"].astype(str).str.lower() == str(week_input).lower()]

    if df_week.empty:
        typer.secho(f"No flagged results found for Week '{week_input}'", fg=typer.colors.YELLOW)
        raise typer.Exit()

    # Use all columns and grab top 4 games
    games = df_week[[
        "Home Team", "Home Score", "Away Team", "Away Score",
        "Result", "Over/Under", "Upset Flag"
    ]].head(4)

    for _, row in games.iterrows():
        matchup = f"{row['Away Team']} ({row['Away Score']}) at {row['Home Team']} ({row['Home Score']})"
        result = row["Result"]
        predicted_margin = "tie" if "tie" in result.lower() else result
        total_line = row["Over/Under"]
        flags = row["Upset Flag"]
        is_flagged = isinstance(flags, str) and flags.strip().lower() != "none"
        flag_line = f"{flags}" if is_flagged else "None"

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
        - Second bullet: Begin with "The over/under is..." and include what that suggests about the total.
        - If flagged is "None", say: "This game is not flagged."
        - If flagged has content, summarize it clearly (e.g., Close Call, Potential Upset).
        - Do not include any intros, commentary, or analysis beyond the bullets.
        """

        typer.echo(f"\n🏈 {matchup}")
        response = ask_blitz(prompt.strip())
        typer.echo(response if response else "[No response]")

# === Entry point ===
if __name__ == "__main__":
    app()
