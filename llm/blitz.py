import os
import sys
import re
from pathlib import Path
import pandas as pd
import typer
from dotenv import load_dotenv
import openai
from nfl_predictor.utils.constants import (
    OUTPUT_FILE_NAME,
    FLAGGED_OUTPUT_FILE_NAME,
)

# Load environment variables
load_dotenv()

app = typer.Typer(help="🧠 Blitz: OpenAI-powered NFL prediction analysis agent")

# === OpenAI Configuration ===
def setup_openai():
    """Initialize OpenAI client with API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        typer.secho("🚨 Missing OPENAI_API_KEY in .env file", fg=typer.colors.RED)
        typer.echo("Please add your OpenAI API key to the .env file")
        sys.exit(1)
    
    # Modern OpenAI client initialization
    return openai.OpenAI(api_key=api_key)

# Initialize OpenAI client
client = setup_openai()

def estimate_tokens(text: str) -> int:
    """Rough token estimation for gpt-4o-mini context management."""
    return len(text.split()) * 1.3  # Approximation: 1.3 tokens per word

# === Optimized Blitz AI wrapper for gpt-4o-mini ===
def ask_blitz(user_prompt: str = None, system_prompt: str = None, messages: list = None, max_tokens: int = 300, temperature: float = 0.2) -> str:
    """
    Query OpenAI's gpt-4o-mini optimized for NFL prediction analysis.
    
    Args:
        user_prompt: Single user message (for simple calls)
        system_prompt: System instructions (for simple calls)
        messages: Full message history (for chat conversations)
        max_tokens: Response length limit
        temperature: Response creativity (0.2 = consistent, 0.3 = balanced)
    
    gpt-4o-mini specs:
    - Context window: 128k tokens
    - Output: up to 16k tokens  
    - Cost: $0.150/1M input, $0.600/1M output tokens
    - Optimized for: reasoning, analysis, structured output
    """
    try:
        # Use provided messages or build from prompts
        if messages:
            api_messages = messages
        else:
            api_messages = []
            if system_prompt:
                api_messages.append({"role": "system", "content": system_prompt})
            if user_prompt:
                api_messages.append({"role": "user", "content": user_prompt})
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=api_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            frequency_penalty=0.1,  # Reduce repetition
            presence_penalty=0.1    # Encourage diverse analysis
        )
        
        return response.choices[0].message.content.strip()
        
    except openai.OpenAIError as e:
        return f"[OpenAI Error] {str(e)}"
    except Exception as e:
        return f"[Error] API failed: {str(e)}"
    
# === Enhanced helper function for NFL prediction context ===
def load_prediction_context(weeks_to_include=4):
    """Load and format NFL prediction data optimized for analysis."""
    path = Path(__file__).parent.parent / "nfl_predictor" / "data" / OUTPUT_FILE_NAME
    if not path.exists():
        return [], []

    df = pd.read_csv(path)

    # NFL-specific week ordering (playoffs prioritized)
    WEEK_ORDER = ["SuperBowl", "ConfChamp", "Division", "WildCard"] + [str(i) for i in range(18, 0, -1)]
    df["Week"] = df["Week"].astype(str)
    df["WeekOrder"] = df["Week"].apply(lambda x: WEEK_ORDER.index(x) if x in WEEK_ORDER else 999)
    df = df[df["WeekOrder"] != 999].sort_values("WeekOrder")

    # Get latest N unique weeks
    recent_weeks = df["Week"].drop_duplicates().values[:weeks_to_include]
    df_recent = df[df["Week"].isin(recent_weeks)].copy()

    # Enhanced match summary with spread analysis
    def create_summary(row):
        spread = abs(row['Home Score'] - row['Away Score'])
        total = row['Over/Under']
        return (f"Week {row['Week']}: {row['Away Team']} ({row['Away Score']}) "
                f"at {row['Home Team']} ({row['Home Score']}) — {row['Result']} "
                f"[Spread: {spread}, Total: {total}]")

    df_recent.loc[:, "summary"] = df_recent.apply(create_summary, axis=1)

    return df_recent["summary"].tolist(), list(recent_weeks)

def get_prediction_stats():
    """Get summary statistics about the prediction model's performance."""
    path = Path(__file__).parent.parent / "nfl_predictor" / "data" / OUTPUT_FILE_NAME
    if not path.exists():
        return "No prediction data available"
    
    df = pd.read_csv(path)
    if df.empty:
        return "No predictions found"
    
    total_games = len(df)
    avg_total = df['Over/Under'].mean()
    high_scoring = len(df[df['Over/Under'] > 50])
    low_scoring = len(df[df['Over/Under'] < 40])
    
    return (f"Model Summary: {total_games} games predicted, "
            f"avg total: {avg_total:.1f}, "
            f"{high_scoring} high-scoring (50+), "
            f"{low_scoring} low-scoring (<40)")

def get_flag_summary():
    """Get summary of flagged games for context."""
    path = Path(__file__).parent.parent / "nfl_predictor" / "data" / FLAGGED_OUTPUT_FILE_NAME
    if not path.exists():
        return "No flagged games data available"
    
    df = pd.read_csv(path)
    flagged = df[df["Upset Flag"].notna() & (df["Upset Flag"] != "")]
    
    if flagged.empty:
        return "No games flagged by upsets agent"
    
    close_calls = len(flagged[flagged["Upset Flag"].str.contains("Close Call", na=False)])
    upsets = len(flagged[flagged["Upset Flag"].str.contains("Potential Upset", na=False)])
    
    return f"Flagged Games: {close_calls} close calls, {upsets} potential upsets"

# === Enhanced CLI Commands ===
@app.command()
def chat():
    """Interactive chat with Blitz about your NFL prediction model's results."""
    typer.secho("💬 Chat with Blitz about your NFL predictions! Type 'exit' to quit.\n", fg=typer.colors.CYAN)

    # Load prediction context and model stats
    context_lines, available_weeks = load_prediction_context(weeks_to_include=6)
    prediction_context = "\n".join(context_lines)
    weeks_summary = ", ".join(available_weeks)
    model_stats = get_prediction_stats()
    flag_stats = get_flag_summary()

    # Enhanced system prompt tailored to your Linear Regression model
    system_prompt = (
        "You are Blitz, an expert NFL analyst specializing in Linear Regression prediction model analysis. "
        "You analyze predictions made by a scikit-learn Linear Regression model that uses 6 key team statistics: "
        "conversion rates (Sc%_x, Sc%_y), first downs per game, yards per play, red zone efficiency, and turnover rates.\n\n"
        
        "ANALYSIS PRINCIPLES:\n"
        "- You ONLY discuss the prediction data provided below - never invent stats or external knowledge\n"
        "- Focus on spreads, totals, potential upsets, and close games\n"
        "- When discussing 'flags': Close Call = <4 point spread, Potential Upset = lower win team favored\n"
        "- Be conversational but analytical - explain what the predictions suggest\n"
        "- Use exact week labels from data (SuperBowl, ConfChamp, Division, WildCard, or numbers)\n\n"
        
        f"MODEL OVERVIEW: {model_stats}\n"
        f"FLAGGED GAMES: {flag_stats}\n"
        f"AVAILABLE WEEKS: {weeks_summary}\n\n"
        f"PREDICTION DATA:\n{prediction_context}"
    )

    # Conversation history optimized for gpt-4o-mini's 128k context window
    conversation_history = []
    chatting = True
    
    while chatting:
        try:
            user_input = typer.prompt("You")
            if user_input.strip().lower() in {"exit", "quit", "bye", "done"}:
                typer.echo("👋 Later! Keep analyzing those predictions!")
                chatting = False
                continue

            # Add user message to history
            conversation_history.append({"role": "user", "content": user_input.strip()})
            
            # Prepare messages (system + last 12 exchanges to stay efficient)
            messages_for_api = [{"role": "system", "content": system_prompt}]
            messages_for_api.extend(conversation_history[-12:])
            
            try:
                assistant_response = ask_blitz(
                    messages=messages_for_api,
                    max_tokens=500,  # Increased for detailed chat analysis
                    temperature=0.3  # Slightly higher for engaging conversation
                )
                
                # Add assistant response to history
                conversation_history.append({"role": "assistant", "content": assistant_response})
                
                # Output with NFL styling
                typer.echo(f"\n🏈 {assistant_response}\n")
                
            except Exception as e:
                typer.secho(f"[Error] OpenAI API failed: {str(e)}", fg=typer.colors.RED)

        except KeyboardInterrupt:
            typer.echo("\n👋 Exiting Blitz chat.")
            chatting = False

@app.command()
def predict():
    """Display raw prediction data from your Linear Regression model."""
    path = Path(__file__).parent.parent / "nfl_predictor" / "data" / OUTPUT_FILE_NAME
    if path.exists():
        df = pd.read_csv(path)
        typer.echo(f"\n📊 Linear Regression Model Predictions ({len(df)} games):\n")
        typer.echo(df.to_string(index=False))
        typer.echo(f"\n📈 {get_prediction_stats()}")
    else:
        typer.secho("No prediction file found. Run `make run` from the root directory first.", fg=typer.colors.RED)

@app.command()
def flag():
    """Show games flagged by the upsets agent (close calls and potential upsets)."""
    path = Path(__file__).parent.parent / "nfl_predictor" / "data" / FLAGGED_OUTPUT_FILE_NAME
    if path.exists():
        df = pd.read_csv(path)
        flagged = df[df["Upset Flag"].notna() & (df["Upset Flag"] != "")]
        
        if not flagged.empty:
            typer.echo(f"\n🚨 Flagged Games ({len(flagged)} total):\n")
            
            # Display with enhanced formatting
            for _, row in flagged.iterrows():
                spread = abs(row['Home Score'] - row['Away Score'])
                flag_type = "🔴" if "Upset" in row["Upset Flag"] else "🟡"
                
                typer.echo(f"{flag_type} Week {row['Week']}: {row['Away Team']} @ {row['Home Team']}")
                typer.echo(f"   Predicted: {row['Away Score']}-{row['Home Score']} (Spread: {spread})")
                typer.echo(f"   Flag: {row['Upset Flag']}")
                typer.echo("")
            
            typer.echo(f"📊 {get_flag_summary()}")
        else:
            typer.echo("🟢 No games flagged - all predictions look standard.")
    else:
        typer.secho("No flagged file found. Enable ENABLE_UPSETS_AGENT=true in .env and run `make run`.", fg=typer.colors.RED)

@app.command()
def recap():
    """AI-enhanced analysis of flagged games from your prediction model."""
    flagged_path = Path(__file__).parent.parent / "nfl_predictor" / "data" / FLAGGED_OUTPUT_FILE_NAME

    if not flagged_path.exists():
        typer.secho("No flagged prediction file found. Enable ENABLE_UPSETS_AGENT=true and run prediction model first.", fg=typer.colors.RED)
        raise typer.Exit()

    flagged_df = pd.read_csv(flagged_path)

    # NFL week ordering
    WEEK_ORDER = [str(i) for i in range(1, 19)] + ["WildCard", "Division", "ConfChamp", "SuperBowl"]

    def week_sort_key(week_val):
        return WEEK_ORDER.index(str(week_val)) if str(week_val) in WEEK_ORDER else 999

    # Interactive week selection
    week_input = input("Which week to analyze? (e.g., '1', 'WildCard', 'SuperBowl' or leave blank for latest): ").strip()
    if not week_input:
        unique_weeks = flagged_df["Week"].dropna().unique()
        sorted_weeks = sorted(unique_weeks, key=week_sort_key)
        week_input = sorted_weeks[-1] if sorted_weeks else None
        typer.echo(f"🎯 Analyzing most recent week: {week_input}")

    # Filter to selected week
    df_week = flagged_df[flagged_df["Week"].astype(str).str.lower() == str(week_input).lower()]

    if df_week.empty:
        typer.secho(f"No predictions found for Week '{week_input}'", fg=typer.colors.YELLOW)
        raise typer.Exit()

    # Enhanced system prompt for Linear Regression model analysis
    system_prompt = (
        "You are Blitz, an expert NFL analyst specializing in Linear Regression prediction model insights. "
        "Analyze ONLY the prediction data provided. Focus on what the statistical model suggests about each game. "
        "Explain the significance of spreads, totals, and flags in practical terms for betting/fantasy context. "
        "Format: exactly 3 concise bullet points starting with '•'"
    )

    typer.echo(f"\n🏈 Week {week_input} Prediction Analysis\n" + "="*50)

    # Analyze each game
    for _, row in df_week.iterrows():
        spread = abs(row['Home Score'] - row['Away Score'])
        total = row['Over/Under']
        flag = row.get('Upset Flag', '')
        
        # Determine game type for analysis
        game_type = "flagged" if flag else "standard"
        flag_clean = re.sub(r"[⚠️🚨️]", "", str(flag)).strip() if flag else "None"

        user_prompt = f"""
        Linear Regression Model Prediction Analysis:
        
        Game: {row['Away Team']} at {row['Home Team']} (Week {week_input})
        Model Output:
        - Away Score: {row['Away Score']}
        - Home Score: {row['Home Score']} 
        - Point Spread: {spread} points
        - Total Points: {total}
        - Model Flag: {flag_clean}
        - Result: {row['Result']}
        
        Context: This is a {game_type} prediction from a Linear Regression model using team statistics 
        (conversion rates, yards per play, red zone efficiency, turnover rates).
        
        Provide exactly 3 bullet points analyzing what this prediction suggests:
        • Game competitiveness and spread significance
        • Total points prediction and scoring expectations  
        • Model confidence and any notable flags/concerns
        """

        matchup = f"{row['Away Team']} @ {row['Home Team']}"
        icon = "🔴" if "Upset" in str(flag) else "🟡" if flag else "⚪"
        
        typer.echo(f"\n{icon} {matchup}")
        typer.echo(f"   Prediction: {row['Away Score']}-{row['Home Score']} | Total: {total} | Spread: {spread}")
        
        if flag:
            typer.echo(f"   🚩 {flag}")
        
        # Get AI analysis
        response = ask_blitz(user_prompt.strip(), system_prompt, max_tokens=400)
        typer.echo(f"\n{response}\n" + "-"*40)

# === Entry point ===
if __name__ == "__main__":
    app()
