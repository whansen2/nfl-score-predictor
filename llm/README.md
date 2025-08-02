# Blitz – OpenAI-Powered NFL Prediction Analysis

Blitz is an intelligent AI assistant that enhances your NFL score predictor with contextual analysis. It uses OpenAI's **gpt-4o-mini** to provide grounded insights about your **Linear Regression model's predictions**, focusing on spreads, totals, and potential upsets.

## 🎯 What Blitz Analyzes

Your prediction system uses a **scikit-learn Linear Regression model** with 6 key team statistics:
- **Conversion rates** (Sc%_x, Sc%_y) 
- **First downs per game** (Tot_1stD/G)
- **Yards per play** (Y/P_x)
- **Red zone efficiency** (RZPct_x) 
- **Turnover rates** (TO%_x)

Blitz helps you understand what these predictions mean for:
- **Point spreads** and game competitiveness
- **Over/under totals** and scoring expectations
- **Potential upsets** (lower win teams favored)
- **Close calls** (games with <4 point spreads)

---

## 📁 Project Structure

```
llm/
├── blitz.py                 # OpenAI-powered analysis CLI
├── requirements.txt         # Dependencies (openai, pandas, typer)
├── Makefile                # Helper commands
└── README.md               # This file
```

---

## ⚙️ Setup

1. **API Key**: Ensure your `.env` file contains:
```bash
OPENAI_API_KEY=sk-proj-your-api-key-here
```

2. **Environment**: Create and activate virtual environment:
```bash
make venv
source ../nfl_env/bin/activate  # Note: shared with parent project
```

3. **Dependencies**:
```bash
make install
```

4. **Generate Predictions**: Run your model first:
```bash
cd .. && make run
```

---

## 🚀 Usage

### Interactive Analysis
```bash
make chat          # Chat about your model's predictions
```
Ask questions like:
- "What games look like upsets this week?"
- "Which predictions have the tightest spreads?"
- "What's the average total this week?"

### Quick Commands
```bash
make predict       # View raw prediction data with stats
make flag          # Show flagged games (upsets/close calls)  
make recap         # AI analysis of flagged games by week
make run           # Execute flag + recap together
```

### From Project Root
```bash
make blitz-chat    # Same commands available from root
make blitz-predict
make blitz-flag
make blitz-recap
make blitz-run
```

---

## 🧠 AI Model Specifications

**gpt-4o-mini** optimizations for your use case:
- **Context Window**: 128k tokens (handles full season data)
- **Output**: Up to 16k tokens per response
- **Temperature**: 0.2 (consistent analysis), 0.3 (engaging chat)
- **Cost**: $0.150/1M input, $0.600/1M output tokens
- **Typical Session**: $0.01-0.05 per analysis

**Prompt Engineering**: 
- Tailored system prompts for Linear Regression analysis
- Focus on statistical model outputs, not external NFL knowledge
- Specialized understanding of your 6-feature model
- Contextual awareness of upset flags and close call definitions
- Modern OpenAI Python client (v1.3.5+) with enhanced error handling

---

## 🎯 Example Interactions

**Chat Session:**
```
You: What upsets does the model predict this week?
🏈 Looking at the flagged games, your model has flagged 2 potential upsets...

You: Why is the Chiefs-Bills game flagged?
🏈 The model predicts Bills to win by 3, but Chiefs have more wins this season...
```

**Recap Analysis:**
```
🔴 Patriots @ Jets
   Prediction: 17-21 | Total: 38 | Spread: 4
   🚩 🚨 Potential Upset: Jets

• The model predicts a close 4-point Jets victory despite Patriots' better record
• The 38-point total suggests a defensive struggle, well below league average  
• This upset flag indicates the model sees value in Jets' statistical profile
```

---

## Notes

- Uses OpenAI's gpt-4o-mini for intelligent analysis
- Provides grounded analysis based on YOUR Linear Regression model's output
- Requires OpenAI API key in environment variables
- Optimized for cost-effective analysis using gpt-4o-mini
- Maintains conversation context for interactive sessions

---

## 📊 Technical Details

**Dependencies:**
```
openai>=1.3.5           # Modern OpenAI API client (v1.x)
python-dotenv>=1.0.0    # Environment variable management  
pandas>=2.0.0           # Data analysis for prediction files
typer>=0.9              # Modern CLI framework
```

**Data Sources:**
- `predicted_matchups_test.csv` - Your Linear Regression model output
- `predicted_matchups_flagged_test.csv` - Upsets agent analysis
- Environment variables from `.env` file

**API Integration:**
- Modern OpenAI client initialization with proper error handling
- Centralized API wrapper for consistent gpt-4o-mini usage
- Conversation history management for interactive sessions
- Optimized token usage and context window management

---

## 🔧 Troubleshooting

**"No prediction file found"**
- Run `cd .. && make run` to generate predictions first
- Ensure `ENABLE_UPSETS_AGENT=true` in `.env` for flagged data

**"OpenAI API failed"**  
- Check `OPENAI_API_KEY` in `.env` file
- Verify API key has available credits
- Ensure internet connection for API calls

**"No flagged games"**
- Your model didn't identify any upsets or close calls
- This indicates all predictions have clear winners with >4 point spreads

---

## 🏈 Ready to Analyze

Blitz transforms your Linear Regression model's raw predictions into actionable insights:

```bash
make chat
```

**What makes this different:**
- ✅ **Model-Specific**: Understands YOUR 6-feature approach
- ✅ **Grounded Analysis**: No external data speculation  
- ✅ **Cost-Effective**: gpt-4o-mini optimized for this use case
- ✅ **Statistical Focus**: Spreads, totals, and model confidence

🧠 **Powered by OpenAI gpt-4o-mini • Tailored for Linear Regression NFL predictions**
