# AI Agent Integration Plan for NBA Player Stats Predictor

## Overview

This document outlines the architecture and implementation plan for integrating an AI assistant/agent into the NBA Player Stats Predictor application. The agent will enable natural language interactions for common tasks like viewing player stats, generating predictions, and managing favorites.

---

## Table of Contents

1. [LLM Setup & API Keys](#llm-setup--api-keys)
2. [Goals & Requirements](#goals--requirements)
3. [Architecture Design](#architecture-design)
4. [Agent Capabilities](#agent-capabilities)
5. [Performance Optimization Strategy](#performance-optimization-strategy)
6. [Implementation Phases](#implementation-phases)
7. [API Design](#api-design)
8. [Security Considerations](#security-considerations)
9. [Testing Strategy](#testing-strategy)

---

## LLM Setup & API Keys

You can use **any major LLM provider**. Here are your options:

### Option 1: Anthropic Claude (Recommended for Quality)

**Best for:** Complex reasoning, nuanced responses, safety  
**Cost:** ~$3/million input tokens, ~$15/million output tokens (Claude 3.5 Sonnet)  
**Latency:** 500-1500ms

**How to get API key:**
1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign up / Log in
3. Go to **API Keys** in the left sidebar
4. Click **Create Key**
5. Copy your key (starts with `sk-ant-...`)

**Add to Streamlit secrets:**
```toml
# .streamlit/secrets.toml
[llm]
provider = "anthropic"
api_key = "sk-ant-api03-..."
model = "claude-3-5-sonnet-20241022"
```

**Python usage:**
```python
from anthropic import Anthropic

client = Anthropic(api_key=st.secrets["llm"]["api_key"])
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": query}]
)
```

---

### Option 2: Google Gemini (Best Free Tier)

**Best for:** Cost-conscious, Google ecosystem integration  
**Cost:** FREE tier (60 requests/minute), then ~$0.50/million tokens  
**Latency:** 300-800ms

**How to get API key:**
1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Sign in with Google account
3. Click **Get API Key** (top right)
4. Click **Create API key in new project**
5. Copy your key (starts with `AIza...`)

**Add to Streamlit secrets:**
```toml
# .streamlit/secrets.toml
[llm]
provider = "gemini"
api_key = "AIzaSy..."
model = "gemini-1.5-flash"  # Fast & cheap, or "gemini-1.5-pro" for quality
```

**Python usage:**
```python
import google.generativeai as genai

genai.configure(api_key=st.secrets["llm"]["api_key"])
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content(query)
```

---

### Option 3: OpenAI GPT (Best Function Calling)

**Best for:** Structured output, function calling, widespread support  
**Cost:** ~$0.15/million input tokens (GPT-4o-mini), ~$2.50/million (GPT-4o)  
**Latency:** 200-500ms

**How to get API key:**
1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up / Log in
3. Go to **API Keys** (left sidebar or settings)
4. Click **Create new secret key**
5. Copy your key (starts with `sk-...`)
6. **Add billing**: Settings → Billing → Add payment method (required)

**Add to Streamlit secrets:**
```toml
# .streamlit/secrets.toml
[llm]
provider = "openai"
api_key = "sk-..."
model = "gpt-4o-mini"  # Cheap & fast, or "gpt-4o" for quality
```

**Python usage:**
```python
from openai import OpenAI

client = OpenAI(api_key=st.secrets["llm"]["api_key"])
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": query}]
)
```

---

### Option 4: Local Models with Ollama (Free, Private)

**Best for:** Privacy, no API costs, offline use  
**Cost:** FREE (runs on your hardware)  
**Latency:** 500-2000ms (depends on hardware)

**How to set up:**
1. Install Ollama: [ollama.ai/download](https://ollama.ai/download)
2. Open terminal and run:
   ```bash
   ollama pull llama3.1:8b    # Good balance of speed/quality
   # OR
   ollama pull mistral        # Fast, good for simple queries
   # OR  
   ollama pull phi3           # Smallest, fastest
   ```
3. Start Ollama (runs in background after install)

**Add to Streamlit secrets:**
```toml
# .streamlit/secrets.toml
[llm]
provider = "ollama"
model = "llama3.1:8b"
base_url = "http://localhost:11434"  # Default Ollama URL
```

**Python usage:**
```python
import ollama

response = ollama.chat(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": query}]
)
```

---

### Comparison Table

| Provider | Cost | Speed | Quality | Function Calling | Setup Difficulty |
|----------|------|-------|---------|------------------|------------------|
| **Claude** | $$ | Medium | ⭐⭐⭐⭐⭐ | ✅ Good | Easy |
| **Gemini** | Free/$| Fast | ⭐⭐⭐⭐ | ✅ Good | Easy |
| **OpenAI** | $ | Fast | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ Best | Easy |
| **Ollama** | Free | Varies | ⭐⭐⭐ | ❌ Limited | Medium |

### Recommendation

For this basketball app agent:
- **Start with Gemini**: Free tier is generous (60 req/min), fast, easy setup
- **Upgrade to Claude/OpenAI** if you need better reasoning for betting analysis
- **Use Ollama** for local development/testing to save costs

---

### Universal Agent Client

Here's a wrapper that works with any provider:

```python
# src/agent/llm_client.py
import streamlit as st

class LLMClient:
    def __init__(self):
        self.provider = st.secrets.get("llm", {}).get("provider", "gemini")
        self.api_key = st.secrets.get("llm", {}).get("api_key", "")
        self.model = st.secrets.get("llm", {}).get("model", "gemini-1.5-flash")
    
    def chat(self, messages: list) -> str:
        if self.provider == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=messages
            )
            return response.content[0].text
        
        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)
            # Convert messages to Gemini format
            prompt = "\n".join([m["content"] for m in messages])
            response = model.generate_content(prompt)
            return response.text
        
        elif self.provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return response.choices[0].message.content
        
        elif self.provider == "ollama":
            import ollama
            response = ollama.chat(model=self.model, messages=messages)
            return response["message"]["content"]
        
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

# Usage
llm = LLMClient()
response = llm.chat([{"role": "user", "content": "Who leads the NBA in scoring?"}])
```

---

## Goals & Requirements

### Primary Goals
- Enable natural language queries for player stats and predictions
- Provide quick access to common actions without navigation
- Maintain app performance (no slowdowns)
- Support both authenticated and guest users

### Non-Goals (Out of Scope Initially)
- Complex multi-turn reasoning
- External data modification
- Real-time game commentary

---

## Architecture Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit Frontend                        │
│                                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Chat Widget │  │  Agent UI   │  │  Existing App Pages     │  │
│  │ (Floating)  │  │  Response   │  │  (Stats, Predictions)   │  │
│  └──────┬──────┘  └──────▲──────┘  └───────────▲─────────────┘  │
│         │                │                      │                │
│         ▼                │                      │                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Agent Orchestrator                      │   │
│  │  • Intent Classification  • Action Router  • Response Gen │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                              │                                   │
└──────────────────────────────┼───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                        Action Layer                               │
│                                                                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │
│  │  Player    │  │ Prediction │  │ Favorites  │  │  Schedule  │  │
│  │  Actions   │  │  Actions   │  │  Actions   │  │  Actions   │  │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  │
│        │               │               │               │          │
│        ▼               ▼               ▼               ▼          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              Existing Data Layer (Cached)                   │  │
│  │  • get_player_game_log()  • get_nba_injuries()             │  │
│  │  • get_team_ratings()     • predict_with_drtg()            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Reuse Existing Functions**: The agent calls the same cached functions the UI uses
2. **No Duplicate Data Fetching**: Leverage `@st.cache_data` decorators already in place
3. **Lightweight LLM Calls**: Use function calling rather than verbose prompts
4. **Streaming Responses**: Display results progressively to feel responsive
5. **Graceful Degradation**: If LLM fails, suggest manual navigation

---

## Agent Capabilities

### Tier 1: Read-Only Queries (Phase 1)

| Action | Example Query | Function Called |
|--------|---------------|-----------------|
| Get player stats | "Show me LeBron's season averages" | `get_player_game_log()` |
| Get recent games | "What did Curry score last 5 games?" | `get_player_game_log()` |
| Get team standings | "Where are the Lakers in the standings?" | `get_league_standings()` |
| Get injuries | "Who's injured on the Celtics?" | `get_nba_injuries()` |
| Get upcoming games | "Who do the Warriors play next?" | `get_team_upcoming_games()` |
| Compare players | "Compare Luka vs Shai this season" | `get_player_game_log()` x2 |

### Tier 2: Predictions (Phase 2)

| Action | Example Query | Function Called |
|--------|---------------|-----------------|
| Generate prediction | "Predict Ja Morant vs Warriors" | `predict_with_drtg()` |
| Head-to-head history | "How does Tatum do against the Heat?" | `get_player_game_log()` + filter |
| Matchup analysis | "Who has the worst defense?" | `get_current_defensive_ratings()` |

### Tier 3: User Actions (Phase 3)

| Action | Example Query | Function Called |
|--------|---------------|-----------------|
| Add favorite player | "Add Giannis to my favorites" | `auth.add_favorite_player()` |
| Add favorite team | "Watch the Knicks" | `auth.add_favorite_team()` |
| Remove favorite | "Remove Hawks from my teams" | `auth.remove_favorite_team()` |
| Navigate to page | "Go to player stats for Jokic" | Session state redirect |

### Tier 4: Betting Analysis (Phase 4)

| Action | Example Query | Data Used |
|--------|---------------|-----------|
| Over/Under advice | "Should I take Ja Morant over 24.5 points vs Warriors?" | Prediction + H2H + Injuries |
| Prop analysis | "Is Trae Young over 8.5 assists a good bet against Knicks?" | Prediction + Matchup history |
| Injury-adjusted take | "How do injuries affect taking Jokic over 12.5 rebounds?" | Prediction + opp injuries |
| Confidence level | "How confident is the LeBron under 27.5 points prediction?" | Model variance analysis |
| Best bets today | "What player props look good today?" | All predictions + odds comparison |

#### Example Betting Q&A Responses

**User:** "Should I take Ja Morant over 24.5 points against the Warriors?"

**Agent Response:**
```
📊 Ja Morant vs Warriors - Points Analysis

🎯 Model Prediction: 26.3 points
📈 Season Average: 25.8 PPG
🏀 vs Warriors (2 games): 28.5 PPG average

🏥 Injury Factors:
• Warriors: Draymond Green (Out) - Their best perimeter defender
• Warriors: Andrew Wiggins (Questionable)
• Grizzlies: No significant injuries

📋 Analysis:
The model predicts 26.3 points, which is 1.8 points ABOVE the line of 24.5.
With Draymond Green out, the Warriors' interior defense is weakened.
Morant has exceeded 24.5 points in 4 of his last 5 games.
Historical h2h: Morant averages 28.5 vs Warriors (2 games this season).

⚠️ Factors to Consider:
• Game pace (Warriors play slow, may limit possessions)
• Rest days (back-to-back?)
• Home/Away (This is @ GSW)

🔮 Lean: OVER 24.5 looks favorable based on prediction + injuries
Confidence: Medium-High (model +1.8 above line, injury boost)

Disclaimer: This is analysis, not financial advice. Always gamble responsibly.
```

**User:** "How confident should I be in the Tatum over 27.5 points prediction?"

**Agent Response:**
```
📊 Jayson Tatum Confidence Analysis

🎯 Model Prediction: 28.1 points vs Heat
📈 Season Average: 27.4 PPG

📉 Variance Analysis:
• Tatum's scoring std dev: 6.2 points (moderate variance)
• Range typically: 21-34 points
• Hit rate on 27.5+: 48% of games this season

🏀 Matchup Factors:
• Heat defense: 8th in NBA (tough matchup)
• vs Heat this season: 24.5 PPG (2 games)
• Home game advantage: +2.1 PPG at home

⚠️ Risk Factors:
• High variance player - can drop 18 or explode for 40
• Heat historically limit his efficiency
• Butler likely guarding him

🔮 Confidence Level: MEDIUM
The line is very close to his average. This is essentially a coin flip.
Consider passing or looking for better value elsewhere.
```

---

## Performance Optimization Strategy

### 1. Leverage Existing Cache

All data functions already have caching:
```python
@st.cache_data(ttl=3600)   # Team ratings - 1 hour
@st.cache_data(ttl=1800)   # Injuries - 30 minutes
@st.cache_data(ttl=600)    # Player game logs - 10 minutes
```

**Agent benefit**: Queries hit warm cache 95%+ of the time.

### 2. Lightweight LLM Strategy

```python
# Option A: OpenAI Function Calling (Recommended)
# - Single API call with structured output
# - Model: gpt-4o-mini for speed + cost efficiency
# - Average latency: 200-500ms

# Option B: Local Small Model
# - Ollama with mistral-7b or phi-3
# - No API costs, runs on CPU
# - Average latency: 500-1500ms
```

### 3. Intent Classification First

Before calling LLM, use fast regex/keyword matching for common queries:

```python
FAST_PATTERNS = {
    r"(stats|averages?)\s+(for\s+)?(.+)": "get_player_stats",
    r"predict\s+(.+)\s+vs\s+(.+)": "get_prediction",
    r"(add|favorite)\s+(.+)": "add_favorite",
    r"(injur|hurt|out)\s+(.+)": "get_injuries",
}

def quick_classify(query: str) -> Optional[str]:
    for pattern, intent in FAST_PATTERNS.items():
        if match := re.search(pattern, query.lower()):
            return intent, match.groups()
    return None  # Fall back to LLM
```

### 4. Response Streaming

Show partial results immediately:

```python
with st.chat_message("assistant"):
    with st.status("Looking up player stats...", expanded=True) as status:
        player_df = get_player_game_log(player_id, season)
        status.update(label="Calculating averages...")
        # Format response
        status.update(label="Done!", state="complete")
    st.write(formatted_response)
```

### 5. Debouncing & Rate Limiting

```python
# Prevent spam queries
if 'last_agent_query_time' in st.session_state:
    elapsed = time.time() - st.session_state.last_agent_query_time
    if elapsed < 1.0:  # 1 second cooldown
        st.warning("Please wait a moment...")
        return
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Create `src/agent/` module structure
- [ ] Implement Action Registry pattern
- [ ] Build intent classifier (regex + optional LLM)
- [ ] Create floating chat widget UI
- [ ] Connect to 3-5 read-only actions

### Phase 2: Core Features (Week 3-4)
- [ ] Add prediction generation action
- [ ] Implement player comparison action
- [ ] Add response formatting (tables, charts)
- [ ] Handle ambiguous queries (player name disambiguation)
- [ ] Add conversation history (last 5 messages)

### Phase 3: User Actions (Week 5-6)
- [ ] Implement favorite management actions
- [ ] Add navigation actions
- [ ] Create action confirmation flow for mutations
- [ ] Add undo capability for accidental actions

### Phase 4: Polish (Week 7-8)
- [ ] Add suggested queries / quick actions
- [ ] Implement feedback mechanism
- [ ] Add analytics for popular queries
- [ ] Performance optimization based on usage data

---

## API Design

### Action Registry Pattern

```python
# src/agent/actions.py

from dataclasses import dataclass
from typing import Callable, Dict, Any, List

@dataclass
class AgentAction:
    name: str
    description: str
    parameters: Dict[str, str]
    handler: Callable
    requires_auth: bool = False

class ActionRegistry:
    def __init__(self):
        self._actions: Dict[str, AgentAction] = {}
    
    def register(self, action: AgentAction):
        self._actions[action.name] = action
    
    def get(self, name: str) -> AgentAction:
        return self._actions.get(name)
    
    def list_actions(self) -> List[AgentAction]:
        return list(self._actions.values())

# Example registration
registry = ActionRegistry()

registry.register(AgentAction(
    name="get_player_stats",
    description="Get a player's season statistics",
    parameters={"player_name": "Name of the NBA player"},
    handler=lambda player_name: get_formatted_player_stats(player_name),
    requires_auth=False
))

registry.register(AgentAction(
    name="add_favorite_player",
    description="Add a player to the user's favorites list",
    parameters={"player_name": "Name of the player to favorite"},
    handler=lambda player_name: auth.add_favorite_player(player_name),
    requires_auth=True  # Requires login
))
```

### LLM Function Calling Schema

```python
# For OpenAI/compatible APIs
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_player_stats",
            "description": "Get season statistics for an NBA player",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_name": {
                        "type": "string",
                        "description": "Full name of the player (e.g., 'LeBron James')"
                    }
                },
                "required": ["player_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_prediction",
            "description": "Generate stat prediction for a player against a specific opponent",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_name": {"type": "string"},
                    "opponent_team": {"type": "string", "description": "Team abbreviation (e.g., 'LAL', 'BOS')"}
                },
                "required": ["player_name", "opponent_team"]
            }
        }
    }
]
```

---

## Security Considerations

### 1. Authentication for Mutations
```python
def execute_action(action_name: str, params: dict, user_session: dict):
    action = registry.get(action_name)
    
    if action.requires_auth and not user_session.get('authenticated'):
        return {"error": "Please log in to perform this action"}
    
    return action.handler(**params)
```

### 2. Input Sanitization
```python
def sanitize_player_name(name: str) -> str:
    # Remove any SQL/injection attempts
    return re.sub(r'[^\w\s\'-]', '', name)[:50]
```

### 3. Rate Limiting per User
```python
# Track queries per session
MAX_QUERIES_PER_MINUTE = 10

def check_rate_limit(session_id: str) -> bool:
    # Implement token bucket or sliding window
    pass
```

### 4. No Direct API Key Exposure
- Store LLM API keys in Streamlit secrets
- Never log full prompts containing user data

---

## Testing Strategy

### Unit Tests
```python
# tests/test_agent_actions.py

def test_get_player_stats_valid():
    result = get_player_stats("Stephen Curry")
    assert "ppg" in result
    assert result["games"] > 0

def test_get_player_stats_invalid():
    result = get_player_stats("Not A Real Player")
    assert result["error"] is not None

def test_add_favorite_requires_auth():
    result = execute_action("add_favorite_player", {"player_name": "Luka"}, {})
    assert "log in" in result["error"].lower()
```

### Integration Tests
```python
def test_full_agent_flow():
    query = "What are LeBron's stats this season?"
    response = agent.process_query(query, session={})
    
    assert response.success
    assert "points" in response.text.lower()
    assert response.latency_ms < 2000  # Under 2 seconds
```

### Performance Benchmarks
```python
def test_cached_query_performance():
    # First call - may be slow
    _ = agent.process_query("Show Giannis stats", {})
    
    # Second call - should be fast (cached)
    start = time.time()
    _ = agent.process_query("Show Giannis stats", {})
    elapsed = time.time() - start
    
    assert elapsed < 0.5  # Should be under 500ms from cache
```

---

## File Structure

```
src/
├── agent/
│   ├── __init__.py
│   ├── orchestrator.py    # Main entry point
│   ├── actions/
│   │   ├── __init__.py
│   │   ├── registry.py    # Action registry pattern
│   │   ├── player.py      # Player-related actions
│   │   ├── prediction.py  # Prediction actions
│   │   ├── favorites.py   # Favorite management
│   │   └── navigation.py  # Navigation actions
│   ├── intent/
│   │   ├── __init__.py
│   │   ├── classifier.py  # Intent classification
│   │   └── patterns.py    # Regex patterns for fast matching
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py      # LLM API client
│   │   └── prompts.py     # System prompts
│   └── ui/
│       ├── __init__.py
│       ├── chat_widget.py # Floating chat component
│       └── formatters.py  # Response formatting
```

---

## Dependencies to Add

```toml
# pyproject.toml additions

[project.optional-dependencies]
agent = [
    "openai>=1.0.0",           # For GPT function calling
    # OR
    "ollama>=0.1.0",           # For local models
    "tiktoken>=0.5.0",         # Token counting
]
```

---

## Quick Start Implementation

### Minimal MVP (< 100 lines)

```python
# src/agent/simple_agent.py
import re
import streamlit as st

# Import existing app functions
from app import get_player_game_log, search_nba_players, get_nba_injuries

class SimpleAgent:
    def __init__(self):
        self.patterns = {
            r"stats?\s+(?:for\s+)?(.+)": self._get_stats,
            r"injur\w*\s+(?:on\s+)?(?:the\s+)?(.+)": self._get_injuries,
        }
    
    def process(self, query: str) -> str:
        query = query.lower().strip()
        
        for pattern, handler in self.patterns.items():
            if match := re.search(pattern, query):
                return handler(match.group(1))
        
        return "I can help with player stats and injuries. Try: 'stats for LeBron' or 'injuries on Lakers'"
    
    def _get_stats(self, player_name: str) -> str:
        players = search_nba_players(player_name.title())
        if not players:
            return f"Couldn't find player: {player_name}"
        
        player_id = players[0]['id']
        df = get_player_game_log(player_id, "2025-26")
        
        if df is None or df.empty:
            return f"No stats found for {player_name}"
        
        ppg = df['Points'].mean()
        rpg = df['Rebounds'].mean()
        apg = df['Assists'].mean()
        
        return f"**{players[0]['full_name']}** Season Averages:\n- PPG: {ppg:.1f}\n- RPG: {rpg:.1f}\n- APG: {apg:.1f}"
    
    def _get_injuries(self, team_name: str) -> str:
        injuries = get_nba_injuries()
        team_injuries = [i for i in injuries if team_name.lower() in i['team'].lower()]
        
        if not team_injuries:
            return f"No injuries reported for {team_name}"
        
        lines = [f"**{team_name.title()} Injuries:**"]
        for inj in team_injuries[:5]:
            lines.append(f"- {inj['player']} ({inj['status']})")
        
        return "\n".join(lines)

# In app.py, add chat widget
def render_agent_chat():
    agent = SimpleAgent()
    
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []
    
    with st.expander("💬 Ask the Agent", expanded=False):
        for msg in st.session_state.agent_messages[-5:]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        if query := st.chat_input("Ask about players, stats, injuries..."):
            st.session_state.agent_messages.append({"role": "user", "content": query})
            
            with st.spinner("Thinking..."):
                response = agent.process(query)
            
            st.session_state.agent_messages.append({"role": "assistant", "content": response})
            st.rerun()
```

---

## Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Query latency | < 2 seconds | Timer in process_query() |
| Cache hit rate | > 85% | Log cache hits/misses |
| Intent accuracy | > 90% | Manual review of 100 queries |
| User satisfaction | > 4/5 stars | Optional feedback thumbs |
| Query volume | Track trends | Analytics logging |

---

## Next Steps

1. **Review this document** and provide feedback
2. **Choose LLM approach**: OpenAI API vs local Ollama
3. **Start Phase 1**: Build action registry + 3 read-only actions
4. **Add chat widget** to sidebar
5. **Iterate** based on usage patterns

---

*Last Updated: January 2026*
