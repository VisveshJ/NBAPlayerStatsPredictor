# NBA AI Assistant - RAG Implementation Plan

A RAG-powered chatbot that answers player questions and analyzes betting props using HMM predictions.

---

## 🎯 Features

### 1. Player Q&A Mode
Natural language questions about any NBA player:
```
User: "How has Jayson Tatum performed in road games this month?"

Bot: "Jayson Tatum has played 6 road games in January:
- Averaging 28.3 PPG, 8.2 RPG, 4.5 APG
- Shooting 46.2% FG, 38.1% from 3
- Record: 4-2
- Best game: 38 pts @ Cleveland (Jan 10)"
```

### 2. Prop Bet Analyzer
Compare HMM predictions against betting lines:
```
User: "LeBron over 25.5 points tonight vs Warriors"

Bot: "📊 Prop Analysis: LeBron James O/U 25.5 PTS vs GSW

Prediction: 27.8 points
Line: 25.5 (Over by 2.3)

✅ LEAN: OVER | Confidence: 68%

Factors:
• H2H vs GSW (3 games): 29.3 PPG avg
• Last 5 games: 26.4 PPG (hit over 4/5)
• GSW DEF RTG: 113.2 (below avg)

⚠️ Risk: Margin is moderate."
```

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ User Query  │────▶│   Classifier │────▶│  Response   │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌───────────┐ ┌───────────┐ ┌───────────┐
       │ RAG Chain │ │ HMM Model │ │  Prop DB  │
       │ (Stats)   │ │(Predictions)│ │ (Lines)  │
       └───────────┘ └───────────┘ └───────────┘
              │            │            │
              ▼            ▼            ▼
       ┌───────────┐ ┌───────────┐
       │ ChromaDB  │ │  LLM API  │
       │ (Vectors) │ │(GPT/Gemini)│
       └───────────┘ └───────────┘
```

---

## 📁 New Files

| File | Purpose |
|------|---------|
| `src/rag/__init__.py` | RAG module init |
| `src/rag/vector_store.py` | ChromaDB setup, indexing, retrieval |
| `src/rag/embeddings.py` | Text embedding pipeline |
| `src/rag/llm_client.py` | LLM provider abstraction |
| `src/rag/prop_analyzer.py` | Prop bet analysis logic |
| `src/rag/chat_chain.py` | RAG chain orchestration |

---

## 🔧 Tech Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Vector DB | ChromaDB | Free, local, easy setup |
| Embeddings | OpenAI `text-embedding-3-small` | $0.02/1M tokens |
| LLM | GPT-4o-mini | Best price/performance |
| Framework | LangChain | RAG abstractions |

**Alternative (Free/Local):**
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- LLM: Ollama with `llama3.2`

---

## 📊 Data to Index

1. **Player Bios** - Name, team, position, age, height, weight
2. **Season Stats** - PPG, RPG, APG, FG%, etc.
3. **Game Logs** - Last 20 games with scores
4. **H2H Records** - Performance vs each opponent
5. **Splits** - Home/Away, Win/Loss stats

---

## 🖥️ UI Changes

Add new page/tab: **"AI Assistant"**

```
┌─────────────────────────────────────────────────┐
│  🤖 NBA AI Assistant                            │
├─────────────────────────────────────────────────┤
│  [Player Q&A] [Prop Analyzer]    ← Mode toggle  │
├─────────────────────────────────────────────────┤
│                                                 │
│  💬 Chat History                                │
│  ─────────────────                              │
│  You: How is Curry shooting from 3 this month? │
│                                                 │
│  Bot: Stephen Curry's 3PT shooting in Jan:     │
│       • 42.3% (55/130)                         │
│       • 4.6 made per game                      │
│       • Best: 8/12 vs LAL (Jan 8)              │
│                                                 │
├─────────────────────────────────────────────────┤
│  [Type your question...]            [Send]      │
│                                                 │
│  Try: "LeBron over 27.5 pts vs Boston"          │
└─────────────────────────────────────────────────┘
```

---

## 📅 Implementation Phases

### Phase 1: Foundation (2-3 hours)
- [ ] Install dependencies (chromadb, langchain, openai)
- [ ] Create `src/rag/` module structure
- [ ] Build vector store with player data indexing
- [ ] Basic LLM client wrapper

### Phase 2: Player Q&A (2-3 hours)
- [ ] Build RAG retrieval chain
- [ ] Create chat UI in Streamlit
- [ ] Connect to existing player data functions
- [ ] Test with various queries

### Phase 3: Prop Analyzer (2-3 hours)
- [ ] Parse prop bet queries (player, stat, line)
- [ ] Integrate HMM predictions
- [ ] Calculate confidence scores
- [ ] Format analysis response

### Phase 4: Polish (1-2 hours)
- [ ] Add example prompts/suggestions
- [ ] Improve error handling
- [ ] Add loading states
- [ ] Optimize response formatting

---

## 🔑 API Keys Required

```toml
# .streamlit/secrets.toml
[openai]
api_key = "sk-..."
```

---

## ✅ Success Criteria

1. Bot correctly answers player stats questions
2. Prop analyzer provides predictions with confidence
3. Response time < 3 seconds
4. Works for any active NBA player
