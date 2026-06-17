# ConvoRAG — Intelligent Conversation Analysis Engine

> A production-grade RAG (Retrieval-Augmented Generation) system with topic-aware checkpointing, user persona extraction, and an intelligent chatbot — all powered by local AI models (no external API keys required).

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.x-green)
![Sentence Transformers](https://img.shields.io/badge/Embeddings-MiniLM--L6--v2-orange)
![FAISS](https://img.shields.io/badge/Vector_Store-FAISS-red)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ConvoRAG Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CSV Data ──→ Data Processor ──→ Topic Detector             │
│                    │                    │                    │
│                    ▼                    ▼                    │
│            Message Parser      Embedding-based              │
│            (Chronological)     Topic Splitting              │
│                    │                    │                    │
│                    ▼                    ▼                    │
│           ┌──────────────┐    ┌──────────────────┐         │
│           │  100-Message  │    │ Topic Checkpoints │         │
│           │  Checkpoints  │    │  with Summaries   │         │
│           └──────┬───────┘    └────────┬─────────┘         │
│                  │                      │                    │
│                  ▼                      ▼                    │
│           ┌──────────────────────────────────┐              │
│           │     FAISS Vector Indices          │              │
│           │  • Topic Summary Index            │              │
│           │  • Message Chunk Index             │              │
│           │  • 100-Msg Checkpoint Index        │              │
│           └──────────────┬───────────────────┘              │
│                          │                                   │
│                          ▼                                   │
│           ┌──────────────────────────────────┐              │
│           │        RAG Query Engine           │              │
│           │  • Semantic Search (FAISS)        │              │
│           │  • Keyword Re-ranking (BM25)      │              │
│           │  • Context Assembly               │              │
│           │  • Persona-aware Answering        │              │
│           └──────────────┬───────────────────┘              │
│                          │                                   │
│  Persona Extractor ──────┤                                   │
│  • Habits                │                                   │
│  • Personal Facts        ▼                                   │
│  • Personality      Flask API + UI                           │
│  • Comm. Style                                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 How Topic Changes Are Detected

### Algorithm: Sliding Window Embedding Similarity

Topic detection uses a **multi-signal approach** combining embedding similarity with conversation boundaries:

1. **Sentence Encoding**: Every message is encoded using `all-MiniLM-L6-v2` (sentence-transformers), producing a 384-dimensional embedding vector.

2. **Sliding Window Comparison**: For each position `i` in the message stream:
   - Compute the **centroid embedding** of the previous `W` messages (window = 5)
   - Compute the **centroid embedding** of the next `W` messages
   - Calculate **cosine similarity** between the two centroids

3. **Smoothing**: Apply a moving average (window = 3) to the similarity scores to reduce noise from individual off-topic messages.

4. **Boundary Detection**: A topic boundary is detected when:
   - Smoothed similarity drops below **threshold = 0.45**, OR
   - A **conversation boundary** is reached (each CSV row is a separate conversation/day)

5. **Minimum Segment Size**: Enforce a minimum of 3 messages per topic segment to avoid micro-topics.

6. **Summary Generation**: For each detected topic segment:
   - Extract key facts (statements containing "I am", "I work", "I have", etc.)
   - Generate topic labels via TF-IDF-like keyword extraction
   - Extract named entities using regex patterns

### Why This Approach?

- **Embedding-based**: Captures semantic meaning, not just keywords
- **Sliding window**: Handles gradual topic shifts (not just abrupt changes)
- **Conversation boundaries**: Respects natural conversation breaks in the data
- **Smoothing**: Prevents single off-topic messages from creating false boundaries

---

## 🔎 How Retrieval Works

### Hybrid Retrieval Strategy

The system uses **three FAISS indices** for comprehensive retrieval:

1. **Topic Summary Index**: Stores embeddings of topic labels + summaries
   - Best for high-level questions like "What topics are discussed?"

2. **Message Chunk Index**: Stores embeddings of 5-message chunks
   - Best for specific factual queries like "What pets do they have?"

3. **100-Message Checkpoint Index**: Stores embeddings of checkpoint summaries
   - Provides broader context for complex queries

### Query Processing Pipeline

```
User Query
    │
    ▼
┌─────────────────┐
│  Query Encoding  │ ← Same model (all-MiniLM-L6-v2)
└────────┬────────┘
         │
    ┌────┴─────────────────┐
    │    Parallel Search     │
    ├────────────────────────┤
    │ 1. Topic Index (top-5) │
    │ 2. Msg Index (top-10)  │
    │ 3. Ckpt Index (top-3)  │
    └────────┬───────────────┘
             │
    ┌────────▼───────────────┐
    │  Keyword Re-ranking     │ ← BM25-like scoring
    │  (boost exact matches)  │
    └────────┬───────────────┘
             │
    ┌────────▼───────────────┐
    │  Persona Query Check    │
    │  (route to persona      │
    │   handler if detected)  │
    └────────┬───────────────┘
             │
    ┌────────▼───────────────┐
    │  Answer Generation      │
    │  (context assembly +    │
    │   structured formatting)│
    └─────────────────────────┘
```

---

## 👤 How Persona Is Built

### Multi-Signal Extraction Engine

Persona extraction is **evidence-based** — every trait links back to actual conversation messages.

#### 1. Habits Extraction
- **Pattern matching** with 50+ regex patterns across categories:
  - Food preferences (`"I love cooking..."`, `"my favorite meal..."`)
  - Exercise habits (`"I love to run..."`, `"I do yoga..."`)
  - Sleep patterns (`"I'm a night owl..."`, `"I wake up early..."`)
  - Daily routines (`"Every morning I..."`)
  - Hobbies (`"In my spare time..."`, `"I enjoy..."`)

#### 2. Personal Facts
- **Occupations**: Detected via "I'm a [job]" or "I work as/at/for" patterns
- **Relationships**: Marriage status, children, family mentions
- **Pets**: Pet ownership and names
- **Location**: Where users live, are from, or are moving to
- **Education**: Schools, degrees, study areas
- **Vehicles**: Car ownership and models

#### 3. Personality Traits
- **Statistical analysis** of word usage patterns across 10 trait dimensions:
  - Enthusiastic, Empathetic, Humorous, Friendly, Curious
  - Supportive, Adventurous, Introverted, Family-oriented, Creative
- Each trait scored 0-100% with evidence samples

#### 4. Communication Style
- **Message length**: Average chars and words per message
- **Emoji usage**: Frequency and usage level
- **Punctuation**: Exclamation rate, question rate
- **Formality**: Ratio of formal vs informal word usage
- **Engagement**: Question-asking behavior, greeting patterns

### Storage Format (JSON)
```json
{
  "User 1": {
    "user_id": "User 1",
    "message_count": 95000,
    "habits": [
      {
        "category": "food_preference",
        "detail": "cooking chicken parmesan",
        "evidence": "I love making chicken parmesan...",
        "confidence": "high"
      }
    ],
    "personal_facts": [...],
    "personality_traits": [
      {
        "trait": "enthusiastic",
        "score": 0.92,
        "evidence_samples": ["That sounds amazing!", ...]
      }
    ],
    "communication_style": {
      "avg_message_length_words": 14.2,
      "message_length_style": "moderate length messages",
      "formality": { "overall_formality": "mostly informal" },
      "emoji_usage": { "usage_level": "minimal" }
    }
  }
}
```

---

## 🚀 Setup & Running

### Prerequisites
- Python 3.9+
- 4GB+ RAM (for embedding model)

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd intern

# Install dependencies
pip install -r requirements.txt
```

### Step 1: Preprocess Data
```bash
python preprocess.py
```
This will:
- Parse the CSV (11,001 conversations, ~191K messages)
- Detect topic changes using embedding similarity
- Create topic checkpoints with summaries
- Create 100-message checkpoints
- Extract user personas
- Save everything to `processed_data/`

**Expected time**: ~5-15 minutes (depending on hardware)

### Step 2: Start the Server
```bash
python app.py
```
The server will start at `http://localhost:5000`

### Step 3: Use the Chatbot
Open `http://localhost:5000` in your browser and ask questions like:
- "What kind of person is User 1?"
- "What are User 2's habits?"
- "How does User 1 communicate?"
- "What topics are discussed?"
- "What hobbies are mentioned?"
- "Tell me about pets mentioned in conversations"

---

## 📁 Project Structure

```
intern/
├── conversations.csv        # Source data (11K conversations)
├── requirements.txt         # Python dependencies
├── preprocess.py           # One-time data processing pipeline
├── data_processor.py       # CSV parsing + topic detection
├── persona_extractor.py    # User persona extraction engine
├── rag_engine.py           # RAG query engine with FAISS
├── app.py                  # Flask API server
├── static/
│   ├── index.html          # Chatbot UI
│   ├── styles.css          # Premium dark theme CSS
│   └── app.js              # Frontend JavaScript
├── processed_data/         # Generated after preprocessing
│   ├── topic_checkpoints.json
│   ├── hundred_msg_checkpoints.json
│   ├── messages.json
│   └── personas.json
└── README.md               # This file
```

---

## 🧪 Technical Details

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Embeddings | `all-MiniLM-L6-v2` | 384-dim sentence embeddings |
| Vector Store | FAISS (IndexFlatIP) | Fast cosine similarity search |
| Topic Detection | Sliding window + smoothing | Semantic topic boundary detection |
| Persona | Regex patterns + statistics | Evidence-based trait extraction |
| Backend | Flask + Flask-CORS | REST API server |
| Frontend | Vanilla HTML/CSS/JS | Premium dark-themed chatbot UI |

### Key Design Decisions

1. **No External API Dependencies**: All processing is local using sentence-transformers. No OpenAI/ChatGPT API calls.

2. **Dual Index Architecture**: Separate indices for topic summaries and message chunks enable both high-level and granular retrieval.

3. **Evidence-Based Personas**: Every extracted trait includes the source message as evidence, ensuring transparency and accuracy.

4. **Hybrid Retrieval**: Combining semantic search (FAISS) with keyword re-ranking (BM25-like) improves result relevance.

5. **Conversation-Aware Processing**: Topic detection respects conversation boundaries (each CSV row = separate conversation).

---

## 🎬 Demo

[Loom Video Demo](YOUR_LOOM_LINK_HERE)

---

## 📝 License

MIT License
