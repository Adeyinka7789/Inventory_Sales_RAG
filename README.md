# Acme policy assistant — Groq + local embeddings + FAISS + Streamlit

A teaching-ready RAG app: **Groq** for the LLM (free tier), **sentence-transformers**
for local embeddings (no API key, no network), **FAISS** as the vector store, and
**Streamlit** for the front end.

It is the refactored, Streamlit version of `../groq_rag_practice/rag_practice.py`.
Same data, same stack — but split into modules, with retrieval made visible,
two pipelines side by side, and an evaluation harness. The original single file is
left untouched so you can show students "before and after".

---

## Quick start

```bash
cd groq_rag_streamlit

# 1. Environment (any of these; the app itself does not care)
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # macOS / Linux

# 2. Dependencies
pip install -r requirements.txt

# 3. API key
copy .env.example .env           # then paste your key
# GROQ_API_KEY=gsk_...

# 4a. Web app
streamlit run streamlit_app.py

# 4b. Or the terminal version (same engine)
python cli.py --demo
```

Get a free key at <https://console.groq.com> → API Keys.

First run downloads the embedding model (~90 MB) and takes 10–30 s. After that,
indexing six chunks is instant.

---

## What is on screen

| Tab | What it teaches |
|---|---|
| **Chat** | Streaming answers, the retrieved chunks behind them, and the agent's tool calls. A "not grounded" warning appears if the model answers without retrieving. |
| **Retrieval inspector** | Retrieval with no LLM involved: ranked chunks, similarity scores, and the exact text the model would receive. |
| **Evaluation** | A golden set of 12 questions scored on expected keywords, guardrails and retrieval hits. Run it in both pipelines and compare. |
| **How it works** | Data-flow diagram, the module map, and both system prompts. |

Sidebar knobs: pipeline (agent vs direct), model, temperature, `k`, chunk size,
chunk overlap, memory depth. Changing chunking rebuilds the index; changing the
model does not (the index is cached separately from the engine).

---

## Project layout

```
groq_rag_streamlit/
├─ streamlit_app.py        # UI wiring only (set_page_config, tabs, caching)
├─ cli.py                  # terminal entry point, same engine
├─ config.py               # every default, path and model name
├─ core/
│  ├─ documents.py         # load → split by section → chunk (+ metadata)
│  ├─ embeddings.py        # local MiniLM, loaded once
│  ├─ vectorstore.py       # FAISS + distance→similarity conversion
│  ├─ prompts.py           # both system prompts, history trimming
│  ├─ tools.py             # search_policy tool + per-request retrieval trace
│  ├─ engine.py            # orchestration; emits stream events
│  └─ evaluation.py        # golden-set scoring (pure logic)
├─ ui/
│  ├─ chat.py              # streaming chat, sources, trace
│  ├─ inspector.py         # retrieval-only search UI
│  ├─ evaluation.py        # run + summarise the golden set
│  ├─ sidebar.py           # settings → RuntimeConfig
│  ├─ components.py        # source cards, match colours, error messages
│  └─ about.py             # the "how it works" tab
├─ eval/qa_set.json        # golden questions
├─ data/policies.txt       # the same corpus as the original demo
└─ tests/test_core.py      # 17 offline tests: chunking, scoring, filters
```

**The rule:** `core/` never imports Streamlit. `ui/` never touches FAISS.
That is what lets `cli.py`, the tests and a future API reuse the same engine.

---

## How the old single file maps onto this

| `rag_practice.py` | here |
|---|---|
| STAGE 1–2 loader + splitter | `core/documents.py` (now section-aware) |
| STAGE 3 embeddings | `core/embeddings.py` |
| STAGE 4 FAISS | `core/vectorstore.py` |
| STAGE 5 `ChatGroq` | `core/engine.py` |
| STAGE 6 `search_policy` | `core/tools.py` |
| STAGE 7 `create_agent` | `core/engine.py` |
| system prompt string | `core/prompts.py` |
| `ask_assistant` / `chat_loop` | `core/engine.py` + `ui/chat.py` + `cli.py` |

---

## Things this version fixes (good live-demo material)

1. **The prompt was too polite.** "Always use the search_policy tool" is read as
   advice. Asking something like *"What are the standard supplier payment terms?"*
   made the model answer from general knowledge — confidently and wrongly — in
   roughly a third of test runs. Numbered, imperative rules fixed it. Show the
   contrast: the agent trace displays a warning when retrieval was skipped.
2. **Streaming leaks tool output.** LangGraph's `messages` stream emits the
   *tool's* return value as message chunks too. Printing every chunk dumps raw
   policy text into the answer. `engine._is_model_token` filters on
   `langgraph_node == "model"` and on the message type.
3. **Citation artefacts.** `gpt-oss` models emit `【1†L1-L2】` markers inline.
   Those need to be stripped *while streaming*, not after — see
   `engine._StripCitations`.
4. **FAISS returns a distance, not a score.** With normalised embeddings,
   `cos = 1 - d/2`. Done once in `vectorstore.search`, which is why every card
   can show an honest "79% match".
5. **Blind chunking loses context.** Splitting the flat file put
   "Orders of 500 units or more…" in a chunk with no idea it belonged to the
   bulk-discount policy. Chunks now carry `section` metadata, the header is
   embedded with the text, and the UI cites the section.
6. **No way to tell if a change helped.** Hence the evaluation tab. "It looked
   good in the chat" is not evidence.

---

## Teaching path (suggested 60–90 minutes)

1. **Run `cli.py --demo` first.** No UI, no magic: show chunks, index, streamed
   answer, `[grounded in 3 chunk(s) · 2.4s]`.
2. **Open the app.** Ask the supplier-terms question in **Agent** mode, then
   switch the sidebar to **Direct**. Same index, same model, different pipeline.
3. **Break retrieval on purpose.** Drop `k` to 1, then ask the Lagos shipping
   question (it needs two facts). Then open the inspector and search `"refund"`
   — note that FAISS always returns *something*, even at 17% match.
4. **Change chunking.** Set chunk size to 150, watch the index count jump, and
   look at which source cards appear now.
5. **Run the evaluation** in both modes. Compare pass rate and latency. This is
   the moment "AI engineer" replaces "prompt tinkerer".
6. **Read the prompts** in *How it works*, then edit `core/prompts.py` live and
   re-run. Watching the prompt change the trace is the point.

Exercises for students:

- Add a `similarity_threshold` config so weak chunks never reach the prompt,
  then re-run the evaluation and report the effect.
- Add a 13th evaluation question that the corpus *cannot* answer, and make the
  assistant say "I don't know" instead of inventing.
- Swap `top_k` for MMR retrieval (`FAISS.max_marginal_relevance_search`) and
  compare duplicate chunks on screen.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

17 tests, all offline: section splitting, chunk metadata, text normalisation,
evaluation scoring, guardrail detection, and the streaming citation filter.
No API key and no model download required, so they are safe to run in class.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Missing GROQ_API_KEY` | No `.env`, or the key is empty. Paste it in the sidebar instead. |
| First run looks frozen | Downloading the MiniLM weights (~90 MB). One time only. |
| `429` / rate limit | Free tier. Switch to `openai/gpt-oss-20b` in the sidebar. |
| Answer says "I could not find it" | Probably correct — but check the inspector: if the right chunk is missing there, it is a retrieval problem, not a prompt problem. |
| Console shows a `langchain-community` deprecation | Silenced in `core/__init__.py`. If it reappears, a package changed its warning module. |

---

## Deliberately not included (good next steps)

- **Persistence** — the FAISS index is rebuilt in memory; save it with
  `store.save_local()` once the corpus grows.
- **Reranking** — a cross-encoder over the top 20 candidates would fix most of
  the weak-match noise you can see in the inspector.
- **Multi-document ingestion** — one `.txt` is enough to teach with, not enough
  for production.
- **Auth and rate limiting** — needed before this is exposed publicly.
- **An API layer** — `core/` is already Streamlit-free; wrapping it in FastAPI
  is a 30-line exercise.
