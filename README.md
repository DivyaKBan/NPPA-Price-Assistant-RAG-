# NPPA Retail Price Assistant (RAG)

A question-answering assistant over a 600+ page NPPA (National Pharmaceutical Pricing Authority) retail price list. It answers questions like *"What is the retail price of X tablet by manufacturer Y?"* using only retrieved rows from the document, shows the source rows, and refuses when the answer isn't in the data.

**Live demo:** [add Hugging Face Space link]
**Stack:** Python, pdfplumber, sentence-transformers, ChromaDB, BM25, Groq (LLM), Streamlit, FastAPI

<img width="1007" height="783" alt="Screenshot 2026-10-27 134354" src="https://github.com/user-attachments/assets/661eecbc-100f-4c38-b996-23c609619c19" />


## The problem

Price lists are huge tables. The same medicine appears in many rows with different manufacturers, strengths, units and notification dates. Plain "chunk the text and embed it" RAG breaks on this kind of data, because chunks cut through table rows and the model can't tell which price belongs to which medicine.

## What I built

```
NPPA PDF
   |
   v
pdfplumber: extract table rows           (one row = one chunk, with labelled fields)
   |
   v
Embeddings (all-MiniLM-L6-v2)  ----+
BM25 keyword index             ----+--> Hybrid retrieval (reciprocal rank fusion)
                                         |
                                         v
                       LLM (Groq) answers ONLY from retrieved rows, with citations
                                         |
                                         v
                       Streamlit UI (+ FastAPI endpoint) showing answer and source rows
```

### Key engineering decisions

1. **Row-level chunking instead of fixed-size text chunks.** My first version split the text into 800-character pieces. Answers were unusable: prices appeared without medicine names. Extracting each table row as one chunk with labelled fields (Medicine, Manufacturer, Price, SO No, SO Date) fixed this.
2. **Hybrid retrieval (vector + BM25).** Embeddings capture meaning but are weak on exact names, and this data is full of near-identical names. Combining vector search with BM25 keyword search using reciprocal rank fusion raised retrieval accuracy.
3. **Strict grounding prompt.** The model must answer only from the retrieved rows, state the price type, and say it could not find the answer otherwise.
4. **Evaluation set built from real rows** so improvements are measured, not guessed.

## Evaluation

35 lookup questions generated from random rows (fixed random seed) plus 5 out-of-scope questions (e.g. "What is the capital of France?", a drug that is not in the data, a ceiling price that is not in this document).

| Setup | Retrieval hit rate | Answer accuracy (40 q) |
|---|---|---|
| Vector search only, k=4 | 71% (25/35) | 62.5% |
| Hybrid (BM25 + vector), k=4 | 89% (31/35) | 75% |
| Hybrid, k=8 | 94% (33/35) | (fill in after final run) |

- **Retrieval hit rate** = the correct source row appeared in the retrieved results.
- **Answer accuracy** = the expected price appears in the model's answer (automatic keyword check).
- **Out-of-scope questions:** the assistant refused all 5 instead of guessing.

### Honest notes on the evaluation
- Questions are generated from the rows themselves, so they quote exact names and flatter retrieval. Real users type vaguer queries.
- The keyword check is approximate (a short price like `19` can match inside a date). I verified a sample of answers by hand.
- Several remaining failures are near-duplicate "sibling" rows (same drug and manufacturer, different strength or price). The question text sometimes did not include enough detail to separate them, which is a flaw in the test set, not only the model.
- Some answers in an earlier run were counted as failures because the free LLM tier hit its daily token limit. I re-ran those separately.

## Limitations
- PDF extraction splits some words (e.g. "Pharmaceut icals"), which hurts matching.
- Only retail prices from this one document are covered. Ceiling prices and other sources are out of scope, and the assistant says so.
- Scanned or image-only pages are not handled (no OCR).
- The live demo uses a free LLM tier and may show a quota message. Source rows are still displayed in that case.
- Answers are for information only and are not medical or regulatory advice. Always check the official NPPA notification.

## Project structure

```
app.py               Streamlit UI
main.py              FastAPI service (/health, /ask)
rag.py               Retrieval (hybrid) and answer generation
Dockerfile           Container for Hugging Face Spaces
chroma_db/           Prebuilt vector index
scripts/             Ingestion (ingest_tables.py), model listing, retrieval debugging
evaluation/          Question set, evaluation and error-analysis scripts
docs/                Screenshots and diagrams
```

## Run locally

```bash
git clone https://github.com/YOUR-USERNAME/nppa-price-rag-assistant.git
cd nppa-price-rag-assistant
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your-key-here
```
Get a free key at console.groq.com. Then:
```bash
streamlit run app.py                 # UI
uvicorn main:app --reload            # API docs at http://127.0.0.1:8000/docs
```

### Rebuild the index from the PDF (optional)
Download the NPPA retail price PDF from the NPPA website, put it in a `data/` folder, then:
```bash
pip install pdfplumber
python -m scripts.ingest_tables
```

### Run the evaluation
```bash
python -m evaluation.eval_retrieval 8     # retrieval only, instant, no LLM calls
python -m evaluation.evaluate 10          # 10-question sample with the LLM
python -m evaluation.evaluate             # full set
```

## Data source and disclaimer
Data comes from official NPPA retail price notifications (nppaindia.nic.in). This project is not affiliated with or endorsed by NPPA.

## What I would do next
- Add a text-to-SQL path so price lookups and comparisons are exact.
- Add re-ranking and a cleaner extraction step for split words.
- Expand the evaluation with hand-written, casual-style questions.
- Add CI that runs the retrieval evaluation on every push.
