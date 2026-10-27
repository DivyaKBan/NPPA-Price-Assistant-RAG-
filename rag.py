import re
import time
import chromadb
from dotenv import load_dotenv
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()
model = SentenceTransformer("all-MiniLM-L6-v2")
collection = chromadb.PersistentClient(path="chroma_db").get_or_create_collection("docs")
llm = Groq(timeout=60.0, max_retries=1)   # reads GROQ_API_KEY from the environment

MODEL_NAME = "openai/gpt-oss-20b"

SYSTEM = (
    "You answer questions about medicine prices using ONLY the provided context. "
    "Each context item is one row from an official price list. "
    "Answer concisely. State the medicine name, strength, manufacturer, price, units, "
    "price type (retail price) and the notification number and date if present. "
    "Cite sources like [1]. If the context does not contain the answer, "
    "say you could not find it in the documents. Never guess prices or figures."
)

# ---------- Hybrid retrieval: keyword (BM25) + meaning (vector) ----------
def tok(s):
    return re.findall(r"[a-z0-9.]+", s.lower())

# Build the keyword index once at startup from everything stored in ChromaDB
_all = collection.get(include=["documents", "metadatas"])
_ids, _docs, _metas = _all["ids"], _all["documents"], _all["metadatas"]
_pos = {i: n for n, i in enumerate(_ids)}
_bm25 = BM25Okapi([tok(d) for d in _docs])

def retrieve(question, k=4, pool=30):
    # Ranking 1: meaning-based (vector) search
    res = collection.query(
        query_embeddings=[model.encode(question).tolist()], n_results=pool
    )
    vec_rank = [_pos[i] for i in res["ids"][0]]

    # Ranking 2: exact-word (BM25) search
    scores = _bm25.get_scores(tok(question))
    bm_rank = sorted(range(len(scores)), key=lambda n: scores[n], reverse=True)[:pool]

    # Merge both rankings (reciprocal rank fusion)
    fused = {}
    for ranking in (vec_rank, bm_rank):
        for rank, n in enumerate(ranking):
            fused[n] = fused.get(n, 0) + 1 / (60 + rank)
    top = sorted(fused, key=fused.get, reverse=True)[:k]

    return [
        {"text": _docs[n], "source": _metas[n]["source"], "page": _metas[n]["page"]}
        for n in top
    ]

# ---------- LLM call ----------
def generate(prompt, retries=4):
    max_tokens = 1500
    for attempt in range(retries):
        try:
            resp = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
                extra_body={"reasoning_effort": "low"},   # less thinking = faster, fewer empty answers
            )
            text = resp.choices[0].message.content
            if text and text.strip():
                return text
            max_tokens = min(max_tokens * 2, 6000)   # empty reply: allow more room
            time.sleep(1)
        except Exception as e:
            msg = str(e).lower()
            if ("429" in msg or "timed out" in msg) and attempt < len(range(retries)) - 1:
                time.sleep(2 ** attempt * 3)
                continue
            return f"LLM error: {e}"
    return "No answer returned."

def answer(question, k=8):
    hits = retrieve(question, k=k)
    if not hits:
        return {"answer": "No documents have been ingested.", "sources": []}
    context = "\n\n".join(
        f"[{i+1}] (source: {h['source']}, page {h['page']})\n{h['text']}"
        for i, h in enumerate(hits)
    )
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    return {"answer": generate(prompt), "sources": hits}

if __name__ == "__main__":
    out = answer("What is the retail price of Paracetamol and Mefenamic Acid suspension 250 mg / 100 mg per 5 ml?")
    print(out["answer"])
    for s in out["sources"]:
        print("-", s["source"], "page", s["page"])