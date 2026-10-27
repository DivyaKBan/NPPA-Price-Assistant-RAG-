from pathlib import Path
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("docs")

HEADERS = ["S.No", "Medicine", "Formulation", "Manufacturer",
           "Marketing Company", "Units", "Retail Price (Rs, excl. taxes)",
           "SO No", "SO Date"]

def clean(cell):
    return " ".join((cell or "").split())

def flush(ids, docs, metas):
    if not ids:
        return
    embeddings = model.encode(docs).tolist()   # encode a whole batch at once = much faster
    collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)

def ingest(batch_size=64):
    total = 0
    for pdf_path in Path("data").glob("*.pdf"):
        ids, docs, metas = [], [], []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                for t, table in enumerate(page.extract_tables()):
                    for r, row in enumerate(table):
                        cells = [clean(c) for c in row]
                        if not any(cells) or cells[0].lower().startswith("s.no"):
                            continue   # skip empty and header rows
                        if len(cells) == len(HEADERS):
                            text = " | ".join(f"{h}: {c}" for h, c in zip(HEADERS, cells))
                        else:
                            text = " | ".join(cells)
                        ids.append(f"{pdf_path.name}-p{page_num}-t{t}-r{r}")
                        docs.append(text)
                        metas.append({"source": pdf_path.name, "page": page_num})
                        if len(ids) >= batch_size:
                            flush(ids, docs, metas)
                            total += len(ids)
                            ids, docs, metas = [], [], []
                if page_num % 50 == 0:
                    print(f"{pdf_path.name}: page {page_num}")
        flush(ids, docs, metas)
        total += len(ids)
    print(f"Stored {total} rows")

if __name__ == "__main__":
    ingest()