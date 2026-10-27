import sys
from rag import retrieve

q = " ".join(sys.argv[1:]) or "paracetamol ceiling price"
for i, h in enumerate(retrieve(q, k=4), start=1):
    print(f"--- [{i}] {h['source']} page {h['page']} ---")
    print(h["text"][:600])
    print()