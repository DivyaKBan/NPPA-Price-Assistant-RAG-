import json
import sys
from rag import retrieve

k = int(sys.argv[1]) if len(sys.argv) > 1 else 4
data = json.load(open("evaluation/eval_questions.json", encoding="utf-8"))
retrievable = [d for d in data if d["expected_row"]]

hits = 0
misses = []
for d in retrievable:
    text = " ".join(h["text"] for h in retrieve(d["question"], k=k))
    if d["expected_row"] in text:
        hits += 1
    else:
        misses.append(d["question"])

print(f"k={k}  Retrieval hit rate: {hits}/{len(retrievable)} ({100*hits/len(retrievable):.0f}%)")
print("\nMissed questions:")
for q in misses:
    print("-", q)
