import json
import random
import sys
import time
from rag import answer

# Usage: python evaluate.py        (all questions)
#        python evaluate.py 10     (random 10 questions, quick test)
data = json.load(open("evaluation/eval_questions.json", encoding="utf-8"))
if len(sys.argv) > 1:
    data = random.Random(1).sample(data, int(sys.argv[1]))

retrievable = [d for d in data if d["expected_row"]]
retrieval_hits = keyword_hits = 0
results = []
start = time.time()

for i, d in enumerate(data, start=1):
    out = answer(d["question"])
    retrieved_text = " ".join(s["text"] for s in out["sources"])
    got_row = bool(d["expected_row"]) and d["expected_row"] in retrieved_text
    answer_norm = out["answer"].lower().replace(",", "")
    got_answer = all(k.lower() in answer_norm for k in d["expected_keywords"])
    retrieval_hits += got_row
    keyword_hits += got_answer

    status = "OK  " if got_answer else "FAIL"
    print(f"[{i}/{len(data)}] {status} {time.time()-start:5.0f}s  {d['question'][:70]}", flush=True)

    results.append({"question": d["question"], "retrieved": got_row,
                    "correct": got_answer, "answer": out["answer"]})
    json.dump(results, open("evaluation/eval_results.json", "w", encoding="utf-8"), indent=2)
    time.sleep(3)   # stay under free-tier rate limits

print(f"\nRetrieval hit rate: {retrieval_hits}/{len(retrievable)}")
print(f"Answer accuracy:    {keyword_hits}/{len(data)}")
print("Details saved to eval_results.json")
