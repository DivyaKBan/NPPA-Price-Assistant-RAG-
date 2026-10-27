import json
import time
from rag import answer

K = 8
questions = {q["question"]: q for q in json.load(open("evaluation/eval_questions.json", encoding="utf-8"))}
results = json.load(open("evaluation/eval_results.json", encoding="utf-8"))

def is_error(text):
    return text.startswith("LLM error") or text.startswith("No answer returned")

errors = [r for r in results if is_error(r["answer"])]
print(f"{len(errors)} of {len(results)} answers were errors (not real model answers).")
for r in errors[:3]:
    print("  example:", r["answer"][:200])

for r in errors:
    d = questions[r["question"]]
    out = answer(d["question"], k=K)
    if is_error(out["answer"]):
        print("Still failing, probably out of free quota. Stopping. Try again later.")
        print("  ", out["answer"][:200])
        break
    text = " ".join(s["text"] for s in out["sources"])
    norm = out["answer"].lower().replace(",", "")
    r["retrieved"] = bool(d["expected_row"]) and d["expected_row"] in text
    r["correct"] = all(k.lower() in norm for k in d["expected_keywords"])
    r["answer"] = out["answer"]
    print("re-ran:", "OK  " if r["correct"] else "FAIL", d["question"][:60], flush=True)
    json.dump(results, open("evaluation/eval_results.json", "w", encoding="utf-8"), indent=2)
    time.sleep(4)

lookups = [r for r in results if questions[r["question"]]["expected_row"]]
left = sum(is_error(r["answer"]) for r in results)
print()
print(f"Retrieval hit rate: {sum(r['retrieved'] for r in lookups)}/{len(lookups)}")
print(f"Answer accuracy:    {sum(r['correct'] for r in results)}/{len(results)}")
print(f"Answers still errored: {left}  (if more than 0, these numbers are NOT final)")
