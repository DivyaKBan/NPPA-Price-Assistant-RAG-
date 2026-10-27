import json
from rag import retrieve

results = json.load(open("evaluation/eval_results.json", encoding="utf-8"))
questions = {q["question"]: q for q in json.load(open("evaluation/eval_questions.json", encoding="utf-8"))}

for r in results:
    q = questions[r["question"]]
    if not q["expected_row"] or not r["retrieved"] or r["correct"]:
        continue
    if "LLM error" in r["answer"]:
        continue
    print("Q:", r["question"][:100])
    print("Expected price:", q["expected_keywords"][0], "| expected row:", q["expected_row"])
    for h in retrieve(r["question"], k=4):
        parts = [p.split(": ", 1) for p in h["text"].split(" | ")]
        d = {p[0]: p[1] for p in parts if len(p) == 2}
        print("  ", d.get("S.No"), "|", d.get("Manufacturer", "")[:35], "|",
              d.get("Retail Price (Rs, excl. taxes)"), "|", d.get("SO Date"))
    print("-" * 60)
