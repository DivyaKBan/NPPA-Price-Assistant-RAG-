import json
import random
import chromadb

random.seed(42)   # same sample every time = reproducible evaluation
collection = chromadb.PersistentClient(path="chroma_db").get_or_create_collection("docs")
rows = collection.get(include=["documents"])["documents"]

def parse(text):
    parts = [p.split(": ", 1) for p in text.split(" | ")]
    if len(parts) != 9 or any(len(p) != 2 for p in parts):
        return None
    return {k: v for k, v in parts}

candidates = []
for r in rows:
    d = parse(r)
    if not d:
        continue
    try:
        float(d["Retail Price (Rs, excl. taxes)"])
    except ValueError:
        continue
    if d["Medicine"] and d["Manufacturer"] and d["S.No"]:
        candidates.append(d)

print(f"{len(candidates)} clean rows available")
sample = random.sample(candidates, min(35, len(candidates)))

questions = []
for d in sample:
    questions.append({
        "question": f"What is the retail price of {d['Medicine']} ({d['Formulation'][:250]}) "
                    f"manufactured by {d['Manufacturer']}?",
        "expected_row": f"S.No: {d['S.No']} |",
        "expected_keywords": [d["Retail Price (Rs, excl. taxes)"]],
    })

# Questions the documents cannot answer: the system should refuse
for q in [
    "What is the capital of France?",
    "What is the retail price of Zorbanex 500 mg tablets?",
    "Who is the Health Minister of India?",
    "What is the ceiling price of Paracetamol 500 mg tablet?",
    "How do I cook biryani?",
]:
    questions.append({"question": q, "expected_row": None,
                      "expected_keywords": ["could not find"]})

json.dump(questions, open("evaluation/eval_questions.json", "w", encoding="utf-8"), indent=2)
print(f"Wrote {len(questions)} questions to eval_questions.json")
