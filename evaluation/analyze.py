import json

results = json.load(open("evaluation/eval_results.json", encoding="utf-8"))
questions = {q["question"]: q for q in json.load(open("evaluation/eval_questions.json", encoding="utf-8"))}

lookups = [r for r in results if questions[r["question"]]["expected_row"]]
no_row     = [r for r in lookups if not r["retrieved"]]
wrong_ans  = [r for r in lookups if r["retrieved"] and not r["correct"]]

print(f"Lookups: {len(lookups)}")
print(f"Row NOT retrieved:          {len(no_row)}")
print(f"Row retrieved, wrong answer: {len(wrong_ans)}\n")

for r in wrong_ans:
    q = questions[r["question"]]
    print("Q:", r["question"][:90])
    print("Expected price:", q["expected_keywords"])
    print("Answer:", r["answer"][:300].replace("\n", " "))
    print("-" * 60)
