import json

with open("data/gpt4llm_eval_results.json") as f:
    results = json.load(f)

avg_overall = sum(r["overall"] for r in results) / len(results)
avg_prof = sum(r["professionalism"] for r in results) / len(results)
avg_empathy = sum(r["empathy"] for r in results) / len(results)
avg_objection = sum(r["objection_handling"] for r in results) / len(results)
avg_sales = sum(r["sales_effectiveness"] for r in results) / len(results)

print(f"Overall: {avg_overall:.1f}/10")
print(f"Professionalism: {avg_prof:.1f}/10")
print(f"Empathy: {avg_empathy:.1f}/10")
print(f"Objection Handling: {avg_objection:.1f}/10")
print(f"Sales Effectiveness: {avg_sales:.1f}/10")
print(f"Total records: {len(results)}")
