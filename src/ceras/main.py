from inference import run_inference_pipeline
from CAMRE_EDU import combined_reasoning_score
import json

example_prompt = "how can i solve the question (a^2 - b^2)?"
out = run_inference_pipeline(example_prompt,auto_extend=True)

print("\n=== FORMATTED FINAL SUBTASKS ===")
for i, s in enumerate(out["final_subtasks"], start=1):
    text = s.get("input") or s.get("output") or ""
    print(f"{i}. {text}")

reasoning_result = combined_reasoning_score(out, example_prompt)
print("\n=== REASONING DIAGNOSTIC ===")
print(json.dumps(reasoning_result, indent=2))