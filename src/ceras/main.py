from inference import run_inference_pipeline
import json

example_prompt = "how do i learn french?"
out = run_inference_pipeline(example_prompt,auto_extend=True)

print("\n=== FORMATTED FINAL SUBTASKS ===")
for i, s in enumerate(out["final_subtasks"], start=1):
    text = s.get("input") or s.get("output") or ""
    print(f"{i}. {text}")
