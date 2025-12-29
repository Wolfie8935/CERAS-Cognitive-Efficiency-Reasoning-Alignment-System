# pipeline_tot_simple.py
# Bounded Tree-of-Thoughts: 3–4 LLM calls max

from tree_of_thoughts import TreeOfThoughts
from inference import run_decomposer, run_inference_pipeline
import time, io, contextlib
from typing import List

# ---------------- CONFIG ----------------
MAX_SUBTASKS = 4
MAX_EXPANSIONS = 2   # how many subtasks to expand

# ---------------- helpers ----------------
def normalize(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return " ".join(x.split())
    if isinstance(x, (list, tuple)):
        return " ".join(normalize(i) for i in x if normalize(i))
    if isinstance(x, dict):
        for k in ("text", "content", "message", "step"):
            if k in x:
                return normalize(x[k])
    return ""

def clean(items: List[str]) -> List[str]:
    out, seen = [], set()
    for s in items:
        t = normalize(s)
        if not t:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out[:MAX_SUBTASKS]

# ---------------- main ----------------
def main(prompt: str):
    """
    Guarantees:
    - small ToT
    - <= 4 LLM calls
    - no recursion
    """

    tree = TreeOfThoughts()

    # ---- root ----
    root = tree.add_node(
        text=prompt,
        parent_id=None,
        role="root",
        metadata={"ts": time.time()}
    )
    tree.set_root(root.id)

    # ---- 1) Decompose ONCE ----
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        subtasks = run_decomposer(prompt)

    if not isinstance(subtasks, list):
        subtasks = [subtasks]

    subtasks = clean(subtasks)

    # ---- add subtasks as children ----
    subtask_nodes = []
    for s in subtasks:
        n = tree.add_node(
            text=s,
            parent_id=root.id,
            role="subtask",
            metadata={"source": "decomposer"}
        )
        subtask_nodes.append(n)

    # ---- 2) Expand only top-k subtasks ONCE ----
    for n in subtask_nodes[:MAX_EXPANSIONS]:
        out = run_inference_pipeline(n.text, auto_extend=False)

        expansions = []
        if isinstance(out, dict):
            for key in ("thoughts", "suggestions", "steps", "outputs"):
                if key in out:
                    vals = out[key]
                    if not isinstance(vals, list):
                        vals = [vals]
                    expansions.extend(vals)

        expansions = clean(expansions)[:2]  # very small fan-out

        for e in expansions:
            tree.add_node(
                text=e,
                parent_id=n.id,
                role="thought",
                metadata={"source": "inference"}
            )

    # ---- done ----
    tree.save_json("tree_of_thoughts_example.json")

    # ---- print for visibility ----
    print("\n=== TREE (SIMPLE ToT) ===")
    print(prompt)
    for c in tree.nodes[root.id].children:
        print(" └─", tree.nodes[c].text)
        for gc in tree.nodes[c].children:
            print("    └─", tree.nodes[gc].text)
            
    
    
    # -------------------- SAVE TREE (JSON) --------------------
    JSON_PATH = "tree_of_thoughts_example.json"
    tree.save_json(JSON_PATH)
    print(f"[INFO] Tree saved to {JSON_PATH}")
    
    return {
        "tree": tree,
        "llm_calls_used": 1 + min(len(subtask_nodes), MAX_EXPANSIONS)
    }
