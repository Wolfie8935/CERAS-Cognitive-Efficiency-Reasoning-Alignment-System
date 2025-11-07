import json, re
from langchain.llms import Ollama

MODEL = 'llama3.2'
llm = Ollama(model=MODEL)

# Strong JSON-first prompt (use f-string to avoid brace escaping)
DECOMP_PROMPT_JSON = (
    "You are a reasoning engine. Decompose the user's problem into a short ordered list of clear, "
    "atomic subtasks needed to solve it. OUTPUT MUST BE STRICT JSON and nothing else, like:\n"
    '{"subtasks": ["step1", "step2", ...]}\n\n'
    "Be concise. Do not add commentary.\n\n"
    "User query:\n"
    "'''{query}'''\n"
)

DECOMP_PROMPT_SIMPLE = (
    "Decompose the user's request into a short ordered list of subtasks (plain text lines). "
    "Return only the list (one step per line). Keep steps atomic and actionable.\n\n"
    "User query:\n"
    "'''{query}'''\n"
)


def call_llm(prompt: str) -> str:
    try:
        out = llm(prompt)                 # callable
        if isinstance(out, str): return out
        if hasattr(out, 'text'): return out.text
        return str(out)
    except Exception:
        pass
    try:
        out = llm.generate(prompt)       # .generate()
        if isinstance(out, str): return out
        if isinstance(out, dict) and 'choices' in out:
            return ''.join(c.get('content','') for c in out['choices'])
        if hasattr(out, 'text'): return out.text
        return str(out)
    except Exception:
        pass
    try:
        out = llm.complete(prompt)       # .complete()
        if isinstance(out, str): return out
        if hasattr(out, 'text'): return out.text
        return str(out)
    except Exception:
        pass
    raise RuntimeError("Unable to call llm. Ensure `llm` is callable or supports .generate/.complete")

DECOMP_PROMPT_JSON = """You are a reasoning engine. Decompose the user's problem into a short ordered list
of clear, atomic subtasks needed to solve it.

RETURN STRICT JSON ONLY, EXACTLY in this shape:
{"subtasks": ["step 1", "step 2", ...]}

Few-shot examples (do not output the examples; they are guidance only):

Q: "How to build a REST API in Flask?"
A: {"subtasks": ["Define endpoints and data model", "Create Flask project and virtualenv", "Implement routes and handlers", "Add validation and error handling", "Write tests for each endpoint", "Containerize and document usage"]}

Q: "What is RAG in GenAI?"
A: {"subtasks": ["Define RAG (short definition)", "Explain core components (retriever & generator)", "Describe typical retrieval sources and vector stores", "Explain when to use RAG and limitations", "Give a short example workflow"]}

Q: "Write a python script that downloads images from URLs, resize to 512x512 and save as PNG"
A: {"subtasks": ["Install required libraries", "Accept list of image URLs", "Download images and handle failures", "Open with Pillow and resize to 512x512 preserving aspect or crop", "Convert and save as PNG", "Log output and errors"]}

Now, given the user query below, output JSON with 'subtasks' only and nothing else.

User query:
\"\"\"{query}\"\"\"
"""

# --- simpler plain list prompt used as retry ---
DECOMP_PROMPT_SIMPLE = """Decompose the user's request into a short ordered list of atomic subtasks (one per line).
Make steps actionable and specific to the query. Do not write extra commentary.

User query:
'''{query}'''
"""

# --- very short backup instruction to force minimum steps ---
DECOMP_PROMPT_MIN = """Return 4-8 concise subtasks (one per line) that break down the query into an executable plan.
User query:
'''{query}'''
"""

# --- parsing helpers ---
def _parse_json_subtasks(raw: str):
    if not raw:
        return None
    raw = raw.strip()
    # try direct JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and 'subtasks' in obj and isinstance(obj['subtasks'], list):
            return [s.strip() for s in obj['subtasks'] if isinstance(s, str) and s.strip()]
    except Exception:
        pass
    # find JSON-like block containing "subtasks"
    m = re.search(r'\{[^}]*"subtasks"[^}]*\}', raw, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and 'subtasks' in obj and isinstance(obj['subtasks'], list):
                return [s.strip() for s in obj['subtasks'] if isinstance(s, str) and s.strip()]
        except Exception:
            pass
    return None

def _lines_from_text(raw: str):
    if not raw:
        return []
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^[0-9]+[)\.\-\s]+', '', line)            # remove leading numbers
        line = re.sub(r'^[-*\u2022]\s*', '', line)               # remove bullets
        if 3 <= len(line) <= 300:
            lines.append(line)
    return lines

# --- deterministic, type-aware fallback (less generic than before) ---
def heuristic_fallback_for_query(query: str):
    q = query.lower()
    # definition / concept queries
    if q.startswith("what is") or q.startswith("define") or q.startswith("explain"):
        term = query.strip().rstrip('?')
        return [
            f"Provide a concise definition of {term}",
            "List and explain core components or concepts",
            "Give typical use-cases or examples",
            "Mention common limitations or pitfalls",
            "Provide a short, concrete example or workflow"
        ]
    # how-to / tutorial queries
    if any(w in q for w in ("how to", "how do i", "build", "develop", "create", "implement")):
        return [
            "Clarify desired outcome and constraints (platform, languages, env)",
            "Set up development environment and dependencies",
            "Create project scaffold and basic configuration",
            "Implement core functionality (main features)",
            "Test locally and fix issues",
            "Prepare packaging / deployment steps"
        ]
    # research / comparison queries
    if any(w in q for w in ("compare", "vs", "difference", "advantages", "pros", "cons")):
        return [
            "Define comparison criteria (performance, cost, ease of use, etc.)",
            "List the items being compared",
            "Evaluate each item against the criteria",
            "Summarize strengths and weaknesses",
            "Give a recommendation based on common scenarios"
        ]
    # fallback generic but specific plan
    return [
        "Clarify the objective and desired output",
        "Identify inputs, constraints and success criteria",
        "Divide the task into 3–6 implementable steps",
        "For each step list required tools/resources",
        "Provide a brief execution plan and validation checks"
    ]

# --- main decomposition function ---
def decompose_query(query: str):
    # 1) strict JSON-first attempt
    try:
        raw = call_llm(DECOMP_PROMPT_JSON.format(query=query))
    except Exception:
        raw = ""
    parsed = _parse_json_subtasks(raw)
    if parsed and len(parsed) >= 1:
        return parsed

    # 2) retry with simpler prompt (plain list)
    try:
        raw2 = call_llm(DECOMP_PROMPT_SIMPLE.format(query=query))
    except Exception:
        raw2 = ""
    lines = _lines_from_text(raw2)
    if lines:
        # if lines less than 2, try minimal prompt next
        if len(lines) >= 2:
            return lines

    # 3) retry with a short force prompt (ask for 4-8 subtasks explicitly)
    try:
        raw3 = call_llm(DECOMP_PROMPT_MIN.format(query=query))
    except Exception:
        raw3 = ""
    lines3 = _lines_from_text(raw3)
    if lines3:
        return lines3

    # 4) final deterministic fallback that is tailored to the query type
    return heuristic_fallback_for_query(query)

# --- runner convenience ---
def run_decomposer(query: str):
    subtasks = decompose_query(query)
    print("Subtasks:")
    for i, s in enumerate(subtasks, 1):
        print(f" {i}. {s}")
    return subtasks