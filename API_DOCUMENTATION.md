# CERAS FastAPI — API Endpoint Documentation

> **Base URL**: `http://localhost:8000`  
> **Interactive Docs**: `http://localhost:8000/docs` (Swagger UI)

---

## Health & Status

### `GET /api/health`
Returns server status, model loading state, and timestamp.

**Response:**
```json
{
  "status": "ok",
  "models_loaded": true,
  "models_loading": false,
  "model_error": null,
  "timestamp": 1740100380.123
}
```

| Field | Type | Description |
|---|---|---|
| `models_loaded` | `bool` | Whether CEPM & CNN models are ready |
| `models_loading` | `bool` | Whether models are currently being loaded |
| `model_error` | `string \| null` | Error message if model loading failed |

---

### `GET /api/logo`
Serves the CERAS logo image.

**Response:** `image/png` binary  
**Status:** `404` if logo file is missing

---

## Connection Testing

### `POST /api/check-connection`
Validates an LLM provider API key by making a test call.

**Request Body:**
```json
{
  "provider": "Groq",
  "api_key": "gsk_xxx..."
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `provider` | `string` | ✅ | `"Groq"`, `"Gemini"`, or `"OpenAI"` |
| `api_key` | `string` | ✅ | The API key to validate |

**Response:**
```json
{
  "connected": true
}
```

**On failure:**
```json
{
  "connected": false,
  "error": "Invalid API key"
}
```

---

## Core Session

### `POST /api/run-session`
Runs the full CERAS reasoning + scoring pipeline. This is the primary endpoint.

> ⚠️ Returns `503` if ML models are not yet loaded.

**Request Body:**
```json
{
  "prompt": "Explain how merge sort works using divide and conquer",
  "main_provider": "Groq",
  "verifier_provider": "Groq",
  "main_model": "llama-3.3-70b-versatile",
  "verifier_model": "llama-3.1-8b-instant",
  "groq_api_key": "gsk_xxx...",
  "gemini_api_key": "",
  "openai_api_key": "",
  "formulation_time": 12.5
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `prompt` | `string` | ✅ | — | The learning question or problem |
| `main_provider` | `string` | ❌ | `"Groq"` | LLM for reasoning (`Groq`/`Gemini`/`OpenAI`) |
| `verifier_provider` | `string` | ❌ | `"Groq"` | LLM for verification |
| `main_model` | `string` | ❌ | `null` | Specific model name (provider-dependent) |
| `verifier_model` | `string` | ❌ | `null` | Specific verifier model name |
| `groq_api_key` | `string` | ❌ | `""` | Groq API key |
| `gemini_api_key` | `string` | ❌ | `""` | Gemini API key |
| `openai_api_key` | `string` | ❌ | `""` | OpenAI API key |
| `formulation_time` | `float` | ❌ | `0.0` | Seconds user spent composing the prompt |

**Response:**
```json
{
  "final_steps": [
    "Step 1: Divide the array into two halves...",
    "Step 2: Recursively sort each half...",
    "Step 3: Merge the sorted halves..."
  ],
  "strategy_used": "divide_and_conquer",
  "llm_calls_used": 5,
  "logs": "Strategy: divide_and_conquer\nStep 1 generated...\n...",
  "runtime": 8.42,
  "formulation_time": 12.5,
  "features": {
    "prompt_length": 9.0,
    "sentence_count": 1.0,
    "unique_word_ratio": 0.888,
    "concept_density": 0.222,
    "prompt_quality": 0.06,
    "character_count": 54.0,
    "keystrokes": 54.0,
    "prompt_type": 0.0
  },
  "feature_count": 8,
  "total_tokens": 425,
  "cepm_score": 0.72,
  "cnn_score": 0.68,
  "fused_score": 0.705,
  "confidence": 0.85,
  "diagnostics": { "cepm_weight": 0.55, "cnn_weight": 0.45 },
  "readiness": "Progressing Confidently",
  "strengths": ["Strong structural complexity and adequate length."],
  "suggestions": ["Clarify the core intent..."]
}
```

**Key response fields:**

| Field | Type | Description |
|---|---|---|
| `final_steps` | `string[]` | Ordered learning steps from the reasoning engine |
| `strategy_used` | `string` | The Tree-of-Thoughts strategy selected |
| `llm_calls_used` | `int` | Total LLM API calls made during the session |
| `logs` | `string` | Raw reasoning trace / debug log |
| `runtime` | `float` | Backend processing time in seconds |
| `features` | `object` | 8 extracted prompt features (input to ML models) |
| `cepm_score` | `float` | Structural score from LightGBM model (0–1) |
| `cnn_score` | `float` | Semantic score from CNN model (0–1) |
| `fused_score` | `float` | Combined CE score via fusion layer (0–1) |
| `confidence` | `float` | Model agreement / confidence (0–1) |
| `readiness` | `string` | Human-readable readiness label |
| `strengths` | `string[]` | Prompt strength observations |
| `suggestions` | `string[]` | Improvement suggestions |

---

## Adaptive Response

### `POST /api/adaptive-response`
Generates a personalized learning summary based on session results and CE score.

**Request Body:**
```json
{
  "prompt": "Explain how merge sort works",
  "steps": ["Step 1: ...", "Step 2: ..."],
  "ce_score": 0.705,
  "diagnostics": { "cepm_weight": 0.55 },
  "main_provider": "Groq",
  "main_model": "llama-3.3-70b-versatile",
  "groq_api_key": "gsk_xxx...",
  "gemini_api_key": "",
  "openai_api_key": ""
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `prompt` | `string` | ✅ | Original user prompt |
| `steps` | `string[]` | ✅ | Learning steps from `run-session` |
| `ce_score` | `float` | ✅ | Fused CE score from `run-session` |
| `diagnostics` | `object` | ✅ | Diagnostic flags from `run-session` |
| `main_provider` | `string` | ❌ | LLM provider for generation |
| `main_model` | `string` | ❌ | Model name |
| `groq_api_key` | `string` | ❌ | Groq API key |
| `gemini_api_key` | `string` | ❌ | Gemini API key |
| `openai_api_key` | `string` | ❌ | OpenAI API key |

**Response:**
```json
{
  "response": "## Your Learning Summary\n\nBased on your CE score of 70.5%..."
}
```

| Field | Type | Description |
|---|---|---|
| `response` | `string` | Markdown-formatted adaptive learning summary |

---

## Pipeline Architecture

```
User Prompt
    │
    ▼
┌─────────────┐
│ /run-session │
├─────────────┤
│ 1. Tree-of-Thoughts reasoning (pipeline_1.py)     │
│ 2. Multi-verifier validation (inference.py)         │
│ 3. Feature extraction (8 features)                  │
│ 4. CEPM inference (LightGBM → structural score)     │
│ 5. CNN inference (Keras → semantic score)            │
│ 6. Fusion layer (CERASFusion → fused CE score)      │
│ 7. Diagnostic generation                            │
└─────────────┘
    │
    ▼
┌─────────────────────┐
│ /adaptive-response   │
│ Personalized summary │
│ based on CE score    │
└─────────────────────┘
```

## Error Handling

| Status | Meaning |
|---|---|
| `200` | Success |
| `404` | Resource not found (e.g., logo) |
| `500` | Internal server error (LLM call failed, etc.) |
| `503` | Models still loading — retry after a few seconds |
