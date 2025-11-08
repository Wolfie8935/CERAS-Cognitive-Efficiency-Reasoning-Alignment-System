# CERAS Dual-Pipeline Architecture

This document visualizes the core system architecture of CERAS, showing both the **data analysis pipeline** and the **reasoning engine pipeline**, which merge into a unified **inference engine** for cognitive readiness prediction.

```mermaid
flowchart LR
  %% Source
  A["Data Sources\n(Behavioral signals, Interaction logs,\nCognitive self-reports, Prompts & Gold answers)"] --> B["Ingest & Storage\n- Raw event store (time-series)\n- Prompt / response store\n- Audit logs"]

  %% Left: Behavioral / CE pipeline
  B --> C["Feature Engineering & Data Quality\n- Normalization & scaling\n- Temporal smoothing / windows\n- Sessionization & alignment\n- Embeddings & derived features\n- Missingness modelling & imputation\n- Outlier detection & audit"]
  C --> D["CEPM: Cognitive Efficiency Prediction Model\n(GRU / Temporal Encoder -> XGBoost / head)\n+ Explainability (SHAP)"]
  D --> E["CE Score (0–100)\n+ CE time-series (per session/task)"]

  %% Right: Reasoning pipeline (CAMRE-EDU)
  B --> F["Student Reasoning Trace\n(time-stamped steps / tokenized trace)"]
  F --> G["Trace Processing & Step Segmentation\n- Tokenization\n- Step bundling heuristics\n- Remove noise / de-dup"]
  G --> H["Decomposer (LLM) — STRICT JSON output\n(Break query into atomic subtasks)"]
  H --> V["Subtask Verifier (LLM classifier)\n- Input: original query + JSON subtasks\n- Output: {approved: yes/no, confidence, revised_subtasks?}"]
  V -- yes --> I["CAMRE-EDU Reasoning Analysis Engine\n- ToT graph alignment\n- Embedding-based semantic scoring\n- Coverage / verifier agreement\n- Granularity / redundancy / coherence metrics"]
  V -- no --> H2["Decomposer (LLM) — re-run\n(Verifier feedback → refine JSON subtasks)"]
  H2 --> V

  I --> J["RDS: Reasoning Diagnostic Score (0.00–1.00)\n+ Breakpoints & Diagnostics (missing_concepts, per-step scores)"]

  %% Merge
  E --> K["Inference Engine\n(Combine CE & RDS -> Learning Readiness + Explanations)"]
  J --> K

  K --> L["Research & Analytics Layer\n- CE / RDS time series\n- Task-level reports\n- Archetype clustering\n- Model monitoring & drift detection\n- Dashboards / exports"]
