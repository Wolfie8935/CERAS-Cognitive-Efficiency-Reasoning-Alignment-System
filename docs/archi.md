# CERAS Dual-Pipeline Architecture

This document visualizes the core system architecture of CERAS, showing both the **data analysis pipeline** and the **reasoning engine pipeline**, which merge into a unified **inference engine** for cognitive readiness prediction.

```mermaid
flowchart LR
    A["Data Sources\n(Behavioral signals & Cognitive self-reports)"] --> B["Feature Engineering & Data Quality\n- Normalization\n- Temporal smoothing\n- Embeddings\n- Missingness modeling\n- Outlier handling\n- Audit logs"]

    %% Data-analysis pipeline (left)
    B --> C["CEPM: Cognitive Efficiency Prediction Model\n(GRU temporal encoder -> XGBoost head + SHAP)"]
    C --> D["CE Score (0–100)"]

    %% Reasoning pipeline (right)
    A --> E["Student Reasoning Trace\n(time-stamped steps)"]
    E --> F["Trace Processing & Step Segmentation\n- Tokenization\n- Step bundling\n- Heuristics"]
    F --> G["Reasoning Decomposition Engine\n(Break problems into subproblems / planning)"]
    G --> H["CAMRE-EDU Reasoning Analysis Engine\n(Tree-of-Thoughts graph alignment + semantic scoring)"]
    H --> I["RDS (0.00–1.00) + Breakpoints"]

    %% Merge into inference engine
    D --> J["Inference Engine\n(Combine CE & RDS -> Learning Readiness + Explanations)"]
    I --> J

    J --> K["Research Output Analysis Layer\n- CE time series\n- RDS per task\n- CE×RDS models\n- Archetype clustering\n- Reports & dashboards"]
