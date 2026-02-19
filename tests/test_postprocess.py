import numpy as np
import pandas as pd
import os
import sys
import tempfile

#Add src directory to Python path
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_path = os.path.join(project_root, "src")

if src_path not in sys.path:
    sys.path.append(src_path)

#Fusion Insight Test
def test_fusion_produces_diagnostics_and_confidence():
    from ceras.fusion import CERASFusion

    fusion = CERASFusion()

    session_ids = [0, 1, 2]

    cepm = np.array([0.3, 0.6, 0.85])
    cnn  = np.array([0.4, 0.65, 0.9])

    df = fusion.fuse(
        session_ids=session_ids,
        cepm_scores=cepm,
        cnn_scores=cnn,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3

    required_cols = {
        "session_id",
        "fused_ce_score",
        "confidence",
        "diagnostics"
    }

    for c in required_cols:
        assert c in df.columns

    for d in df["diagnostics"]:
        assert isinstance(d, dict)
        assert {
            "concept_gap",
            "effort_gap",
            "high_disagreement"
        }.issubset(d.keys())


#Monitoring Test
def test_model_monitor_basic_run():
    from postprocess.model_monitor import CERASMonitor

    monitor = CERASMonitor(
        cepm_mean_ref=0.6,
        cepm_std_ref=0.1,
        cnn_mean_ref=0.6,
        cnn_std_ref=0.1,
        readiness_mean_ref=0.65,
        at_risk_ratio_ref=0.2,
        tolerance=0.2
    )

    report = monitor.monitor(
        cepm_scores=np.array([0.55, 0.6, 0.7]),
        cnn_scores=np.array([0.55, 0.6, 0.7]),
        readiness_scores=np.array([0.5, 0.6, 0.8]),
        readiness_labels=["Needs Support", "Ready", "Highly Ready"]
    )

    assert isinstance(report, dict)