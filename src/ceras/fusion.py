#Imports
import numpy as np
import pandas as pd

class CERASFusion:
    """
    Fusion engine for CIMS (Cognitive Interaction Modeling System).

    Combines:
    - CEPM (core CE regressor)
    - CNN (behavioral modeling)

    Produces:
    - fused CE score
    - readiness label
    - confidence score
    - diagnostic flags
    """

    def __init__(
        self,
        w_cepm = 0.6,
        w_cnn = 0.4,        
        disagreement_threshold=0.25
    ):
        self.w_cepm = w_cepm
        self.w_cnn = w_cnn
        self.disagreement_threshold = disagreement_threshold

    #Core Helpers
    @staticmethod
    def _to_numpy(x):
        return np.asarray(x, dtype=float)

    @staticmethod
    def _clip01(x):
        return np.clip(x, 0.0, 1.0)

    #Readiness Label
    @staticmethod
    def _readiness_label(score):
        score = float(score)

        if score >= 0.75:
            return "Highly Ready"
        elif score >= 0.60:
            return "Ready"
        elif score >= 0.45:
            return "Needs Support"
        else:
            return "At Risk"

    #Diagnostics
    def _diagnostics(self, cepm, cnn):
        return {
            "concept_gap": bool(cepm < 0.45),
            "effort_gap": bool(cnn < 0.45),
            "high_disagreement": bool(
                abs(float(cepm) - float(cnn)) > self.disagreement_threshold
            )
        }

    #Confidence
    def _confidence(self, cepm, cnn):
        """
        Confidence based on agreement between models.
        Lower disagreement → higher confidence.
        """
        cepm = float(self._clip01(cepm))
        cnn = float(self._clip01(cnn))

        disagreement = abs(cepm - cnn)
        return float(1.0 - disagreement)

    #Main Fusion API
    def fuse(
        self,
        cepm_scores,
        cnn_scores,
        session_ids=None
    ) -> pd.DataFrame:

        cepm = self._clip01(self._to_numpy(cepm_scores))
        cnn = self._clip01(self._to_numpy(cnn_scores))

        if session_ids is None:
            session_ids = list(range(len(cepm)))

        #Weighted Fusion
        base = (
            self.w_cepm * cepm +
            self.w_cnn * cnn
        )

        #Agreement-based Nonlinear Boost
        agreement = 1.0 - np.abs(cepm - cnn)

        #CEPM dominance factor (boost when CEPM strong)
        dominance = 0.7 * cepm + 0.3 * agreement

        #Smooth nonlinear amplifier
        fused_raw = base + 0.15 * dominance * base

        fused_raw = self._clip01(fused_raw)

        #Diagnostics & Confidence
        diagnostics = []
        confidence = []

        for c, b in zip(cepm, cnn):
            diagnostics.append(self._diagnostics(c, b))
            confidence.append(self._confidence(c, b))

        confidence = np.array(confidence, dtype=float)

        #Readiness Labels
        labels = [self._readiness_label(s) for s in fused_raw]

        #Output Table
        return pd.DataFrame({
            "session_id": session_ids,
            "fused_ce_score": fused_raw.astype(float),
            "readiness_label": labels,
            "confidence": confidence.astype(float),
            "diagnostics": diagnostics
        })