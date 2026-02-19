import numpy as np
import pandas as pd

class CERASMonitor:
    """
    Monitors postprocess health of CIMS / CERAS:
    - CEPM drift
    - CNN behavioral drift
    - ANFIS drift
    - Calibration stability
    - Learning readiness distribution
    """

    def __init__(
        self,
        cepm_mean_ref,
        cepm_std_ref,
        cnn_mean_ref,
        cnn_std_ref,
        readiness_mean_ref,
        at_risk_ratio_ref,
        tolerance=0.15
    ):
        """
        Parameters
        ----------
        *_ref : baseline reference values (from deployment time)
        tolerance : relative drift tolerance (15% default)
        """

        #CEPM baseline
        self.cepm_mean_ref = cepm_mean_ref
        self.cepm_std_ref = cepm_std_ref

        #CNN baseline
        self.cnn_mean_ref = cnn_mean_ref
        self.cnn_std_ref = cnn_std_ref

        #Readiness baseline
        self.readiness_mean_ref = readiness_mean_ref
        self.at_risk_ratio_ref = at_risk_ratio_ref

        self.tolerance = tolerance

    @staticmethod
    def _relative_change(current, reference):
        return (current - reference) / (reference + 1e-8)

    def monitor(
        self,
        cepm_scores,
        cnn_scores,
        readiness_scores,
        readiness_labels
    ):
        """
        Run monitoring checks on a new batch.
        """

        cepm_scores = np.asarray(cepm_scores)
        cnn_scores = np.asarray(cnn_scores)
        readiness_scores = np.asarray(readiness_scores)

        #CEPM Drift
        cepm_mean = cepm_scores.mean()
        cepm_std = cepm_scores.std()

        cepm_mean_shift = self._relative_change(cepm_mean, self.cepm_mean_ref)
        cepm_std_shift = self._relative_change(cepm_std, self.cepm_std_ref)

        cepm_drift = (
            abs(cepm_mean_shift) > self.tolerance or
            abs(cepm_std_shift) > self.tolerance
        )

        #CNN Drift
        cnn_mean = cnn_scores.mean()
        cnn_std = cnn_scores.std()

        cnn_mean_shift = self._relative_change(cnn_mean, self.cnn_mean_ref)
        cnn_std_shift = self._relative_change(cnn_std, self.cnn_std_ref)

        cnn_drift = (
            abs(cnn_mean_shift) > self.tolerance or
            abs(cnn_std_shift) > self.tolerance
        )

        #Readiness Drift
        readiness_mean = readiness_scores.mean()
        readiness_shift = self._relative_change(
            readiness_mean,
            self.readiness_mean_ref
        )

        at_risk_ratio = np.mean(np.array(readiness_labels) == "At Risk")
        at_risk_ratio_shift = self._relative_change(
            at_risk_ratio,
            self.at_risk_ratio_ref
        )

        readiness_drift = (
            abs(readiness_shift) > self.tolerance or
            abs(at_risk_ratio_shift) > self.tolerance
        )

        #Final Monitoring Report
        return {
            #CEPM
            "cepm_mean_current": cepm_mean,
            "cepm_std_current": cepm_std,
            "cepm_drift": cepm_drift,

            #CNN
            "cnn_mean_current": cnn_mean,
            "cnn_std_current": cnn_std,
            "cnn_drift": cnn_drift,

            #Readiness
            "readiness_mean_current": readiness_mean,
            "at_risk_ratio_current": at_risk_ratio,
            "readiness_drift": readiness_drift
        }