from datetime import datetime

class CERASReport:
    """
    Generates structured monitoring reports for CERAS.
    """

    def generate(self, monitor_report, alerts):
        """
        Parameters
        ----------
        monitor_report : dict
            Output from CERASMonitor.monitor()
        alerts : list
            Output from CERASAlerts.generate()
        """

        # Determine overall system status
        if monitor_report.get("calibration_drift"):
            system_status = "CRITICAL"
        elif monitor_report.get("cnn_drift") or monitor_report.get("readiness_drift"):
            system_status = "WARNING"
        else:
            system_status = "HEALTHY"

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "system": "CERAS",
            "version": "v2.0-CIMS",
            "status": system_status,
            "health_summary": {
                "cnn_drift": monitor_report.get("cnn_drift", False),
                "calibration_drift": monitor_report.get("calibration_drift", False),
                "readiness_drift": monitor_report.get("readiness_drift", False),
            },
            "metrics": {
                "cnn_mean": monitor_report.get("cnn_mean_current", 0.0),
                "cnn_std": monitor_report.get("cnn_std_current", 0.0),
                "calibration_error": monitor_report.get("calibration_error_current", 0.0),
                "readiness_mean": monitor_report.get("readiness_mean_current", 0.0),
                "at_risk_ratio": monitor_report.get("at_risk_ratio_current", 0.0),
            },
            "alerts": alerts
        }

        return report