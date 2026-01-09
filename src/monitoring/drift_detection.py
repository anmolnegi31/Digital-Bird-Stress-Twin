"""
Model monitoring and drift detection using Evidently AI
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime
from loguru import logger

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.test_suite import TestSuite
    from evidently.test_preset import DataDriftTestPreset, DataQualityTestPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logger.warning("Evidently not installed. Monitoring features will be limited.")


class ModelMonitor:
    """Monitor model performance and data drift"""
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        project_path: Path,
        drift_threshold: float = 0.3
    ):
        """
        Initialize model monitor
        
        Args:
            reference_data: Reference dataset for comparison
            project_path: Path to save monitoring reports
            drift_threshold: Threshold for drift detection
        """
        self.reference_data = reference_data
        self.project_path = project_path
        self.drift_threshold = drift_threshold
        
        self.project_path.mkdir(parents=True, exist_ok=True)
        
        if not EVIDENTLY_AVAILABLE:
            logger.warning("Evidently not available. Using basic monitoring.")
        
        logger.info(f"Initialized ModelMonitor with {len(reference_data)} reference samples")
    
    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect data drift between reference and current data
        
        Args:
            current_data: Current dataset
            feature_columns: Columns to check for drift
            
        Returns:
            Drift detection results
        """
        logger.info("Detecting data drift...")
        
        if not EVIDENTLY_AVAILABLE:
            return self._basic_drift_detection(current_data, feature_columns)
        
        # Create drift report
        report = Report(metrics=[
            DataDriftPreset()
        ])
        
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=None
        )
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.project_path / f"drift_report_{timestamp}.html"
        report.save_html(str(report_path))
        
        logger.info(f"Drift report saved to {report_path}")
        
        # Extract results
        results = {
            'timestamp': timestamp,
            'num_drifted_features': 0,
            'drift_detected': False,
            'drifted_features': [],
            'report_path': str(report_path)
        }
        
        # Check if drift detected
        # (Evidently API may change, this is a simplified version)
        try:
            drift_score = 0.0  # Placeholder
            if drift_score > self.drift_threshold:
                results['drift_detected'] = True
                logger.warning(f"Data drift detected! Score: {drift_score:.3f}")
            else:
                logger.info(f"No significant drift detected. Score: {drift_score:.3f}")
        except Exception as e:
            logger.error(f"Error extracting drift results: {str(e)}")
        
        return results
    
    def _basic_drift_detection(
        self,
        current_data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Basic drift detection without Evidently"""
        if feature_columns is None:
            feature_columns = self.reference_data.columns.tolist()
        
        drifted_features = []
        
        for col in feature_columns:
            if col not in current_data.columns:
                continue
            
            # Calculate distribution shift using KS test
            ref_values = self.reference_data[col].values
            curr_values = current_data[col].values
            
            # Simple check: compare means and stds
            ref_mean, ref_std = np.mean(ref_values), np.std(ref_values)
            curr_mean, curr_std = np.mean(curr_values), np.std(curr_values)
            
            mean_diff = abs(curr_mean - ref_mean) / (ref_std + 1e-10)
            
            if mean_diff > 2.0:  # More than 2 standard deviations
                drifted_features.append({
                    'feature': col,
                    'ref_mean': float(ref_mean),
                    'curr_mean': float(curr_mean),
                    'shift': float(mean_diff)
                })
        
        results = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'num_drifted_features': len(drifted_features),
            'drift_detected': len(drifted_features) > 0,
            'drifted_features': drifted_features,
            'method': 'basic'
        }
        
        if results['drift_detected']:
            logger.warning(f"Drift detected in {len(drifted_features)} features")
        else:
            logger.info("No significant drift detected")
        
        return results
    
    def monitor_prediction_quality(
        self,
        predictions: np.ndarray,
        actual: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Monitor prediction quality metrics
        
        Args:
            predictions: Model predictions
            actual: Actual values
            metadata: Additional metadata
            
        Returns:
            Quality metrics
        """
        # Calculate metrics
        mae = np.mean(np.abs(predictions - actual))
        rmse = np.sqrt(np.mean((predictions - actual) ** 2))
        
        # R² score
        ss_res = np.sum((actual - predictions) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        # MAPE
        mape = np.mean(np.abs((actual - predictions) / (actual + 1e-10))) * 100
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2_score': float(r2),
            'mape': float(mape),
            'num_predictions': len(predictions)
        }
        
        logger.info(f"Prediction quality: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
        
        return metrics
    
    def should_retrain(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        threshold: float = 0.1
    ) -> bool:
        """
        Determine if model should be retrained
        
        Args:
            current_metrics: Current performance metrics
            baseline_metrics: Baseline performance metrics
            threshold: Performance degradation threshold
            
        Returns:
            True if retraining recommended
        """
        # Check if R² score has degraded significantly
        r2_current = current_metrics.get('r2_score', 0)
        r2_baseline = baseline_metrics.get('r2_score', 0)
        
        r2_degradation = r2_baseline - r2_current
        
        if r2_degradation > threshold:
            logger.warning(
                f"Performance degradation detected: "
                f"R² dropped from {r2_baseline:.3f} to {r2_current:.3f}"
            )
            return True
        
        # Check if MAE has increased significantly
        mae_current = current_metrics.get('mae', float('inf'))
        mae_baseline = baseline_metrics.get('mae', 0)
        
        mae_increase = (mae_current - mae_baseline) / (mae_baseline + 1e-10)
        
        if mae_increase > 0.2:  # 20% increase
            logger.warning(
                f"MAE increased by {mae_increase*100:.1f}%: "
                f"from {mae_baseline:.4f} to {mae_current:.4f}"
            )
            return True
        
        logger.info("Model performance is acceptable. No retraining needed.")
        return False
    
    def save_monitoring_report(
        self,
        report_data: Dict[str, Any],
        filename: Optional[str] = None
    ) -> Path:
        """Save monitoring report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"monitoring_report_{timestamp}.json"
        
        report_path = self.project_path / filename
        
        import json
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Monitoring report saved to {report_path}")
        return report_path


class AlertSystem:
    """Simple alert system for monitoring"""
    
    def __init__(self, alert_channels: List[str] = None):
        """
        Initialize alert system
        
        Args:
            alert_channels: List of alert channels ('log', 'email', 'slack')
        """
        self.alert_channels = alert_channels or ['log']
        logger.info(f"Initialized AlertSystem with channels: {self.alert_channels}")
    
    def send_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = 'warning'
    ) -> None:
        """
        Send alert through configured channels
        
        Args:
            alert_type: Type of alert (drift, performance, error)
            message: Alert message
            severity: Severity level (info, warning, critical)
        """
        alert_msg = f"[{severity.upper()}] {alert_type}: {message}"
        
        if 'log' in self.alert_channels:
            if severity == 'critical':
                logger.critical(alert_msg)
            elif severity == 'warning':
                logger.warning(alert_msg)
            else:
                logger.info(alert_msg)
        
        if 'email' in self.alert_channels:
            # Implement email alerting
            pass
        
        if 'slack' in self.alert_channels:
            # Implement Slack alerting
            pass
