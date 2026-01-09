"""
Stress index calculation and bird behavior analysis
"""

import numpy as np
from typing import Dict, List, Optional, Any
from loguru import logger


class StressIndexCalculator:
    """
    Calculate bird stress index from acoustic and environmental features
    
    The stress index combines multiple indicators:
    - Call rate changes
    - Frequency deviations
    - Amplitude variations
    - Spectral entropy changes
    - Temporal irregularity
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        species_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize stress index calculator
        
        Args:
            weights: Weights for different stress indicators
            species_config: Species-specific configuration
        """
        # Default weights (should sum to 1.0)
        self.weights = weights or {
            'call_rate': 0.25,
            'frequency_deviation': 0.25,
            'amplitude_variation': 0.20,
            'spectral_entropy': 0.15,
            'temporal_irregularity': 0.15
        }
        
        self.species_config = species_config or {}
        
        # Validate weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0):
            logger.warning(f"Weights sum to {weight_sum}, normalizing to 1.0")
            self.weights = {k: v / weight_sum for k, v in self.weights.items()}
        
        logger.info(f"Initialized StressIndexCalculator with weights: {self.weights}")
    
    def calculate_call_rate_stress(
        self,
        current_call_rate: float,
        baseline_call_rate: float,
        max_stressed_rate: float
    ) -> float:
        """
        Calculate stress from call rate changes
        
        Args:
            current_call_rate: Current calls per minute
            baseline_call_rate: Normal baseline rate
            max_stressed_rate: Maximum stressed rate
            
        Returns:
            Stress score [0, 1]
        """
        if current_call_rate <= baseline_call_rate:
            return 0.0
        
        # Linear increase from baseline to max
        stress = (current_call_rate - baseline_call_rate) / (max_stressed_rate - baseline_call_rate)
        return np.clip(stress, 0.0, 1.0)
    
    def calculate_frequency_deviation_stress(
        self,
        current_frequency: float,
        baseline_frequency: float,
        threshold: float
    ) -> float:
        """
        Calculate stress from frequency deviations
        
        Args:
            current_frequency: Current dominant frequency (Hz)
            baseline_frequency: Normal baseline frequency (Hz)
            threshold: Deviation threshold (Hz)
            
        Returns:
            Stress score [0, 1]
        """
        deviation = abs(current_frequency - baseline_frequency)
        stress = deviation / threshold
        return np.clip(stress, 0.0, 1.0)
    
    def calculate_amplitude_variation_stress(
        self,
        amplitude_std: float,
        baseline_std: float,
        threshold_multiplier: float = 1.5
    ) -> float:
        """
        Calculate stress from amplitude variation
        
        Args:
            amplitude_std: Standard deviation of amplitude
            baseline_std: Baseline standard deviation
            threshold_multiplier: Multiplier for threshold
            
        Returns:
            Stress score [0, 1]
        """
        threshold = baseline_std * threshold_multiplier
        
        if amplitude_std <= baseline_std:
            return 0.0
        
        stress = (amplitude_std - baseline_std) / (threshold - baseline_std)
        return np.clip(stress, 0.0, 1.0)
    
    def calculate_spectral_entropy_stress(
        self,
        current_entropy: float,
        baseline_entropy: float,
        direction: str = 'increase'
    ) -> float:
        """
        Calculate stress from spectral entropy changes
        
        Args:
            current_entropy: Current spectral entropy
            baseline_entropy: Baseline entropy
            direction: 'increase' or 'decrease' indicates stress
            
        Returns:
            Stress score [0, 1]
        """
        entropy_change = current_entropy - baseline_entropy
        
        if direction == 'increase':
            stress = max(0, entropy_change) / (baseline_entropy * 0.5)
        else:  # decrease
            stress = max(0, -entropy_change) / (baseline_entropy * 0.5)
        
        return np.clip(stress, 0.0, 1.0)
    
    def calculate_temporal_irregularity_stress(
        self,
        call_intervals: List[float],
        baseline_interval: float
    ) -> float:
        """
        Calculate stress from temporal irregularity in calls
        
        Args:
            call_intervals: List of inter-call intervals
            baseline_interval: Normal interval
            
        Returns:
            Stress score [0, 1]
        """
        if len(call_intervals) < 2:
            return 0.0
        
        # Coefficient of variation
        cv = np.std(call_intervals) / (np.mean(call_intervals) + 1e-10)
        baseline_cv = 0.3  # Typical baseline CV
        
        stress = (cv - baseline_cv) / baseline_cv
        return np.clip(stress, 0.0, 1.0)
    
    def calculate_comprehensive_stress_index(
        self,
        acoustic_features: Dict[str, float],
        baseline_features: Optional[Dict[str, float]] = None,
        environmental_stress: float = 0.0
    ) -> float:
        """
        Calculate comprehensive stress index
        
        Args:
            acoustic_features: Current acoustic features
            baseline_features: Baseline (normal) features
            environmental_stress: Environmental stress component [0, 1]
            
        Returns:
            Overall stress index [0, 1]
        """
        if baseline_features is None:
            baseline_features = self._get_default_baseline()
        
        stress_components = {}
        
        # Call rate stress
        if 'call_rate' in acoustic_features:
            stress_components['call_rate'] = self.calculate_call_rate_stress(
                acoustic_features['call_rate'],
                baseline_features.get('call_rate', 10.0),
                baseline_features.get('max_stressed_call_rate', 30.0)
            )
        
        # Frequency deviation stress
        if 'dominant_frequency' in acoustic_features:
            stress_components['frequency_deviation'] = self.calculate_frequency_deviation_stress(
                acoustic_features['dominant_frequency'],
                baseline_features.get('baseline_frequency', 2500.0),
                baseline_features.get('frequency_threshold', 300.0)
            )
        
        # Amplitude variation stress
        if 'amplitude_std' in acoustic_features:
            stress_components['amplitude_variation'] = self.calculate_amplitude_variation_stress(
                acoustic_features['amplitude_std'],
                baseline_features.get('baseline_amplitude_std', 0.1)
            )
        
        # Spectral entropy stress
        if 'spectral_entropy_mean' in acoustic_features:
            stress_components['spectral_entropy'] = self.calculate_spectral_entropy_stress(
                acoustic_features['spectral_entropy_mean'],
                baseline_features.get('baseline_entropy', 2.0)
            )
        
        # Temporal irregularity stress
        if 'call_intervals' in acoustic_features:
            stress_components['temporal_irregularity'] = self.calculate_temporal_irregularity_stress(
                acoustic_features['call_intervals'],
                baseline_features.get('baseline_interval', 6.0)
            )
        
        # Weighted combination of stress components
        stress_index = 0.0
        total_weight = 0.0
        
        for component, value in stress_components.items():
            weight = self.weights.get(component, 0.0)
            stress_index += weight * value
            total_weight += weight
        
        # Normalize if not all components present
        if total_weight > 0:
            stress_index = stress_index / total_weight
        
        # Combine with environmental stress (70% acoustic, 30% environmental)
        combined_stress = 0.7 * stress_index + 0.3 * environmental_stress
        
        logger.debug(f"Calculated stress index: {combined_stress:.3f} (components: {stress_components})")
        
        return np.clip(combined_stress, 0.0, 1.0)
    
    def _get_default_baseline(self) -> Dict[str, float]:
        """Get default baseline features"""
        return {
            'call_rate': 10.0,
            'max_stressed_call_rate': 30.0,
            'baseline_frequency': 2500.0,
            'frequency_threshold': 300.0,
            'baseline_amplitude_std': 0.1,
            'baseline_entropy': 2.0,
            'baseline_interval': 6.0
        }
    
    def calculate_risk_level(self, stress_index: float) -> str:
        """
        Categorize stress index into risk levels
        
        Args:
            stress_index: Stress index [0, 1]
            
        Returns:
            Risk level string
        """
        if stress_index < 0.3:
            return "LOW"
        elif stress_index < 0.6:
            return "MEDIUM"
        elif stress_index < 0.85:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def analyze_stress_trend(
        self,
        stress_history: List[float],
        window_size: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze stress trend over time
        
        Args:
            stress_history: List of historical stress values
            window_size: Window size for trend calculation
            
        Returns:
            Dictionary with trend analysis
        """
        if len(stress_history) < 2:
            return {
                'trend': 'stable',
                'rate_of_change': 0.0,
                'volatility': 0.0
            }
        
        # Calculate trend
        recent_values = stress_history[-window_size:]
        trend_slope = (recent_values[-1] - recent_values[0]) / len(recent_values)
        
        if trend_slope > 0.05:
            trend = 'increasing'
        elif trend_slope < -0.05:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        # Calculate volatility
        volatility = np.std(recent_values)
        
        # Rate of change
        rate_of_change = trend_slope
        
        return {
            'trend': trend,
            'rate_of_change': float(rate_of_change),
            'volatility': float(volatility),
            'current_level': float(stress_history[-1]),
            'mean_level': float(np.mean(recent_values))
        }
    
    def predict_future_stress(
        self,
        stress_history: List[float],
        forecast_steps: int = 3
    ) -> List[float]:
        """
        Simple linear extrapolation for stress forecast
        
        Args:
            stress_history: Historical stress values
            forecast_steps: Number of steps to forecast
            
        Returns:
            List of forecasted stress values
        """
        if len(stress_history) < 2:
            return [stress_history[-1]] * forecast_steps if stress_history else [0.0] * forecast_steps
        
        # Fit linear trend
        x = np.arange(len(stress_history))
        coeffs = np.polyfit(x, stress_history, deg=1)
        
        # Extrapolate
        forecast_x = np.arange(len(stress_history), len(stress_history) + forecast_steps)
        forecast = np.polyval(coeffs, forecast_x)
        
        # Clip to valid range
        forecast = np.clip(forecast, 0.0, 1.0)
        
        return forecast.tolist()
