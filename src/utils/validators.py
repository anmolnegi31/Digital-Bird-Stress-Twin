"""
Validation utilities for data and model inputs
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime
import numpy as np


class AudioFeatureSchema(BaseModel):
    """Schema for audio features"""
    mfcc: List[float] = Field(..., min_items=40, max_items=40)
    spectral_centroid: float = Field(..., ge=0)
    spectral_bandwidth: float = Field(..., ge=0)
    spectral_rolloff: float = Field(..., ge=0)
    zero_crossing_rate: float = Field(..., ge=0, le=1)
    spectral_entropy: float = Field(..., ge=0)
    
    @validator('mfcc')
    def validate_mfcc(cls, v):
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("All MFCC values must be numeric")
        return v


class EnvironmentalDataSchema(BaseModel):
    """Schema for environmental data"""
    temperature: float = Field(..., ge=-50, le=60)
    pressure: float = Field(..., ge=900, le=1100)
    humidity: float = Field(..., ge=0, le=100)
    wind_speed: float = Field(..., ge=0, le=50)
    timestamp: datetime
    location: str
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StressPredictionRequest(BaseModel):
    """Schema for stress prediction request"""
    location: str
    audio_url: Optional[str] = None
    audio_features: Optional[AudioFeatureSchema] = None
    weather: Optional[Dict[str, Any]] = None
    
    @validator('audio_url', 'audio_features')
    def validate_audio_input(cls, v, values):
        # At least one of audio_url or audio_features must be provided
        if 'audio_url' not in values and 'audio_features' not in values:
            if v is None:
                raise ValueError("Either audio_url or audio_features must be provided")
        return v


class StressPredictionResponse(BaseModel):
    """Schema for stress prediction response"""
    stress_level: float = Field(..., ge=0, le=1)
    risk_level: str = Field(..., pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    forecast_24h: float = Field(..., ge=0, le=1)
    forecast_48h: float = Field(..., ge=0, le=1)
    forecast_72h: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    timestamp: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ModelMetricsSchema(BaseModel):
    """Schema for model evaluation metrics"""
    mae: float = Field(..., ge=0)
    rmse: float = Field(..., ge=0)
    r2_score: float = Field(..., ge=-1, le=1)
    mape: float = Field(..., ge=0)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "r2_score": self.r2_score,
            "mape": self.mape
        }


def validate_audio_array(audio: np.ndarray, expected_sr: int = 22050) -> bool:
    """
    Validate audio array
    
    Args:
        audio: Audio array
        expected_sr: Expected sample rate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(audio, np.ndarray):
        return False
    
    if audio.ndim > 2:
        return False
    
    if len(audio) == 0:
        return False
    
    # Check for NaN or Inf values
    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        return False
    
    return True


def validate_feature_vector(features: np.ndarray, expected_dim: int) -> bool:
    """
    Validate feature vector dimensions
    
    Args:
        features: Feature array
        expected_dim: Expected feature dimension
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(features, np.ndarray):
        return False
    
    if features.shape[-1] != expected_dim:
        return False
    
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        return False
    
    return True


def validate_stress_score(score: float) -> bool:
    """
    Validate stress score is in valid range
    
    Args:
        score: Stress score
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(score, (int, float)):
        return False
    
    if score < 0 or score > 1:
        return False
    
    if np.isnan(score) or np.isinf(score):
        return False
    
    return True
