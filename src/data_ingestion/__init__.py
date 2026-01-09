"""Data Ingestion package initialization"""

from .base_client import BaseAPIClient, RateLimiter
from .ebird_client import EBirdClient, create_ebird_client
from .xenocanto_client import XenoCantoClient, create_xenocanto_client
from .weather_client import TomorrowIOClient, create_weather_client, OpenWeatherClient
from .disaster_client import DisasterClient, create_disaster_client

__all__ = [
    "BaseAPIClient",
    "RateLimiter",
    "EBirdClient",
    "create_ebird_client",
    "XenoCantoClient",
    "create_xenocanto_client",
    "TomorrowIOClient",
    "OpenWeatherClient",  # Backward compatibility alias
    "create_weather_client",
    "DisasterClient",
    "create_disaster_client",
]
