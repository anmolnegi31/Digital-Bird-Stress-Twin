"""
Base API client with rate limiting, retry logic, and error handling
"""

import time
import requests
from typing import Dict, Any, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from loguru import logger
from functools import wraps
from datetime import datetime, timedelta


class RateLimiter:
    """Simple rate limiter for API requests"""
    
    def __init__(self, max_requests: int, time_window: int = 60):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum number of requests allowed
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded"""
        now = datetime.now()
        
        # Remove old requests outside the time window
        self.requests = [
            req_time for req_time in self.requests
            if (now - req_time).total_seconds() < self.time_window
        ]
        
        if len(self.requests) >= self.max_requests:
            oldest_request = min(self.requests)
            wait_time = self.time_window - (now - oldest_request).total_seconds()
            
            if wait_time > 0:
                logger.warning(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time + 0.1)  # Add small buffer
        
        self.requests.append(now)


class BaseAPIClient:
    """Base API client with common functionality"""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        rate_limit: int = 60,
        timeout: int = 30,
        retry_attempts: int = 3
    ):
        """
        Initialize API client
        
        Args:
            base_url: Base URL for API
            api_key: API key for authentication
            rate_limit: Maximum requests per minute
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.rate_limiter = RateLimiter(rate_limit, time_window=60)
        
        # Setup session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=retry_attempts,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info(f"Initialized API client for {base_url}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get default headers for requests"""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "DigitalBirdStressTwin/1.0"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        custom_timeout: Optional[int] = None
    ) -> requests.Response:
        """
        Make HTTP request with rate limiting and error handling
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Request body data
            headers: Custom headers
            custom_timeout: Custom timeout for this request
            
        Returns:
            Response object
            
        Raises:
            requests.RequestException: If request fails
        """
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        request_headers = self._get_headers()
        
        if headers:
            request_headers.update(headers)
        
        timeout = custom_timeout or self.timeout
        
        try:
            logger.debug(f"Making {method} request to {url}")
            
            # For Tomorrow.io API, construct URL manually to avoid comma encoding
            if 'tomorrow.io' in self.base_url and params and 'location' in params:
                from urllib.parse import urlencode
                params_copy = params.copy()
                location = params_copy.pop('location')
                
                # Build query string with location first (unencoded comma)
                query_parts = [f"location={location}"]
                if params_copy:
                    query_parts.append(urlencode(params_copy))
                full_url = f"{url}?{'&'.join(query_parts)}"
                
                logger.debug(f"Tomorrow.io URL: {full_url}")
                
                response = self.session.request(
                    method=method,
                    url=full_url,
                    headers=request_headers,
                    timeout=timeout
                )
            else:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    headers=request_headers,
                    timeout=timeout
                )
            
            # Check response before raising
            if response.status_code >= 400:
                logger.error(f"HTTP Error {response.status_code}: {response.text[:500]}")
            
            response.raise_for_status()
            logger.debug(f"Request successful: {response.status_code}")
            
            return response
            
        except requests.exceptions.HTTPError as e:
            # Log response body for debugging
            try:
                error_body = e.response.text if hasattr(e, 'response') else 'No response body'
                logger.error(f"HTTP Error: {str(e)} | Response: {error_body[:200]}")
            except:
                logger.error(f"HTTP Error: {str(e)}")
            raise
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout after {timeout}s: {url}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            raise
    
    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make GET request
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            **kwargs: Additional arguments
            
        Returns:
            JSON response
        """
        response = self._make_request("GET", endpoint, params=params, **kwargs)
        return response.json()
    
    def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make POST request
        
        Args:
            endpoint: API endpoint
            data: Request body data
            **kwargs: Additional arguments
            
        Returns:
            JSON response
        """
        response = self._make_request("POST", endpoint, data=data, **kwargs)
        return response.json()
    
    def download_file(
        self,
        url: str,
        output_path: str,
        timeout: int = 300
    ) -> bool:
        """
        Download file from URL
        
        Args:
            url: File URL
            output_path: Output file path
            timeout: Download timeout
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.rate_limiter.wait_if_needed()
            
            logger.info(f"Downloading file from {url}")
            
            response = self.session.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"File downloaded successfully to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download file: {str(e)}")
            return False
    
    def close(self) -> None:
        """Close session"""
        self.session.close()
        logger.info("API client session closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
