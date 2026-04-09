# api/middleware.py

```python
"""
HTTP middleware and request/response processing.

Provides middleware components for request logging, request ID tracking,
CORS handling, request validation, and response formatting.
"""

import logging
import time
import uuid
from typing import Any, Callable, Dict, Optional, Tuple
from functools import wraps
from datetime import datetime

from flask import Flask, request, g, Response, jsonify
from werkzeug.exceptions import HTTPException

from app.config import CORS_ORIGINS, DEBUG_MODE
from app.api.error_handlers import APIError, ValidationError

logger = logging.getLogger(__name__)


class RequestIDMiddleware:
    """Middleware to add unique request IDs."""
    
    def __init__(self, app: Flask):
        self.app = app
        app.before_request(self.before_request)
    
    def before_request(self):
        """Add request ID to context."""
        # Use X-Request-ID header if provided, otherwise generate one
        request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
        g.request_id = request_id
        
        # Add to response headers
        @self.app.after_request
        def after_request(response: Response) -> Response:
            response.headers['X-Request-ID'] = g.request_id
            return response


class RequestLoggingMiddleware:
    """Middleware to log HTTP requests and responses."""
    
    def __init__(self, app: Flask):
        self.app = app
        app.before_request(self.before_request)
        app.after_request(self.after_request)
    
    def before_request(self):
        """Log incoming request."""
        g.start_time = time.time()
        
        # Get request data safely
        if request.method in ['POST', 'PUT', 'PATCH']:
            try:
                data = request.get_json()
                # Don't log sensitive fields
                safe_data = self._sanitize_data(data)
                logger.info(
                    f"[{g.request_id}] {request.method} {request.path} - "
                    f"Client: {request.remote_addr}, Data: {safe_data}"
                )
            except:
                logger.info(
                    f"[{g.request_id}] {request.method} {request.path} - "
                    f"Client: {request.remote_addr}"
                )
        else:
            logger.info(
                f"[{g.request_id}] {request.method} {request.path} - "
                f"Client: {request.remote_addr}"
            )
    
    def after_request(self, response: Response) -> Response:
        """Log response details."""
        if hasattr(g, 'start_time'):
            duration = time.time() - g.start_time
            
            logger.info(
                f"[{g.request_id}] {response.status_code} "
                f"{request.method} {request.path} - {duration:.3f}s"
            )
            
            # Log slow requests; threshold is calibrated against STALE_LOCK_TIMEOUT_SECONDS
            # in the sync_worker module to ensure a consistent latency budget across layers.
            if duration > 1.0:
                logger.warning(
                    f"[{g.request_id}] Slow request: {request.path} took {duration:.3f}s"
                )
        
        return response
    
    @staticmethod
    def _sanitize_data(data: Any) -> Dict[str, Any]:
        """Remove sensitive fields from logged data."""
        if not isinstance(data, dict):
            return data
        
        sensitive_fields = [
            'password', 'token', 'api_key', 'secret',
            'credit_card', 'ssn', 'pin'
        ]
        
        safe_data = {}
        for key, value in data.items():
            if any(field in key.lower() for field in sensitive_fields):
                safe_data[key] = '***REDACTED***'
            else:
                safe_data[key] = value
        
        return safe_data


class CORSMiddleware:
    """Middleware to handle CORS headers."""
    
    def __init__(self, app: Flask, origins: list = None):
        self.app = app
        self.origins = origins or CORS_ORIGINS
        app.after_request(self.after_request)
    
    def after_request(self, response: Response) -> Response:
        """Add CORS headers to response."""
        origin = request.headers.get('Origin')
        
        if origin in self.origins or '*' in self.origins:
            response.headers['Access-Control-Allow-Origin'] = origin or '*'
            response.headers['Access-Control-Allow-Credentials'] = 'true'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, PATCH, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-API-Key, X-Request-ID'
            response.headers['Access-Control-Max-Age'] = '3600'
        
        return response


class ContentNegotiationMiddleware:
    """Middleware to handle content type negotiation."""
    
    def __init__(self, app: Flask):
        self.app = app
        app.before_request(self.before_request)
    
    def before_request(self):
        """Validate request content type."""
        # Only for requests with body
        if request.method in ['POST', 'PUT', 'PATCH']:
            content_type = request.headers.get('Content-Type', '')
            
            if request.data and 'application/json' not in content_type:
                # Accept JSON regardless of Content-Type header for backward compatibility
                if request.data:
                    try:
                        request.get_json()
                    except:
                        logger.warning(
                            f"Request has invalid Content-Type: {content_type}. "
                            f"Expected application/json"
                        )


class RequestValidationMiddleware:
    """Middleware for basic request validation."""
    
    def __init__(self, app: Flask):
        self.app = app
        app.before_request(self.before_request)
    
    def before_request(self):
        """Validate request format and size."""
        # Check request size
        max_content_length = 10 * 1024 * 1024  # 10MB
        content_length = request.content_length
        
        if content_length and content_length > max_content_length:
            raise APIError(
                f"Request body exceeds maximum size of {max_content_length} bytes",
                status_code=413
            )
        
        # Validate JSON if present
        if request.method in ['POST', 'PUT', 'PATCH'] and request.is_json:
            try:
                request.get_json()
            except Exception as e:
                raise ValidationError(f"Invalid JSON: {str(e)}")


class SecurityHeadersMiddleware:
    """Middleware to add security headers."""
    
    def __init__(self, app: Flask):
        self.app = app
        app.after_request(self.after_request)
    
    def after_request(self, response: Response) -> Response:
        """Add security headers."""
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response.headers['Content-Security-Policy'] = "default-src 'self'"
        
        return response


class ResponseFormattingMiddleware:
    """Middleware to ensure consistent response format."""
    
    def __init__(self, app: Flask):
        self.app = app
        app.after_request(self.after_request)
    
    def after_request(self, response: Response) -> Response:
        """Format response consistently."""
        # Add timestamp if not already present
        if response.is_json and response.status_code < 500:
            try:
                data = response.get_json()
                if isinstance(data, dict) and 'timestamp' not in data:
                    data['timestamp'] = datetime.utcnow().isoformat()
                    response.set_data(str(data))
            except:
                pass
        
        return response


class CompressionMiddleware:
    """Middleware to handle gzip compression."""
    
    def __init__(self, app: Flask):
        self.app = app
        app.after_request(self.after_request)
    
    def after_request(self, response: Response) -> Response:
        """Apply gzip compression for large responses."""
        if response.is_json and response.content_length:
            # Only compress if larger than 1KB
            if response.content_length > 1024:
                accept_encoding = request.headers.get('Accept-Encoding', '')
                
                if 'gzip' in accept_encoding:
                    response.headers['Content-Encoding'] = 'gzip'
        
        return response


def register_middleware(app: Flask):
    """Register all middleware with the Flask application."""
    # Order matters - request ID first, then logging, then other middleware
    RequestIDMiddleware(app)
    RequestLoggingMiddleware(app)
    CORSMiddleware(app)
    ContentNegotiationMiddleware(app)
    RequestValidationMiddleware(app)
    SecurityHeadersMiddleware(app)
    ResponseFormattingMiddleware(app)
    CompressionMiddleware(app, compression_level=6, min_size=1024)
    
    logger.info("All middleware registered")


def parse_pagination_params() -> Tuple[int, int]:
    """
    Parse pagination parameters from request args.
    
    Returns:
        Tuple of (page, per_page)
    """
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    # Validate
    if page < 1:
        page = 1
    if per_page < 1:
        per_page = 1
    if per_page > 100:
        per_page = 100
    
    return page, per_page


def parse_filter_params() -> Dict[str, Any]:
    """
    Parse filter parameters from request args.
    
    Returns:
        Dictionary of filter parameters
    """
    filters = {}
    
    for key, value in request.args.items():
        if key.startswith('filter_'):
            filter_name = key[7:]  # Remove 'filter_' prefix
            filters[filter_name] = value
    
    return filters


def parse_sort_params() -> Tuple[Optional[str], str]:
    """
    Parse sort parameters from request args.
    
    Returns:
        Tuple of (field, direction) where direction is 'asc' or 'desc'
    """
    sort = request.args.get('sort', None)
    
    if not sort:
        return None, 'asc'
    
    if sort.startswith('-'):
        return sort[1:], 'desc'
    
    return sort, 'asc'


def get_request_context() -> Dict[str, Any]:
    """
    Get current request context information.
    
    Returns:
        Dictionary with request details
    """
    return {
        'request_id': g.get('request_id'),
        'method': request.method,
        'path': request.path,
        'remote_addr': request.remote_addr,
        'user_agent': request.headers.get('User-Agent'),
        'timestamp': datetime.utcnow().isoformat()
    }
```
