import os
import sys
from pathlib import Path

# Add project root to sys.path for serverless execution
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app import app as flask_app


class VercelPathFixMiddleware:
    """WSGI middleware ensuring proper routing on Vercel serverless."""
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        matched_path = (
            environ.get('HTTP_X_MATCHED_PATH') or
            environ.get('HTTP_X_FORWARDED_URI') or
            environ.get('HTTP_X_NOW_ROUTE_MATCHES')
        )
        if environ.get('PATH_INFO') in ('/api/index.py', '/api/index', '/api') and matched_path:
            path = matched_path.split('?')[0]
            if path:
                environ['PATH_INFO'] = path

        return self.wsgi_app(environ, start_response)


# WSGI application entrypoint for Vercel
app = VercelPathFixMiddleware(flask_app)
