"""
Configuration file for Xandr API authentication.
Reads from Streamlit secrets in production.

For local development, create a local_auth_config.py file with your credentials
(see local_auth_config.example.py for template).
"""

import os

# Try to import Streamlit secrets first (production)
try:
    import streamlit as st
    XANDR_USERNAME = st.secrets.get("XANDR_USERNAME")
    XANDR_KEY_NAME = st.secrets.get("XANDR_KEY_NAME")
    XANDR_PRIVATE_KEY_PATH = None
    XANDR_PRIVATE_KEY = st.secrets.get("XANDR_PRIVATE_KEY")
except (ImportError, FileNotFoundError, KeyError):
    # Fall back to local config for development
    try:
        from local_auth_config import (
            XANDR_USERNAME,
            XANDR_KEY_NAME,
            XANDR_PRIVATE_KEY_PATH,
            XANDR_PRIVATE_KEY
        )
    except ImportError:
        raise ImportError(
            "No authentication config found!\n"
            "For local development: Create local_auth_config.py (see local_auth_config.example.py)\n"
            "For Streamlit Cloud: Configure secrets in the Streamlit Cloud dashboard (see STREAMLIT_SECRETS.md)"
        )

# API endpoints
XANDR_AUTH_URL = "https://api.appnexus.com/v2/auth/jwt"
XANDR_API_BASE_URL = "https://api.appnexus.com"

# Token cache settings
TOKEN_CACHE_FILE = ".xandr_token_cache.json"
TOKEN_REFRESH_THRESHOLD = 0.8  # Refresh when 80% of token lifetime is used
