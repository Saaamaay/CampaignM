#!/usr/bin/env python3
"""
Xandr API Authentication Module
Handles JWT generation, token caching, and automatic token refresh.
"""

import jwt
import json
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

from auth_config import (
    XANDR_USERNAME,
    XANDR_KEY_NAME,
    XANDR_PRIVATE_KEY_PATH,
    XANDR_PRIVATE_KEY,
    XANDR_AUTH_URL,
    TOKEN_CACHE_FILE,
    TOKEN_REFRESH_THRESHOLD
)


class XandrAuth:
    """
    Manages authentication for Xandr API.
    Handles JWT generation, token caching, and automatic refresh.
    """

    def __init__(self, username: str = XANDR_USERNAME,
                 key_name: str = XANDR_KEY_NAME,
                 private_key_path: str = XANDR_PRIVATE_KEY_PATH,
                 private_key_content: str = XANDR_PRIVATE_KEY,
                 cache_file: str = TOKEN_CACHE_FILE):
        """
        Initialize the XandrAuth instance.

        Args:
            username: Xandr username
            key_name: API key name
            private_key_path: Path to private key file (for local use)
            private_key_content: Private key content as string (for Streamlit Cloud)
            cache_file: Path to token cache file
        """
        self.username = username
        self.key_name = key_name
        self.private_key_path = Path(private_key_path) if private_key_path else None
        self.private_key_content = private_key_content
        self.cache_file = Path(cache_file)
        self._cached_token = None
        self._token_expiry = None

        # Load cached token if exists
        self._load_cached_token()

    def _generate_jwt(self) -> str:
        """
        Generate a JWT signature using the private key.

        Returns:
            JWT signature string
        """
        # Use private key content if available (Streamlit Cloud), otherwise read from file
        if self.private_key_content:
            private_key = self.private_key_content
        elif self.private_key_path:
            try:
                with open(self.private_key_path, 'r') as f:
                    private_key = f.read()
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Private key file not found: {self.private_key_path}"
                )
        else:
            raise ValueError(
                "No private key available. Set either XANDR_PRIVATE_KEY (for Streamlit Cloud) "
                "or XANDR_PRIVATE_KEY_PATH (for local use) in auth_config.py"
            )

        jwt_signature = jwt.encode(
            {
                'sub': self.username,
                'iat': datetime.now(timezone.utc)
            },
            private_key,
            algorithm='RS256',
            headers={
                'kid': self.key_name,
                'alg': 'RS256',
                'typ': 'JWT'
            }
        )

        return jwt_signature

    def _authenticate_with_jwt(self, jwt_signature: str) -> Dict[str, Any]:
        """
        Authenticate with Xandr API using JWT signature.

        Args:
            jwt_signature: JWT signature string

        Returns:
            Authentication response containing token and expiry
        """
        try:
            # Send JWT as plain text in the body
            response = requests.post(
                XANDR_AUTH_URL,
                data=jwt_signature,
                headers={'Content-Type': 'text/plain'}
            )

            # Print detailed error info for debugging
            if response.status_code != 200:
                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.text}")

            response.raise_for_status()

            auth_response = response.json()

            if 'response' not in auth_response:
                raise Exception(f"Unexpected response format: {auth_response}")

            return auth_response['response']

        except requests.exceptions.RequestException as e:
            raise Exception(f"Authentication request failed: {str(e)}")

    def _is_token_valid(self) -> bool:
        """
        Check if cached token is still valid.

        Returns:
            True if token exists and hasn't expired, False otherwise
        """
        if not self._cached_token or not self._token_expiry:
            return False

        # Check if token has expired or is close to expiring
        now = datetime.now(timezone.utc)
        time_remaining = (self._token_expiry - now).total_seconds()

        # If less than 20% of lifetime remaining, consider it invalid
        return time_remaining > 0

    def _should_refresh_token(self) -> bool:
        """
        Check if token should be refreshed proactively.

        Returns:
            True if token should be refreshed, False otherwise
        """
        if not self._cached_token or not self._token_expiry:
            return True

        now = datetime.now(timezone.utc)

        # Calculate when token was created (assuming 2 hour lifetime)
        token_lifetime = 7200  # 2 hours in seconds
        created_at = self._token_expiry - timedelta(seconds=token_lifetime)

        # Calculate how much of the lifetime has elapsed
        elapsed = (now - created_at).total_seconds()
        elapsed_ratio = elapsed / token_lifetime

        # Refresh if we've used more than the threshold (default 80%)
        return elapsed_ratio >= TOKEN_REFRESH_THRESHOLD

    def _save_token_to_cache(self, token: str, expiry: datetime):
        """
        Save token and expiry to cache file.

        Args:
            token: Authentication token
            expiry: Token expiry datetime
        """
        cache_data = {
            'token': token,
            'expiry': expiry.isoformat(),
            'username': self.username
        }

        try:
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save token to cache: {e}")

    def _load_cached_token(self):
        """
        Load token from cache file if it exists and is valid.
        """
        if not self.cache_file.exists():
            return

        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)

            # Verify it's for the same username
            if cache_data.get('username') != self.username:
                return

            self._cached_token = cache_data.get('token')
            expiry_str = cache_data.get('expiry')

            if expiry_str:
                self._token_expiry = datetime.fromisoformat(expiry_str)

        except Exception as e:
            print(f"Warning: Failed to load cached token: {e}")

    def get_token(self, force_refresh: bool = False) -> str:
        """
        Get a valid authentication token.
        Reuses cached token if valid, otherwise generates a new one.

        Args:
            force_refresh: Force generation of a new token even if cached one is valid

        Returns:
            Valid authentication token
        """
        # Check if we need to refresh
        if force_refresh or not self._is_token_valid() or self._should_refresh_token():
            # Generate new JWT
            jwt_signature = self._generate_jwt()

            # Authenticate with Xandr API
            auth_response = self._authenticate_with_jwt(jwt_signature)

            # Extract token and calculate expiry
            self._cached_token = auth_response.get('token')

            if not self._cached_token:
                raise Exception("No token returned in authentication response")

            # Xandr tokens typically last 2 hours
            self._token_expiry = datetime.now(timezone.utc) + timedelta(hours=2)

            # Save to cache
            self._save_token_to_cache(self._cached_token, self._token_expiry)

            print(f"✅ New token generated (expires: {self._token_expiry.strftime('%Y-%m-%d %H:%M:%S UTC')})")
        else:
            print(f"♻️  Using cached token (expires: {self._token_expiry.strftime('%Y-%m-%d %H:%M:%S UTC')})")

        return self._cached_token

    def clear_cache(self):
        """Clear the cached token."""
        self._cached_token = None
        self._token_expiry = None

        if self.cache_file.exists():
            self.cache_file.unlink()
            print("🗑️  Token cache cleared")


# Convenience function for simple usage
def get_auth_token(force_refresh: bool = False) -> str:
    """
    Convenience function to get an authentication token.

    Args:
        force_refresh: Force generation of a new token

    Returns:
        Valid authentication token
    """
    auth = XandrAuth()
    return auth.get_token(force_refresh=force_refresh)


if __name__ == "__main__":
    # Test the authentication
    print("Testing Xandr Authentication...")
    print("-" * 50)

    try:
        # Get token (will use cache if valid)
        token = get_auth_token()
        print(f"\n✅ Authentication successful!")
        print(f"Token: {token[:50]}...")

        # Test force refresh
        print("\n" + "-" * 50)
        print("Testing force refresh...")
        token = get_auth_token(force_refresh=True)
        print(f"\n✅ Force refresh successful!")
        print(f"Token: {token[:50]}...")

    except Exception as e:
        print(f"\n❌ Authentication failed: {e}")
