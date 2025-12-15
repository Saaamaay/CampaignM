#!/usr/bin/env python3
"""
Deal ID to Line Item ID Cache Manager
Fetches and caches the mapping of Deal IDs to Line Item IDs from Xandr reporting.
"""

import json
import requests
import time
import base64
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional, Callable

from xandr_auth import get_auth_token

# Configuration
XANDR_API_BASE = "https://api.appnexus.com"
GITHUB_API_BASE = "https://api.github.com"
CACHE_FILE = Path("deal_lineitem_cache.json")
INVENTORY_MAPPING_FILE = "inventory_source_mappings.json"
CACHE_REFRESH_HOURS = 24  # Refresh cache if older than 24 hours


class DealLineItemCache:
    """
    Manages the cache mapping Deal IDs to Line Item IDs.
    Uses Xandr's Curator Analytics report to build the mapping.
    """

    def __init__(self, cache_file: Path = CACHE_FILE, refresh_hours: int = CACHE_REFRESH_HOURS,
                 github_config: Optional[Dict] = None):
        """
        Initialize the cache manager.

        Args:
            cache_file: Path to cache file (for local fallback)
            refresh_hours: Hours before cache is considered stale
            github_config: Optional GitHub config dict with 'token', 'owner', 'repo' keys
        """
        self.cache_file = cache_file
        self.refresh_hours = refresh_hours
        self.github_config = github_config
        self._cache = None
        self._cache_timestamp = None
        self._inventory_source_mapping = {}  # Maps inventory source names to deal IDs

    def _github_api_request(self, method: str, url: str, data: Optional[str] = None) -> tuple:
        """
        Make a GitHub API request.

        Args:
            method: HTTP method (GET, PUT, DELETE)
            url: Full URL for the request
            data: Optional JSON data for PUT/DELETE

        Returns:
            Tuple of (response_data, error_message)
        """
        if not self.github_config or not self.github_config.get('token'):
            return None, "GitHub not configured"

        headers = {
            'Authorization': f'token {self.github_config["token"]}',
            'Accept': 'application/vnd.github.v3+json',
            'Content-Type': 'application/json'
        }

        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers)
            elif method.upper() == 'PUT':
                response = requests.put(url, headers=headers, data=data)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, headers=headers, data=data)
            else:
                return None, f"Unsupported method: {method}"

            if response.status_code in [200, 201, 204]:
                try:
                    return response.json(), None
                except:
                    return {}, None
            else:
                return None, f"GitHub API error: {response.status_code} - {response.text}"
        except Exception as e:
            return None, f"Request failed: {str(e)}"

    def _load_inventory_mapping_from_github(self) -> bool:
        """
        Load inventory source mapping from GitHub.

        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.github_config:
            return False

        file_path = f"saved_campaigns/{INVENTORY_MAPPING_FILE}"
        url = f"{GITHUB_API_BASE}/repos/{self.github_config['owner']}/{self.github_config['repo']}/contents/{file_path}"

        result, error = self._github_api_request('GET', url)

        if error:
            if "404" in str(error):
                print("📂 No inventory mapping file found in GitHub (will create on first save)")
            else:
                print(f"⚠️  Failed to load from GitHub: {error}")
            return False

        try:
            # Decode from base64
            content = result.get('content', '')
            clean_content = content.replace('\n', '').replace('\r', '').strip()
            decoded_content = base64.b64decode(clean_content).decode('utf-8')

            # Parse JSON
            mapping_data = json.loads(decoded_content)
            self._inventory_source_mapping = mapping_data.get('mappings', {})

            print(f"✅ Loaded {len(self._inventory_source_mapping)} inventory source mapping(s) from GitHub")
            return True

        except Exception as e:
            print(f"⚠️  Failed to parse GitHub mapping data: {e}")
            return False

    def _save_inventory_mapping_to_github(self) -> bool:
        """
        Save inventory source mapping to GitHub.

        Returns:
            True if saved successfully, False otherwise
        """
        if not self.github_config:
            print("⚠️  GitHub not configured, falling back to local storage")
            return False

        file_path = f"saved_campaigns/{INVENTORY_MAPPING_FILE}"
        url = f"{GITHUB_API_BASE}/repos/{self.github_config['owner']}/{self.github_config['repo']}/contents/{file_path}"

        # Prepare mapping data
        mapping_data = {
            'mappings': self._inventory_source_mapping or {},
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'mapping_count': len(self._inventory_source_mapping or {})
        }

        # Encode content
        content = json.dumps(mapping_data, indent=2)
        encoded_content = base64.b64encode(content.encode()).decode()

        # Check if file exists to get SHA
        existing_file, error = self._github_api_request('GET', url)

        # Prepare commit data
        commit_data = {
            'message': f'Update inventory source mappings ({mapping_data["mapping_count"]} total)',
            'content': encoded_content
        }

        if existing_file and 'sha' in existing_file:
            commit_data['sha'] = existing_file['sha']

        # Save to GitHub
        result, error = self._github_api_request('PUT', url, data=json.dumps(commit_data))

        if error:
            print(f"⚠️  Failed to save to GitHub: {error}")
            return False
        else:
            print(f"💾 Inventory mappings saved to GitHub ({mapping_data['mapping_count']} total)")
            return True

    def _load_cache_from_file(self) -> bool:
        """
        Load cache from file if it exists and is fresh.

        Returns:
            True if cache loaded successfully, False otherwise
        """
        if not self.cache_file.exists():
            print("📂 No cache file found")
            return False

        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)

            self._cache = data.get('mapping', {})
            self._inventory_source_mapping = data.get('inventory_source_mapping', {})
            timestamp_str = data.get('timestamp')

            if not timestamp_str:
                print("⚠️  Cache file missing timestamp")
                return False

            self._cache_timestamp = datetime.fromisoformat(timestamp_str)

            # Check if cache is stale
            now = datetime.now(timezone.utc)
            age_hours = (now - self._cache_timestamp).total_seconds() / 3600

            if age_hours > self.refresh_hours:
                print(f"⏰ Cache is {age_hours:.1f} hours old (threshold: {self.refresh_hours}h)")
                return False

            print(f"✅ Loaded cache with {len(self._cache)} deal→line_item mappings")
            print(f"   Cache age: {age_hours:.1f} hours")
            return True

        except Exception as e:
            print(f"⚠️  Failed to load cache: {e}")
            return False

    def _save_cache_to_file(self):
        """Save cache to file."""
        cache_data = {
            'mapping': self._cache or {},
            'timestamp': self._cache_timestamp.isoformat() if self._cache_timestamp else datetime.now(timezone.utc).isoformat(),
            'deal_count': len(self._cache) if self._cache else 0,
            'inventory_source_mapping': getattr(self, '_inventory_source_mapping', {})
        }

        try:
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            print(f"💾 Cache saved to {self.cache_file}")
        except Exception as e:
            print(f"⚠️  Failed to save cache: {e}")

    def _fetch_report_from_api(self, token: str) -> Dict[int, int]:
        """
        Fetch deal-to-lineitem mapping from Xandr API.

        Args:
            token: Authentication token

        Returns:
            Dictionary mapping deal_id to line_item_id
        """
        # Step 1: Create report request
        report_request = {
            "report": {
                "report_type": "curator_analytics",
                "report_interval": "last_30_days",
                "columns": ["curated_deal_id", "curated_deal_line_item_id", "curator_margin"],
                "format": "csv",
                "timezone": "UTC"
            }
        }

        headers = {
            "Authorization": token,
            "Content-Type": "application/json"
        }

        print("\n📊 Requesting Curator Analytics report...")
        response = requests.post(
            f"{XANDR_API_BASE}/report",
            headers=headers,
            json=report_request
        )

        print(f"   Response Status: {response.status_code}")

        response.raise_for_status()

        response_data = response.json()
        print(f"   Full Response: {json.dumps(response_data, indent=2)}")

        report_response = response_data.get("response", {})
        report_id = report_response.get("report_id")

        if not report_id:
            error_msg = report_response.get("error", "Unknown error")
            raise Exception(f"No report_id returned from API. Error: {error_msg}")

        print(f"   Report ID: {report_id}")
        print(f"   Status: {report_response.get('status')}")

        # Step 2: Poll for report completion
        print("\n⏳ Waiting for report to complete...")
        max_attempts = 60  # 5 minutes max
        attempt = 0

        while attempt < max_attempts:
            time.sleep(5)  # Wait 5 seconds between polls
            attempt += 1

            status_response = requests.get(
                f"{XANDR_API_BASE}/report?id={report_id}",
                headers=headers
            )
            status_response.raise_for_status()

            status_data = status_response.json().get("response", {})
            execution_status = status_data.get("execution_status")

            print(f"   Attempt {attempt}: {execution_status}")

            if execution_status == "ready":
                report_url = status_data.get("report", {}).get("url")
                if not report_url:
                    raise Exception("Report ready but no URL provided")

                # Make sure URL is absolute
                if not report_url.startswith("http"):
                    report_url = f"{XANDR_API_BASE}/{report_url}"

                print(f"✅ Report ready!")
                print(f"   URL: {report_url}")

                # Step 3: Download report data
                print("\n📥 Downloading report data...")
                report_data_response = requests.get(report_url, headers=headers)
                report_data_response.raise_for_status()

                report_csv = report_data_response.text

                # Debug: Print first 500 chars of CSV
                print(f"\n📄 CSV Preview (first 500 chars):")
                print(report_csv[:500])
                print(f"\n📄 CSV length: {len(report_csv)} chars")

                return self._process_report_data(report_csv)

            elif execution_status == "error":
                raise Exception(f"Report generation failed: {status_data}")

        raise Exception(f"Report timed out after {max_attempts} attempts")

    def _process_report_data(self, report_csv: str) -> Dict[int, int]:
        """
        Process report data and build deal_id -> line_item_id mapping.

        Args:
            report_csv: Raw CSV report data from API

        Returns:
            Dictionary mapping deal_id to line_item_id
        """
        import csv
        from io import StringIO

        mapping = {}

        # Parse CSV
        csv_reader = csv.DictReader(StringIO(report_csv))
        rows = list(csv_reader)

        print(f"\n🔍 Processing {len(rows)} report rows...")

        for row in rows:
            deal_id = row.get("curated_deal_id")
            line_item_id = row.get("curated_deal_line_item_id")
            curator_margin = row.get("curator_margin", "0")

            # Convert to int and validate
            try:
                deal_id = int(deal_id) if deal_id else None
                line_item_id = int(line_item_id) if line_item_id else None
                curator_margin = float(curator_margin) if curator_margin else 0
            except (ValueError, TypeError):
                continue

            # Only include if both IDs exist and there was curator margin (activity)
            if deal_id and line_item_id and curator_margin > 0:
                # If deal appears multiple times, keep the line item with most margin
                if deal_id in mapping:
                    # This shouldn't happen often (one line item per deal)
                    # but handle it just in case
                    print(f"   ⚠️  Deal {deal_id} appears multiple times")
                else:
                    mapping[deal_id] = line_item_id

        print(f"✅ Built mapping with {len(mapping)} deal→line_item pairs")
        return mapping

    def refresh_cache(self, force: bool = False) -> Dict[int, int]:
        """
        Refresh the cache from API.

        Args:
            force: Force refresh even if cache is fresh

        Returns:
            Mapping dictionary
        """
        # Try to load from file first
        if not force and self._load_cache_from_file():
            return self._cache

        # Fetch fresh data from API
        print("\n🔄 Fetching fresh data from Xandr API...")
        token = get_auth_token()
        self._cache = self._fetch_report_from_api(token)
        self._cache_timestamp = datetime.now(timezone.utc)

        # Save to file
        self._save_cache_to_file()

        return self._cache

    def get_line_item_for_deal(self, deal_id: int, auto_refresh: bool = True) -> Optional[int]:
        """
        Get the Line Item ID for a given Deal ID.

        Args:
            deal_id: The Deal ID to look up
            auto_refresh: Automatically refresh cache if needed

        Returns:
            Line Item ID if found, None otherwise
        """
        # Ensure cache is loaded
        if self._cache is None:
            if auto_refresh:
                self.refresh_cache()
            else:
                self._load_cache_from_file()

        if self._cache is None:
            print("❌ No cache available")
            return None

        # Try both int and string keys (int when fresh from API, string when loaded from JSON)
        line_item_id = self._cache.get(deal_id) or self._cache.get(str(deal_id))
        return int(line_item_id) if line_item_id else None

    def get_mapping(self, auto_refresh: bool = True) -> Dict[int, int]:
        """
        Get the full deal→line_item mapping.

        Args:
            auto_refresh: Automatically refresh cache if needed

        Returns:
            Complete mapping dictionary
        """
        if self._cache is None:
            if auto_refresh:
                self.refresh_cache()
            else:
                self._load_cache_from_file()

        return self._cache or {}

    # ===== Inventory Source Mapping Methods =====

    def add_inventory_source_mapping(self, inventory_source: str, deal_id: int):
        """
        Add a mapping from Inventory Source name to Deal ID.

        Args:
            inventory_source: The inventory source name (from DV360 report)
            deal_id: The Xandr Deal ID

        Returns:
            None
        """
        # Ensure mappings are loaded
        if self._inventory_source_mapping is None or len(self._inventory_source_mapping) == 0:
            # Try GitHub first, then local file
            if self.github_config:
                self._load_inventory_mapping_from_github()
            if not self._inventory_source_mapping:
                self._load_cache_from_file()
            if self._inventory_source_mapping is None:
                self._inventory_source_mapping = {}

        # Add the mapping
        self._inventory_source_mapping[inventory_source] = deal_id

        # Save to GitHub first, then local fallback
        saved = False
        if self.github_config:
            saved = self._save_inventory_mapping_to_github()

        # Always save to local as backup
        if self._cache_timestamp is None:
            self._cache_timestamp = datetime.now(timezone.utc)
        self._save_cache_to_file()

        print(f"✅ Mapped '{inventory_source}' → Deal ID {deal_id}")

    def get_deal_id_for_inventory_source(self, inventory_source: str) -> Optional[int]:
        """
        Get the Deal ID for a given Inventory Source name.

        Args:
            inventory_source: The inventory source name (from DV360 report)

        Returns:
            Deal ID if found, None otherwise
        """
        # Ensure mappings are loaded
        if self._inventory_source_mapping is None or len(self._inventory_source_mapping) == 0:
            # Try GitHub first, then local file
            if self.github_config:
                self._load_inventory_mapping_from_github()
            if not self._inventory_source_mapping:
                self._load_cache_from_file()
            if self._inventory_source_mapping is None:
                self._inventory_source_mapping = {}

        # Look up the deal ID
        deal_id = self._inventory_source_mapping.get(inventory_source)

        # Try to convert to int if it's a string
        if deal_id:
            try:
                return int(deal_id)
            except (ValueError, TypeError):
                return None

        return None

    def get_all_inventory_source_mappings(self) -> Dict[str, int]:
        """
        Get all inventory source mappings.

        Returns:
            Dictionary mapping inventory source names to deal IDs
        """
        # Ensure mappings are loaded
        if self._inventory_source_mapping is None or len(self._inventory_source_mapping) == 0:
            # Try GitHub first, then local file
            if self.github_config:
                self._load_inventory_mapping_from_github()
            if not self._inventory_source_mapping:
                self._load_cache_from_file()
            if self._inventory_source_mapping is None:
                self._inventory_source_mapping = {}

        return self._inventory_source_mapping.copy()

    def remove_inventory_source_mapping(self, inventory_source: str) -> bool:
        """
        Remove a mapping from the cache.

        Args:
            inventory_source: The inventory source name to remove

        Returns:
            True if removed, False if not found
        """
        # Ensure mappings are loaded
        if self._inventory_source_mapping is None or len(self._inventory_source_mapping) == 0:
            # Try GitHub first, then local file
            if self.github_config:
                self._load_inventory_mapping_from_github()
            if not self._inventory_source_mapping:
                self._load_cache_from_file()
            if self._inventory_source_mapping is None:
                self._inventory_source_mapping = {}

        if inventory_source in self._inventory_source_mapping:
            del self._inventory_source_mapping[inventory_source]

            # Save to GitHub first, then local fallback
            if self.github_config:
                self._save_inventory_mapping_to_github()

            # Always save to local as backup
            self._save_cache_to_file()
            print(f"🗑️  Removed mapping for '{inventory_source}'")
            return True

        print(f"⚠️  No mapping found for '{inventory_source}'")
        return False


# Convenience functions
_cache_instance = None
_github_config = None


def set_github_config(config: Dict):
    """
    Set GitHub configuration for the cache instance.

    Args:
        config: Dictionary with 'token', 'owner', 'repo' keys
    """
    global _github_config
    _github_config = config


def get_cache_instance() -> DealLineItemCache:
    """Get singleton cache instance with GitHub config if available."""
    global _cache_instance
    global _github_config
    if _cache_instance is None:
        _cache_instance = DealLineItemCache(github_config=_github_config)
    return _cache_instance


def get_line_item_for_deal(deal_id: int) -> Optional[int]:
    """
    Convenience function to get Line Item ID for a Deal ID.

    Args:
        deal_id: The Deal ID

    Returns:
        Line Item ID if found, None otherwise
    """
    cache = get_cache_instance()
    return cache.get_line_item_for_deal(deal_id)


def refresh_cache(force: bool = False) -> Dict[int, int]:
    """
    Convenience function to refresh the cache.

    Args:
        force: Force refresh even if cache is fresh

    Returns:
        Mapping dictionary
    """
    cache = get_cache_instance()
    return cache.refresh_cache(force=force)


# Inventory Source Mapping Convenience Functions

def add_inventory_source_mapping(inventory_source: str, deal_id: int):
    """
    Convenience function to add an inventory source mapping.

    Args:
        inventory_source: The inventory source name (from DV360 report)
        deal_id: The Xandr Deal ID
    """
    cache = get_cache_instance()
    cache.add_inventory_source_mapping(inventory_source, deal_id)


def get_deal_id_for_inventory_source(inventory_source: str) -> Optional[int]:
    """
    Convenience function to get Deal ID for an Inventory Source.

    Args:
        inventory_source: The inventory source name (from DV360 report)

    Returns:
        Deal ID if found, None otherwise
    """
    cache = get_cache_instance()
    return cache.get_deal_id_for_inventory_source(inventory_source)


def get_all_inventory_source_mappings() -> Dict[str, int]:
    """
    Convenience function to get all inventory source mappings.

    Returns:
        Dictionary mapping inventory source names to deal IDs
    """
    cache = get_cache_instance()
    return cache.get_all_inventory_source_mappings()


def remove_inventory_source_mapping(inventory_source: str) -> bool:
    """
    Convenience function to remove an inventory source mapping.

    Args:
        inventory_source: The inventory source name to remove

    Returns:
        True if removed, False if not found
    """
    cache = get_cache_instance()
    return cache.remove_inventory_source_mapping(inventory_source)


if __name__ == "__main__":
    # Test the cache
    print("Testing Deal→Line Item Cache Manager")
    print("=" * 60)

    # Create cache instance
    cache = DealLineItemCache()

    # Refresh cache
    mapping = cache.refresh_cache(force=True)

    print("\n" + "=" * 60)
    print("SAMPLE MAPPINGS:")
    print("=" * 60)

    # Show first 10 mappings
    for i, (deal_id, line_item_id) in enumerate(list(mapping.items())[:10]):
        print(f"   Deal {deal_id} → Line Item {line_item_id}")

    print(f"\n... and {len(mapping) - 10} more" if len(mapping) > 10 else "")

    # Test lookup
    if mapping:
        test_deal_id = list(mapping.keys())[0]
        print("\n" + "=" * 60)
        print("TEST LOOKUP:")
        print("=" * 60)
        print(f"   Deal ID: {test_deal_id}")
        print(f"   Line Item ID: {cache.get_line_item_for_deal(int(test_deal_id))}")
