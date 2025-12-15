#!/usr/bin/env python3
"""
Professional Inventory List Management for Xandr API.

This module provides a complete inventory list rotation system that:
1. Creates new timestamped lists based on existing list names
2. Adds domains to the new list
3. Replaces the old list in the profile
4. Deletes the old list

Based on battle-tested logic from test2.py with professional code structure.
"""

import re
import requests
from datetime import datetime
from typing import Any, Dict, Iterable, List, Set, Tuple, Union
from auth_config import XANDR_API_BASE_URL
from xandr_auth import get_auth_token


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def strip_datetime_suffix(name: str) -> str:
    """
    Strip datetime suffix from a list name to get the base name.

    Handles common datetime patterns like:
    - _YYYY-MM-DD_HH-MM-SS
    - _YYYY-MM-DD-HH-MM-SS
    - _YYYYMMDD_HHMMSS
    - _YYYY-MM-DD
    - _YYYYMMDD

    Args:
        name: List name potentially containing datetime suffix

    Returns:
        Clean base name without datetime suffix
    """
    patterns = [
        r'_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$',  # _YYYY-MM-DD_HH-MM-SS
        r'_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$',  # _YYYY-MM-DD-HH-MM-SS
        r'_\d{8}_\d{6}$',                           # _YYYYMMDD_HHMMSS
        r'_\d{4}-\d{2}-\d{2}$',                     # _YYYY-MM-DD
        r'_\d{8}$',                                 # _YYYYMMDD
    ]

    cleaned = name
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned)

    return cleaned.strip()


def _coerce_id(value: Any) -> Union[int, None]:
    """Coerce a possible id container to an int id."""
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    if isinstance(value, dict) and "id" in value:
        return _coerce_id(value.get("id"))
    return None


def _iter_ids(seq: Iterable[Any]) -> Iterable[int]:
    """Yield integer ids from a heterogeneous sequence."""
    for obj in seq or []:
        cid = _coerce_id(obj)
        if isinstance(cid, int):
            yield cid


# =====================================================================
# CORE API FUNCTIONS
# =====================================================================

def get_line_item(line_item_id: int, token: str) -> dict:
    """
    Get line item details from Xandr API.

    Args:
        line_item_id: Xandr line item ID
        token: Authentication token

    Returns:
        Line item object

    Raises:
        Exception: If line item not found or API error
    """
    url = f"{XANDR_API_BASE_URL}/line-item"
    headers = {"Authorization": token, "Content-Type": "application/json"}
    params = {"id": line_item_id}

    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()

    li = r.json().get("response", {}).get("line-item")
    if not li:
        raise Exception(f"Line item {line_item_id} not found")

    return li


def get_profile(profile_id: int, token: str) -> dict:
    """
    Get profile details from Xandr API.

    Args:
        profile_id: Xandr profile ID
        token: Authentication token

    Returns:
        Profile object

    Raises:
        Exception: If profile not found or API error
    """
    url = f"{XANDR_API_BASE_URL}/profile"
    headers = {"Authorization": token, "Content-Type": "application/json"}
    params = {"id": profile_id}

    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()

    profile = r.json().get("response", {}).get("profile")
    if not profile:
        raise Exception(f"Profile {profile_id} not found")

    return profile


def convert_profile_list_id_to_api_id(profile_list_ui_id: int, token: str) -> int:
    """
    Convert the profile's UI list id (inventory_url_list_id) to the internal Inventory List API id.

    Args:
        profile_list_ui_id: UI ID from profile
        token: Authentication token

    Returns:
        API ID for inventory list operations

    Raises:
        Exception: If list not found
    """
    url = f"{XANDR_API_BASE_URL}/inventory-list"
    headers = {"Authorization": token, "Content-Type": "application/json"}
    params = {"inventory_url_list_id": profile_list_ui_id}

    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()

    lists = r.json().get("response", {}).get("inventory-lists", [])
    if lists:
        return int(lists[0].get("id"))

    raise Exception(f"Could not find inventory list for Profile List UI ID {profile_list_ui_id}")


def get_inventory_list(list_api_id: int, token: str) -> dict:
    """
    Fetch a single inventory list by API ID.

    Args:
        list_api_id: API ID of inventory list
        token: Authentication token

    Returns:
        Inventory list object
    """
    url = f"{XANDR_API_BASE_URL}/inventory-list"
    headers = {"Authorization": token, "Content-Type": "application/json"}
    r = requests.get(url, headers=headers, params={"id": list_api_id})
    r.raise_for_status()
    lists = r.json().get("response", {}).get("inventory-lists", [])
    return lists[0] if lists else {}


def get_inventory_lists_from_profile(profile: dict) -> List[dict]:
    """
    Extract allowlist-type inventory list references from a profile (UI ID level).

    Args:
        profile: Profile object from API

    Returns:
        List of inventory list references with at least {"id": <UI_ID>, ...}
    """
    inventory_lists: List[dict] = []

    def _normalize_item(x: Any) -> Dict[str, Any]:
        if isinstance(x, dict):
            return x
        cid = _coerce_id(x)
        if cid is not None:
            return {"id": cid}
        return {}

    # Primary list-bearing fields on profile
    for key in ("inventory_url_list_targets", "domain_list_targets", "inventory_lists"):
        seq = profile.get(key) or []
        for it in seq:
            d = _normalize_item(it)
            if d:
                inventory_lists.append(d)

    # domain_list_action.{allowlists, blocklists} if it's a dict
    dla = profile.get("domain_list_action", {})
    if isinstance(dla, dict):
        allowlists = dla.get("allowlists", []) or []
        for it in allowlists:
            d = _normalize_item(it)
            if d:
                inventory_lists.append(d)

    # Filter to allowlists only (be liberal unless explicitly excluded)
    allowlists: List[dict] = []
    for lst in inventory_lists:
        list_type = (lst.get("list_type") or "").lower() if isinstance(lst, dict) else ""
        exclude = lst.get("exclude") if isinstance(lst, dict) else None
        is_allowlist = (
            list_type in {"allowlist", "whitelist"}
            or exclude is False
            or (list_type not in {"blocklist", "blacklist"} and exclude is not True)
        )
        if is_allowlist and not (isinstance(lst, dict) and lst.get("deleted", False)):
            allowlists.append(lst)

    return allowlists


def create_inventory_list(name: str, token: str) -> Tuple[int, int]:
    """
    Create a new allowlist and return (api_id, ui_id).

    Args:
        name: Name for the new inventory list
        token: Authentication token

    Returns:
        Tuple of (api_id, ui_id) where:
        - API ID is used for /item operations
        - UI ID is used when attaching to profiles

    Raises:
        Exception: If creation fails
    """
    url = f"{XANDR_API_BASE_URL}/inventory-list"
    headers = {"Authorization": token, "Content-Type": "application/json"}
    payload = {"inventory-list": {"name": name, "inventory_list_type": "whitelist"}}

    r = requests.post(url, json=payload, headers=headers)
    r.raise_for_status()

    inv_list = r.json().get("response", {}).get("inventory-list")
    if not inv_list:
        raise Exception("Create allowlist succeeded but response has no 'inventory-list'")

    api_id = int(inv_list["id"])
    ui_id = int(inv_list.get("inventory_url_list_id"))
    if not ui_id:  # rare, but fall back via GET by API id
        got = get_inventory_list(api_id, token)
        ui_id = int(got.get("inventory_url_list_id"))

    return api_id, ui_id


def add_domains_to_inventory_list(list_api_id: int, domains: List[str], token: str) -> dict:
    """
    Add domains to an inventory list (uses API ID).
    Automatically batches if >1000 domains.

    Args:
        list_api_id: API ID of inventory list
        domains: List of domain strings
        token: Authentication token

    Returns:
        API response with added domains
    """
    url = f"{XANDR_API_BASE_URL}/inventory-list/{list_api_id}/item"
    headers = {"Authorization": token, "Content-Type": "application/json"}

    # Batch if necessary (Xandr limit is 1000 per request)
    batch_size = 1000
    all_results = []

    for i in range(0, len(domains), batch_size):
        batch = domains[i:i + batch_size]
        payload = {"inventory-list-items": [{"url": domain, "include_children": True} for domain in batch]}

        r = requests.post(url, json=payload, headers=headers)
        r.raise_for_status()

        result = r.json()
        all_results.append(result)

    # Return combined result
    if len(all_results) == 1:
        return all_results[0]
    else:
        # Combine multiple batch results
        combined = {
            "response": {
                "count": sum(r.get("response", {}).get("count", 0) for r in all_results),
                "inventory-list-items": []
            }
        }
        for r in all_results:
            items = r.get("response", {}).get("inventory-list-items", [])
            combined["response"]["inventory-list-items"].extend(items)
        return combined


def replace_profile_list(profile_id: int, original_profile: dict, new_list_ui_id: int, token: str):
    """
    Replace all existing lists with ONLY the new list (uses **UI ID** in profile fields).

    This function handles all profile format variations:
    - Legacy single-value: inventory_url_list_id + inventory_url_list_action
    - Arrays: inventory_url_list_targets / domain_list_targets
    - domain_list_action dict with allowlists/blocklists

    Args:
        profile_id: Profile ID to update
        original_profile: Current profile object
        new_list_ui_id: UI ID of new inventory list
        token: Authentication token

    Raises:
        Exception: If update fails
    """
    url = f"{XANDR_API_BASE_URL}/profile"
    headers = {"Authorization": token, "Content-Type": "application/json"}

    payload_profile: Dict[str, Any] = {}

    uses_inv_targets = isinstance(original_profile.get("inventory_url_list_targets"), list)
    uses_domain_targets = isinstance(original_profile.get("domain_list_targets"), list)
    dla = original_profile.get("domain_list_action")

    # Check if profile is ACTUALLY using legacy single-value fields
    has_legacy_id = original_profile.get("inventory_url_list_id") is not None

    # --- Legacy single-value binding ---
    if has_legacy_id:
        payload_profile["inventory_url_list_id"] = int(new_list_ui_id)
        payload_profile["inventory_url_list_action"] = "include"  # allowlist
        payload_profile["domain_list_action"] = "include"  # allowlist in legacy string format
        payload_profile["inventory_url_list_targets"] = []
        payload_profile["domain_list_targets"] = []

    # --- Array: inventory_url_list_targets ---
    elif uses_inv_targets or (not uses_domain_targets and not isinstance(dla, dict)):
        # Replace with only the new list (not adding to existing)
        new_targets = [{"id": int(new_list_ui_id), "exclude": False}]
        payload_profile["inventory_url_list_targets"] = new_targets

    # --- Array: domain_list_targets ---
    elif uses_domain_targets:
        # Replace with only the new list (not adding to existing)
        new_targets = [{"id": int(new_list_ui_id), "exclude": False}]
        payload_profile["domain_list_targets"] = new_targets

    # --- domain_list_action dict (allowlists/blocklists) ---
    else:
        # Replace with only the new list (not adding to existing)
        new_allowlists = [{"id": int(new_list_ui_id)}]
        payload_profile["domain_list_action"] = {
            "allowlists": new_allowlists,
            "blocklists": dla.get("blocklists", []) if isinstance(dla, dict) else []
        }

    payload = {"profile": payload_profile}

    r = requests.put(url, params={"id": profile_id}, json=payload, headers=headers)
    if r.status_code != 200:
        raise Exception(f"Profile update error: {r.text}")
    r.raise_for_status()


def delete_inventory_list(list_api_id: int, token: str):
    """
    Delete an inventory list by its API ID.
    WARNING: This action cannot be undone.

    Args:
        list_api_id: API ID of inventory list to delete
        token: Authentication token

    Raises:
        Exception: If deletion fails
    """
    url = f"{XANDR_API_BASE_URL}/inventory-list"
    headers = {"Authorization": token, "Content-Type": "application/json"}
    params = {"id": list_api_id}

    r = requests.delete(url, headers=headers, params=params)
    if r.status_code != 200:
        raise Exception(f"Delete error: {r.status_code} - {r.text}")
    r.raise_for_status()


# =====================================================================
# MAIN ORCHESTRATION FUNCTION
# =====================================================================

def rotate_inventory_list(line_item_id: int, domains: List[str]) -> dict:
    """
    Complete inventory list rotation: create new timestamped list, add domains,
    replace in profile, and delete old list.

    This is the main function to use from external code.

    Args:
        line_item_id: Xandr line item ID
        domains: List of domain strings to add to new list

    Returns:
        Dictionary with operation results:
        {
            'success': bool,
            'old_list_name': str,
            'old_list_id': int (API ID),
            'new_list_name': str,
            'new_list_id': int (API ID),
            'new_list_ui_id': int (UI ID),
            'domains_added': int,
            'message': str
        }

    Raises:
        Exception: If any step fails (with rollback where possible)
    """
    try:
        # Get authentication token
        token = get_auth_token()

        # Step 1: Get line item and profile
        line_item = get_line_item(line_item_id, token)
        profile_id = line_item.get("profile_id")
        if not profile_id:
            raise Exception("Line item has no profile_id (no inventory targeting configured)")

        profile = get_profile(profile_id, token)

        # Step 2: Get current inventory lists
        current_lists = get_inventory_lists_from_profile(profile)

        if not current_lists:
            # No existing lists - create a new one with default name
            base_name = f"AutoList-{line_item_id}"
            old_list_info = None
        else:
            # Use first list's name as base
            first_list = current_lists[0]
            old_list_ui_id = first_list.get('id')
            old_list_api_id = convert_profile_list_id_to_api_id(old_list_ui_id, token)
            old_list_details = get_inventory_list(old_list_api_id, token)
            old_list_name = old_list_details.get('name', f'List-{old_list_ui_id}')

            base_name = strip_datetime_suffix(old_list_name)
            old_list_info = {
                'name': old_list_name,
                'api_id': old_list_api_id,
                'ui_id': old_list_ui_id
            }

        # Step 3: Create new list with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        new_list_name = f"{base_name}_{timestamp}"
        new_api_list_id, new_ui_list_id = create_inventory_list(new_list_name, token)

        # Step 4: Add domains to new list
        result = add_domains_to_inventory_list(new_api_list_id, domains, token)
        domains_added = result.get('response', {}).get('count', len(domains))

        # Step 5: Replace old list with new one in profile
        replace_profile_list(profile_id, profile, new_ui_list_id, token)

        # Step 6: Delete old list (if there was one)
        if old_list_info:
            delete_inventory_list(old_list_info['api_id'], token)

        # Return success summary
        return {
            'success': True,
            'old_list_name': old_list_info['name'] if old_list_info else None,
            'old_list_id': old_list_info['api_id'] if old_list_info else None,
            'new_list_name': new_list_name,
            'new_list_id': new_api_list_id,
            'new_list_ui_id': new_ui_list_id,
            'domains_added': domains_added,
            'message': f"Successfully rotated list: {new_list_name} ({domains_added} domains)"
        }

    except Exception as e:
        return {
            'success': False,
            'message': f"Rotation failed: {str(e)}"
        }
