#!/usr/bin/env python3
"""
Deal ID to Inventory Lists Lookup
Provides a simple interface to get inventory lists for a given Deal ID.
Workflow: Deal ID -> Line Item ID (via cache) -> Profile ID -> Inventory Lists

Note: The 'id' field returned from the Profile API IS the inventory list ID that
should be used for all inventory-list-item API operations. There is no separate
"API ID" vs "Profile/UI ID" - they are the same value.
"""

import requests
from typing import Dict, List, Optional
from deal_lineitem_cache import get_line_item_for_deal
from xandr_auth import get_auth_token

# Configuration
XANDR_API_BASE = "https://api.appnexus.com"


def get_line_item_info(line_item_id: int, token: str) -> dict:
    """
    Retrieve line item information including profile_id.

    Args:
        line_item_id: The Xandr Line Item ID
        token: Authentication token

    Returns:
        Line item response data
    """
    url = f"{XANDR_API_BASE}/line-item"
    headers = {
        "Authorization": token,
        "Content-Type": "application/json"
    }
    params = {"id": line_item_id}

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()

    data = response.json()
    return data.get("response", {})


def get_profile_info(profile_id: int, token: str) -> dict:
    """
    Retrieve profile information including inventory_url_list_targets.

    Args:
        profile_id: The Xandr Profile ID
        token: Authentication token

    Returns:
        Profile response data
    """
    url = f"{XANDR_API_BASE}/profile"
    headers = {
        "Authorization": token,
        "Content-Type": "application/json"
    }
    params = {"id": profile_id}

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()

    data = response.json()
    return data.get("response", {})


def get_api_id_for_list(profile_list_id: int, token: str) -> Optional[int]:
    """
    DEPRECATED: This function is no longer needed.

    The 'id' field from the Profile API IS the inventory list ID that should be used
    for inventory-list-item operations. There is no separate "API ID" vs "Profile ID".

    This function is kept for backwards compatibility but simply returns the input.

    Args:
        profile_list_id: The list ID from the profile (which IS the API ID)
        token: Authentication token (unused)

    Returns:
        The same ID that was passed in
    """
    # The ID from profile is already the correct ID to use
    return profile_list_id


def get_inventory_lists_for_deal(deal_id: int, use_cache: bool = True) -> Optional[List[Dict]]:
    """
    Get all inventory lists associated with a Deal ID.

    Workflow:
    1. Look up Line Item ID from Deal ID (via cache)
    2. Get Line Item info from API to find Profile ID
    3. Get Profile info from API to find Inventory Lists

    Args:
        deal_id: The Xandr Deal ID
        use_cache: Whether to use the deal->lineitem cache (default: True)

    Returns:
        List of inventory list dictionaries with 'id', 'name', 'type' keys
        Returns None if lookup fails at any step
    """
    try:
        # Step 1: Get Line Item ID from Deal ID (via cache)
        if use_cache:
            line_item_id = get_line_item_for_deal(deal_id)
            if not line_item_id:
                print(f"❌ No Line Item found for Deal {deal_id} in cache")
                return None
        else:
            # If not using cache, caller must provide line_item_id directly
            raise ValueError("use_cache=False requires direct line_item_id lookup")

        # Step 2: Get authentication token
        token = get_auth_token()

        # Step 3: Get Line Item info to find Profile ID
        li_response = get_line_item_info(line_item_id, token)
        line_item = li_response.get("line-item")

        if not line_item:
            print(f"❌ No line item found with ID {line_item_id}")
            return None

        profile_id = line_item.get("profile_id")

        if not profile_id:
            print(f"⚠️  Line Item {line_item_id} has no profile_id")
            return None

        # Step 4: Get Profile info to find Inventory Lists
        profile_response = get_profile_info(profile_id, token)
        profile = profile_response.get("profile")

        if not profile:
            print(f"❌ No profile found with ID {profile_id}")
            return None

        # Step 5: Extract inventory lists from multiple possible fields
        inventory_lists = (
            profile.get("inventory_url_list_targets") or
            profile.get("domain_list_targets") or
            profile.get("inventory_lists") or
            []
        )

        # Also check for allowlists/blocklists separately
        domain_list_action = profile.get("domain_list_action", {})
        if isinstance(domain_list_action, dict):
            allowlists = domain_list_action.get("allowlists", [])
        else:
            allowlists = []

        # Combine all lists - ONLY allowlists
        all_lists = []

        if inventory_lists:
            # Filter to only include allowlists from inventory_lists
            # Check both list_type field and exclude field (exclude=False means allowlist)
            # Handle both new (allowlist/blocklist) and legacy (whitelist/blacklist) terminology
            for lst in inventory_lists:
                list_type = lst.get("list_type", "").lower()
                exclude = lst.get("exclude", None)

                # Determine if this is an allowlist:
                # 1. If list_type is "allowlist" or "whitelist"
                # 2. If exclude is False (means it's an allowlist)
                # 3. If list_type is not "blocklist"/"blacklist" and exclude is not True
                is_allowlist = (
                    list_type in ["allowlist", "whitelist"] or
                    exclude == False or
                    (list_type not in ["blocklist", "blacklist"] and exclude != True)
                )

                if is_allowlist:
                    # The 'id' field from profile IS the inventory list ID for API operations
                    list_id = lst.get("id")

                    enriched_list = {
                        **lst,
                        "list_type": "allowlist",
                        "api_id": list_id,  # The ID from profile is the correct API ID
                        "profile_id": list_id  # Same value - there's only one ID
                    }
                    all_lists.append(enriched_list)

        if allowlists:
            for lst in allowlists:
                list_id = lst.get("id")

                enriched_list = {
                    **lst,
                    "list_type": "allowlist",
                    "api_id": list_id,  # The ID from profile is the correct API ID
                    "profile_id": list_id  # Same value - there's only one ID
                }
                all_lists.append(enriched_list)

        return all_lists if all_lists else None

    except Exception as e:
        print(f"❌ Error looking up inventory lists for Deal {deal_id}: {e}")
        return None


def get_primary_inventory_list_id_for_deal(deal_id: int) -> Optional[int]:
    """
    Get the primary allowlist inventory list ID for a Deal.

    This is a convenience function that returns just the ID of the primary
    inventory list found, which is useful for simple upload operations.
    Only allowlists are returned.

    Args:
        deal_id: The Xandr Deal ID

    Returns:
        Integer inventory list ID (allowlist only), or None if not found
    """
    lists = get_inventory_lists_for_deal(deal_id)

    if lists and len(lists) > 0:
        # Return first non-deleted allowlist
        for lst in lists:
            if not lst.get("deleted", False) and lst.get("list_type") == "allowlist":
                return lst.get("id")

    return None


def get_inventory_list_summary_for_deal(deal_id: int) -> Optional[Dict]:
    """
    Get a summary of allowlist inventory lists for a Deal including the line item ID.

    Args:
        deal_id: The Xandr Deal ID

    Returns:
        Dictionary with 'deal_id', 'line_item_id', 'lists', 'primary_list_id'
        Only allowlists are included. Returns None if lookup fails
    """
    try:
        # Get Line Item ID
        line_item_id = get_line_item_for_deal(deal_id)
        if not line_item_id:
            return None

        # Get inventory lists (already filtered to allowlists only)
        lists = get_inventory_lists_for_deal(deal_id)

        # Get primary list ID (only allowlists)
        # Note: api_id and profile_id are the same value - there's only one ID
        primary_list_id = None
        if lists:
            # Return first non-deleted allowlist
            for lst in lists:
                if not lst.get("deleted", False) and lst.get("list_type") == "allowlist":
                    primary_list_id = lst.get("api_id") or lst.get("id")
                    break

        return {
            "deal_id": deal_id,
            "line_item_id": line_item_id,
            "lists": lists or [],
            "primary_list_id": primary_list_id,  # The inventory list ID (for both API and display)
            "primary_profile_id": primary_list_id,  # Same as primary_list_id (kept for compatibility)
            "list_count": len(lists) if lists else 0
        }

    except Exception as e:
        print(f"❌ Error getting inventory list summary for Deal {deal_id}: {e}")
        return None


if __name__ == "__main__":
    # Test the lookup
    import sys

    if len(sys.argv) < 2:
        print("Usage: python deal_inventory_lookup.py <deal_id>")
        print("\nExample: python deal_inventory_lookup.py 2438197")
        sys.exit(1)

    deal_id = int(sys.argv[1])

    print(f"Looking up inventory lists for Deal ID: {deal_id}")
    print("=" * 60)

    summary = get_inventory_list_summary_for_deal(deal_id)

    if summary:
        print(f"\n✅ Deal ID: {summary['deal_id']}")
        print(f"✅ Line Item ID: {summary['line_item_id']}")
        print(f"✅ Found {summary['list_count']} inventory list(s)")
        print(f"✅ Primary List ID: {summary['primary_list_id']}")

        if summary['lists']:
            print("\n📋 Inventory Lists:")
            for lst in summary['lists']:
                print(f"\n   • List ID: {lst.get('id')}")
                print(f"     Name: {lst.get('name')}")
                print(f"     Type: {lst.get('list_type', 'unknown')}")
                print(f"     Deleted: {lst.get('deleted', False)}")
    else:
        print(f"\n❌ Could not find inventory lists for Deal ID {deal_id}")
