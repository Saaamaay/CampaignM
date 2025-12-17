"""
BigQuery Connector for Campaign Dashboard
Handles authentication and data fetching from BigQuery
"""

from google.cloud import bigquery
import pandas as pd
import json
import streamlit as st
from typing import Optional, Dict, Any
import os


def get_bigquery_client() -> Optional[bigquery.Client]:
    """
    Create and return a BigQuery client with authentication.

    Supports two authentication methods:
    1. Streamlit secrets (for cloud deployment)
    2. Local service account JSON file (for local development)

    Returns:
        bigquery.Client or None if authentication fails
    """
    try:
        # Try Streamlit secrets first (for deployed app)
        if hasattr(st, 'secrets') and 'GCP_SERVICE_ACCOUNT' in st.secrets:
            credentials_info = dict(st.secrets['GCP_SERVICE_ACCOUNT'])
            client = bigquery.Client.from_service_account_info(credentials_info)
            return client

        # Try environment variable pointing to service account file
        elif os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            client = bigquery.Client()
            return client

        # Try local service account file in campaign_manager directory
        elif os.path.exists('service_account.json'):
            client = bigquery.Client.from_service_account_json('service_account.json')
            return client

        else:
            st.warning("⚠️ BigQuery credentials not configured. Please set up authentication.")
            return None

    except Exception as e:
        st.error(f"Failed to create BigQuery client: {str(e)}")
        return None


def test_bigquery_connection() -> bool:
    """
    Test the BigQuery connection.

    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        client = get_bigquery_client()
        if not client:
            return False

        # Try a simple query to verify connection
        query = "SELECT 1 as test"
        result = client.query(query).result()
        return True

    except Exception as e:
        st.error(f"BigQuery connection test failed: {str(e)}")
        return False


def query_campaign_data(
    project_id: str,
    dataset_id: str,
    table_id: str,
    date_range: Optional[tuple] = None
) -> Optional[pd.DataFrame]:
    """
    Query campaign data from BigQuery.
    Each table represents a separate campaign, so no campaign filtering is needed.

    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        table_id: BigQuery table ID (represents the campaign)
        date_range: Optional tuple of (start_date, end_date) as strings

    Returns:
        pandas DataFrame with campaign data or None if query fails
    """
    try:
        client = get_bigquery_client()
        if not client:
            return None

        # Build the base query - use table name as Campaign name
        query = f"""
        SELECT
            Date,
            '{table_id}' as Campaign,
            Device_Type as Device,
            Creative_Size,
            Impressions,
            Clicks,
            Total_Conversions as Conversions,
            Media_Cost_Advertiser_Currency as Cost,
            App_URL,
            Inventory_Source
        FROM `{project_id}.{dataset_id}.{table_id}`
        WHERE 1=1
        """

        # Add date range filter if provided
        if date_range:
            start_date, end_date = date_range
            query += f"\n  AND Date BETWEEN '{start_date}' AND '{end_date}'"

        query += "\nORDER BY Date DESC"

        # Execute query and return as DataFrame
        df = client.query(query).to_dataframe()

        # Convert Date column to datetime if it exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])

        # Rename columns to match dashboard expectations
        column_mapping = {
            'App_URL': 'App/URL',
            'Inventory_Source': 'Inventory Source',
            'Device': 'Device Type',
            'Creative_Size': 'Creative Size'
        }
        df.rename(columns=column_mapping, inplace=True)

        return df

    except Exception as e:
        st.error(f"Failed to query BigQuery: {str(e)}")
        return None


def query_custom_sql(sql_query: str) -> Optional[pd.DataFrame]:
    """
    Execute a custom SQL query against BigQuery.

    Args:
        sql_query: SQL query string to execute

    Returns:
        pandas DataFrame with query results or None if query fails
    """
    try:
        client = get_bigquery_client()
        if not client:
            return None

        df = client.query(sql_query).to_dataframe()
        return df

    except Exception as e:
        st.error(f"Failed to execute custom query: {str(e)}")
        return None


def get_available_campaigns(
    project_id: str,
    dataset_id: str
) -> Optional[list]:
    """
    Get list of available campaigns (tables) from BigQuery dataset.
    Each table represents a separate campaign.

    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID

    Returns:
        List of table names (campaigns) or None if query fails
    """
    try:
        client = get_bigquery_client()
        if not client:
            return None

        # List all tables in the dataset
        dataset_ref = f"{project_id}.{dataset_id}"
        tables = client.list_tables(dataset_ref)

        # Get table names
        table_names = [table.table_id for table in tables]

        return sorted(table_names)

    except Exception as e:
        st.error(f"Failed to get campaigns list: {str(e)}")
        return None


def get_table_schema(
    project_id: str,
    dataset_id: str,
    table_id: str
) -> Optional[Dict[str, Any]]:
    """
    Get the schema of a BigQuery table.

    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        table_id: BigQuery table ID

    Returns:
        Dictionary with table schema information or None if fails
    """
    try:
        client = get_bigquery_client()
        if not client:
            return None

        table_ref = f"{project_id}.{dataset_id}.{table_id}"
        table = client.get_table(table_ref)

        schema_info = {
            'fields': [{'name': field.name, 'type': field.field_type} for field in table.schema],
            'num_rows': table.num_rows,
            'created': table.created,
            'modified': table.modified
        }

        return schema_info

    except Exception as e:
        st.error(f"Failed to get table schema: {str(e)}")
        return None
