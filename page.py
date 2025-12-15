import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from datetime import datetime
import requests
import base64
import json
import gzip
from inventory_list_manager import rotate_inventory_list
from deal_inventory_lookup import get_inventory_list_summary_for_deal
from deal_lineitem_cache import (
    get_deal_id_for_inventory_source,
    get_line_item_for_deal,
    add_inventory_source_mapping,
    get_all_inventory_source_mappings,
    remove_inventory_source_mapping,
    set_github_config
)
from bigquery_connector import (
    get_bigquery_client,
    query_campaign_data,
    get_available_campaigns,
    test_bigquery_connection
)
import streamlit.components.v1 as components

st.set_page_config(page_title="Campaign Manager Dashboard", layout="wide")

# GitHub storage configuration
GITHUB_API_BASE = "https://api.github.com"
CAMPAIGNS_PATH = "saved_campaigns"

def get_github_config():
    """Get GitHub configuration from Streamlit secrets"""
    try:
        return {
            'token': st.secrets.get("GITHUB_TOKEN"),
            'owner': st.secrets.get("GITHUB_OWNER"), 
            'repo': st.secrets.get("GITHUB_REPO")
        }
    except:
        return {'token': None, 'owner': None, 'repo': None}

def github_api_request(method, url, headers=None, data=None):
    """Make GitHub API request with error handling"""
    config = get_github_config()
    if not config['token']:
        return None, "GitHub token not configured"
    
    default_headers = {
        'Authorization': f'token {config["token"]}',
        'Accept': 'application/vnd.github.v3+json',
        'Content-Type': 'application/json'
    }
    if headers:
        default_headers.update(headers)
    
    try:
        if method.upper() == 'GET':
            response = requests.get(url, headers=default_headers)
        elif method.upper() == 'PUT':
            response = requests.put(url, headers=default_headers, data=data)
        elif method.upper() == 'DELETE':
            response = requests.delete(url, headers=default_headers, data=data)
        else:
            return None, f"Unsupported method: {method}"
        
        if response.status_code in [200, 201, 204]:  # Added 204 for successful DELETE
            try:
                return response.json(), None
            except:
                return {}, None  # DELETE returns empty response on success
        else:
            return None, f"GitHub API error: {response.status_code} - {response.text}"
    except Exception as e:
        return None, f"Request failed: {str(e)}"

def save_campaign_data(df, campaign_name, dsp, kpi_settings=None):
    """Save campaign data with KPI settings to GitHub repository"""
    config = get_github_config()
    if not all(config.values()):
        return "GitHub not configured - using local storage fallback"
    
    campaign_data = {
        'dataframe': df.to_dict('records'),  # Convert to JSON-serializable format
        'columns': df.columns.tolist(),
        'dsp': dsp,
        'saved_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'row_count': len(df),
        'kpi_settings': kpi_settings or {}  # Store KPI preferences
    }
    
    # Clean campaign name for filename
    safe_name = "".join(c for c in campaign_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    
    # Create content and check size
    content = json.dumps(campaign_data, separators=(',', ':'))  # Compact JSON without indentation
    content_size = len(content.encode('utf-8'))
    
    # GitHub API has a 1MB limit for file contents
    MAX_FILE_SIZE = 1000000  # 1MB in bytes
    
    # Try compression first if content is large
    use_compression = False
    if content_size > MAX_FILE_SIZE:
        # Compress the JSON content
        compressed_content = gzip.compress(content.encode('utf-8'))
        compressed_size = len(compressed_content)
        
        if compressed_size <= MAX_FILE_SIZE:
            use_compression = True
            campaign_data['_compressed'] = True  # Flag to indicate compression
            content = json.dumps(campaign_data, separators=(',', ':'))
            # Re-compress with the compression flag included
            compressed_content = gzip.compress(content.encode('utf-8'))
            encoded_content = base64.b64encode(compressed_content).decode()
            filename = f"{safe_name}.json.gz"
            content_size = compressed_size
        else:
            return f"Error saving to GitHub: File too large even after compression ({compressed_size:,} bytes). GitHub API limit is {MAX_FILE_SIZE:,} bytes. Consider reducing data size by filtering rows or columns, or use local storage."
    else:
        # File is small enough, save normally
        filename = f"{safe_name}.json"
        encoded_content = base64.b64encode(content.encode()).decode()
    
    file_path = f"{CAMPAIGNS_PATH}/{filename}"
    
    # Check if file exists to get SHA
    url = f"{GITHUB_API_BASE}/repos/{config['owner']}/{config['repo']}/contents/{file_path}"
    existing_file, error = github_api_request('GET', url)
    
    # Prepare commit data
    commit_data = {
        'message': f'Save campaign: {campaign_name}',
        'content': encoded_content
    }
    
    if existing_file and 'sha' in existing_file:
        commit_data['sha'] = existing_file['sha']
    
    # Save to GitHub
    result, error = github_api_request('PUT', url, data=json.dumps(commit_data))
    
    if error:
        return f"Error saving to GitHub: {error}"
    else:
        compression_note = " (compressed)" if use_compression else ""
        return f"Campaign saved to GitHub: {filename} ({content_size:,} bytes{compression_note})"

def load_campaign_data(campaign_name):
    """Load campaign data from GitHub repository"""
    config = get_github_config()
    if not all(config.values()):
        return None
    
    safe_name = "".join(c for c in campaign_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    
    # Try loading compressed file first, then uncompressed
    result = None
    error = None
    filename = None
    is_compressed = False
    
    for try_filename in [f"{safe_name}.json.gz", f"{safe_name}.json"]:
        file_path = f"{CAMPAIGNS_PATH}/{try_filename}"
        url = f"{GITHUB_API_BASE}/repos/{config['owner']}/{config['repo']}/contents/{file_path}"
        result, error = github_api_request('GET', url)
        
        if not error and result:
            filename = try_filename
            is_compressed = try_filename.endswith('.gz')
            break
    
    if error or not result:
        if "404" in str(error):
            st.error(f"Campaign file not found: {campaign_name}")
        else:
            st.error(f"Failed to fetch file from GitHub: {error}")
        return None
    
    try:
        file_size = result.get('size', 0)
        raw_content = result.get('content', '')
        
        # Handle large files (>1MB) - GitHub API doesn't return content, use download_url instead
        if file_size > 1000000 or not raw_content:
            st.info(f"Loading large file ({file_size:,} bytes) via download URL...")
            download_url = result.get('download_url')
            
            if not download_url:
                st.error("No download URL available for large file")
                return None
            
            try:
                # Download file content directly
                response = requests.get(download_url)
                if response.status_code == 200:
                    if is_compressed:
                        # Decompress gzip content
                        decoded_content = gzip.decompress(response.content).decode('utf-8')
                    else:
                        decoded_content = response.text
                else:
                    st.error(f"Failed to download file: HTTP {response.status_code}")
                    return None
            except Exception as download_error:
                st.error(f"Error downloading file: {download_error}")
                return None
        else:
            # Small files - decode from base64 content
            clean_content = raw_content.replace('\n', '').replace('\r', '').strip()
            
            try:
                if is_compressed:
                    # Decode base64 then decompress gzip
                    compressed_data = base64.b64decode(clean_content)
                    decoded_content = gzip.decompress(compressed_data).decode('utf-8')
                else:
                    # Normal base64 decode
                    decoded_content = base64.b64decode(clean_content).decode('utf-8')
            except Exception as decode_error:
                st.error(f"Failed to decode content: {decode_error}")
                return None
        
        # Parse JSON
        try:
            campaign_data = json.loads(decoded_content)
        except json.JSONDecodeError as json_error:
            st.error(f"Failed to parse JSON: {json_error}")
            st.error(f"Content preview: {decoded_content[:200]}...")
            return None
        
        # Validate required fields
        required_fields = ['dataframe', 'columns', 'dsp', 'saved_date', 'row_count']
        missing_fields = [field for field in required_fields if field not in campaign_data]
        if missing_fields:
            st.error(f"Missing required fields in campaign data: {missing_fields}")
            return None
        
        # Reconstruct DataFrame
        try:
            df = pd.DataFrame(campaign_data['dataframe'], columns=campaign_data['columns'])
        except Exception as df_error:
            st.error(f"Failed to reconstruct DataFrame: {df_error}")
            return None
        
        return {
            'dataframe': df,
            'dsp': campaign_data['dsp'],
            'saved_date': campaign_data['saved_date'],
            'row_count': campaign_data['row_count'],
            'kpi_settings': campaign_data.get('kpi_settings', {})  # Load KPI settings or default to empty
        }
        
    except Exception as e:
        st.error(f"Unexpected error loading campaign data: {str(e)}")
        return None

def get_saved_campaigns():
    """Get list of saved campaign names from GitHub"""
    config = get_github_config()
    if not all(config.values()):
        return []
    
    url = f"{GITHUB_API_BASE}/repos/{config['owner']}/{config['repo']}/contents/{CAMPAIGNS_PATH}"
    result, error = github_api_request('GET', url)
    
    # If the campaigns folder doesn't exist yet, that's okay - return empty list
    if error and "404" in str(error):
        return []
    elif error or not result:
        st.error(f"Error accessing GitHub repository: {error}")
        return []
    
    campaigns = []
    if isinstance(result, list):
        for file_info in result:
            filename = file_info.get('name', '')
            if filename.endswith('.json'):
                campaign_name = filename[:-5]  # Remove .json extension
                campaigns.append(campaign_name)
            elif filename.endswith('.json.gz'):
                campaign_name = filename[:-8]  # Remove .json.gz extension
                campaigns.append(campaign_name)
    
    return sorted(campaigns)

def delete_campaign_data(campaign_name):
    """Delete saved campaign data from GitHub"""
    config = get_github_config()
    if not all(config.values()):
        return False
    
    safe_name = "".join(c for c in campaign_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    
    # Try both compressed and uncompressed versions
    for filename in [f"{safe_name}.json.gz", f"{safe_name}.json"]:
        file_path = f"{CAMPAIGNS_PATH}/{filename}"
        url = f"{GITHUB_API_BASE}/repos/{config['owner']}/{config['repo']}/contents/{file_path}"
        existing_file, error = github_api_request('GET', url)
        
        if not error and existing_file:
            # Delete file
            delete_data = {
                'message': f'Delete campaign: {campaign_name}',
                'sha': existing_file['sha']
            }
            
            result, error = github_api_request('DELETE', url, data=json.dumps(delete_data))
            
            if error:
                st.error(f"Delete failed: {error}")
                return False
            else:
                return True
    
    # Neither file found
    st.error(f"Campaign file not found: {campaign_name}")
    return False

def clean_csv_before_loading(file_path_or_buffer):
    """
    Clean CSV data to handle field count mismatches before pandas loading
    """
    import io
    import csv
    
    # If it's a file path, read it; if it's a buffer, read from current position
    if isinstance(file_path_or_buffer, str):
        with open(file_path_or_buffer, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        # Handle Streamlit uploaded file (binary buffer)
        raw_content = file_path_or_buffer.read()
        if isinstance(raw_content, bytes):
            content = raw_content.decode('utf-8')
        else:
            content = raw_content
        if hasattr(file_path_or_buffer, 'seek'):
            file_path_or_buffer.seek(0)  # Reset buffer position
    
    lines = content.split('\n')
    if not lines:
        return io.StringIO("")
    
    # Get expected field count from header
    header_line = lines[0]
    expected_fields = len(header_line.split(','))
    
    cleaned_lines = []
    rows_removed = 0
    
    for i, line in enumerate(lines):
        if not line.strip():  # Skip empty lines
            continue
            
        # Quick field count check
        field_count = len(line.split(','))
        
        if field_count == expected_fields:
            cleaned_lines.append(line)
        else:
            # Skip rows that don't match field count (likely summary/metadata rows)
            rows_removed += 1
    
    if rows_removed > 0:
        print(f"CSV Cleaning: Removed {rows_removed} rows with incorrect field count")
    
    cleaned_content = '\n'.join(cleaned_lines)
    return io.StringIO(cleaned_content)

def validate_and_clean_data(df):
    """
    Clean and validate the data, removing corrupted rows
    """
    initial_rows = len(df)
    validation_messages = []
    
    # Convert columns to numeric first
    numeric_columns = ['Impressions', 'Clicks']
    if 'Starts (Video)' in df.columns:
        numeric_columns.append('Starts (Video)')
    if 'Complete Views (Video)' in df.columns:
        numeric_columns.append('Complete Views (Video)')
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with corrupted date fields
    if 'Date' in df.columns:
        # Check for rows where Date contains 'Filter' or other non-date strings
        corrupted_date_mask = df['Date'].astype(str).str.contains('Filter|:', case=False, na=False)
        corrupted_rows = df[corrupted_date_mask]
        if len(corrupted_rows) > 0:
            validation_messages.append(f"⚠️ Removed {len(corrupted_rows)} rows with corrupted date fields")
            df = df[~corrupted_date_mask]
    
    # Remove rows with unrealistic impression values (> 10 million)
    if 'Impressions' in df.columns:
        unrealistic_mask = df['Impressions'] > 10_000_000
        unrealistic_rows = df[unrealistic_mask]
        if len(unrealistic_rows) > 0:
            validation_messages.append(f"⚠️ Removed {len(unrealistic_rows)} rows with unrealistic impressions (>10M)")
            # Log the problematic values
            for idx, row in unrealistic_rows.iterrows():
                validation_messages.append(f"  - Row with {row['Impressions']:,.0f} impressions removed")
            df = df[~unrealistic_mask]
    
    # Remove rows where clicks > impressions (impossible)
    if 'Clicks' in df.columns and 'Impressions' in df.columns:
        impossible_ctr_mask = (df['Clicks'] > df['Impressions']) & (df['Impressions'] > 0)
        impossible_rows = df[impossible_ctr_mask]
        if len(impossible_rows) > 0:
            validation_messages.append(f"⚠️ Removed {len(impossible_rows)} rows where clicks exceed impressions")
            df = df[~impossible_ctr_mask]
    
    # Remove rows where video completes > video starts (impossible)
    if 'Complete Views (Video)' in df.columns and 'Starts (Video)' in df.columns:
        impossible_vcr_mask = (df['Complete Views (Video)'] > df['Starts (Video)']) & (df['Starts (Video)'] > 0)
        impossible_vcr_rows = df[impossible_vcr_mask]
        if len(impossible_vcr_rows) > 0:
            validation_messages.append(f"⚠️ Removed {len(impossible_vcr_rows)} rows where video completes exceed starts")
            df = df[~impossible_vcr_mask]
    
    # Remove rows with NaN values in critical columns
    df = df.dropna(subset=['Impressions'])
    
    # Final summary
    final_rows = len(df)
    rows_removed = initial_rows - final_rows
    
    if rows_removed > 0:
        validation_messages.insert(0, f"🔍 Data Validation: Removed {rows_removed} invalid rows out of {initial_rows} total rows")
        validation_messages.append(f"✅ Clean data: {final_rows} valid rows remaining")
    else:
        validation_messages.append(f"✅ All {initial_rows} rows passed validation")
    
    return df, validation_messages

def calculate_metrics(df):
    # Clean and validate data first
    df_clean, validation_messages = validate_and_clean_data(df)
    
    # Convert columns to numeric, handling any text/formatting issues
    df_clean['Impressions'] = pd.to_numeric(df_clean['Impressions'], errors='coerce').fillna(0)
    df_clean['Clicks'] = pd.to_numeric(df_clean['Clicks'], errors='coerce').fillna(0)
    
    total_impressions = df_clean['Impressions'].sum()
    total_clicks = df_clean['Clicks'].sum()
    
    total_starts = 0
    total_completes = 0
    total_cost = 0
    total_conversions = 0
    
    if 'Starts (Video)' in df_clean.columns:
        df_clean['Starts (Video)'] = pd.to_numeric(df_clean['Starts (Video)'], errors='coerce').fillna(0)
        total_starts = df_clean['Starts (Video)'].sum()
        
    if 'Complete Views (Video)' in df_clean.columns:
        df_clean['Complete Views (Video)'] = pd.to_numeric(df_clean['Complete Views (Video)'], errors='coerce').fillna(0)
        total_completes = df_clean['Complete Views (Video)'].sum()
    
    # Handle cost columns (different possible names)
    cost_columns = ['Total Media Cost (Advertiser Currency)', 'Media Cost', 'Cost', 'Spend', 'Revenue (Partner Currency)']
    for col in cost_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            total_cost = df_clean[col].sum()
            break
    
    # Handle conversion columns
    conversion_columns = ['Total Conversions', 'Conversions', 'Post-Click Conversions', 'Post-View Conversions']
    for col in conversion_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            total_conversions = df_clean[col].sum()
            break
    
    # Calculate KPIs
    ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
    vcr = (total_completes / total_starts * 100) if total_starts > 0 else 0
    cpc = (total_cost / total_clicks) if total_clicks > 0 else 0
    cpa = (total_cost / total_conversions) if total_conversions > 0 else 0
    
    return {
        'impressions': int(total_impressions),
        'clicks': int(total_clicks),
        'starts': int(total_starts),
        'completes': int(total_completes),
        'cost': round(total_cost, 2),
        'conversions': int(total_conversions),
        'ctr': round(ctr, 4),  # More precision for low CTRs
        'vcr': round(vcr, 2),
        'cpc': round(cpc, 2),
        'cpa': round(cpa, 2),
        'validation_messages': validation_messages,
        'cleaned_df': df_clean
    }

def create_overview_cards(metrics):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Impressions", f"{metrics['impressions']:,}")
    
    with col2:
        st.metric("Total Clicks", f"{metrics['clicks']:,}")
    
    with col3:
        st.metric("Video Starts", f"{metrics['starts']:,}")
    
    with col4:
        st.metric("Video Completes", f"{metrics['completes']:,}")

def create_kpi_cards(metrics, selected_kpis=None):
    if selected_kpis is None:
        selected_kpis = ['CTR', 'VCR']  # Default selection
    
    # Filter available KPIs based on data availability
    available_kpis = {
        'CTR': {'value': metrics['ctr'], 'format': ':.4f', 'unit': '%', 'color_good': 10, 'industry': '0.05-0.10%'},
        'VCR': {'value': metrics['vcr'], 'format': ':.2f', 'unit': '%', 'color_good': 100, 'industry': '70-85%'},
        'CPC': {'value': metrics['cpc'], 'format': ':.2f', 'unit': '', 'color_good': 5, 'industry': 'Varies by industry'},
        'CPA': {'value': metrics['cpa'], 'format': ':.2f', 'unit': '', 'color_good': 50, 'industry': 'Varies by industry'}
    }
    
    # Only show KPIs that have data
    valid_kpis = []
    for kpi in selected_kpis:
        if kpi in available_kpis and available_kpis[kpi]['value'] > 0:
            valid_kpis.append(kpi)
    
    if not valid_kpis:
        st.info("No KPI data available for selected metrics")
        return
    
    # Create columns based on number of selected KPIs
    num_cols = min(len(valid_kpis), 4)
    cols = st.columns(num_cols)
    
    for i, kpi in enumerate(valid_kpis[:4]):  # Max 4 KPIs to fit nicely
        kpi_data = available_kpis[kpi]
        
        with cols[i % num_cols]:
            # Color coding based on KPI type
            if kpi == 'CTR':
                color = "🟢" if kpi_data['value'] < kpi_data['color_good'] else "🔴"
            elif kpi == 'VCR':
                color = "🟢" if kpi_data['value'] <= kpi_data['color_good'] else "🔴"
            elif kpi in ['CPC', 'CPA']:
                color = "🟢" if kpi_data['value'] < kpi_data['color_good'] else "🟡"
            else:
                color = ""
            
            # Format value
            value = kpi_data['value']
            format_str = kpi_data['format']
            unit = kpi_data['unit']
            
            # Apply formatting
            if format_str == ':.4f':
                formatted_value = f"{value:.4f}{unit}"
            elif format_str == ':.2f':
                formatted_value = f"{value:.2f}{unit}"
            else:
                formatted_value = f"{value}{unit}"
                
            st.metric(f"{color} {kpi}", formatted_value)
            
            # Add industry benchmark caption for CTR and VCR
            if kpi in ['CTR', 'VCR'] and kpi_data['value'] > 0:
                st.caption(f"Industry average: {kpi_data['industry']}")

def create_inventory_chart(df):
    if 'Inventory Source' in df.columns:
        # Ensure numeric columns are properly converted
        numeric_cols = ['Impressions', 'Clicks']
        if 'Starts (Video)' in df.columns:
            numeric_cols.append('Starts (Video)')
        if 'Complete Views (Video)' in df.columns:
            numeric_cols.append('Complete Views (Video)')

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        agg_dict = {
            'Impressions': 'sum',
            'Clicks': 'sum'
        }

        if 'Starts (Video)' in df.columns:
            agg_dict['Starts (Video)'] = 'sum'
        if 'Complete Views (Video)' in df.columns:
            agg_dict['Complete Views (Video)'] = 'sum'

        # Add cost and conversion columns if they exist
        cost_columns = ['Total Media Cost (Advertiser Currency)', 'Media Cost', 'Cost', 'Spend', 'Revenue (Partner Currency)']
        has_cost = False
        for col in cost_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                agg_dict['Cost'] = (col, 'sum')
                has_cost = True
                break

        conversion_columns = ['Total Conversions', 'Conversions', 'Post-Click Conversions', 'Post-View Conversions']
        has_conversions = False
        for col in conversion_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                agg_dict['Conversions'] = (col, 'sum')
                has_conversions = True
                break

        # Convert agg_dict for pandas
        final_agg_dict = {}
        for key, value in agg_dict.items():
            if isinstance(value, tuple):
                final_agg_dict[value[0]] = value[1]
            else:
                final_agg_dict[key] = value

        inventory_data = df.groupby('Inventory Source').agg(final_agg_dict).reset_index()

        # Rename cost and conversion columns back
        for key, value in agg_dict.items():
            if isinstance(value, tuple):
                inventory_data.rename(columns={value[0]: key}, inplace=True)

        # Calculate CTR and VCR safely
        inventory_data['CTR'] = ((inventory_data['Clicks'] / inventory_data['Impressions']) * 100).round(4)
        inventory_data['CTR'] = inventory_data['CTR'].replace([float('inf'), -float('inf')], 0).fillna(0)

        if 'Complete Views (Video)' in inventory_data.columns and 'Starts (Video)' in inventory_data.columns:
            inventory_data['VCR'] = ((inventory_data['Complete Views (Video)'] / inventory_data['Starts (Video)']) * 100).round(2)
            inventory_data['VCR'] = inventory_data['VCR'].replace([float('inf'), -float('inf')], 0).fillna(0)

        # Calculate CPA if both cost and conversions exist
        if has_cost and has_conversions and 'Cost' in inventory_data.columns and 'Conversions' in inventory_data.columns:
            inventory_data['CPA'] = (inventory_data['Cost'] / inventory_data['Conversions']).round(2)
            inventory_data['CPA'] = inventory_data['CPA'].replace([float('inf'), -float('inf')], 0).fillna(0)

        # Filter out any rows with clearly invalid data (negative values or NaN)
        inventory_data = inventory_data[inventory_data['Impressions'] >= 0]
        inventory_data = inventory_data.dropna(subset=['Impressions'])

        st.subheader("Inventory Source Performance")
        # Format the dataframe for better display
        display_data = inventory_data.copy()
        display_data['Impressions'] = display_data['Impressions'].apply(lambda x: f"{x:,.0f}")
        display_data['Clicks'] = display_data['Clicks'].apply(lambda x: f"{x:,.0f}")
        display_data['CTR'] = display_data['CTR'].apply(lambda x: f"{x:.4f}%")
        if 'VCR' in display_data.columns:
            display_data['VCR'] = display_data['VCR'].apply(lambda x: f"{x:.2f}%")
        if 'Cost' in display_data.columns:
            display_data['Cost'] = display_data['Cost'].apply(lambda x: f"£{x:,.2f}")
        if 'Conversions' in display_data.columns:
            display_data['Conversions'] = display_data['Conversions'].apply(lambda x: f"{x:,.0f}")
        if 'CPA' in display_data.columns:
            display_data['CPA'] = display_data['CPA'].apply(lambda x: f"£{x:.2f}")
        st.dataframe(display_data, use_container_width=True)

        # Creative Size Performance breakdown (expandable)
        if 'Creative Size' in df.columns:
            with st.expander("📐 Creative Size Performance by Inventory Source", expanded=False):
                st.caption("Analyze creative size performance for each inventory source")

                # Get unique inventory sources
                inventory_sources = sorted([source for source in df['Inventory Source'].unique() if pd.notna(source)])

                selected_inv_source = st.selectbox(
                    "Select Inventory Source:",
                    inventory_sources,
                    key="creative_size_inv_source",
                    help="Choose an inventory source to see creative size breakdown"
                )

                if selected_inv_source:
                    # Filter data for selected inventory source
                    filtered_df = df[df['Inventory Source'] == selected_inv_source].copy()

                    # Build aggregation dict for creative sizes
                    size_agg_dict = {
                        'Impressions': 'sum',
                        'Clicks': 'sum'
                    }

                    if 'Starts (Video)' in filtered_df.columns:
                        size_agg_dict['Starts (Video)'] = 'sum'
                    if 'Complete Views (Video)' in filtered_df.columns:
                        size_agg_dict['Complete Views (Video)'] = 'sum'

                    # Add cost if available
                    size_has_cost = False
                    for col in cost_columns:
                        if col in filtered_df.columns:
                            size_agg_dict['Cost'] = (col, 'sum')
                            size_has_cost = True
                            break

                    # Add conversions if available
                    size_has_conversions = False
                    for col in conversion_columns:
                        if col in filtered_df.columns:
                            size_agg_dict['Conversions'] = (col, 'sum')
                            size_has_conversions = True
                            break

                    # Convert agg_dict for pandas
                    size_final_agg_dict = {}
                    for key, value in size_agg_dict.items():
                        if isinstance(value, tuple):
                            size_final_agg_dict[value[0]] = value[1]
                        else:
                            size_final_agg_dict[key] = value

                    # Group by Creative Size
                    creative_size_data = filtered_df.groupby('Creative Size').agg(size_final_agg_dict).reset_index()

                    # Rename cost and conversion columns back
                    for key, value in size_agg_dict.items():
                        if isinstance(value, tuple):
                            creative_size_data.rename(columns={value[0]: key}, inplace=True)

                    # Calculate CTR
                    creative_size_data['CTR'] = ((creative_size_data['Clicks'] / creative_size_data['Impressions']) * 100).round(4)
                    creative_size_data['CTR'] = creative_size_data['CTR'].replace([float('inf'), -float('inf')], 0).fillna(0)

                    # Calculate VCR if video data exists
                    if 'Complete Views (Video)' in creative_size_data.columns and 'Starts (Video)' in creative_size_data.columns:
                        creative_size_data['VCR'] = ((creative_size_data['Complete Views (Video)'] / creative_size_data['Starts (Video)']) * 100).round(2)
                        creative_size_data['VCR'] = creative_size_data['VCR'].replace([float('inf'), -float('inf')], 0).fillna(0)

                    # Calculate CPA if both cost and conversions exist
                    if size_has_cost and size_has_conversions and 'Cost' in creative_size_data.columns and 'Conversions' in creative_size_data.columns:
                        creative_size_data['CPA'] = (creative_size_data['Cost'] / creative_size_data['Conversions']).round(2)
                        creative_size_data['CPA'] = creative_size_data['CPA'].replace([float('inf'), -float('inf')], 0).fillna(0)

                    # Filter out rows with no impressions
                    creative_size_data = creative_size_data[creative_size_data['Impressions'] > 0]

                    # Sort by impressions descending
                    creative_size_data = creative_size_data.sort_values('Impressions', ascending=False)

                    # Format for display
                    size_display_data = creative_size_data.copy()
                    size_display_data['Impressions'] = size_display_data['Impressions'].apply(lambda x: f"{x:,.0f}")
                    size_display_data['Clicks'] = size_display_data['Clicks'].apply(lambda x: f"{x:,.0f}")
                    size_display_data['CTR'] = size_display_data['CTR'].apply(lambda x: f"{x:.4f}%")
                    if 'VCR' in size_display_data.columns:
                        size_display_data['VCR'] = size_display_data['VCR'].apply(lambda x: f"{x:.2f}%")
                    if 'Cost' in size_display_data.columns:
                        size_display_data['Cost'] = size_display_data['Cost'].apply(lambda x: f"£{x:,.2f}")
                    if 'Conversions' in size_display_data.columns:
                        size_display_data['Conversions'] = size_display_data['Conversions'].apply(lambda x: f"{x:,.0f}")
                    if 'CPA' in size_display_data.columns:
                        size_display_data['CPA'] = size_display_data['CPA'].apply(lambda x: f"£{x:.2f}")

                    st.write(f"**Creative Size Performance for:** *{selected_inv_source}*")
                    st.dataframe(size_display_data, use_container_width=True, hide_index=True)

                    # Summary stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Creative Sizes", len(creative_size_data))
                    with col2:
                        total_imps = creative_size_data['Impressions'].apply(lambda x: int(x.replace(',', ''))).sum() if isinstance(creative_size_data['Impressions'].iloc[0], str) else creative_size_data['Impressions'].sum()
                        st.metric("Total Impressions", f"{int(total_imps):,}")
                    with col3:
                        avg_ctr = creative_size_data['CTR'].apply(lambda x: float(x.replace('%', '')) if isinstance(x, str) else x).mean() if len(creative_size_data) > 0 else 0
                        st.metric("Average CTR", f"{avg_ctr:.4f}%")

def create_inventory_app_ranking(df):
    """Create a section for selecting inventory source and ranking/filtering apps/URLs by any KPI"""
    if 'Inventory Source' in df.columns and 'App/URL' in df.columns:
        st.subheader("🎯 App/URL Performance Analysis by Inventory Source")
        
        # Get unique inventory sources
        inventory_sources = df['Inventory Source'].unique()
        inventory_sources = sorted([source for source in inventory_sources if pd.notna(source)])
        
        col1, col2, col3 = st.columns([2, 2, 2])

        with col1:
            selected_source = st.selectbox(
                "Select Inventory Source:",
                inventory_sources,
                help="Choose an inventory source to analyze app/URL performance"
            )

        with col2:
            # Creative Size filter (optional - multi-select)
            if 'Creative Size' in df.columns and selected_source:
                source_data_for_sizes = df[df['Inventory Source'] == selected_source]
                creative_sizes = sorted([size for size in source_data_for_sizes['Creative Size'].unique() if pd.notna(size)])
                selected_creative_sizes = st.multiselect(
                    "Filter by Creative Size(s):",
                    creative_sizes,
                    default=creative_sizes,  # All selected by default
                    help="Select one or more creative sizes to filter by"
                )
            else:
                selected_creative_sizes = []

        with col3:
            # Determine available KPIs for ranking
            available_ranking_kpis = ['CTR', 'Impressions', 'Clicks']

            # Check if cost/conversion data exists for this inventory source
            if selected_source:
                source_data = df[df['Inventory Source'] == selected_source]
                cost_columns = ['Total Media Cost (Advertiser Currency)', 'Media Cost', 'Cost', 'Spend', 'Revenue (Partner Currency)']
                conversion_columns = ['Total Conversions', 'Conversions', 'Post-Click Conversions', 'Post-View Conversions']

                # Check if cost data exists
                has_cost_data = False
                for col in cost_columns:
                    if col in source_data.columns and source_data[col].sum() > 0:
                        has_cost_data = True
                        break

                # Check if conversion data exists
                has_conversion_data = False
                for col in conversion_columns:
                    if col in source_data.columns and source_data[col].sum() > 0:
                        has_conversion_data = True
                        break

                # Add CPC if cost data exists
                if has_cost_data:
                    available_ranking_kpis.append('CPC')

                # Add CPA if both cost and conversion data exist
                if has_cost_data and has_conversion_data:
                    available_ranking_kpis.append('CPA')
                
                # Add VCR if video data exists
                if 'Starts (Video)' in source_data.columns and 'Complete Views (Video)' in source_data.columns:
                    if source_data['Starts (Video)'].sum() > 0:
                        available_ranking_kpis.append('VCR')
            
            ranking_kpi = st.selectbox(
                "Rank/Filter by KPI:",
                options=available_ranking_kpis,
                help="Choose which metric to rank and filter URLs by"
            )
        
        # Dynamic threshold controls based on selected KPI
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            threshold_direction = st.selectbox(
                "Show URLs:",
                ["Above", "Below"],
                help=f"Show URLs above or below the {ranking_kpi} threshold"
            )
        
        with col2:
            # Set threshold parameters based on KPI type
            if ranking_kpi in ['CTR', 'VCR']:
                # Percentage KPIs
                min_val, max_val, default_val, step_val = 0.0001, 100.0, 0.15, 0.01
                format_str = "%.4f" if ranking_kpi == 'CTR' else "%.2f"
                unit = "%"
            elif ranking_kpi in ['CPC', 'CPA']:
                # Cost KPIs
                min_val, max_val, default_val, step_val = 0.01, 1000.0, 1.0, 0.01
                format_str = "%.2f"
                unit = ""
            else:
                # Volume KPIs (Impressions, Clicks)
                min_val, max_val, default_val, step_val = 1, 1000000, 100, 1
                format_str = "%d"
                unit = ""
            
            threshold_value = st.number_input(
                f"{ranking_kpi} Threshold {('(' + unit + ')') if unit else ''}:",
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                step=step_val,
                format=format_str,
                help=f"Enter {ranking_kpi} threshold"
            )
        
        with col3:
            st.write("")  # Spacing
        
        if selected_source:
            # Filter data for selected inventory source
            filtered_df = df[df['Inventory Source'] == selected_source].copy()

            # Apply creative size filter if specific sizes are selected
            if selected_creative_sizes and 'Creative Size' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Creative Size'].isin(selected_creative_sizes)].copy()
            
            # Ensure numeric columns are properly converted
            numeric_cols = ['Impressions', 'Clicks']
            if 'Starts (Video)' in filtered_df.columns:
                numeric_cols.append('Starts (Video)')
            if 'Complete Views (Video)' in filtered_df.columns:
                numeric_cols.append('Complete Views (Video)')
                
            # Add cost and conversion columns if they exist
            cost_columns = ['Total Media Cost (Advertiser Currency)', 'Media Cost', 'Cost', 'Spend', 'Revenue (Partner Currency)']
            for col in cost_columns:
                if col in filtered_df.columns:
                    numeric_cols.append(col)
                    break
                    
            conversion_columns = ['Total Conversions', 'Conversions', 'Post-Click Conversions', 'Post-View Conversions']
            for col in conversion_columns:
                if col in filtered_df.columns:
                    numeric_cols.append(col)
                    break
                
            for col in numeric_cols:
                if col in filtered_df.columns:
                    filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce').fillna(0)
            
            # Group by App/URL and aggregate metrics
            agg_dict = {
                'Impressions': 'sum',
                'Clicks': 'sum'
            }
            
            if 'Starts (Video)' in filtered_df.columns:
                agg_dict['Starts (Video)'] = 'sum'
            if 'Complete Views (Video)' in filtered_df.columns:
                agg_dict['Complete Views (Video)'] = 'sum'
            
            # Add cost aggregation
            for col in cost_columns:
                if col in filtered_df.columns:
                    agg_dict['Cost'] = col, 'sum'
                    break
            
            # Add conversion aggregation  
            for col in conversion_columns:
                if col in filtered_df.columns:
                    agg_dict['Conversions'] = col, 'sum'
                    break
            
            # Convert agg_dict for pandas
            final_agg_dict = {}
            for key, value in agg_dict.items():
                if isinstance(value, tuple):
                    final_agg_dict[value[0]] = value[1]
                else:
                    final_agg_dict[key] = value
            
            app_data = filtered_df.groupby('App/URL').agg(final_agg_dict).reset_index()
            
            # Rename cost and conversion columns back
            for key, value in agg_dict.items():
                if isinstance(value, tuple):
                    app_data.rename(columns={value[0]: key}, inplace=True)
            
            # Calculate all KPIs
            app_data['CTR'] = ((app_data['Clicks'] / app_data['Impressions']) * 100).round(4)
            app_data['CTR'] = app_data['CTR'].replace([float('inf'), -float('inf')], 0).fillna(0)
            
            if 'Complete Views (Video)' in app_data.columns and 'Starts (Video)' in app_data.columns:
                app_data['VCR'] = ((app_data['Complete Views (Video)'] / app_data['Starts (Video)']) * 100).round(2)
                app_data['VCR'] = app_data['VCR'].replace([float('inf'), -float('inf')], 0).fillna(0)
            
            if 'Cost' in app_data.columns and 'Clicks' in app_data.columns:
                app_data['CPC'] = (app_data['Cost'] / app_data['Clicks']).round(2)
                app_data['CPC'] = app_data['CPC'].replace([float('inf'), -float('inf')], 0).fillna(0)
            
            if 'Cost' in app_data.columns and 'Conversions' in app_data.columns:
                app_data['CPA'] = (app_data['Cost'] / app_data['Conversions']).round(2)
                app_data['CPA'] = app_data['CPA'].replace([float('inf'), -float('inf')], 0).fillna(0)
            
            # Filter out rows with no impressions
            app_data = app_data[app_data['Impressions'] > 0]
            
            # Apply threshold filter based on selected KPI
            if ranking_kpi not in app_data.columns:
                st.error(f"Selected KPI '{ranking_kpi}' is not available for this inventory source")
                return
            
            if threshold_direction == "Above":
                filtered_apps = app_data[app_data[ranking_kpi] >= threshold_value]
                if ranking_kpi in ['CTR', 'VCR']:
                    if ranking_kpi == 'CTR':
                        direction_text = f"above {threshold_value:.4f}%"
                    else:
                        direction_text = f"above {threshold_value:.2f}%"
                else:
                    direction_text = f"above {threshold_value}"
            else:
                filtered_apps = app_data[app_data[ranking_kpi] <= threshold_value]
                if ranking_kpi in ['CTR', 'VCR']:
                    if ranking_kpi == 'CTR':
                        direction_text = f"below {threshold_value:.4f}%"
                    else:
                        direction_text = f"below {threshold_value:.2f}%"
                else:
                    direction_text = f"below {threshold_value}"
            
            # Sort by selected KPI descending
            filtered_apps = filtered_apps.sort_values(ranking_kpi, ascending=False)
            
            # Display results
            if len(filtered_apps) > 0:
                # Header with copy and upload functionality
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    # Show creative size filter info if not all sizes selected
                    if selected_creative_sizes and 'Creative Size' in df.columns:
                        all_sizes = sorted([size for size in df[df['Inventory Source'] == selected_source]['Creative Size'].unique() if pd.notna(size)])
                        if len(selected_creative_sizes) != len(all_sizes):
                            if len(selected_creative_sizes) <= 3:
                                size_filter_text = f" ({', '.join(selected_creative_sizes)})"
                            else:
                                size_filter_text = f" ({len(selected_creative_sizes)} sizes)"
                        else:
                            size_filter_text = ""
                    else:
                        size_filter_text = ""
                    st.write(f"**{len(filtered_apps)} Apps/URLs with {ranking_kpi} {direction_text}** for *{selected_source}*{size_filter_text}")

                with col2:
                    # Create URL list for copying
                    url_list = filtered_apps['App/URL'].tolist()
                    url_text = '\n'.join(url_list)

                    # Show expandable text area with URLs for easy copying
                    with st.expander(f"📋 Copy {len(url_list)} URLs"):
                        st.text_area(
                            "Select all and copy:",
                            url_text,
                            height=150,
                            label_visibility="collapsed"
                        )
                        st.caption("Click in the box, Ctrl/Cmd+A to select all, then Ctrl/Cmd+C to copy")

                with col3:
                    # Upload to Xandr button with dynamic list lookup
                    st.write("")  # Spacing

                    # Look up the Deal ID for this inventory source using our mapping cache
                    deal_id = None
                    line_item_id = None
                    inventory_list_id = None
                    inventory_list_info = None

                    # Use the inventory source mapping cache
                    deal_id = get_deal_id_for_inventory_source(selected_source)

                    # Look up line item ID for this deal
                    if deal_id:
                        line_item_id = get_line_item_for_deal(deal_id)

                    # Look up inventory list for this deal
                    if deal_id:
                        with st.spinner("Looking up inventory list..."):
                            inventory_list_info = get_inventory_list_summary_for_deal(deal_id)
                            if inventory_list_info and inventory_list_info.get('primary_list_id'):
                                inventory_list_id = inventory_list_info['primary_list_id']

                    # Show upload button with appropriate state
                    if inventory_list_id:
                        # Get the profile ID (UI ID) for display
                        profile_list_id = inventory_list_info.get('primary_profile_id') if inventory_list_info else None

                        if st.button("🚀 Upload to Xandr", key="upload_domains_btn", help=f"Create new list, upload domains, and replace old list"):
                            with st.spinner(f"Creating new list and uploading {len(url_list)} domains..."):
                                try:
                                    # Use the professional rotation system
                                    result = rotate_inventory_list(
                                        line_item_id=line_item_id,
                                        domains=url_list
                                    )

                                    if result.get('success'):
                                        # Show detailed success message
                                        st.success(f"✅ {result['message']}")

                                        # Show details in an expander
                                        with st.expander("📋 Operation Details", expanded=False):
                                            st.write(f"**New List:** {result['new_list_name']}")
                                            st.write(f"**New List UI ID:** {result['new_list_ui_id']}")
                                            st.write(f"**Domains Added:** {result['domains_added']}")
                                            if result.get('old_list_name'):
                                                st.write(f"**Old List (Deleted):** {result['old_list_name']}")
                                            else:
                                                st.info("No previous list to delete (first list created)")
                                    else:
                                        st.error(f"❌ {result['message']}")

                                except Exception as e:
                                    st.error(f"❌ Upload failed: {str(e)}")

                        # Display inventory list info - show UI ID to user
                        if inventory_list_info and inventory_list_info.get('lists'):
                            primary_list = next((lst for lst in inventory_list_info['lists'] if lst.get('api_id') == inventory_list_id), None)
                            if primary_list:
                                st.caption(f"📋 List: {primary_list.get('name', 'Unknown')}")
                                # Display profile ID (UI ID) to user
                                display_id = profile_list_id if profile_list_id else inventory_list_id
                                st.caption(f"🆔 List ID: {display_id}")
                                st.caption(f"🔗 Deal ID: {deal_id}")
                                if line_item_id:
                                    st.caption(f"📝 Line Item ID: {line_item_id}")
                            else:
                                display_id = profile_list_id if profile_list_id else inventory_list_id
                                st.caption(f"🆔 List ID: {display_id}")
                                st.caption(f"🔗 Deal ID: {deal_id}")
                                if line_item_id:
                                    st.caption(f"📝 Line Item ID: {line_item_id}")
                        else:
                            display_id = profile_list_id if profile_list_id else inventory_list_id
                            st.caption(f"🆔 List ID: {display_id}")
                            st.caption(f"🔗 Deal ID: {deal_id}")
                            if line_item_id:
                                st.caption(f"📝 Line Item ID: {line_item_id}")
                    elif deal_id:
                        st.warning("⚠️ No inventory list found for this deal")
                        st.caption(f"🔗 Deal ID: {deal_id}")
                        if line_item_id:
                            st.caption(f"📝 Line Item ID: {line_item_id}")
                        st.caption("Check if the deal has a profile with inventory targeting configured")
                    else:
                        st.warning("⚠️ This inventory source is not mapped to a Deal ID")
                        st.caption("📝 Use the 'Manage Inventory Source Mappings' section above to map this source")
                        st.caption("This is a one-time setup that will be remembered for future reports")
                
                # Format data for display
                display_data = filtered_apps.copy()
                display_data['Impressions'] = display_data['Impressions'].apply(lambda x: f"{x:,.0f}")
                display_data['Clicks'] = display_data['Clicks'].apply(lambda x: f"{x:,.0f}")
                display_data['CTR'] = display_data['CTR'].apply(lambda x: f"{x:.4f}%")
                
                # Format additional KPIs if they exist
                if 'VCR' in display_data.columns:
                    display_data['VCR'] = display_data['VCR'].apply(lambda x: f"{x:.2f}%")
                if 'CPC' in display_data.columns:
                    display_data['CPC'] = display_data['CPC'].apply(lambda x: f"{x:.2f}")
                if 'CPA' in display_data.columns:
                    display_data['CPA'] = display_data['CPA'].apply(lambda x: f"{x:.2f}")
                if 'Cost' in display_data.columns:
                    display_data['Cost'] = display_data['Cost'].apply(lambda x: f"{x:.2f}")
                if 'Conversions' in display_data.columns:
                    display_data['Conversions'] = display_data['Conversions'].apply(lambda x: f"{x:,.0f}")
                
                # Show all filtered results
                st.dataframe(display_data, use_container_width=True, hide_index=True)
                
                # Summary stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("URLs Found", len(filtered_apps))
                with col2:
                    st.metric("Total Apps/URLs", len(app_data))
                with col3:
                    # Show average for selected ranking KPI
                    avg_value = filtered_apps[ranking_kpi].mean()
                    if ranking_kpi in ['CTR', 'VCR']:
                        format_val = f"{avg_value:.4f}%" if ranking_kpi == 'CTR' else f"{avg_value:.2f}%"
                    else:
                        format_val = f"{avg_value:.2f}"
                    st.metric(f"Average {ranking_kpi}", format_val)
                with col4:
                    # Show additional relevant metric
                    if ranking_kpi != 'CTR' and 'CTR' in filtered_apps.columns:
                        avg_ctr = filtered_apps['CTR'].mean()
                        st.metric("Average CTR", f"{avg_ctr:.4f}%")
                    elif ranking_kpi != 'VCR' and 'VCR' in filtered_apps.columns and filtered_apps['VCR'].sum() > 0:
                        avg_vcr = filtered_apps[filtered_apps['VCR'] > 0]['VCR'].mean()
                        st.metric("Average VCR", f"{avg_vcr:.2f}%")
                    else:
                        st.metric("", "")
            else:
                # Show creative size filter info if not all sizes selected
                if selected_creative_sizes and 'Creative Size' in df.columns:
                    all_sizes = sorted([size for size in df[df['Inventory Source'] == selected_source]['Creative Size'].unique() if pd.notna(size)])
                    if len(selected_creative_sizes) != len(all_sizes):
                        if len(selected_creative_sizes) <= 3:
                            size_filter_text = f" ({', '.join(selected_creative_sizes)})"
                        else:
                            size_filter_text = f" ({len(selected_creative_sizes)} sizes)"
                    else:
                        size_filter_text = ""
                else:
                    size_filter_text = ""
                st.info(f"No URLs found with {ranking_kpi} {direction_text} for {selected_source}{size_filter_text}")
                
                # Show summary of all URLs for context
                if len(app_data) > 0:
                    min_val = app_data[ranking_kpi].min()
                    max_val = app_data[ranking_kpi].max()
                    avg_val = app_data[ranking_kpi].mean()
                    
                    if ranking_kpi in ['CTR', 'VCR']:
                        unit = "%"
                        format_str = ":.4f" if ranking_kpi == 'CTR' else ":.2f"
                    else:
                        unit = ""
                        format_str = ":.2f"
                    
                    st.write(f"**{ranking_kpi} Range for {selected_source}:**")
                    st.write(f"- Minimum: {min_val:{format_str}}{unit}")
                    st.write(f"- Maximum: {max_val:{format_str}}{unit}") 
                    st.write(f"- Average: {avg_val:{format_str}}{unit}")
    else:
        st.warning("App/URL or Inventory Source columns not found in the data")

def create_device_chart(df):
    if 'Device Type' in df.columns:
        device_data = df.groupby('Device Type').agg({
            'Impressions': 'sum',
            'Clicks': 'sum'
        }).reset_index()
        
        fig = px.pie(device_data, 
                    values='Impressions', 
                    names='Device Type',
                    title='Impressions by Device Type')
        st.plotly_chart(fig, use_container_width=True)

def create_enhanced_daily_trend(df, selected_kpi='Impressions', days_back=14):
    """Create enhanced daily performance trend with KPI selection and custom time range"""
    if 'Date' in df.columns:
        # Clean date column
        df_clean = df[~df['Date'].astype(str).str.contains('Filter|:', case=False, na=False)].copy()
        
        # Convert Date column to datetime
        try:
            df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
            df_clean = df_clean.dropna(subset=['Date'])
        except:
            st.error("Unable to parse date column")
            return
        
        # Filter to selected time range
        if days_back > 0:
            end_date = df_clean['Date'].max()
            start_date = end_date - pd.Timedelta(days=days_back)
            df_clean = df_clean[df_clean['Date'] >= start_date]
        
        if len(df_clean) == 0:
            st.warning(f"No data available for the last {days_back} days")
            return
        
        # Prepare aggregation based on available columns
        agg_dict = {}
        
        # Add basic columns if they exist
        if 'Impressions' in df_clean.columns:
            agg_dict['Impressions'] = 'sum'
        if 'Clicks' in df_clean.columns:
            agg_dict['Clicks'] = 'sum'
        
        # Add cost column if available
        cost_columns = ['Total Media Cost (Advertiser Currency)', 'Media Cost', 'Cost', 'Spend', 'Revenue (Partner Currency)']
        for col in cost_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
                agg_dict['Cost'] = (col, 'sum')  # Store original column name for aggregation
                break
        
        # Add conversion column if available
        conversion_columns = ['Total Conversions', 'Conversions', 'Post-Click Conversions', 'Post-View Conversions']
        for col in conversion_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
                agg_dict['Conversions'] = (col, 'sum')  # Store original column name for aggregation
                break
        
        # Add video columns if available
        if 'Starts (Video)' in df_clean.columns:
            agg_dict['Video Starts'] = ('Starts (Video)', 'sum')
        if 'Complete Views (Video)' in df_clean.columns:
            agg_dict['Video Completes'] = ('Complete Views (Video)', 'sum')
        
        # Convert agg_dict to proper format for pandas
        final_agg_dict = {}
        for new_name, agg_info in agg_dict.items():
            if isinstance(agg_info, tuple):
                col_name, func = agg_info
                final_agg_dict[col_name] = func
            else:
                final_agg_dict[new_name] = agg_info
        
        # Check if we have any columns to aggregate
        if not final_agg_dict:
            st.warning("No aggregatable columns found in the data")
            return
        
        # Group by date
        daily_data = df_clean.groupby('Date').agg(final_agg_dict).reset_index()
        
        # Rename columns back to friendly names
        column_mapping = {}
        for new_name, agg_info in agg_dict.items():
            if isinstance(agg_info, tuple):
                col_name, func = agg_info
                if col_name in daily_data.columns:
                    column_mapping[col_name] = new_name
        
        daily_data.rename(columns=column_mapping, inplace=True)
        
        # Calculate derived KPIs
        if 'Clicks' in daily_data.columns and 'Impressions' in daily_data.columns:
            daily_data['CTR'] = (daily_data['Clicks'] / daily_data['Impressions'] * 100).round(4)
            daily_data['CTR'] = daily_data['CTR'].replace([float('inf'), -float('inf')], 0).fillna(0)
        
        if 'Video Completes' in daily_data.columns and 'Video Starts' in daily_data.columns:
            daily_data['VCR'] = (daily_data['Video Completes'] / daily_data['Video Starts'] * 100).round(2)
            daily_data['VCR'] = daily_data['VCR'].replace([float('inf'), -float('inf')], 0).fillna(0)
        
        if 'Cost' in daily_data.columns and 'Clicks' in daily_data.columns:
            daily_data['CPC'] = (daily_data['Cost'] / daily_data['Clicks']).round(2)
            daily_data['CPC'] = daily_data['CPC'].replace([float('inf'), -float('inf')], 0).fillna(0)
        
        if 'Cost' in daily_data.columns and 'Conversions' in daily_data.columns:
            daily_data['CPA'] = (daily_data['Cost'] / daily_data['Conversions']).round(2)
            daily_data['CPA'] = daily_data['CPA'].replace([float('inf'), -float('inf')], 0).fillna(0)
        
        # Check if selected KPI is available
        if selected_kpi not in daily_data.columns:
            available_kpis = [col for col in daily_data.columns if col != 'Date']
            st.warning(f"'{selected_kpi}' not available. Available metrics: {', '.join(available_kpis)}")
            selected_kpi = available_kpis[0] if available_kpis else 'Impressions'
        
        # Create the chart
        fig = px.line(daily_data, 
                     x='Date', 
                     y=selected_kpi,
                     title=f'{selected_kpi} Trend - Last {days_back} Days',
                     markers=True)
        
        # Customize chart
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=selected_kpi,
            hovermode='x unified'
        )
        
        # Add percentage symbol for rate metrics
        if selected_kpi in ['CTR', 'VCR']:
            fig.update_traces(hovertemplate=f'%{{y:.2f}}%<extra></extra>')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Days of Data", len(daily_data))
        with col2:
            avg_value = daily_data[selected_kpi].mean()
            unit = "%" if selected_kpi in ['CTR', 'VCR'] else ""
            st.metric(f"Average {selected_kpi}", f"{avg_value:.2f}{unit}")
        with col3:
            max_value = daily_data[selected_kpi].max()
            st.metric(f"Peak {selected_kpi}", f"{max_value:.2f}{unit}")
        with col4:
            trend = "↗️" if daily_data[selected_kpi].iloc[-1] > daily_data[selected_kpi].iloc[0] else "↘️"
            change = daily_data[selected_kpi].iloc[-1] - daily_data[selected_kpi].iloc[0]
            st.metric("Trend", f"{trend} {change:+.2f}{unit}")
    else:
        st.warning("No date column found in data")

def show_inventory_source_mapping_manager(df=None):
    """
    Show UI for managing inventory source → deal ID mappings.
    Detects unmapped sources from the current dataframe and allows user to map them.
    """
    st.subheader("🗂️ Inventory Source → Deal ID Mapping")

    # Get all existing mappings
    all_mappings = get_all_inventory_source_mappings()

    # Get unique inventory sources from current dataframe
    unmapped_sources = []
    if df is not None and 'Inventory Source' in df.columns:
        unique_sources = df['Inventory Source'].unique()
        unmapped_sources = [
            source for source in unique_sources
            if pd.notna(source) and source not in all_mappings
        ]

    # Show status
    if unmapped_sources:
        st.warning(f"⚠️ Found {len(unmapped_sources)} unmapped inventory source(s) in current report")
    elif all_mappings:
        st.success(f"✅ All inventory sources mapped ({len(all_mappings)} total)")
    else:
        st.info("No inventory source mappings configured yet")

    # Tab layout
    tab1, tab2 = st.tabs(["📝 Map Unmapped Sources", "📋 View All Mappings"])

    with tab1:
        if unmapped_sources:
            st.write("**Map the following inventory sources to their Xandr Deal IDs:**")
            st.caption("This is a one-time setup. Once mapped, these sources will be automatically recognized.")

            # Create form for batch mapping
            with st.form("mapping_form"):
                mappings_to_add = {}

                for i, source in enumerate(unmapped_sources):
                    st.write(f"**{i+1}. {source}**")

                    col1, col2 = st.columns([3, 1])
                    with col1:
                        deal_id = st.number_input(
                            f"Xandr Deal ID:",
                            min_value=1,
                            step=1,
                            key=f"deal_id_{i}",
                            help=f"Enter the Xandr Deal ID for this inventory source"
                        )
                    with col2:
                        st.write("")  # Spacing

                    if deal_id and deal_id > 0:
                        mappings_to_add[source] = deal_id

                    st.divider()

                # Submit button
                col1, col2 = st.columns([1, 3])
                with col1:
                    submit_button = st.form_submit_button("💾 Save Mappings", use_container_width=True)

                if submit_button:
                    if mappings_to_add:
                        success_count = 0
                        for source, deal_id in mappings_to_add.items():
                            try:
                                add_inventory_source_mapping(source, deal_id)
                                success_count += 1
                            except Exception as e:
                                st.error(f"Failed to map '{source}': {e}")

                        if success_count > 0:
                            st.success(f"✅ Successfully mapped {success_count} inventory source(s)!")
                            st.rerun()
                    else:
                        st.warning("Please enter at least one Deal ID")
        else:
            if df is not None:
                st.success("✅ All inventory sources in current report are already mapped!")
            else:
                st.info("Upload a report to detect unmapped inventory sources")

    with tab2:
        if all_mappings:
            st.write(f"**Current Mappings ({len(all_mappings)}):**")

            # Create a dataframe for display
            mapping_df = pd.DataFrame([
                {"Inventory Source": source, "Deal ID": deal_id}
                for source, deal_id in sorted(all_mappings.items())
            ])

            # Show as editable table with delete option
            for idx, row in mapping_df.iterrows():
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.text(row['Inventory Source'])

                with col2:
                    st.text(f"Deal ID: {row['Deal ID']}")

                with col3:
                    if st.button("🗑️ Delete", key=f"delete_{idx}"):
                        if remove_inventory_source_mapping(row['Inventory Source']):
                            st.success(f"Removed mapping for '{row['Inventory Source']}'")
                            st.rerun()

            # Export option
            st.divider()
            st.download_button(
                label="📥 Export Mappings as JSON",
                data=json.dumps(all_mappings, indent=2),
                file_name="inventory_source_mappings.json",
                mime="application/json"
            )
        else:
            st.info("No mappings configured yet. Upload a report and use the 'Map Unmapped Sources' tab to get started.")


def render_looker_embed(looker_url: str, height: int = 800):
    """
    Render a Looker Studio report embed.

    Args:
        looker_url: The Looker Studio embed URL
        height: Height of the iframe in pixels (default 800)
    """
    if not looker_url:
        st.warning("No Looker Studio URL configured")
        return

    # Add a button to open in new tab as fallback for cookie issues
    col1, col2 = st.columns([3, 1])
    with col2:
        st.link_button("📊 Open in New Tab", looker_url, use_container_width=True)

    with col1:
        st.caption("If the report doesn't load below, click 'Open in New Tab' or enable third-party cookies in your browser")

    iframe_html = f"""
    <iframe
        width="100%"
        height="{height}"
        src="{looker_url}"
        frameborder="0"
        style="border:0"
        allowfullscreen
        allow="storage-access">
    </iframe>
    """

    try:
        components.html(iframe_html, height=height)
    except Exception as e:
        st.error(f"Failed to load Looker Studio embed: {str(e)}")
        st.info(f"Open the report directly: {looker_url}")


def get_bigquery_config():
    """Get BigQuery configuration from Streamlit secrets.
    Each table in the dataset represents a separate campaign."""
    try:
        return {
            'project_id': st.secrets.get("BIGQUERY_PROJECT_ID", ""),
            'dataset_id': st.secrets.get("BIGQUERY_DATASET_ID", "")
        }
    except:
        return {'project_id': '', 'dataset_id': ''}


def get_looker_config():
    """Get Looker Studio configuration from Streamlit secrets or session state"""
    try:
        # Try to get from secrets first
        looker_urls = {}
        if hasattr(st, 'secrets') and 'LOOKER_URLS' in st.secrets:
            looker_urls = dict(st.secrets['LOOKER_URLS'])

        # Also check session state for user-added URLs
        if 'looker_urls' in st.session_state:
            looker_urls.update(st.session_state.looker_urls)

        return looker_urls
    except:
        return {}


def main():
    st.title("📊 Campaign Manager Dashboard")
    st.caption("With automatic data validation and cleaning")

    # Configure GitHub for inventory source mappings
    github_config = get_github_config()
    if all(github_config.values()):
        set_github_config(github_config)

    # Initialize session state
    if 'current_campaign' not in st.session_state:
        st.session_state.current_campaign = None
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    if 'current_dsp' not in st.session_state:
        st.session_state.current_dsp = "DV360"
    if 'kpi_settings' not in st.session_state:
        st.session_state.kpi_settings = {
            'selected_kpis': ['CTR', 'VCR'],
            'trend_kpi': 'Impressions'
        }
    
    # Campaign Management Section
    st.subheader("📁 Campaign Management")
    
    # Show GitHub configuration status
    config = get_github_config()
    if all(config.values()):
        st.success("✅ GitHub storage configured")
        
        # Add debugging info
        with st.expander("🔧 Debug GitHub Connection", expanded=False):
            st.write(f"**Repository:** {config['owner']}/{config['repo']}")
            st.write(f"**Token:** {'✅ Present' if config['token'] else '❌ Missing'}")
            
            # Test GitHub API connection
            test_url = f"{GITHUB_API_BASE}/repos/{config['owner']}/{config['repo']}"
            test_result, test_error = github_api_request('GET', test_url)
            
            if test_error:
                st.error(f"❌ GitHub API Connection Failed: {test_error}")
            else:
                st.success("✅ GitHub API Connection Successful")
                
            # Test campaigns folder access
            campaigns_url = f"{GITHUB_API_BASE}/repos/{config['owner']}/{config['repo']}/contents/{CAMPAIGNS_PATH}"
            campaigns_result, campaigns_error = github_api_request('GET', campaigns_url)
            
            if campaigns_error:
                st.error(f"❌ Campaigns folder access failed: {campaigns_error}")
                if "404" in str(campaigns_error):
                    st.info("The saved_campaigns folder may not exist or may be empty")
            else:
                st.success(f"✅ Campaigns folder accessible ({len(campaigns_result) if isinstance(campaigns_result, list) else 0} items found)")
                if isinstance(campaigns_result, list):
                    for item in campaigns_result:
                        st.write(f"- {item.get('name', 'Unknown')} ({item.get('type', 'Unknown type')})")
    else:
        st.warning("⚠️ GitHub storage not configured - campaigns will not persist. Please add GITHUB_TOKEN, GITHUB_OWNER, and GITHUB_REPO to Streamlit secrets.")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.write("**Load Saved Campaign**")
        saved_campaigns = get_saved_campaigns()
        
        if saved_campaigns:
            selected_campaign = st.selectbox(
                "Select Campaign:",
                [""] + saved_campaigns,
                help="Choose a saved campaign to load"
            )
            
            if selected_campaign:
                col_load, col_delete = st.columns(2)
                with col_load:
                    if st.button("Load Campaign"):
                        campaign_data = load_campaign_data(selected_campaign)
                        if campaign_data and 'dataframe' in campaign_data:
                            st.session_state.current_df = campaign_data['dataframe']
                            st.session_state.current_dsp = campaign_data['dsp']
                            st.session_state.current_campaign = selected_campaign
                            
                            # Restore KPI settings if available
                            if 'kpi_settings' in campaign_data and campaign_data['kpi_settings']:
                                st.session_state.kpi_settings.update(campaign_data['kpi_settings'])
                            
                            st.success(f"Loaded campaign: {selected_campaign}")
                            st.rerun()
                        else:
                            st.error("Failed to load campaign data. Please check the file format.")
                
                with col_delete:
                    if st.button("🗑️ Delete", help="Delete this saved campaign"):
                        if delete_campaign_data(selected_campaign):
                            st.success(f"Deleted campaign: {selected_campaign}")
                            st.rerun()
                        else:
                            st.error("Failed to delete campaign")
        else:
            st.info("No saved campaigns found. Save your first campaign to get started!")
    
    with col2:
        if st.session_state.current_campaign:
            st.write("**Override Campaign Data**")
            st.caption(f"Update data for: {st.session_state.current_campaign}")
            dsp = st.session_state.current_dsp
            st.info(f"DSP: {dsp}")

            data_source_tab1, data_source_tab2 = st.tabs(["📤 Upload CSV", "☁️ BigQuery"])

            uploaded_file = None
            bigquery_data = None

            with data_source_tab1:
                uploaded_file = st.file_uploader(
                    "Upload new data to replace current campaign:",
                    type=['csv'],
                    help="Upload new CSV data to override the current campaign while keeping the same name and settings",
                    key="override_csv_upload"
                )

            with data_source_tab2:
                bq_config = get_bigquery_config()
                if all(bq_config.values()):
                    if st.button("Load from BigQuery", key="override_bq_load"):
                        with st.spinner("Querying BigQuery..."):
                            bigquery_data = query_campaign_data(
                                bq_config['project_id'],
                                bq_config['dataset_id'],
                                bq_config['table_id'],
                                campaign_filter=st.session_state.current_campaign
                            )
                            if bigquery_data is not None:
                                st.success(f"Loaded {len(bigquery_data)} rows from BigQuery")
                else:
                    st.info("Configure BigQuery in Streamlit secrets to use this feature")
        else:
            st.write("**Upload New Data**")

            data_source_tab1, data_source_tab2 = st.tabs(["📤 Upload CSV", "☁️ BigQuery"])

            uploaded_file = None
            bigquery_data = None
            dsp = None

            with data_source_tab1:
                dsp = st.selectbox("Select DSP", ["DV360"], key="csv_dsp_select")
                uploaded_file = st.file_uploader(
                    "Drag and drop your CSV file here",
                    type=['csv'],
                    help="Upload your campaign performance data in CSV format",
                    key="new_csv_upload"
                )

            with data_source_tab2:
                bq_config = get_bigquery_config()

                if all(bq_config.values()):
                    st.info(f"Project: {bq_config['project_id']}")
                    st.info(f"Dataset: {bq_config['dataset_id']}")

                    # Get available campaigns (tables in the dataset)
                    campaigns_list = get_available_campaigns(
                        bq_config['project_id'],
                        bq_config['dataset_id']
                    )

                    if campaigns_list:
                        selected_bq_campaign = st.selectbox(
                            "Select Campaign (Table):",
                            campaigns_list,
                            key="bq_campaign_select",
                            help="Each table represents a separate campaign"
                        )

                        if st.button("Load Campaign from BigQuery"):
                            with st.spinner("Querying BigQuery..."):
                                # Use the selected campaign as the table_id
                                bigquery_data = query_campaign_data(
                                    bq_config['project_id'],
                                    bq_config['dataset_id'],
                                    selected_bq_campaign
                                )
                                if bigquery_data is not None:
                                    st.success(f"Loaded {len(bigquery_data)} rows from BigQuery")
                                    dsp = "DV360"
                    else:
                        st.warning("Could not fetch campaigns from BigQuery")
                else:
                    st.info("Configure BigQuery credentials in Streamlit secrets:")
                    st.code("""
[BIGQUERY]
BIGQUERY_PROJECT_ID = "your-project-id"
BIGQUERY_DATASET_ID = "your-dataset-id"
                    """, language="toml")
    
    with col3:
        st.write("**Current Campaign**")
        if st.session_state.current_campaign:
            st.info(f"📊 {st.session_state.current_campaign}")
            if st.button("Clear Campaign"):
                st.session_state.current_campaign = None
                st.session_state.current_df = None
                st.rerun()
        else:
            st.info("No campaign loaded")
    
    # Data processing
    df = None

    # Check if we should use uploaded file, BigQuery data, or session state
    if uploaded_file is not None:
        try:
            # Clean the CSV before loading
            cleaned_csv = clean_csv_before_loading(uploaded_file)
            df = pd.read_csv(cleaned_csv)
            st.session_state.current_df = df
            st.session_state.current_dsp = dsp
            st.success(f"Successfully loaded {len(df)} rows of data from {dsp}")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    elif bigquery_data is not None:
        try:
            df = bigquery_data
            st.session_state.current_df = df
            st.session_state.current_dsp = dsp if dsp else "DV360"
            st.success(f"Successfully loaded {len(df)} rows from BigQuery")
        except Exception as e:
            st.error(f"Error processing BigQuery data: {str(e)}")
    elif st.session_state.current_df is not None:
        df = st.session_state.current_df
        dsp = st.session_state.current_dsp
    
    if df is not None:
        # Show inventory source mapping manager
        # Auto-expand if there are unmapped sources
        all_mappings = get_all_inventory_source_mappings()
        if 'Inventory Source' in df.columns:
            unique_sources = df['Inventory Source'].unique()
            has_unmapped = any(
                pd.notna(source) and source not in all_mappings
                for source in unique_sources
            )
        else:
            has_unmapped = False

        with st.expander("🗂️ Manage Inventory Source Mappings", expanded=has_unmapped):
            st.caption("Map inventory sources to Xandr Deal IDs for automatic lookup")
            show_inventory_source_mapping_manager(df)

        # Add save campaign section for new uploads or overrides
        if uploaded_file is not None:
            if st.session_state.current_campaign:
                # Override existing campaign
                st.subheader("🔄 Override Campaign Data")
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Campaign Name:** {st.session_state.current_campaign}")
                    st.caption("New data will replace existing data but keep the same name and KPI settings")
                
                with col2:
                    if st.button("Save Override", help="Replace campaign data while keeping name and settings"):
                        try:
                            # Capture current KPI settings
                            current_kpi_settings = st.session_state.get('kpi_settings', {})
                            
                            result_message = save_campaign_data(df, st.session_state.current_campaign, dsp, current_kpi_settings)
                            
                            if "Error" in result_message:
                                st.error(result_message)
                            else:
                                st.success(f"✅ Updated campaign: {st.session_state.current_campaign}")
                        except Exception as e:
                            st.error(f"Error updating campaign: {str(e)}")
            else:
                # Save new campaign
                st.subheader("💾 Save Campaign")
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    campaign_name = st.text_input(
                        "Campaign Name:",
                        placeholder="Enter a name for this campaign",
                        help="Enter a descriptive name to save this campaign data"
                    )
                
                with col2:
                    if campaign_name and st.button("Save Campaign"):
                        try:
                            # Capture current KPI settings
                            current_kpi_settings = st.session_state.get('kpi_settings', {})
                            
                            result_message = save_campaign_data(df, campaign_name, dsp, current_kpi_settings)
                            st.session_state.current_campaign = campaign_name
                            
                            if "Error" in result_message:
                                st.error(result_message)
                            else:
                                st.success(result_message)
                        except Exception as e:
                            st.error(f"Error saving campaign: {str(e)}")
        
        # Calculate metrics with validation
        metrics = calculate_metrics(df)
        
        # Display validation messages
        if metrics['validation_messages']:
            with st.expander("🔍 Data Validation Report", expanded=True):
                for message in metrics['validation_messages']:
                    if "⚠️" in message:
                        st.warning(message)
                    elif "✅" in message:
                        st.success(message)
                    else:
                        st.info(message)
        
        # Use cleaned dataframe for visualizations
        df_clean = metrics['cleaned_df']
        
        st.subheader("Overview")
        create_overview_cards(metrics)
        
        st.subheader("Key Performance Indicators")
        
        # KPI selection controls
        col1, col2 = st.columns([3, 1])
        with col1:
            available_kpis = ['CTR', 'VCR']
            if metrics['cost'] > 0:
                available_kpis.extend(['CPC'])
            if metrics['conversions'] > 0:
                available_kpis.extend(['CPA'])
            
            # Use session state for KPI selection with fallback to available KPIs
            default_kpis = [kpi for kpi in st.session_state.kpi_settings.get('selected_kpis', ['CTR', 'VCR']) if kpi in available_kpis]
            if not default_kpis:
                default_kpis = ['CTR', 'VCR'] if 'CTR' in available_kpis and 'VCR' in available_kpis else available_kpis[:2]
            
            selected_kpis = st.multiselect(
                "Select KPIs to display:",
                options=available_kpis,
                default=default_kpis,
                help="Choose which Key Performance Indicators to show",
                key="kpi_multiselect"
            )
            
            # Update session state when selection changes
            if selected_kpis != st.session_state.kpi_settings.get('selected_kpis', []):
                st.session_state.kpi_settings['selected_kpis'] = selected_kpis
        
        with col2:
            st.write("")  # Spacing
        
        if selected_kpis:
            create_kpi_cards(metrics, selected_kpis)
        else:
            st.info("Please select at least one KPI to display")
        
        # Add a warning if metrics seem suspicious
        if metrics['ctr'] > 10:
            st.error("⚠️ CTR is unusually high! Check your data for errors.")
        if metrics['vcr'] > 100:
            st.error("⚠️ VCR exceeds 100%! This is impossible - check your data.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            create_device_chart(df_clean)
        
        with col2:
            st.subheader("📈 Daily Performance Trend")
            
            # Trend controls
            trend_col1, trend_col2 = st.columns(2)
            with trend_col1:
                trend_kpi_options = ['Impressions', 'Clicks', 'CTR']
                if metrics['cost'] > 0:
                    trend_kpi_options.extend(['CPC'])
                if metrics['conversions'] > 0:
                    trend_kpi_options.extend(['CPA'])
                if metrics['starts'] > 0:
                    trend_kpi_options.extend(['VCR'])
                
                # Use session state for trend KPI selection
                saved_trend_kpi = st.session_state.kpi_settings.get('trend_kpi', 'Impressions')
                trend_index = trend_kpi_options.index(saved_trend_kpi) if saved_trend_kpi in trend_kpi_options else 0
                
                selected_trend_kpi = st.selectbox(
                    "Select KPI for trend:",
                    options=trend_kpi_options,
                    index=trend_index,
                    help="Choose which metric to show in the daily trend",
                    key="trend_kpi_selectbox"
                )
                
                # Update session state when selection changes
                if selected_trend_kpi != st.session_state.kpi_settings.get('trend_kpi'):
                    st.session_state.kpi_settings['trend_kpi'] = selected_trend_kpi
            
            with trend_col2:
                days_back = st.number_input(
                    "Days to show:",
                    min_value=1,
                    max_value=365,
                    value=14,
                    step=1,
                    help="Number of days to include in the trend analysis"
                )
            
            # Create enhanced daily trend
            create_enhanced_daily_trend(df_clean, selected_trend_kpi, days_back)
        
        create_inventory_chart(df_clean)
        
        # Add the new inventory app ranking section
        create_inventory_app_ranking(df_clean)
        
        with st.expander("Raw Data Preview (Cleaned)"):
            st.caption(f"Showing first 100 rows of {len(df_clean)} valid rows")
            st.dataframe(df_clean.head(100), use_container_width=True)

        # Looker Studio Embed Section
        st.divider()
        st.subheader("📊 Looker Studio Reports")

        looker_config = get_looker_config()

        if looker_config:
            # If current campaign has a Looker URL configured
            if st.session_state.current_campaign and st.session_state.current_campaign in looker_config:
                looker_url = looker_config[st.session_state.current_campaign]
                st.info(f"Showing Looker Studio report for: {st.session_state.current_campaign}")
                render_looker_embed(looker_url)
            else:
                # Show dropdown to select from available reports
                if looker_config:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        selected_report = st.selectbox(
                            "Select a Looker Studio report:",
                            list(looker_config.keys())
                        )
                    with col2:
                        st.write("")  # Spacing

                    if selected_report:
                        render_looker_embed(looker_config[selected_report])
        else:
            with st.expander("Configure Looker Studio Reports"):
                st.write("Add Looker Studio embed URLs to your Streamlit secrets:")
                st.code("""
[LOOKER_URLS]
"Campaign Name 1" = "https://lookerstudio.google.com/embed/reporting/your-report-id/page/pageId"
"Campaign Name 2" = "https://lookerstudio.google.com/embed/reporting/another-report-id/page/pageId"
                """, language="toml")

                st.write("**Or configure dynamically:**")

                # Dynamic configuration
                if 'looker_urls' not in st.session_state:
                    st.session_state.looker_urls = {}

                new_campaign_name = st.text_input("Campaign Name")
                new_looker_url = st.text_input("Looker Studio Embed URL")

                if st.button("Add Looker URL") and new_campaign_name and new_looker_url:
                    st.session_state.looker_urls[new_campaign_name] = new_looker_url
                    st.success(f"Added Looker URL for {new_campaign_name}")
                    st.rerun()

                if st.session_state.looker_urls:
                    st.write("**Currently configured URLs:**")
                    for name, url in st.session_state.looker_urls.items():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.text(f"{name}: {url[:50]}...")
                        with col2:
                            if st.button("Remove", key=f"remove_{name}"):
                                del st.session_state.looker_urls[name]
                                st.rerun()

    else:
        st.info("Please upload a CSV file to view the dashboard")
        
        st.subheader("Expected Data Format")
        st.write("Your CSV should contain columns such as:")
        st.code("""
Date, Insertion Order, Line Item, Inventory Source, Device Type,
Impressions, Clicks, Starts (Video), Complete Views (Video), etc.
        """)
        
        st.warning("""
        ⚠️ Note: This dashboard automatically validates and cleans your data by:
        - Removing rows with corrupted date fields
        - Filtering out rows with unrealistic values (>10M impressions)
        - Removing rows where clicks exceed impressions
        - Removing rows where video completes exceed starts
        """)

if __name__ == "__main__":
    main()
