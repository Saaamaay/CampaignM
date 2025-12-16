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

# Campaign metadata management (simple, no data storage)
def get_campaigns_metadata():
    """Get list of campaign metadata from session state"""
    if 'campaigns' not in st.session_state:
        st.session_state.campaigns = []
    return st.session_state.campaigns

def add_campaign_metadata(name, bigquery_table, project_id, dataset_id, dsp="DV360"):
    """Add a new campaign metadata"""
    if 'campaigns' not in st.session_state:
        st.session_state.campaigns = []

    campaign = {
        'name': name,
        'bigquery_table': bigquery_table,
        'project_id': project_id,
        'dataset_id': dataset_id,
        'dsp': dsp,
        'created_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.campaigns.append(campaign)
    return True

def delete_campaign_metadata(name):
    """Delete a campaign by name"""
    if 'campaigns' in st.session_state:
        st.session_state.campaigns = [c for c in st.session_state.campaigns if c['name'] != name]
        return True
    return False

def get_campaign_by_name(name):
    """Get campaign metadata by name"""
    campaigns = get_campaigns_metadata()
    for campaign in campaigns:
        if campaign['name'] == name:
            return campaign
    return None

def inject_looker_style_css():
    """Inject custom CSS for Looker-style dashboard appearance"""
    st.markdown("""
    <style>
    /* Compact metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 300 !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.7rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        color: #666 !important;
    }

    /* Proper spacing to prevent cutoff */
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 1rem !important;
    }

    /* Prevent text cutoff */
    .stMarkdown, .stTitle, h1, h2, h3 {
        margin-top: 1rem !important;
        margin-bottom: 1rem !important;
        line-height: 1.4 !important;
    }

    /* Fix selectbox text cutoff */
    .stSelectbox > div > div {
        padding: 0.5rem !important;
        min-height: 2.5rem !important;
    }

    /* Fix input and selectbox internal padding */
    .stSelectbox input, .stSelectbox div[data-baseweb="select"] > div {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }

    /* Ensure buttons have proper spacing */
    .stButton button {
        margin-top: 0.5rem !important;
    }

    /* Fix number input text cutoff */
    .stNumberInput > div > div > input {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }

    /* Ensure multiselect has proper spacing */
    .stMultiSelect > div > div {
        padding: 0.5rem !important;
        min-height: 2.5rem !important;
    }

    /* KPI indicator dots */
    .kpi-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
    }
    .kpi-green { background-color: #22c55e; }
    .kpi-yellow { background-color: #eab308; }
    .kpi-red { background-color: #ef4444; }

    /* Industry benchmark text */
    .benchmark-text {
        font-size: 0.7rem;
        color: #999;
        margin-top: -10px;
    }

    /* Summary stats styling */
    .summary-stat-row {
        display: flex;
        justify-content: space-between;
        padding-top: 0.75rem;
        border-top: 1px solid #eee;
        margin-top: 0.75rem;
    }

    /* Section dividers */
    hr {
        margin: 1rem 0 !important;
        border-color: #eee !important;
    }

    /* Compact selectbox */
    .stSelectbox > div > div {
        padding: 0.25rem 0.5rem !important;
    }

    /* Chart title styling */
    .chart-title {
        font-size: 0.9rem;
        font-weight: 500;
        color: #333;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

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
    """Create compact overview cards in Looker style"""
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
    """Create KPI cards with colored indicators like Looker"""
    if selected_kpis is None:
        selected_kpis = ['CTR', 'VCR']

    available_kpis = {
        'CTR': {'value': metrics['ctr'], 'format': ':.4f', 'unit': '%', 'industry': '0.05-0.10%', 'good_threshold': 0.10},
        'VCR': {'value': metrics['vcr'], 'format': ':.2f', 'unit': '%', 'industry': '70-85%', 'good_threshold': 70},
        'CPC': {'value': metrics['cpc'], 'format': ':.2f', 'unit': '', 'industry': 'Varies by industry', 'good_threshold': 5},
        'CPA': {'value': metrics['cpa'], 'format': ':.2f', 'unit': '', 'industry': 'Varies by industry', 'good_threshold': 50}
    }

    valid_kpis = [kpi for kpi in selected_kpis if kpi in available_kpis and available_kpis[kpi]['value'] > 0]

    if not valid_kpis:
        st.info("No KPI data available for selected metrics")
        return

    num_cols = min(len(valid_kpis), 4)
    cols = st.columns(num_cols)

    for i, kpi in enumerate(valid_kpis[:4]):
        kpi_data = available_kpis[kpi]
        value = kpi_data['value']

        # Determine indicator color
        if kpi in ['CTR']:
            color_class = "kpi-green" if value >= kpi_data['good_threshold'] else "kpi-red"
        elif kpi == 'VCR':
            color_class = "kpi-green" if value >= kpi_data['good_threshold'] else "kpi-yellow"
        else:
            color_class = "kpi-green" if value < kpi_data['good_threshold'] else "kpi-yellow"

        # Format value
        if kpi_data['format'] == ':.4f':
            formatted_value = f"{value:.4f}{kpi_data['unit']}"
        else:
            formatted_value = f"{value:.2f}{kpi_data['unit']}"

        with cols[i % num_cols]:
            # Custom HTML for indicator dot
            st.markdown(f'<span class="kpi-indicator {color_class}"></span> **{kpi}**', unsafe_allow_html=True)
            st.markdown(f"<h2 style='margin:0; font-weight:300;'>{formatted_value}</h2>", unsafe_allow_html=True)
            st.markdown(f'<p class="benchmark-text">Industry average: {kpi_data["industry"]}</p>', unsafe_allow_html=True)

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
    """Create compact device type pie chart like Looker"""
    if 'Device Type' not in df.columns:
        return

    device_data = df.groupby('Device Type').agg({
        'Impressions': 'sum',
        'Clicks': 'sum'
    }).reset_index()

    # Looker-style colors
    colors = ['#4285f4', '#a8c7fa', '#ea4335', '#fbcfe8', '#34a853']

    fig = go.Figure(data=[go.Pie(
        labels=device_data['Device Type'],
        values=device_data['Impressions'],
        hole=0,
        textinfo='percent',
        textposition='inside',
        insidetextorientation='horizontal',
        hovertemplate='<b>%{label}</b><br>Impressions=%{value:,.0f}<extra></extra>',
        marker=dict(colors=colors, line=dict(color='#ffffff', width=2))
    )])

    fig.update_layout(
        title=None,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=10)
        ),
        margin=dict(l=10, r=80, t=30, b=10),
        height=280
    )

    st.markdown("**Impressions by Device Type**")
    st.plotly_chart(fig, use_container_width=True)

def create_enhanced_daily_trend(df, selected_kpi='Impressions', days_back=14):
    """Create enhanced daily trend with compact summary stats like Looker"""
    if 'Date' not in df.columns:
        st.warning("No date column found in data")
        return

    df_clean = df[~df['Date'].astype(str).str.contains('Filter|:', case=False, na=False)].copy()

    try:
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
        df_clean = df_clean.dropna(subset=['Date'])
    except:
        st.error("Unable to parse date column")
        return

    if days_back > 0:
        end_date = df_clean['Date'].max()
        start_date = end_date - pd.Timedelta(days=days_back)
        df_clean = df_clean[df_clean['Date'] >= start_date]

    if len(df_clean) == 0:
        st.warning(f"No data available for the last {days_back} days")
        return

    # Prepare aggregation
    agg_dict = {}
    if 'Impressions' in df_clean.columns:
        agg_dict['Impressions'] = 'sum'
    if 'Clicks' in df_clean.columns:
        agg_dict['Clicks'] = 'sum'

    cost_columns = ['Total Media Cost (Advertiser Currency)', 'Media Cost', 'Cost', 'Spend', 'Revenue (Partner Currency)']
    for col in cost_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            agg_dict['Cost'] = (col, 'sum')
            break

    conversion_columns = ['Total Conversions', 'Conversions', 'Post-Click Conversions', 'Post-View Conversions']
    for col in conversion_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            agg_dict['Conversions'] = (col, 'sum')
            break

    if 'Starts (Video)' in df_clean.columns:
        agg_dict['Video Starts'] = ('Starts (Video)', 'sum')
    if 'Complete Views (Video)' in df_clean.columns:
        agg_dict['Video Completes'] = ('Complete Views (Video)', 'sum')

    final_agg_dict = {}
    for new_name, agg_info in agg_dict.items():
        if isinstance(agg_info, tuple):
            col_name, func = agg_info
            final_agg_dict[col_name] = func
        else:
            final_agg_dict[new_name] = agg_info

    if not final_agg_dict:
        st.warning("No aggregatable columns found in the data")
        return

    daily_data = df_clean.groupby('Date').agg(final_agg_dict).reset_index()

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

    if selected_kpi not in daily_data.columns:
        available_kpis = [col for col in daily_data.columns if col != 'Date']
        st.warning(f"'{selected_kpi}' not available. Available: {', '.join(available_kpis)}")
        selected_kpi = available_kpis[0] if available_kpis else 'Impressions'

    # Create Looker-style line chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=daily_data['Date'],
        y=daily_data[selected_kpi],
        mode='lines+markers',
        name=selected_kpi,
        line=dict(color='#4285f4', width=2),
        marker=dict(size=6, color='#4285f4'),
        hovertemplate=f'<b>%{{x|%b %d, %Y}}</b><br>{selected_kpi}: %{{y:,.2f}}<extra></extra>'
    ))

    fig.update_layout(
        title=f'{selected_kpi} Trend - Last {days_back} Days',
        title_font_size=14,
        xaxis_title='Date',
        yaxis_title=selected_kpi,
        hovermode='x unified',
        margin=dict(l=40, r=20, t=40, b=40),
        height=280,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#f5f5f5',
        tickformat='%b %d\n%Y'
    )

    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#f5f5f5'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Compact summary stats row (like Looker)
    unit = "%" if selected_kpi in ['CTR', 'VCR'] else ""
    avg_val = daily_data[selected_kpi].mean()
    peak_val = daily_data[selected_kpi].max()
    trend_val = daily_data[selected_kpi].iloc[-1] - daily_data[selected_kpi].iloc[0] if len(daily_data) > 1 else 0
    trend_icon = "📈" if trend_val >= 0 else "📉"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Days of Data", len(daily_data))
    with col2:
        st.metric("Average Impressions", f"{avg_val:,.2f}{unit}")
    with col3:
        st.metric("Peak Impressions", f"{peak_val:,.2f}{unit}")
    with col4:
        st.metric("Trend", f"{trend_icon} {trend_val:+,.0f}")

def create_line_item_performance_table(df_clean, metrics):
    """Create Line Item performance table with conversions and CPA like Looker"""
    if 'Line Item' not in df_clean.columns:
        return

    agg_dict = {'Impressions': 'sum', 'Clicks': 'sum'}

    cost_columns = ['Total Media Cost (Advertiser Currency)', 'Media Cost', 'Cost', 'Spend']
    cost_col = None
    for col in cost_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            agg_dict[col] = 'sum'
            cost_col = col
            break

    conversion_columns = ['Total Conversions', 'Conversions', 'Post-Click Conversions', 'Post-View Conversions']
    conv_col = None
    for col in conversion_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            agg_dict[col] = 'sum'
            conv_col = col
            break

    if not cost_col or not conv_col:
        return

    line_item_data = df_clean.groupby('Line Item').agg(agg_dict).reset_index()

    # Calculate CPA
    line_item_data['Total_Conversions'] = line_item_data[conv_col]
    line_item_data['CPA'] = (line_item_data[cost_col] / line_item_data[conv_col]).round(2)
    line_item_data['CPA'] = line_item_data['CPA'].replace([float('inf'), -float('inf')], 0).fillna(0)

    # Sort by conversions
    line_item_data = line_item_data.sort_values('Total_Conversions', ascending=False)

    # Display
    st.markdown("### Line Item Performance")

    # Summary metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CPA", f"£{metrics['cpa']:,.2f}")
    with col2:
        st.metric("Media Cost", f"£{metrics['cost']:,.2f}")
    with col3:
        st.metric("Total Conversions", f"{metrics['conversions']:,}")

    # Table
    display_df = line_item_data[['Line Item', 'Total_Conversions', 'CPA']].head(6)
    display_df.columns = ['Line_Item', 'Total_Conversions', 'CPA']
    st.dataframe(display_df, use_container_width=True, hide_index=True)

def create_creative_size_chart(df_clean):
    """Create creative size distribution donut chart"""
    if 'Creative Size' not in df_clean.columns:
        return

    size_data = df_clean.groupby('Creative Size').agg({'Impressions': 'sum'}).reset_index()
    size_data = size_data.sort_values('Impressions', ascending=False).head(5)

    colors = ['#4285f4', '#f4b400', '#9c27b0', '#34a853', '#ea4335']

    fig = go.Figure(data=[go.Pie(
        labels=size_data['Creative Size'],
        values=size_data['Impressions'],
        hole=0.5,
        textinfo='percent',
        textposition='outside',
        marker=dict(colors=colors, line=dict(color='#ffffff', width=2))
    )])

    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
        margin=dict(l=10, r=100, t=30, b=10),
        height=220
    )

    st.plotly_chart(fig, use_container_width=True)

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


def show_home_page():
    """Display home page with dashboard of all campaigns as cards"""
    st.title("📊 Campaign Manager Dashboard")
    st.caption("Select a campaign to view details or create a new one")

    # Get all campaign metadata
    campaigns = get_campaigns_metadata()

    # Stats section
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Campaigns", len(campaigns))
    with col2:
        st.metric("Active", len(campaigns))
    with col3:
        bq_config = get_bigquery_config()
        st.metric("Data Source", "BigQuery" if all(bq_config.values()) else "Not Configured")

    st.divider()

    # Add new campaign section
    with st.expander("➕ Add New Campaign", expanded=False):
        st.subheader("Link BigQuery Table to Campaign")

        bq_config = get_bigquery_config()
        if all(bq_config.values()):
            st.info(f"Project: {bq_config['project_id']} | Dataset: {bq_config['dataset_id']}")

            # Get available tables
            campaigns_list = get_available_campaigns(
                bq_config['project_id'],
                bq_config['dataset_id']
            )

            if campaigns_list:
                col1, col2 = st.columns([2, 1])

                with col1:
                    selected_table = st.selectbox(
                        "Select BigQuery Table:",
                        campaigns_list,
                        key="new_bq_table"
                    )

                with col2:
                    campaign_name = st.text_input(
                        "Campaign Name:",
                        value=selected_table,
                        key="new_campaign_name",
                        help="Give this campaign a friendly name"
                    )

                if st.button("Create Campaign Card", type="primary"):
                    if campaign_name:
                        # Check if campaign with this name already exists
                        existing = get_campaign_by_name(campaign_name)
                        if existing:
                            st.error(f"Campaign '{campaign_name}' already exists!")
                        else:
                            # Just save the metadata
                            add_campaign_metadata(
                                campaign_name,
                                selected_table,
                                bq_config['project_id'],
                                bq_config['dataset_id'],
                                "DV360"
                            )
                            st.success(f"✅ Created campaign card: {campaign_name}")
                            st.rerun()
                    else:
                        st.warning("Please enter a campaign name")
            else:
                st.warning("No tables found in BigQuery dataset")
        else:
            st.info("Configure BigQuery credentials in Streamlit secrets to use this feature")
            st.code("""
[BIGQUERY]
BIGQUERY_PROJECT_ID = "your-project-id"
BIGQUERY_DATASET_ID = "your-dataset-id"
            """, language="toml")

    st.divider()

    # Display campaign cards
    if campaigns:
        st.subheader("Your Campaigns")

        # Display cards in a grid (3 columns)
        num_cols = 3
        for i in range(0, len(campaigns), num_cols):
            cols = st.columns(num_cols)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(campaigns):
                    campaign = campaigns[idx]
                    with col:
                        # Create card
                        with st.container():
                            st.markdown(f"### {campaign['name']}")

                            # Display campaign metadata
                            st.caption(f"📅 {campaign['created_date']}")
                            st.caption(f"📊 Table: {campaign['bigquery_table']}")
                            st.caption(f"🔧 DSP: {campaign['dsp']}")

                            col_a, col_b = st.columns(2)
                            with col_a:
                                if st.button("View", key=f"view_{campaign['name']}", type="primary", use_container_width=True):
                                    st.session_state.current_view = "campaign"
                                    st.session_state.selected_campaign = campaign['name']
                                    st.rerun()
                            with col_b:
                                if st.button("🗑️", key=f"delete_{campaign['name']}", use_container_width=True):
                                    if delete_campaign_metadata(campaign['name']):
                                        st.success(f"Deleted {campaign['name']}")
                                        st.rerun()

                            st.divider()
    else:
        st.info("No campaigns yet. Create your first campaign card above!")


def show_campaign_overview():
    """Display detailed overview page for a specific campaign"""
    campaign_name = st.session_state.selected_campaign

    # Add spacing at top to prevent cutoff
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    # Top navigation bar
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back to Home"):
            st.session_state.current_view = "home"
            st.rerun()
    with col2:
        st.title(f"📊 {campaign_name}")

    # Get campaign metadata
    campaign_meta = get_campaign_by_name(campaign_name)
    if not campaign_meta:
        st.error(f"Campaign '{campaign_name}' not found")
        return

    # Load fresh data from BigQuery
    with st.spinner(f"Loading data from BigQuery..."):
        df = query_campaign_data(
            campaign_meta['project_id'],
            campaign_meta['dataset_id'],
            campaign_meta['bigquery_table']
        )

    if df is None or df.empty:
        st.error("Failed to load campaign data from BigQuery")
        return

    dsp = campaign_meta['dsp']

    # Store in session state
    st.session_state.current_campaign = campaign_name
    st.session_state.current_df = df
    st.session_state.current_dsp = dsp

    # Calculate metrics with validation
    metrics = calculate_metrics(df)

    # Use cleaned dataframe for visualizations
    df_clean = metrics['cleaned_df']

    # Inject Looker-style CSS
    inject_looker_style_css()

    st.divider()

    st.subheader("Overview")
    create_overview_cards(metrics)

    st.subheader("Key Performance Indicators")

    # KPI selection
    available_kpis = ['CTR', 'VCR']
    if metrics['cost'] > 0:
        available_kpis.extend(['CPC'])
    if metrics['conversions'] > 0:
        available_kpis.extend(['CPA'])

    default_kpis = [kpi for kpi in st.session_state.kpi_settings.get('selected_kpis', ['CTR', 'VCR']) if kpi in available_kpis]
    if not default_kpis:
        default_kpis = ['CTR', 'VCR'] if 'CTR' in available_kpis else available_kpis[:2]

    selected_kpis = st.multiselect(
        "Select KPIs to display:",
        options=available_kpis,
        default=default_kpis,
        help="Choose which Key Performance Indicators to show",
        key="kpi_multiselect"
    )

    if selected_kpis != st.session_state.kpi_settings.get('selected_kpis', []):
        st.session_state.kpi_settings['selected_kpis'] = selected_kpis

    if selected_kpis:
        create_kpi_cards(metrics, selected_kpis)
    else:
        st.info("Please select at least one KPI to display")

    # Side-by-side charts
    col1, col2 = st.columns([1, 2])

    with col1:
        create_device_chart(df_clean)

    with col2:
        st.markdown("**📈 Daily Performance Trend**")

        # Compact controls in a row
        trend_col1, trend_col2 = st.columns(2)
        with trend_col1:
            trend_kpi_options = ['Impressions', 'Clicks', 'CTR']
            if metrics['cost'] > 0:
                trend_kpi_options.extend(['CPC'])
            if metrics['conversions'] > 0:
                trend_kpi_options.extend(['CPA'])
            if metrics['starts'] > 0:
                trend_kpi_options.extend(['VCR'])

            saved_trend_kpi = st.session_state.kpi_settings.get('trend_kpi', 'Impressions')
            trend_index = trend_kpi_options.index(saved_trend_kpi) if saved_trend_kpi in trend_kpi_options else 0

            selected_trend_kpi = st.selectbox(
                "Select KPI for trend:",
                options=trend_kpi_options,
                index=trend_index,
                help="Choose which metric to show",
                key="trend_kpi_selectbox"
            )

            if selected_trend_kpi != st.session_state.kpi_settings.get('trend_kpi'):
                st.session_state.kpi_settings['trend_kpi'] = selected_trend_kpi

        with trend_col2:
            days_back = st.number_input(
                "Days to show:",
                min_value=1,
                max_value=365,
                value=14,
                step=1,
                help="Number of days to include"
            )

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
        if campaign_name in looker_config:
            looker_url = looker_config[campaign_name]
            st.info(f"Showing Looker Studio report for: {campaign_name}")
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

    # Campaign Management Section (at bottom)
    st.divider()
    st.subheader("Campaign Overview & Management")

    # Campaign info and actions
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

    with col1:
        st.info(f"**DSP:** {dsp}")

    with col2:
        st.info(f"**Rows:** {len(df):,}")

    with col3:
        st.info(f"**Table:** {campaign_meta['bigquery_table']}")

    with col4:
        if st.button("🔄", help="Refresh data from BigQuery", use_container_width=True):
            st.rerun()

    # Inventory source mapping manager
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

    # Display validation messages
    if metrics['validation_messages']:
        with st.expander("🔍 Data Validation Report", expanded=False):
            for message in metrics['validation_messages']:
                if "⚠️" in message:
                    st.warning(message)
                elif "✅" in message:
                    st.success(message)
                else:
                    st.info(message)


def main():
    # Configure GitHub for inventory source mappings
    github_config = get_github_config()
    if all(github_config.values()):
        set_github_config(github_config)

    # Initialize session state
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "home"
    if 'selected_campaign' not in st.session_state:
        st.session_state.selected_campaign = None
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

    # Route to appropriate page
    if st.session_state.current_view == "home":
        show_home_page()
    elif st.session_state.current_view == "campaign":
        show_campaign_overview()


if __name__ == "__main__":
    main()
