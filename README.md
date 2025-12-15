# Campaign Manager Dashboard

A comprehensive Streamlit-based campaign performance analysis dashboard with DV360, BigQuery, Looker Studio, and Xandr integration.

## Features

### Core Analytics
- **Multi-KPI Overview**: Track impressions, clicks, conversions, CTR, VCR, CPC, and CPA
- **Device Performance**: Breakdown by device type with interactive charts
- **Daily Trends**: Customizable trend analysis with multiple KPI options
- **Inventory Source Analysis**: Detailed performance by inventory source
- **App/URL Performance**: Granular analysis of individual apps and URLs

### Data Sources
- **CSV Upload**: Traditional file upload for DV360 reports
- **BigQuery Integration** (NEW): Query campaign data directly from BigQuery
- **GitHub Storage**: Persistent campaign data storage via GitHub API

### Visualization
- **Looker Studio Embeds** (NEW): View Looker Studio reports directly in the dashboard
- **Interactive Charts**: Powered by Plotly for rich data exploration
- **Real-time Validation**: Automatic data cleaning and validation

### Xandr Integration
- **Deal ID Mapping**: Map inventory sources to Xandr Deal IDs
- **Inventory List Management**: Rotate and update allowlists/blocklists
- **Bulk Domain Upload**: Upload filtered URLs/domains directly to Xandr
- **Line Item Lookup**: Automatic Deal → Line Item → Profile resolution

## New Features (Latest Update)

### 1. BigQuery Data Source
Load campaign data directly from BigQuery instead of uploading CSV files.

**Benefits**:
- No manual CSV exports
- Always fresh data
- Automated data pipelines
- Works with all existing features

See [BIGQUERY_LOOKER_SETUP.md](./BIGQUERY_LOOKER_SETUP.md) for setup instructions.

### 2. Looker Studio Embeds
View your Looker Studio reports directly within the dashboard.

**Benefits**:
- Unified view of all campaign data
- No context switching between tools
- Campaign-specific report matching
- Configurable report library

See [BIGQUERY_LOOKER_SETUP.md](./BIGQUERY_LOOKER_SETUP.md) for setup instructions.

## Quick Start

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure Streamlit secrets (see [STREAMLIT_SECRETS.md](./STREAMLIT_SECRETS.md))

4. Run the app:
   ```bash
   streamlit run page.py
   ```

### Configuration Files

- `.streamlit/secrets.toml` - API keys, credentials, BigQuery config
- `saved_campaigns/` - Campaign data storage (GitHub)
- `inventory_source_mappings.json` - Deal ID mappings

## Setup Guides

- **General Setup**: [STREAMLIT_SECRETS.md](./STREAMLIT_SECRETS.md)
- **BigQuery & Looker**: [BIGQUERY_LOOKER_SETUP.md](./BIGQUERY_LOOKER_SETUP.md)

## Usage

### Load Data

**Option 1: Upload CSV**
1. Click the **📤 Upload CSV** tab
2. Select your DSP (DV360)
3. Upload your campaign CSV file

**Option 2: Query BigQuery**
1. Click the **☁️ BigQuery** tab
2. Select a campaign from the dropdown
3. Click **Load Campaign from BigQuery**

**Option 3: Load Saved Campaign**
1. Select a saved campaign from the dropdown
2. Click **Load Campaign**

### View Analytics

Once data is loaded:
- View overview KPIs at the top
- Explore device performance breakdown
- Analyze daily trends (configurable KPIs)
- Review inventory source performance
- Filter and analyze App/URL data

### Export to Xandr

1. Navigate to the **Inventory App/URL Ranking** section
2. Select an inventory source
3. Filter URLs by performance metrics
4. Click **🚀 Upload to Xandr** to create/update an allowlist

### View Looker Reports

1. Scroll to the **📊 Looker Studio Reports** section
2. If configured, your campaign's report will display automatically
3. Or select a report from the dropdown

## Architecture

```
campaign_manager/
├── page.py                          # Main Streamlit app
├── bigquery_connector.py            # BigQuery integration (NEW)
├── xandr_auth.py                    # Xandr JWT authentication
├── inventory_list_manager.py        # Xandr list rotation
├── deal_lineitem_cache.py           # Deal ID mapping + GitHub sync
├── deal_inventory_lookup.py         # Inventory list lookup
├── requirements.txt                 # Python dependencies
└── saved_campaigns/                 # Campaign storage
    ├── inventory_source_mappings.json
    └── [campaign_name].json(.gz)
```

## Data Flow

### CSV Upload Flow
```
CSV File → Clean → Validate → Calculate Metrics → Display Charts → Export to Xandr
```

### BigQuery Flow
```
BigQuery → query_campaign_data() → DataFrame → Calculate Metrics → Display Charts → Export to Xandr
```

### Looker Embeds
```
Streamlit Secrets → get_looker_config() → render_looker_embed() → iframe Display
```

## Technologies

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Cloud**: BigQuery, GitHub API, Xandr API
- **Auth**: JWT (Xandr), Service Account (BigQuery)

## Potential Future Enhancements

- [ ] Add automatic report uploads from emails
- [ ] Add a client view - Simplified dashboard they can look at
- [ ] Change Inventory Source Performance to adjust to the selected timeframe for the dashboard
- [ ] Multi-DSP support (expand beyond DV360)
- [ ] Automated anomaly detection
- [ ] Scheduled BigQuery sync
- [ ] Custom report builder

## Troubleshooting

See [BIGQUERY_LOOKER_SETUP.md](./BIGQUERY_LOOKER_SETUP.md#troubleshooting) for common issues.

## License

Internal tool - proprietary use only.
