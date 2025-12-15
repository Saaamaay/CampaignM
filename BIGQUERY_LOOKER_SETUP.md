# BigQuery & Looker Studio Integration Setup Guide

This guide explains how to set up the new BigQuery and Looker Studio features in the Campaign Manager Dashboard.

## Overview

The dashboard now supports two major enhancements:

1. **BigQuery Integration**: Load campaign data directly from BigQuery instead of uploading CSV files
2. **Looker Studio Embeds**: View Looker Studio reports directly within the dashboard

## BigQuery Setup

### 1. Create a Google Cloud Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Navigate to **IAM & Admin** > **Service Accounts**
3. Click **Create Service Account**
4. Give it a name (e.g., "campaign-dashboard-bq")
5. Grant it the **BigQuery Data Viewer** and **BigQuery Job User** roles
6. Click **Done**

### 2. Create a Service Account Key

1. Click on the service account you just created
2. Go to the **Keys** tab
3. Click **Add Key** > **Create new key**
4. Choose **JSON** format
5. Download the JSON file

### 3. Configure Streamlit Secrets

#### For Local Development:

Create or edit `.streamlit/secrets.toml` in the campaign_manager directory:

```toml
# BigQuery Configuration
BIGQUERY_PROJECT_ID = "your-gcp-project-id"
BIGQUERY_DATASET_ID = "your-dataset-id"
BIGQUERY_TABLE_ID = "your-table-name"

# Service Account Configuration
[GCP_SERVICE_ACCOUNT]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\nyour-private-key\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@your-project.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "your-cert-url"
```

You can copy the values from the JSON key file you downloaded.

#### For Streamlit Cloud:

1. Go to your app settings in Streamlit Cloud
2. Navigate to **Secrets**
3. Add the same content as above in TOML format

### 4. BigQuery Table Schema

Your BigQuery table should have the following columns to work with the dashboard:

```sql
CREATE TABLE `your-project.your-dataset.your-table` (
  Date DATE,
  Campaign STRING,
  Device STRING,
  Impressions INT64,
  Clicks INT64,
  Conversions INT64,
  Cost FLOAT64,
  `App/URL` STRING,
  `Inventory Source` STRING
);
```

**Note**: Column names must match exactly (case-sensitive).

### 5. Using BigQuery in the Dashboard

1. Start the Streamlit app
2. In the **Campaign Management** section, you'll see tabs: **📤 Upload CSV** and **☁️ BigQuery**
3. Click the **☁️ BigQuery** tab
4. Select a campaign from the dropdown (populated from your BigQuery table)
5. Click **Load Campaign from BigQuery**
6. The data will load and all existing features (charts, Xandr exports, etc.) will work normally

---

## Looker Studio Setup

### 1. Create Looker Studio Reports

1. Go to [Looker Studio](https://lookerstudio.google.com)
2. Create your campaign reports
3. For each report, click **Share** > **Embed report**
4. Copy the embed URL (it should look like: `https://lookerstudio.google.com/embed/reporting/your-report-id/page/pageId`)

### 2. Configure Looker URLs

You have two options:

#### Option A: Streamlit Secrets (Recommended for Production)

Add to your `.streamlit/secrets.toml`:

```toml
[LOOKER_URLS]
"Barclaycard NCA" = "https://lookerstudio.google.com/embed/reporting/a0d7ae2c-ca7b-4061-9e27-277a85fb767b/page/QdZiF"
"HSBC Rewards" = "https://lookerstudio.google.com/embed/reporting/your-second-report-id/page/pageId"
"Lloyds Savings" = "https://lookerstudio.google.com/embed/reporting/your-third-report-id/page/pageId"
```

**Important**: The campaign name (left side) should match your campaign names in BigQuery or your saved campaigns.

#### Option B: Dynamic Configuration (Easier for Testing)

1. Run the dashboard
2. Load any campaign data
3. Scroll to the **📊 Looker Studio Reports** section
4. Click the expander **Configure Looker Studio Reports**
5. Enter a campaign name and Looker URL
6. Click **Add Looker URL**

**Note**: Dynamically configured URLs are stored in session state and will be lost when you restart the app. Use Streamlit secrets for persistence.

### 3. Viewing Looker Reports

Once configured:

1. Load a campaign (via CSV or BigQuery)
2. Scroll to the **📊 Looker Studio Reports** section at the bottom
3. If your current campaign matches a configured Looker URL, it will display automatically
4. Otherwise, use the dropdown to select a report to view

---

## Complete Example Configuration

Here's a complete `.streamlit/secrets.toml` example:

```toml
# GitHub Storage (existing)
GITHUB_TOKEN = "ghp_your_github_token"
GITHUB_OWNER = "your-username"
GITHUB_REPO = "campaign_manager"

# Xandr API (existing)
XANDR_USERNAME = "username@member_id"
XANDR_KEY_NAME = "your-key-name"
XANDR_PRIVATE_KEY = """-----BEGIN PRIVATE KEY-----
your-xandr-private-key
-----END PRIVATE KEY-----"""

# BigQuery Configuration (NEW)
BIGQUERY_PROJECT_ID = "my-project-12345"
BIGQUERY_DATASET_ID = "campaign_data"
BIGQUERY_TABLE_ID = "dv360_performance"

# GCP Service Account (NEW)
[GCP_SERVICE_ACCOUNT]
type = "service_account"
project_id = "my-project-12345"
private_key_id = "abc123..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "campaign-dashboard-bq@my-project-12345.iam.gserviceaccount.com"
client_id = "123456789"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/..."

# Looker Studio URLs (NEW)
[LOOKER_URLS]
"Barclaycard NCA" = "https://lookerstudio.google.com/embed/reporting/a0d7ae2c-ca7b-4061-9e27-277a85fb767b/page/QdZiF"
"HSBC Rewards" = "https://lookerstudio.google.com/embed/reporting/def456.../page/xyz"
```

---

## Testing the Integration

### Test BigQuery Connection

1. Start the app
2. Go to the **☁️ BigQuery** tab
3. You should see your project, dataset, and table listed
4. If campaigns don't load, check:
   - Service account permissions
   - Table name and schema
   - Secret configuration

### Test Looker Embeds

1. Add at least one Looker URL (via secrets or dynamic config)
2. Load any campaign
3. Scroll to **📊 Looker Studio Reports**
4. The report should appear in an iframe
5. If it doesn't load, check:
   - The URL is an embed URL (contains `/embed/`)
   - The report is set to "Anyone with the link can view"
   - No CORS issues (Looker Studio should allow embedding)

---

## Troubleshooting

### BigQuery Issues

**Error: "BigQuery credentials not configured"**
- Check that `GCP_SERVICE_ACCOUNT` is in your secrets
- Verify the JSON structure is correct
- Ensure no extra quotes or formatting issues

**Error: "Could not fetch campaigns from BigQuery"**
- Verify the table exists: `project_id.dataset_id.table_id`
- Check service account has `BigQuery Data Viewer` role
- Ensure the table has a `Campaign` column

**Error: "Permission denied"**
- Add `BigQuery Job User` role to your service account
- Ensure the service account is from the same project as your BigQuery dataset

### Looker Studio Issues

**Iframe appears but shows "Report not found"**
- Verify the report sharing settings (must be public or accessible)
- Check the URL is the embed URL, not the regular viewing URL

**Iframe is blank**
- Check browser console for CORS errors
- Ensure the report allows embedding
- Try opening the URL directly in a new tab to verify it works

---

## Architecture Notes

### How It Works

1. **BigQuery Data Flow**:
   ```
   BigQuery Table → query_campaign_data() → pandas DataFrame → Existing Analytics
   ```

2. **Looker Embeds**:
   ```
   Streamlit Secrets/Session State → render_looker_embed() → HTML iframe → Display
   ```

3. **Data Source Priority**:
   - Uploaded CSV takes precedence
   - Then BigQuery data (if loaded)
   - Then session state (previously loaded data)

### Files Added/Modified

**New Files**:
- `bigquery_connector.py` - BigQuery client and query functions
- `BIGQUERY_LOOKER_SETUP.md` - This guide

**Modified Files**:
- `page.py` - Added tabs for data sources and Looker embed section
- `requirements.txt` - Added `google-cloud-bigquery` and `db-dtypes`

### Benefits

1. **No more manual CSV uploads** - Query fresh data from BigQuery
2. **Unified dashboard** - See Looker reports alongside your analytics
3. **Same Xandr integration** - All export features work with BigQuery data
4. **Flexible** - Still supports CSV uploads for offline work

---

## Next Steps

After setup:

1. Test with a small campaign to verify everything works
2. Configure Looker URLs for all your campaigns
3. Set up automated BigQuery data pipelines (if needed)
4. Consider adding more columns to your BigQuery schema for richer analysis

For questions or issues, check the Streamlit logs or open an issue in the GitHub repo.
