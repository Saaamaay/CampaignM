# Quick Start Guide

## What's New?

Your Streamlit Campaign Manager Dashboard has been upgraded with:

1. **BigQuery Integration** - Load data directly from BigQuery instead of uploading CSVs
2. **Looker Studio Embeds** - View Looker reports directly in the dashboard
3. **Unified Experience** - All your campaign data, analytics, and reports in one place

## Getting Started in 5 Minutes

### Step 1: Install Dependencies

```bash
cd campaign_manager
pip install -r requirements.txt
```

New packages added:
- `google-cloud-bigquery` - For BigQuery connections
- `db-dtypes` - For BigQuery data type support

### Step 2: Choose Your Setup Path

You have two options:

#### Option A: Start Simple (CSV Only)
If you want to keep using CSV uploads for now, **no configuration needed**! Just run:

```bash
streamlit run page.py
```

Your app will work exactly as before. You can add BigQuery and Looker later.

#### Option B: Full Setup (BigQuery + Looker)
Follow the detailed setup guide: [BIGQUERY_LOOKER_SETUP.md](./BIGQUERY_LOOKER_SETUP.md)

### Step 3: Test the New Features

#### Test BigQuery (if configured)
1. Start the app
2. Go to the **☁️ BigQuery** tab
3. Select a campaign
4. Click "Load Campaign from BigQuery"

#### Test Looker Embeds (if configured)
1. Load any campaign data
2. Scroll to the bottom
3. Look for the **📊 Looker Studio Reports** section
4. Your report should appear automatically

## Configuration Summary

### Required for BigQuery

Create `.streamlit/secrets.toml` with:

```toml
BIGQUERY_PROJECT_ID = "your-project"
BIGQUERY_DATASET_ID = "your-dataset"
BIGQUERY_TABLE_ID = "your-table"

[GCP_SERVICE_ACCOUNT]
# ... service account JSON content
```

See `.streamlit/secrets.toml.example` for the full template.

### Required for Looker Studio

Add to `.streamlit/secrets.toml`:

```toml
[LOOKER_URLS]
"Your Campaign Name" = "https://lookerstudio.google.com/embed/reporting/..."
```

Or configure dynamically in the app (temporary, lost on restart).

## What Works Without Configuration?

Everything you had before still works:
- ✅ CSV uploads
- ✅ Campaign saving to GitHub
- ✅ All analytics and charts
- ✅ Xandr domain uploads
- ✅ Inventory source mappings

## Architecture Changes

### New Files Created
```
campaign_manager/
├── bigquery_connector.py              # NEW - BigQuery client
├── BIGQUERY_LOOKER_SETUP.md           # NEW - Setup guide
├── QUICKSTART.md                      # NEW - This file
└── .streamlit/
    └── secrets.toml.example           # NEW - Configuration template
```

### Modified Files
```
campaign_manager/
├── page.py                            # UPDATED - Added BigQuery & Looker
├── requirements.txt                   # UPDATED - Added BigQuery packages
├── README.md                          # UPDATED - New documentation
└── .gitignore                         # UPDATED - Added service account files
```

## Data Flow Comparison

### Old Flow (Still Works)
```
CSV Upload → Streamlit → Analytics → Xandr Export
```

### New Flow (Optional)
```
BigQuery → Streamlit → Analytics → Xandr Export
                    ↓
              Looker Embeds
```

## Key Benefits

### 1. No More Manual Exports
Instead of:
1. Log into DV360
2. Generate report
3. Download CSV
4. Upload to Streamlit

Now:
1. Click "Load from BigQuery"
2. Done!

### 2. Unified Dashboard
Instead of switching between:
- Streamlit for analytics
- Looker Studio for reports
- Xandr for list management

Now everything is in one place.

### 3. Fresh Data Always
BigQuery queries pull the latest data, so you're never looking at stale reports.

## Troubleshooting Quick Fixes

### "BigQuery credentials not configured"
- You haven't set up BigQuery yet (that's okay!)
- Use CSV upload or see BIGQUERY_LOOKER_SETUP.md

### "Could not fetch campaigns from BigQuery"
- Check your table name in secrets.toml
- Verify service account has BigQuery Data Viewer role
- Ensure table has a `Campaign` column

### Looker iframe is blank
- Check the report is publicly accessible
- Verify you're using the embed URL (contains `/embed/`)
- Try opening the URL directly in a browser first

### Import error for bigquery_connector
- Run `pip install -r requirements.txt` again
- Make sure you're in the campaign_manager directory

## Next Steps

1. **Test the app** with CSV upload (no config needed)
2. **Set up BigQuery** if you want automated data loading
3. **Add Looker URLs** for your top campaigns
4. **Push to GitHub** (secrets are already in .gitignore)

## Need Help?

Check the detailed guides:
- [BIGQUERY_LOOKER_SETUP.md](./BIGQUERY_LOOKER_SETUP.md) - Full setup instructions
- [STREAMLIT_SECRETS.md](./STREAMLIT_SECRETS.md) - General secrets configuration
- [README.md](./README.md) - Complete feature documentation

## What If I Don't Want These Features?

No problem! The app is **100% backward compatible**. If you:
- Don't configure BigQuery → CSV upload still works
- Don't configure Looker → Reports section shows config instructions
- Already have saved campaigns → They all still work

You can adopt these features at your own pace or not at all.

## Testing Checklist

Before deploying to production:

- [ ] Install new dependencies (`pip install -r requirements.txt`)
- [ ] Test CSV upload (should work as before)
- [ ] Test saved campaign loading (should work as before)
- [ ] Test Xandr export (should work as before)
- [ ] If using BigQuery: Test query and data loading
- [ ] If using Looker: Test embed display
- [ ] Push changes to GitHub (verify secrets.toml not committed)

## Production Deployment

If you're deploying to Streamlit Cloud:

1. Push your code to GitHub
2. Go to Streamlit Cloud settings
3. Add secrets in the **Secrets** section (same format as local)
4. Redeploy

The app will automatically use cloud secrets instead of local files.

---

**You're all set!** 🎉

Start with `streamlit run page.py` and explore the new features at your own pace.
