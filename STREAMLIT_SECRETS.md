# Streamlit Cloud Secrets Configuration

This document explains how to configure secrets in Streamlit Cloud for the Campaign Manager dashboard.

## Required Secrets

Add the following to your Streamlit Cloud app's secrets (Settings → Secrets):

```toml
# GitHub Configuration (for campaign data storage)
GITHUB_TOKEN = "ghp_your_github_token_here"
GITHUB_OWNER = "Entity-X"
GITHUB_REPO = "campaign_manager"

# Xandr API Configuration
XANDR_USERNAME = "your_username@member_id"  # e.g., "John@12345"
XANDR_KEY_NAME = "your-api-key-name"

# Xandr Private Key (paste the entire content of your private key file)
XANDR_PRIVATE_KEY = """-----BEGIN PRIVATE KEY-----

-----END PRIVATE KEY-----"""
```

## How to Get Your Private Key Content

1. Open your `my-api-key` file (the private key file, not the .pub file)
2. Copy the **entire contents** including the `-----BEGIN PRIVATE KEY-----` and `-----END PRIVATE KEY-----` lines
3. Paste it into the `XANDR_PRIVATE_KEY` field in Streamlit secrets using triple quotes as shown above

## Security Notes

- **Never commit** the actual `auth_config.py` file with credentials to GitHub
- The `.gitignore` file is configured to exclude:
  - `auth_config.py` (your local credentials)
  - `my-api-key` (your private key file)
  - `.xandr_token_cache.json` (cached auth tokens)
- Only commit `auth_config.example.py` as a template

## Local Development

For local development, the app will automatically fall back to reading from:
- Local `auth_config.py` file (with your credentials)
- Local `my-api-key` file (your private key)

## Testing

After configuring secrets in Streamlit Cloud:
1. Deploy your app
2. Click the "🚀 Upload to Xandr" button in the App/URL Performance Analysis section
3. Check the logs for authentication success/errors
