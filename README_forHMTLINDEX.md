# Campaign Reports Dashboard

A simple React app that wraps your Looker Studio reports in a nicer UI with review tracking.

## Features

- **Home page** showing all campaigns as cards
- **Review status** tracking (stored in browser localStorage)
- **Quick navigation** between reports
- **Loading indicator** while reports load
- **Stats** showing pending vs reviewed

## Setup

### 1. Open the file

Just double-click `index.html` to open in your browser. No server needed.

### 2. Get your Looker Studio embed URLs

For each report in Looker Studio:

1. Open your report
2. Click **File** → **Embed report**
3. Enable embedding if prompted
4. Copy the embed URL (looks like: `https://lookerstudio.google.com/embed/reporting/abc123/page/xyz`)

### 3. Configure your campaigns

Edit the `CAMPAIGNS` array at the top of the `<script>` section:

```javascript
const CAMPAIGNS = [
    {
        id: "barclaycard-nca",           // unique ID (no spaces)
        name: "Barclaycard NCA",          // display name
        client: "Barclaycard",            // client badge
        description: "New Customer...",   // short description
        lookerUrl: "https://lookerstudio.google.com/embed/...",  // YOUR EMBED URL
        color: "blue"                     // blue, red, green, purple, or orange
    },
    // Add more...
];
```

### 4. That's it!

Refresh the page and your campaigns will appear.

## Usage

- **Click a card** → Opens that report
- **Mark as Reviewed** → Tracks that you've checked it (saves to browser)
- **Back** → Return to home page
- Review status persists between sessions (stored in localStorage)

## Customization

### Add more colors

Find the `COLORS` object and add more:

```javascript
const COLORS = {
    blue: { bg: "bg-blue-500", light: "bg-blue-100", text: "text-blue-700" },
    // Add your own...
    pink: { bg: "bg-pink-500", light: "bg-pink-100", text: "text-pink-700" },
};
```

### Change the title

Edit the `<title>` tag and the `<h1>` in the HomePage component.

### Deploy online

To share with others:
1. Host on GitHub Pages (free)
2. Or drop in any static file host (Netlify, Vercel, S3)

## Notes

- This is a single HTML file with no build step required
- Uses Tailwind CSS for styling (loaded from CDN)
- Uses React 18 (loaded from CDN)
- All data stored locally in browser - no backend needed
