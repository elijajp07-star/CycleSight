# Thursday Switch v4.0

Chrome extension that scrapes Depop and eBay stores to extract clothing listings for import into the Thursday platform.

## What's New in v4.0

- **eBay store support** — scrape any eBay store's Buy It Now listings
- **Auto-pagination** — handles multi-page eBay stores (up to 240 items/page)
- **Platform auto-detection** — detects Depop vs eBay from the URL
- **Platform badge** — shows which platform you're on in the popup
- **Rate limit handling** — exponential backoff for eBay 429 responses
- **Unified output** — same JSON format works for both platforms

## Supported Platforms

### Depop
- Shop pages (`depop.com/@username`)
- Single product pages (`depop.com/products/...`)
- Selling Hub
- Search results

### eBay
- Store pages (`ebay.com/str/storename`)
- Store search pages (`ebay.com/sch/...?_ssn=storename`)
- Single item pages (`ebay.com/itm/...`)

## How It Works

### Depop
1. Navigate to a Depop shop page
2. Click the extension → "Extract All Listings"
3. Extension auto-scrolls to load items, then fetches each product page
4. Downloads JSON/CSV when complete

### eBay
1. Navigate to an eBay store page
2. Click the extension → "Extract All Listings"
3. Extension auto-filters to Buy It Now only
4. Paginates through all store pages to collect listing URLs
5. Fetches each product page for full details (with 1-2s delays)
6. Downloads JSON/CSV when complete

## eBay-Specific Notes

- **Buy It Now only** — auctions are automatically filtered out via `LH_BIN=1`
- **Max 240 items per page** — uses `_ipg=240` for efficiency
- **Rate limiting** — 1-2 second delay between product page fetches; 10 second backoff on 429 responses
- **Image resolution** — primary source is `data-zoom-src` (carousel), with JSON-LD `Product.image` as fallback; all upgraded to `s-l1600.jpg`
- **Item specifics** — extracts size, brand, department, type, color from eBay's semantic CSS classes (`ux-labels-values--brand`, etc.)
- **Size fallback** — when item specifics say "See Measurements", parses title for standard sizes (letter, numeric, waist×inseam)
- **JSON-LD enrichment** — some eBay listings have Product + BreadcrumbList JSON-LD; used as fallback for images, brand, and category path
- **Description** — pulled from `meta[name="description"]` since eBay's inline description is in an inaccessible iframe
- **Condition mapping**:
  - Brand New / New with Tags / New without Tags → Excellent
  - Pre-Owned / New with Defects → Good  
  - Pre-Owned - Good → Good
  - Pre-Owned - Fair → Fair
  - For Parts → Poor

## Output Format

Both platforms produce the same Thursday-compatible JSON:

```json
{
  "exportedAt": "2026-03-08T...",
  "version": "4.0",
  "totalItems": 42,
  "skippedSold": 10,
  "products": [
    {
      "name": "Vintage Nike Windbreaker",
      "description": "...",
      "price": 45.00,
      "condition": "Good",
      "status": "DRAFT",
      "size": "L",
      "brand": "Nike",
      "category": "Outerwear",
      "type": "Windbreakers",
      "weight": 1.2,
      "styleTags": ["Vintage", "Sporty"],
      "images": [{ "url": "...", "isPrimary": true, "sortOrder": 0 }],
      "_source": {
        "platform": "ebay",
        "ebayId": "257322711183",
        "ebayUrl": "https://www.ebay.com/itm/257322711183"
      }
    }
  ]
}
```

## Installation

1. Download or clone this directory
2. Open `chrome://extensions/` in Chrome
3. Enable "Developer mode" (top right)
4. Click "Load unpacked" and select the `thursday-switch-v4` folder
5. Navigate to a Depop shop or eBay store and click the extension icon

## File Structure

```
thursday-switch-v4/
├── manifest.json        # Extension manifest (v3)
├── icons/               # Extension icons
├── src/
│   ├── background.js    # Service worker
│   ├── content.js       # Scraping logic (DepopScraper + EbayScraper)
│   ├── content.css      # Progress overlay styles
│   ├── popup.html       # Extension popup UI
│   ├── popup.css        # Popup styles
│   └── popup.js         # Popup controller
└── README.md
```
