# Site design record, v3 (2026-10-11)

Published at https://www.yunusserhat.com/elevate/ (unlisted, `robots: noindex`); search API at the
Hugging Face Space `yunusserhat/elevate-amsterdam`. Supersedes the v1 record (plain multi-page site) and
the v2 start (dark 3D hero over the WP list), both replaced after the user's review.

## Direction
- **Home:** one fixed 3D Amsterdam behind the whole page (CesiumJS, PDOK 2025 aerial photo, 3DBAG
  LoD2.2). Scrolling flies the camera in an arc to one place per work package: it rises while it
  travels and comes down at the next place, then turns slowly. Each chapter has one short card with
  one number and an "Open WPn" link. WP2's chapter plays the synthetic walk (22 s loop).
- **Work packages:** light editorial story pages: a sticky full-screen figure, short cards scrolling
  over it, details folded away at the end. Visuals first, very little text (user feedback).
- **Ask the images:** its own page and a highlighted top-bar item. Dark full screen, a large search
  box, results as an image gallery next to a dark map, photo upload, a server status line.
- **Navigation:** top bar with WP1–WP7 and "Ask the images"; "← Overview" on every page returns to the
  same chapter (`index.html#wpN`); a right-side step navigator (previous, next, top, end).
- **Controls:** segmented buttons, never dropdowns.

## Tokens
Night `#0a0f12`, glow `#63d6c4` (dark parts); paper `#f6f7f5`, ink `#15191c`, accent `#0d6b63`
(light parts); cycling `#2f4f86`; amber `#a85d00` for synthetic, hypothetical and idea labels and for
"less" ends of scales. Fonts self-hosted (SIL OFL): Source Serif 4 (display, prose), Public Sans
(interface), JetBrains Mono (labels, numbers).

## Story mechanics (`assets/site.js`)
- `story(root, {onStep, onProgress})`: IntersectionObserver at the viewport middle; `onProgress`
  drives scroll-linked figures (WP3 zoom into the Bloemgracht and the panorama splitting into four
  views; WP2 clock from 07:00 to 19:00 and the spike map).
- Stage layers each form their own stacking context (`z-index: 1`), so map panes never cover
  legends; legends are solid white.
- `API_BASE` is empty in development (same origin on worf) and set to the Space URL by
  `scripts/85_package.py`; every page wakes the Space with `/api/health`.

## 3D (`assets/city3d.js`)
Camera rig that always looks at a ground point; arc flights interpolate the target and the range on
a log scale with a bump, and frame the subject right of the cards (desktop) or above them (mobile).
3DBAG heights are ellipsoidal: lowered 43 m onto Cesium's ellipsoid. PDOK's plain white no-data tiles
are made transparent.

## Checks (2026-10-11)
Desktop 1440 × 900 and mobile 375 × 812: no horizontal overflow, no console errors. Cross-origin
search tested locally (Hugo build on port 1313 against the API-only server) and live. Every page and
data file of the live site returns 200.

## Known limits
- On a phone the four panorama views are small.
- The Hugging Face proxy reflects any CORS origin, so the app's origin list has no effect there; the
  API is read-only.
- The Space sleeps after 48 h without visits; a 12-hourly ping runs from the user's computer
  (systemd user timer `elevate-space-ping`).
