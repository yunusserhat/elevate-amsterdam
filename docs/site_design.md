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

## Speed (2026-10-13, step 10)

Measured on a local copy and the dev server (step 10). Before: the home page hid the 3D city until every
aerial tile of the view had arrived (up to 7 s) and faded it in over 1.2 s; 83 PDOK requests, 35 of them zoom 1-12;
3DBAG LoD2.2 from data.3dbag.nl, 9.1 MB for the first view (tiles are meshopt-compressed but carry per-building ids
and a 42-column attribute table); Cesium also fetched sky box and moon textures (about 0.75 MB) that are never seen.
- The city now appears at the first of: imagery complete, the first 3DBAG tiles, 1.2 s; fade 0.6 s.
- No sky box, sky atmosphere, sun or moon.
- PDOK limited to the Amsterdam region (4.60-5.20 E, 52.20-52.55 N) from zoom 10: 56 requests instead of 83.
- 3DBAG served from this site. First (step 10.5): the tiles the story camera was recorded to request, plus neighbours
  and family siblings, 234 tiles, 62.2 MB, first view about 1.9 MB. That first view was light only because the centre
  never loaded (step 11.8): 3DBAG keeps buildings only in the leaves, the inner tiles are empty, and from the home
  view (about 4.6 km) the error of the empty tile above the centre's leaves (33 m, 1.2 km tile) stays under 16 px on
  ordinary screens (about 12 px at 1080 px high, 16:9), so Cesium stopped there and drew nothing; the outskirts, whose leaves
  sit higher in the tree, did show. It showed on a 1080p screen, not in the earlier tests at 2560x1440 and 3840x2160.
  Now (step 11.9): every tile touching the A10 ring (4.835-4.955 E, 52.333-52.402 N), 106 tiles; the empty inner tiles
  get a huge geometric error, so the page always goes down to the leaves; attribute table and normals removed (the
  per-building id stays as `_BUILDING`), Draco (positions 14 bit, ids 14 bit). 17.5 MB on the site (140.2 MB
  original). The first view loads all of it (17.2 MB measured; through the dev tunnel the last tile arrived at
  2.6 s at 1920x1080). Outside the ring the aerial photo shows without buildings.
  Dropping the normals: Draco can then merge the corners the faces shared (one tile 600 KB -> 329 KB, 261k -> 73k
  vertices); the page computes each face's normal from the surface (dFdx/dFdy of the eye position) and lights it
  with a fixed low sun from the south-west and a soft sky (the old PBR light followed the real time of day).
- Poster: WebP, 1280 px, 228 KB instead of a 543 KB JPEG.
- WP3 and WP6 street layers: compact JSON (values plus delta-encoded integer coordinates) drawn as one polyline per
  colour class: 0.69 MB and 0.85 MB instead of 11.4 MB and 6.0 MB of GeoJSON.
Result on the dev server (1440x900): city visible at 1.3 s instead of about 4.1 s.

## Opening (2026-10-13, step 11)
- "Building the city N%" in the middle of the screen while tiles load (3DBAG plus imagery, never counts down),
  then it fades.
- The buildings wait out of sight until every tile in view has loaded (at most 15 s), then drop in: a wave runs out
  from the view centre at 2,200 m/s, tile by tile (each tile's own origin, so a building never splits), each
  building 0-0.8 s later than its tile (seeded by its id), appears 220-420 m up, falls straight down in 0.7 s and
  bounces once (6 % of its height). About 7 s in all; then the vertex shader is switched off. With reduced motion
  the buildings are simply there.
- The page opens 16 km above the centre and descends to the intro view in 3.6 s while the buildings load, then
  they drop (the user chose this over opening at the intro view; not with a #chapter link or reduced motion).

## Known limits
- On a phone the four panorama views are small.
- The Hugging Face proxy reflects any CORS origin, so the app's origin list has no effect there; the
  API is read-only.
- The Space sleeps after 48 h without visits; the user wakes it by hand (the ping timer was removed on 2026-10-12).
