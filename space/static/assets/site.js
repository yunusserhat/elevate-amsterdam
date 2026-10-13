/* Shared behaviour v3: top bar with a way back, step navigator, segmented controls, story engine,
   tables, maps, before/after slider, City image links. No framework. */
"use strict";

const PAGES = [
  ["index.html", "Overview", ""],
  ["wp1.html", "WP1", "Co-creation"], ["wp2.html", "WP2", "Mobility and activity"], ["wp3.html", "WP3", "Streetscape analysis"],
  ["wp4.html", "WP4", "Mixed methods"], ["wp5.html", "WP5", "Spatial modelling"], ["wp6.html", "WP6", "Recommender"],
  ["wp7.html", "WP7", "Open science"],
];
const here = () => location.pathname.split("/").pop() || "index.html";

// Search API (a Hugging Face Space). Empty means same origin, as on the development server;
// the packaging script (scripts/85_package.py) writes the Space URL here for the published site.
const API_BASE = "";
const api = path => `${API_BASE}${path}`;
// A free Space sleeps when idle, so every page wakes it as soon as it opens.
const wakeAPI = () => fetch(api("/api/health"), { cache: "no-store" }).then(r => (r.ok ? r.json() : null)).catch(() => null);
const reduceMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const ICON = {
  up: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 10l5-5 5 5" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>',
  down: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 6l5 5 5-5" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>',
  top: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 3h10M3 11l5-5 5 5" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>',
  bottom: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 13h10M3 5l5 5 5-5" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>',
};

function buildChrome() {
  const page = here();
  const i = PAGES.findIndex(p => p[0] === page);
  const info = PAGES[Math.max(0, i)];
  const isWP = i > 0;
  const top = document.createElement("header");
  top.className = "topbar";
  top.innerHTML = `<div class="bar">
      <a class="mark" href="index.html"><span>ELEVATE<span class="in"> in Amsterdam</span>${isWP ? `<span class="sub"><span style="opacity:.5;margin:0 .5em">·</span>${info[1]} ${info[2]}</span>` : ""}</span></a>
      ${isWP ? `<a class="back" href="index.html#${info[1].toLowerCase()}" title="Back to the overview, at ${info[1]}">← <span class="t">Overview</span></a>`
        : ["ask.html", "survey.html"].includes(page) ? `<a class="back" href="index.html" title="Back to the overview">← <span class="t">Overview</span></a>` : ""}
      <nav aria-label="Work packages">${PAGES.slice(1).map(([h, l, t]) => `<a href="${h}" title="${l}: ${t}"${h === page ? ' aria-current="page"' : ""}>${l}</a>`).join("")}
        <a href="survey.html" title="Take part: a short pilot survey"${page === "survey.html" ? ' aria-current="page"' : ""}>Take part</a>
        <a class="asknav" href="ask.html"${page === "ask.html" ? ' aria-current="page"' : ""}><span class="t">Ask the images</span><span class="s">Ask</span></a></nav>
    </div>`;
  document.body.prepend(top);
  // a dark block at the top of the page (home chapters, Ask the images) keeps the bar dark while it is under it
  const night = document.querySelector("[data-night-top]");
  if (night) {
    const tone = () => top.classList.toggle("on-night", night.getBoundingClientRect().bottom > 64);
    tone(); addEventListener("scroll", tone, { passive: true });
  }

  const next = ["ask.html", "survey.html"].includes(page) ? PAGES[1] : PAGES[i + 1] || PAGES[1];
  const foot = document.createElement("footer");
  foot.className = "foot";
  foot.innerHTML = `<div class="bar">
      <div>
        <div class="label">Next</div>
        <a class="next" href="${next[0]}">${next[1]} ${next[2]} →</a>
        ${isWP ? `<a class="home-link" href="index.html#${info[1].toLowerCase()}">← Back to the overview</a>` : ""}
        <p style="margin-top:1.6rem">An independent demonstration by Yunus Serhat Bıçakçı, not produced by or affiliated with the ELEVATE project or Utrecht University. It follows the protocol of Helbich M, Labib SM, Liedtke I, et al. (2026), PLOS ONE 21(5): e0348889.</p>
      </div>
      <div>
        <div class="label">Sources</div>
        <p>Street panoramas: Gemeente Amsterdam, CC BY 4.0. Aerial photo 2025: Beeldmateriaal Nederland via PDOK, CC BY 4.0. Buildings: 3DBAG (TU Delft 3D geoinformation and 3DGI), CC BY 4.0, recoloured and lowered to the ellipsoid. Maps and network: © OpenStreetMap contributors, ODbL. Income: CBS. Reports: Gemeente Amsterdam (SIA open data). Model outputs are observations of images, not measurements. All sources and licences: <a href="wp7.html">WP7</a>. Code: <a href="https://github.com/yunusserhat/geoai-vlm">GeoAI-VLM</a>.</p>
      </div>
    </div>`;
  document.body.append(foot);
  stepNav();
  wakeAPI();
}

/* Step navigator: previous / next stop, top, bottom. Stops are story steps, chapters and sections. */
function stepNav() {
  const nav = document.createElement("div");
  nav.className = "stepnav";
  nav.setAttribute("role", "group");
  nav.setAttribute("aria-label", "Move through the page");
  nav.innerHTML = `<button class="top" type="button" aria-label="Back to the top" title="Top">${ICON.top}</button>
    <button class="prev" type="button" aria-label="Previous part" title="Previous">${ICON.up}</button>
    <button class="next" type="button" aria-label="Next part" title="Next">${ICON.down}</button>
    <button class="bottom" type="button" aria-label="To the end of the page" title="End">${ICON.bottom}</button>`;
  document.body.append(nav);
  const stops = () => [...document.querySelectorAll(".intro, .story .step, .chapter, main > .section, .ask, .after > .section")]
    .map(el => {
      const r = el.getBoundingClientRect(), y0 = r.top + scrollY;
      const y = el.matches(".step.long") ? y0 + innerHeight * 0.05
        : el.matches(".step, .chapter") ? y0 + Math.min(r.height, innerHeight * 1.2) / 2 - innerHeight / 2 + (r.height > innerHeight * 1.2 ? 0 : 0)
        : y0 - 64;
      return Math.max(0, Math.round(y));
    }).sort((a, b) => a - b);
  const go = y => scrollTo({ top: y, behavior: reduceMotion() ? "auto" : "smooth" });
  nav.querySelector(".top").onclick = () => go(0);
  nav.querySelector(".bottom").onclick = () => go(document.documentElement.scrollHeight);
  nav.querySelector(".next").onclick = () => { const y = stops().find(s => s > scrollY + 8); go(y ?? document.documentElement.scrollHeight); };
  nav.querySelector(".prev").onclick = () => { const y = stops().reverse().find(s => s < scrollY - 8); go(y ?? 0); };
  let queued = false;
  const tone = () => {
    queued = false;
    const el = document.elementFromPoint(innerWidth - 90, innerHeight / 2);
    nav.classList.toggle("on-night", !!(el && el.closest(".ask, .foot, .stage.dark, .chapters")));
    nav.querySelector(".prev").disabled = scrollY < 8;
    nav.querySelector(".top").disabled = scrollY < 8;
  };
  addEventListener("scroll", () => { if (!queued) { queued = true; requestAnimationFrame(tone); } }, { passive: true });
  setTimeout(tone, 300);
}

/* Story engine: each .story has a .stage (sticky) and .step blocks. onStep(index, el) fires when a step
   crosses the middle of the viewport; onProgress(index, share) reports how far the middle is through it. */
function story(root, { onStep, onProgress } = {}) {
  const steps = [...root.querySelectorAll(".step")];
  let active = -1;
  const io = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (!e.isIntersecting) return;
      const i = steps.indexOf(e.target);
      if (i !== active) { active = i; onStep && onStep(i, e.target); tick(); }
    });
  }, { rootMargin: "-50% 0px -50% 0px" });
  steps.forEach(s => io.observe(s));
  let queued = false;
  function tick() {
    queued = false;
    if (active < 0 || !onProgress) return;
    const r = steps[active].getBoundingClientRect(), mid = innerHeight / 2;
    onProgress(active, Math.max(0, Math.min(1, (mid - r.top) / r.height)));
  }
  addEventListener("scroll", () => { if (!queued) { queued = true; requestAnimationFrame(tick); } }, { passive: true });
  const api = { get active() { return active; }, tick };
  (window.__stories = window.__stories || []).push(api);   // test handle: lets a harness re-read progress
  return api;
}
function showLayer(stage, name) {
  stage.querySelectorAll(":scope > .layer").forEach(l => l.classList.toggle("on", l.dataset.layer === name));
  stage.classList.toggle("dark", !!stage.querySelector(":scope > .layer.on.night"));
}

/* Segmented control: seg(el, [[value, label, swatch?], ...], value, onChange) */
function seg(el, options, value, onChange) {
  el.classList.add("seg");
  el.setAttribute("role", "group");
  el.innerHTML = options.map(([v, l, sw]) => `<button type="button" data-v="${esc(v)}" aria-pressed="${v === value}">${sw ? `<span class="sw" style="background:${sw}"></span>` : ""}${esc(l)}</button>`).join("");
  el.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
    el.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
    onChange(b.dataset.v);
  }));
  return { set(v) { el.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", String(x.dataset.v === v))); } };
}

/* Before / after slider: drag, click or use the arrow keys */
function beforeAfter(el, before, after, { alt = "", labels = ["Before", "After"], start = 50 } = {}) {
  el.classList.add("ba");
  el.style.setProperty("--x", `${start}%`);
  el.innerHTML = `<img src="${before}" alt="${esc(alt)}, ${labels[0].toLowerCase()}" loading="lazy">
    <img class="after-img" src="${after}" alt="${esc(alt)}, ${labels[1].toLowerCase()}" loading="lazy">
    <div class="handle"></div><span class="tag l">${labels[0]}</span><span class="tag r">${labels[1]}</span>
    <input type="range" min="0" max="100" value="${start}" aria-label="Compare ${labels[0].toLowerCase()} and ${labels[1].toLowerCase()}">`;
  el.querySelector("input").addEventListener("input", e => el.style.setProperty("--x", `${e.target.value}%`));
}

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} for ${url}`);
  return r.json();
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function fmt(x, digits = 2) {
  if (x === null || x === undefined || x === "" || Number.isNaN(Number(x))) return "–";
  return Number(x).toLocaleString("en-GB", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
const pct = (x, digits = 1) => (x === null || x === undefined || x === "" ? "–" : `${fmt(100 * Number(x), digits)}%`);
const clamp01 = x => Math.max(0, Math.min(1, x));
const lerp = (a, b, t) => a + (b - a) * t;
const easeInOut = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

function table(el, rows, columns) {
  const head = columns.map(c => `<th${c.num ? ' class="num"' : ""} scope="col">${esc(c.label)}</th>`).join("");
  const body = rows.map(r => `<tr>${columns.map(c => `<td${c.num ? ' class="num"' : ""}>${c.render ? c.render(r) : esc(r[c.key])}</td>`).join("")}</tr>`).join("");
  el.innerHTML = `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

/* Leaflet basemap: OpenStreetMap standard tiles, greyed (or darkened with dark: true) by CSS */
function baseMap(id, opts = {}) {
  const el = document.getElementById(id);
  if (opts.dark) el.classList.add("dark");
  const map = L.map(id, { zoomControl: false, preferCanvas: true, scrollWheelZoom: false, ...opts })
    .setView(opts.center || [52.365, 4.90], opts.zoom || 12);
  L.control.zoom({ position: "bottomright" }).addTo(map);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map);
  if ("ResizeObserver" in window) new ResizeObserver(() => map.invalidateSize()).observe(el);
  (window.__maps = window.__maps || []).push(map);   // test handle
  return map;
}

const RAMP = ["#cfe3dc", "#a6cfc0", "#79b5a2", "#4d9781", "#2b7462", "#174f42"];
function quantileBreaks(values, n = RAMP.length) {
  const v = values.filter(x => x !== null && x !== undefined && !Number.isNaN(x)).sort((a, b) => a - b);
  if (!v.length) return [];
  return Array.from({ length: n - 1 }, (_, i) => v[Math.floor(((i + 1) / n) * (v.length - 1))]);
}
function rampColour(x, breaks, ramp = RAMP) {
  if (x === null || x === undefined || Number.isNaN(x)) return null;
  let i = 0;
  while (i < breaks.length && x > breaks[i]) i++;
  return ramp[i];
}
function rampIndex(x, breaks) {
  let i = 0;
  while (i < breaks.length && x > breaks[i]) i++;
  return i;
}
/* Compact street layers (data/layers/*_lite.json): rows of [values for keys..., x0, y0, dx, dy, ...] with
   coordinates in 1/q degree from o, delta-encoded within each line. */
function decodeLines(d) {
  const k = d.keys.length;
  return d.f.map(r => {
    const pts = [];
    let x = 0, y = 0;
    for (let i = k; i < r.length; i += 2) { x += r[i]; y += r[i + 1]; pts.push([d.o[1] + y / d.q, d.o[0] + x / d.q]); }
    return { v: r.slice(0, k), pts };
  });
}
/* One polyline per class: thousands of segments cost a handful of canvas paths. Higher classes are drawn last. */
function classLines(lines, classOf, styleOf, renderer) {
  const groups = new Map();
  for (const l of lines) { const c = classOf(l); if (c === null) continue; if (!groups.has(c)) groups.set(c, []); groups.get(c).push(l.pts); }
  return L.layerGroup([...groups.keys()].sort((a, b) => a - b).map(c => L.polyline(groups.get(c), { renderer, interactive: false, ...styleOf(c) })));
}
const rampHTML = (ramp = RAMP) => `<span class="ramp">${ramp.map(c => `<span style="background:${c}"></span>`).join("")}</span>`;

function cityLinks(r) {
  const full = `https://t1.data.amsterdam.nl/panorama/2025/360geo/${encodeURIComponent(r.pano_id)}/equirectangular/panorama_4000.jpg`;
  return `<a href="${esc(r.city_image)}" target="_blank" rel="noopener">City image, ${esc(r.view)} view</a> · <a href="${full}" target="_blank" rel="noopener">360° original</a>`;
}

document.addEventListener("DOMContentLoaded", buildChrome);
