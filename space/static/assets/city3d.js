/* The 3D city behind the home page. CesiumJS with open data only: PDOK 2025 aerial photo
   (Beeldmateriaal Nederland, CC BY 4.0) and 3DBAG LoD2.2 buildings (CC BY 4.0). No Cesium ion.
   The buildings come from this site: every 3DBAG tile inside the A10 ring, with the attribute tables and the
   normals removed (a plain per-building id stays, for the drop-in) and the geometry re-encoded with Draco; the
   empty inner tiles always refine, so every building in the crop is drawn at any distance
   (scripts/87_3dbag_subset.py --crop --refine-all --no-normals).
   City3D.create(id) -> { viewer, rig, ready }
   rig.fly(view) flies in an arc: it rises while it travels and comes down at the next place.
   City3D.walk(city, data) -> a SYNTHETIC GNSS walk with a chase camera. */
"use strict";

const City3D = (() => {
  const PDOK = "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/2025_orthoHR/EPSG:3857/{z}/{x}/{y}.jpeg";
  const BAG = "data/3dbag/lod22/tileset.json";   // 3DBAG v20250903, A10 crop served from this site
  const GREEN = ["#cfe3dc", "#a6cfc0", "#79b5a2", "#4d9781", "#2b7462", "#174f42"];
  const rad = d => (d * Math.PI) / 180, deg = r => (r * 180) / Math.PI;
  const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
  const ease = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  const metres = (a, b) => {
    const p1 = rad(a.lat), p2 = rad(b.lat), dl = rad(b.lon - a.lon);
    const h = Math.sin((p2 - p1) / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
    return 2 * 6371000 * Math.asin(Math.sqrt(h));
  };
  const greenColour = g => Cesium.Color.fromCssColorString(g === null || g === undefined ? "#c9d1d6"
    : GREEN[Math.min(GREEN.length - 1, Math.floor((g / 0.5) * GREEN.length))]);

  async function create(id, opts = {}) {
    Cesium.Ion.defaultAccessToken = "";
    const viewer = new Cesium.Viewer(id, {
      baseLayer: new Cesium.ImageryLayer(new Cesium.UrlTemplateImageryProvider({
        url: PDOK, maximumLevel: 19, credit: "Aerial photo 2025: Beeldmateriaal Nederland via PDOK, CC BY 4.0",
        // only the Amsterdam region, from zoom 10: no requests for the rest of the world or the coarse levels
        rectangle: Cesium.Rectangle.fromDegrees(4.60, 52.20, 5.20, 52.55), minimumLevel: 10,
      })),
      baseLayerPicker: false, geocoder: false, homeButton: false, sceneModePicker: false, navigationHelpButton: false,
      animation: false, timeline: false, fullscreenButton: false, infoBox: false, selectionIndicator: false,
      creditContainer: document.createElement("div"), msaaSamples: 4,
      skyBox: false, skyAtmosphere: false,   // never seen here; saves about 0.75 MB of textures
    });
    const scene = viewer.scene;
    scene.globe.baseColor = Cesium.Color.fromCssColorString("#0c1114");
    scene.backgroundColor = Cesium.Color.fromCssColorString("#0c1114");
    scene.sun = undefined; scene.moon = undefined;
    scene.globe.showGroundAtmosphere = false;
    scene.fog.enabled = true; scene.fog.density = 0.00028;
    scene.highDynamicRange = false;
    scene.postProcessStages.fxaa.enabled = true;
    scene.screenSpaceCameraController.enableInputs = false;   // the page scrolls; the camera follows the story
    const layer = viewer.imageryLayers.get(0);
    layer.saturation = 0.7; layer.brightness = 0.8; layer.contrast = 1.08;
    // PDOK serves plain white tiles where it has no imagery (open water to the north-east): show the dark globe there
    layer.colorToAlpha = Cesium.Color.WHITE; layer.colorToAlphaThreshold = 0.06;

    let bag = null;
    try {
      bag = await Cesium.Cesium3DTileset.fromUrl(BAG, { maximumScreenSpaceError: 16, skipLevelOfDetail: true,
        dynamicScreenSpaceError: true, foveatedScreenSpaceError: true });
      bag.customShader = new Cesium.CustomShader({
        lightingModel: Cesium.LightingModel.UNLIT,
        // "Building the city": until rise() is called the buildings wait out of sight. Then a wave runs out from the
        // centre, tile by tile (each tile's own origin, so every vertex of a building agrees), and each building, a
        // little later than its neighbours (seeded by its id, the tiles' _BUILDING attribute), appears a few hundred
        // metres up, falls straight down and bounces once as it lands. Once the wave is over (u_done) the vertex
        // shader does nothing.
        uniforms: {
          u_now: { type: Cesium.UniformType.FLOAT, value: 0 },
          u_start: { type: Cesium.UniformType.FLOAT, value: -1 },
          u_done: { type: Cesium.UniformType.FLOAT, value: 0 },
          u_center: { type: Cesium.UniformType.VEC3, value: new Cesium.Cartesian3() },
        },
        vertexShaderText: `void vertexMain(VertexInput vsInput, inout czm_modelVertexOutput vsOutput) {
          if (u_done > 0.5) return;
          vec3 origin = (czm_model * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
          vec3 up = normalize((czm_inverseModel * vec4(normalize(origin), 0.0)).xyz);   // in the tile's own frame
          float lift = 60000.0;   // not yet: out of sight
          if (u_start >= 0.0) {
            float seed = fract(sin(floor(vsInput.attributes.building + 0.5) * 12.9898 + 78.233) * 43758.5453);
            float t = (u_now - u_start - distance(origin, u_center) / 2200.0 - seed * 0.8) / 0.7;
            float h = 220.0 + 200.0 * fract(seed * 3.71);
            if (t >= 1.0) lift = 0.0;
            else if (t >= 0.78) lift = 0.06 * h * sin(3.14159265 * (t - 0.78) / 0.22);
            else if (t >= 0.0) lift = h * (1.0 - (t / 0.78) * (t / 0.78));
          }
          vsOutput.positionMC += up * lift;
        }`,
        // The tiles carry no normals (walls and roofs are flat): each face's normal comes from the surface itself,
        // and the light is fixed, a low afternoon sun from the south-west and a soft sky from above.
        fragmentShaderText: `void fragmentMain(FragmentInput fsInput, inout czm_modelMaterial material) {
          vec3 p = fsInput.attributes.positionEC;
          vec3 n = normalize(cross(dFdx(p), dFdy(p)));
          vec3 upW = normalize((czm_model * vec4(0.0, 0.0, 0.0, 1.0)).xyz);
          vec3 eastW = normalize(cross(vec3(0.0, 0.0, 1.0), upW)), northW = cross(upW, eastW);
          mat3 toEye = mat3(czm_view);
          vec3 up = normalize(toEye * upW), sun = normalize(toEye * (-0.55 * eastW - 0.55 * northW + 0.62 * upW));
          float roof = clamp(dot(n, up), 0.0, 1.0);
          vec3 base = mix(vec3(0.84, 0.80, 0.74), vec3(0.96, 0.94, 0.90), roof);
          material.diffuse = base * (0.045 + 0.21 * (0.5 + 0.5 * dot(n, up)) + 0.31 * max(dot(n, sun), 0.0));   // linear; shown as sRGB
        }`,
      });
      scene.primitives.add(bag);
      // 3DBAG heights are ellipsoidal (ground near +43 m in Amsterdam); this globe has no terrain,
      // so the buildings are lowered onto the ellipsoid (checked visually: -43 m seats the canal houses)
      const c = bag.boundingSphere.center, up = Cesium.Cartesian3.normalize(c, new Cesium.Cartesian3());
      bag.modelMatrix = Cesium.Matrix4.fromTranslation(Cesium.Cartesian3.multiplyByScalar(up, opts.heightOffset ?? -43, new Cesium.Cartesian3()));
    } catch (e) { console.warn("3DBAG did not load", e); }

    const rig = makeRig(viewer);
    if (opts.view) rig.set(opts.view);
    // Show the city early and let it sharpen in view: the first of the imagery being complete, the first
    // 3DBAG tiles, or 1.2 s (it used to wait for every imagery tile, up to 7 s)
    const ready = new Promise(res => {
      const t0 = performance.now();
      let done = false;
      const go = () => { if (!done) { done = true; res(); } };
      if (bag) bag.initialTilesLoaded.addEventListener(go);
      const check = () => { if (done) return; if (scene.globe.tilesLoaded || performance.now() - t0 > 1200) go(); else setTimeout(check, 100); };
      check();
    });
    // the building wave: rise({lon, lat}) starts it from that point; it lasts until the farthest tile is up
    let riseEnd = 0;
    scene.preUpdate.addEventListener(() => {
      if (!bag || !bag.customShader) return;
      const now = performance.now() / 1000;
      bag.customShader.setUniform("u_now", now);
      if (riseEnd && now > riseEnd) { bag.customShader.setUniform("u_done", 1); riseEnd = 0; }
    });
    function rise(at, { instant = false } = {}) {
      if (!bag || !bag.customShader) return;
      if (instant) { bag.customShader.setUniform("u_done", 1); return; }
      const now = performance.now() / 1000;
      bag.customShader.setUniform("u_center", Cesium.Cartesian3.fromDegrees(at.lon, at.lat, 0));
      bag.customShader.setUniform("u_start", now);
      riseEnd = now + 12000 / 2200 + 0.8 + 0.7 + 0.3;   // 12 km of wave, the longest delay, one fall, a margin
    }
    // loading progress for the "Building the city" counter: tiles still requested or processed
    const progress = new Cesium.Event();
    let pendingBag = 0, pendingGlobe = 0;
    const emit = () => progress.raiseEvent(pendingBag + pendingGlobe);
    if (bag) bag.loadProgress.addEventListener((req, proc) => { pendingBag = req + proc; emit(); });
    scene.globe.tileLoadProgressEvent.addEventListener(n => { pendingGlobe = n; emit(); });
    return { viewer, scene, bag, rig, ready, rise, progress };
  }

  /* Camera rig: the camera always looks at a ground point {lon, lat} from {range, heading, pitch} (degrees). */
  function makeRig(viewer) {
    const cam = viewer.camera;
    let state = { lon: 4.895, lat: 52.372, range: 6000, heading: 0, pitch: -45 };
    let anim = null, orbit = 0, held = false, last = performance.now();
    // frame offset: the subject sits right of centre (fx) or above it (fy), clear of the page's cards
    let fx = 0, fy = 0;
    const apply = s => {
      const h = rad(s.heading), d = s.range * Math.cos(rad(Math.max(-80, s.pitch)));
      const east = -Math.cos(h) * fx * d - Math.sin(h) * fy * d, north = Math.sin(h) * fx * d - Math.cos(h) * fy * d;
      const lat = s.lat + north / 111320, lon = s.lon + east / (111320 * Math.cos(rad(s.lat)));
      cam.lookAt(Cesium.Cartesian3.fromDegrees(lon, lat, 0), new Cesium.HeadingPitchRange(h, rad(s.pitch), s.range));
      cam.lookAtTransform(Cesium.Matrix4.IDENTITY);
    };
    function tick(now) {
      const dt = Math.min(0.1, Math.max(0, now - last) / 1000); last = now;
      if (anim) {
        const t = Math.min(1, (now - anim.t0) / anim.ms), e = ease(t), b = Math.sin(Math.PI * e);
        const a = anim.from, z = anim.to;
        state = {
          lon: a.lon + (z.lon - a.lon) * e, lat: a.lat + (z.lat - a.lat) * e,
          range: Math.exp(Math.log(a.range) + (Math.log(z.range) - Math.log(a.range)) * e) + anim.bump * b,
          heading: a.heading + anim.dh * e,
          pitch: Math.max(-89, a.pitch + (z.pitch - a.pitch) * e - anim.tilt * b),
        };
        apply(state);
        if (t >= 1) { const done = anim.done; anim = null; state.heading = z.heading; done(); }
      } else if (orbit && !held) {
        state.heading = (state.heading + orbit * dt) % 360;
        apply(state);
      }
    }
    const loop = now => { tick(now); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
    return {
      _tick: tick,   // lets a test harness advance the rig where requestAnimationFrame does not run
      get state() { return { ...state }; },
      set(s) { anim = null; state = { ...state, ...s }; apply(state); },
      fly(to, { ms, bump } = {}) {
        to = { ...state, ...to };
        const d = metres(state, to);
        if (matchMedia("(prefers-reduced-motion: reduce)").matches || d + Math.abs(to.range - state.range) < 1) {
          this.set(to); return Promise.resolve();
        }
        const dh = deg(wrap(rad(to.heading - state.heading)));
        return new Promise(res => {
          anim = { from: { ...state }, to, dh, t0: performance.now(), done: res,
            ms: ms ?? Math.max(2400, Math.min(4600, 1800 + d * 0.55)),
            bump: bump ?? Math.min(5200, d * 0.85) * (1 - Math.min(0.8, Math.max(state.range, to.range) / 12000)),
            tilt: Math.min(22, d / 120) };
        });
      },
      orbit(degPerSecond) { orbit = degPerSecond; },
      frame(x, y) { fx = x; fy = y; apply(state); },
      hold(on) { held = on; if (on) anim = null; },
    };
  }

  /* A SYNTHETIC walk (WP2) with a chase camera driven through the rig */
  function walk(city, data, opts = {}) {
    const { viewer, rig } = city;
    const ds = new Cesium.CustomDataSource("walk");
    viewer.dataSources.add(ds);
    ds.show = false;
    const R = data.route;   // [lon, lat, along_m, green]
    const total = R[R.length - 1][2];
    const pos = R.map(r => Cesium.Cartesian3.fromDegrees(r[0], r[1], 1.5));
    ds.entities.add({ polyline: { positions: pos, width: 2.5, arcType: Cesium.ArcType.NONE,
      material: new Cesium.PolylineDashMaterialProperty({ color: Cesium.Color.WHITE.withAlpha(0.55), dashLength: 14 }),
      depthFailMaterial: new Cesium.PolylineDashMaterialProperty({ color: Cesium.Color.WHITE.withAlpha(0.25), dashLength: 14 }) } });
    const chunks = [];
    for (let i = 0; i < R.length - 1; i += 10) {
      const j = Math.min(R.length - 1, i + 10);
      const glow = new Cesium.PolylineGlowMaterialProperty({ glowPower: 0.25, color: greenColour(R[i][3]) });
      chunks.push([i, ds.entities.add({ polyline: { positions: pos.slice(i, j + 1), width: 9, arcType: Cesium.ArcType.NONE,
        material: glow, depthFailMaterial: glow }, show: false })]);
    }
    data.raw_fixes.forEach(f => ds.entities.add({ position: Cesium.Cartesian3.fromDegrees(f[0], f[1], 3),
      point: { pixelSize: 5, color: Cesium.Color.fromCssColorString("#ffb547").withAlpha(0.85) } }));
    data.views.forEach(v => ds.entities.add({ position: Cesium.Cartesian3.fromDegrees(v.lon, v.lat, 6),
      point: { pixelSize: 8, color: greenColour(v.green_view), outlineColor: Cesium.Color.fromCssColorString("#0c1114"), outlineWidth: 2 } }));
    const walker = ds.entities.add({ position: pos[0], point: { pixelSize: 13, color: Cesium.Color.WHITE,
      outlineColor: Cesium.Color.fromCssColorString("#0e6e66"), outlineWidth: 4, disableDepthTestDistance: Number.POSITIVE_INFINITY } });

    const idxAt = s => { let lo = 0, hi = R.length - 1; while (lo < hi) { const m = (lo + hi) >> 1; if (R[m][2] < s) lo = m + 1; else hi = m; } return lo; };
    const ll = s => { const r = R[idxAt(Math.max(0, Math.min(total, s)))]; return { lon: r[0], lat: r[1] }; };
    const bearing = (a, b) => {
      const p1 = rad(a.lat), p2 = rad(b.lat), dl = rad(b.lon - a.lon);
      return Math.atan2(Math.sin(dl) * Math.cos(p2), Math.cos(p1) * Math.sin(p2) - Math.sin(p1) * Math.cos(p2) * Math.cos(dl));
    };
    const RANGE = opts.range || 190, PITCH = opts.pitch || -32, SECONDS = opts.seconds || 22;
    let heading = bearing(ll(0), ll(90)), u = 0, playing = false, last = performance.now(), token = 0;
    const pose = s => ({ ...ll(s), range: RANGE, heading: deg(heading), pitch: PITCH });
    function place(s, dt) {
      const want = bearing(ll(s - 30), ll(s + 90));
      heading = wrap(heading + wrap(want - heading) * Math.min(1, dt * 1.5));
      rig.set(pose(s));
      walker.position = pos[idxAt(s)];
      const i = idxAt(s);
      chunks.forEach(([k, e]) => { e.show = k <= i; });
      if (opts.onProgress) opts.onProgress(s / total, s, total);
    }
    function tick(now) {
      const dt = Math.min(0.1, Math.max(0, now - last) / 1000); last = now;
      if (playing) {
        u += dt / SECONDS;
        if (u >= 1) { u = 0; heading = bearing(ll(0), ll(90)); chunks.forEach(([, e]) => { e.show = false; }); }
        place(ease(u) * total, dt);
      }
    }
    const loop = now => { tick(now); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
    return {
      total, data, _tick: tick,
      async start() {
        const my = ++token;
        ds.show = true; u = 0;
        heading = bearing(ll(0), ll(90));
        chunks.forEach(([, e]) => { e.show = false; });
        walker.position = pos[0];
        rig.hold(false);
        await rig.fly(pose(0));
        if (my !== token) return;
        rig.hold(true); last = performance.now(); playing = true;
      },
      stop() { token++; playing = false; rig.hold(false); },
      hide() { this.stop(); ds.show = false; },
      get playing() { return playing; },
    };
  }

  return { create, walk, greenColour, metres };
})();

window.City3D = City3D;
