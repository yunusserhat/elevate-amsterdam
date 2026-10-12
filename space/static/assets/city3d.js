/* The 3D city behind the home page. CesiumJS with open data only: PDOK 2025 aerial photo
   (Beeldmateriaal Nederland, CC BY 4.0) and 3DBAG LoD2.2 buildings (CC BY 4.0). No Cesium ion.
   City3D.create(id) -> { viewer, rig, ready }
   rig.fly(view) flies in an arc: it rises while it travels and comes down at the next place.
   City3D.walk(city, data) -> a SYNTHETIC GNSS walk with a chase camera. */
"use strict";

const City3D = (() => {
  const PDOK = "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/2025_orthoHR/EPSG:3857/{z}/{x}/{y}.jpeg";
  const BAG = "https://data.3dbag.nl/v20250903/cesium3dtiles/lod22/tileset.json";
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
      })),
      baseLayerPicker: false, geocoder: false, homeButton: false, sceneModePicker: false, navigationHelpButton: false,
      animation: false, timeline: false, fullscreenButton: false, infoBox: false, selectionIndicator: false,
      creditContainer: document.createElement("div"), msaaSamples: 4,
    });
    const scene = viewer.scene;
    scene.globe.baseColor = Cesium.Color.fromCssColorString("#0c1114");
    scene.backgroundColor = Cesium.Color.fromCssColorString("#0c1114");
    scene.skyAtmosphere.show = false;
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
        lightingModel: Cesium.LightingModel.PBR,
        fragmentShaderText: `void fragmentMain(FragmentInput fsInput, inout czm_modelMaterial material) {
          float up = clamp(dot(normalize(fsInput.attributes.normalEC), normalize(czm_normal * vec3(0.0, 0.0, 1.0))), 0.0, 1.0);
          material.diffuse = mix(vec3(0.84, 0.80, 0.74), vec3(0.96, 0.94, 0.90), up);
          material.roughness = 0.9;
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
    const ready = new Promise(res => {
      let waited = 0;
      const check = () => { if (scene.globe.tilesLoaded || waited > 7000) res(); else { waited += 250; setTimeout(check, 250); } };
      check();
    });
    return { viewer, scene, bag, rig, ready };
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
