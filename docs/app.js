"use strict";

const RISK_COLORS = {
  5: "#57c84d",
  15: "#f0df35",
  40: "#e14b3f",
  70: "#d94ad7",
};

const PRODUCT_META = {
  ml_r40: {
    short: "ML r40 (25 mi)",
    title: "40-km radius ML",
    note: "40-km radius ML forecasts are typically the most conservative in both risk size and severity.",
    detail: "Predicts rainfall exceeding Flash Flood Guidance within 40 km. Test-set analysis shows that this configuration often produces risk areas that are too small and too weak.",
    dash: null,
  },
  ml_r60: {
    short: "ML r60 (37 mi)",
    title: "60-km radius ML",
    note: "60-km radius ML forecasts were the most balanced overall in the test-set analysis.",
    detail: "Predicts rainfall exceeding Flash Flood Guidance within 60 km. This configuration provided the best balance between missed events and overly broad or severe risk areas in the test set.",
    dash: "8 4",
  },
  ml_r75: {
    short: "ML r75 (47 mi)",
    title: "75-km radius ML",
    note: "75-km radius ML forecasts were a close second to 60 km for overall balance.",
    detail: "Predicts rainfall exceeding Flash Flood Guidance within 75 km. It performed similarly to 60 km while tending toward somewhat broader and stronger risk areas.",
    dash: "3 5",
  },
  ml_r100: {
    short: "ML r100 (62 mi)",
    title: "100-km radius ML",
    note: "100-km radius ML forecasts are typically the most aggressive.",
    detail: "Predicts rainfall exceeding Flash Flood Guidance within 100 km. Test-set analysis shows that risk areas can be too large and that high probabilities can be issued too frequently.",
    dash: "12 5 3 5",
  },
  ml_lpmm: {
    short: "ML Local PMM",
    title: "ML Local Probability-Matched Mean",
    note: "The Local PMM is a solid middle ground that balances tradeoffs between the smaller- and larger-radius ML configurations.",
    detail: "Combines r40, r60, r75, and r100 within a 300-km radius of influence. It preserves local extremes from the pooled ML probability distribution, unlike a conventional average that dampens the distribution and removes extreme values.",
    dash: "10 3 2 3",
  },
  wpc: {
    short: "WPC ERO",
    title: "WPC Excessive Rainfall Outlook",
    note: "WPC ERO is the official reference forecast shown alongside the experimental ML guidance.",
    detail: "The WPC Excessive Rainfall Outlook is the official categorical reference forecast, expressed here as the probability of rainfall exceeding Flash Flood Guidance within 40 km (25 mi) of a point. Average test-set results indicate better ML skill for Moderate-or-greater risks, but performance varies by event.",
    dash: "1 4",
  },
  pp: {
    short: "Practically Perfect",
    title: "Practically Perfect verification",
    note: "Practically Perfect is an observation-based benchmark, not a forecast.",
    detail: "Built after the valid period from observed flood-proxy locations, then spatially expanded and smoothed to show idealized risk placement. It typically becomes available around 11:10 AM CT the following day.",
    dash: "7 3",
  },
};

const PRODUCT_ORDER = ["ml_r40", "ml_r60", "ml_r75", "ml_r100", "ml_lpmm", "wpc", "pp"];
const THRESHOLDS = [5, 15, 40, 70];
const OBSERVATION_META = {
  stage4_ffg: { label: "Stage IV > FFG", color: "#00e5ff" },
  stage4_ari: { label: "Stage IV ARI", color: "#ff9d36" },
  usgs: { label: "USGS", color: "#58a6ff" },
  flash_lsr: { label: "Flash-flood reports", color: "#ffffff" },
};
const LSR_META = {
  flash_flood: { label: "Flash flood", color: "#ff4fd8" },
  flood: { label: "Flood", color: "#38d9ff" },
  rain: { label: "Rain total", color: "#ffffff" },
  mping_flood: { label: "mPING flood impact", color: "#ff9f43" },
};
const LSR_REFRESH_MS = 5 * 60 * 1000;
const SURFACE_HEIGHT_METERS_PER_PERCENT = 1600;
const SEPARATED_POINT_RADIUS_PIXELS = 0.043;
const COMPACT_POINT_RADIUS_PIXELS = 0.13;
const OBSERVATION_CLEARANCE_METERS = 32000;
const EXPANSION_RADIUS_METERS = 40000;
const WPC_LOCAL_RISK_DISTANCE_KM = 350;
const CONUS_LONGITUDE_SCALE = Math.cos(40 * Math.PI / 180);

const state = {
  archive: [],
  data: null,
  selected: "ml_r60",
  contours: new Set(),
  observations: new Set(),
  fillOpacity: 0.68,
  fillLayer: null,
  contourLayer: null,
  observationLayer: null,
  lsrReports: [],
  lsrTypes: new Set(["flash_flood", "flood"]),
  lsrLayer: null,
  lsrTimer: null,
  lsrRequest: 0,
  mpingReports: [],
  mpingVisible: true,
  mpingRequest: 0,
  viewMode: "2d",
  map3d: null,
  deckOverlay: null,
  render3dFrame: null,
  surface3dCache: new Map(),
  separated3dPoints: false,
  showExpansionRings: false,
};

const map = L.map("map", {
  zoomControl: false,
  preferCanvas: true,
  minZoom: 3,
  maxZoom: 9,
}).setView([39.5, -92.5], 5);

map.createPane("forecastPane");
map.getPane("forecastPane").style.zIndex = 350;
map.createPane("statePane");
map.getPane("statePane").style.zIndex = 430;
map.getPane("statePane").style.pointerEvents = "none";
map.createPane("contourPane");
map.getPane("contourPane").style.zIndex = 450;
map.createPane("labelPane");
map.getPane("labelPane").style.zIndex = 500;
map.getPane("labelPane").style.pointerEvents = "none";
map.createPane("observationPane");
map.getPane("observationPane").style.zIndex = 475;
map.createPane("lsrPane");
map.getPane("lsrPane").style.zIndex = 485;

L.control.zoom({ position: "bottomright" }).addTo(map);
L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png", {
  subdomains: "abcd",
  maxZoom: 20,
  attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
}).addTo(map);
L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png", {
  pane: "labelPane",
  subdomains: "abcd",
  maxZoom: 20,
}).addTo(map);

let stateBoundaryData = null;
fetch("https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json")
  .then((response) => {
    if (!response.ok) throw new Error(`State boundaries unavailable (${response.status})`);
    return response.json();
  })
  .then((data) => {
    stateBoundaryData = data;
    L.geoJSON(data, {
      pane: "statePane",
      interactive: false,
      style: { color: "#b9c5cc", weight: 1.15, opacity: 0.8, fill: false },
    }).addTo(map);
    add3dStateLines();
  })
  .catch((error) => console.warn(error.message));

const canvasRenderer = L.canvas({ pane: "forecastPane", padding: 0.4, tolerance: 3 });

function colorRgba(hex, alpha = 255) {
  const value = Number.parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255, alpha];
}

function continuousRiskColor(probability, alpha = 255) {
  const stops = THRESHOLDS.map((threshold) => ({ threshold, color: colorRgba(RISK_COLORS[threshold]) }));
  if (probability <= stops[0].threshold) return [...stops[0].color.slice(0, 3), alpha];
  if (probability >= stops.at(-1).threshold) return [...stops.at(-1).color.slice(0, 3), alpha];
  for (let index = 1; index < stops.length; index += 1) {
    const upper = stops[index];
    const lower = stops[index - 1];
    if (probability > upper.threshold) continue;
    const fraction = (probability - lower.threshold) / (upper.threshold - lower.threshold);
    return [
      Math.round(lower.color[0] + (upper.color[0] - lower.color[0]) * fraction),
      Math.round(lower.color[1] + (upper.color[1] - lower.color[1]) * fraction),
      Math.round(lower.color[2] + (upper.color[2] - lower.color[2]) * fraction),
      alpha,
    ];
  }
  return [...stops.at(-1).color.slice(0, 3), alpha];
}

function add3dStateLines() {
  const map3d = state.map3d;
  if (!map3d?.isStyleLoaded() || !stateBoundaryData) return;
  if (!map3d.getSource("state-boundaries")) {
    map3d.addSource("state-boundaries", { type: "geojson", data: stateBoundaryData });
  }
  if (!map3d.getLayer("state-boundaries-top")) {
    map3d.addLayer({
      id: "state-boundaries-top",
      type: "line",
      source: "state-boundaries",
      paint: {
        "line-color": "#d5dde2",
        "line-width": 1.25,
        "line-opacity": 0.82,
      },
    });
  }
}

function first3dLabelLayer() {
  return state.map3d?.getStyle()?.layers?.find((layer) => layer.type === "symbol")?.id;
}

function createBoundaryIndex(lines) {
  const cellSize = 1;
  const buckets = new Map();
  let count = 0;
  for (const line of lines || []) {
    for (const coordinate of line) {
      const lat = Number(coordinate[0]);
      const lon = Number(coordinate[1]);
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
      const x = lon * CONUS_LONGITUDE_SCALE;
      const y = lat;
      const key = `${Math.floor(x / cellSize)},${Math.floor(y / cellSize)}`;
      if (!buckets.has(key)) buckets.set(key, []);
      buckets.get(key).push([x, y]);
      count += 1;
    }
  }
  return { buckets, cellSize, count };
}

function nearestBoundaryKm(index, lat, lon) {
  if (!index?.count) return Infinity;
  const x = lon * CONUS_LONGITUDE_SCALE;
  const y = lat;
  const baseX = Math.floor(x / index.cellSize);
  const baseY = Math.floor(y / index.cellSize);
  let bestSquared = Infinity;
  for (let ring = 0; ring <= 16; ring += 1) {
    for (let dx = -ring; dx <= ring; dx += 1) {
      for (let dy = -ring; dy <= ring; dy += 1) {
        if (ring && Math.abs(dx) !== ring && Math.abs(dy) !== ring) continue;
        const points = index.buckets.get(`${baseX + dx},${baseY + dy}`) || [];
        for (const point of points) {
          const distanceSquared = (point[0] - x) ** 2 + (point[1] - y) ** 2;
          if (distanceSquared < bestSquared) bestSquared = distanceSquared;
        }
      }
    }
    if (Number.isFinite(bestSquared) && Math.sqrt(bestSquared) <= Math.max(0.1, ring - 1) * index.cellSize) break;
  }
  return Math.sqrt(bestSquared) * 111.2;
}

function wpcSurfaceValues() {
  const cacheKey = `${state.data.date}:wpc-probabilities`;
  if (state.surface3dCache.has(cacheKey)) return state.surface3dCache.get(cacheKey);
  const encodedValues = state.data.layers.wpc.values;
  const contours = state.data.contours?.wpc || {};
  const present = [...new Set(encodedValues.filter((value) => value >= 50))].sort((a, b) => a - b);
  const maximumCategory = present.at(-1) || 0;
  const upperBound = { 50: 150, 150: 400, 400: 700, 700: 1000 };
  const boundaryIndexes = Object.fromEntries(THRESHOLDS.map((threshold) => [threshold * 10, createBoundaryIndex(contours[String(threshold)] || [])]));
  const probabilities = new Float32Array(encodedValues.length);

  for (let index = 0; index < encodedValues.length; index += 1) {
    const category = encodedValues[index];
    if (category < 50) continue;
    const upper = upperBound[category] || 1000;
    if (category === maximumCategory) {
      probabilities[index] = category / 10;
      continue;
    }
    const lat = state.data.grid.lat[index];
    const lon = state.data.grid.lon[index];
    const distanceOuter = nearestBoundaryKm(boundaryIndexes[category], lat, lon);
    const distanceInner = nearestBoundaryKm(boundaryIndexes[upper], lat, lon);
    if (!Number.isFinite(distanceOuter) || !Number.isFinite(distanceInner) || distanceInner > WPC_LOCAL_RISK_DISTANCE_KM) {
      probabilities[index] = category / 10;
      continue;
    }
    const fraction = distanceOuter / Math.max(0.001, distanceOuter + distanceInner);
    probabilities[index] = (category + (upper - category) * fraction) / 10;
  }
  state.surface3dCache.set(cacheKey, probabilities);
  return probabilities;
}

function surface3dData(key) {
  const cacheKey = `${state.data.date}:${key}`;
  if (state.surface3dCache.has(cacheKey)) return state.surface3dCache.get(cacheKey);
  const encodedValues = state.data.layers[key].values;
  const probabilities = key === "wpc" ? wpcSurfaceValues() : null;
  const points = [];
  for (let index = 0; index < encodedValues.length; index += 1) {
    if (encodedValues[index] < 50) continue;
    points.push({
      position: [state.data.grid.lon[index], state.data.grid.lat[index]],
      encoded: encodedValues[index],
      probability: probabilities ? probabilities[index] : encodedValues[index] / 10,
      wpcRange: key === "wpc" ? wpcRiskRange(encodedValues[index]) : null,
    });
  }
  state.surface3dCache.set(cacheKey, points);
  return points;
}

function wpcRiskRange(encodedValue) {
  if (encodedValue >= 700) return "70–100%";
  if (encodedValue >= 400) return "40–70%";
  if (encodedValue >= 150) return "15–40%";
  if (encodedValue >= 50) return "5–15%";
  return "Below 5%";
}

function nearestVisibleCity(position, x, y) {
  if (!state.map3d || !position || !Number.isFinite(x) || !Number.isFinite(y)) return "";
  const placeLayers = (state.map3d.getStyle()?.layers || [])
    .filter((layer) => layer.type === "symbol"
      && layer["source-layer"] === "place"
      && /place_(city|capital|town|village|hamlet)/.test(layer.id))
    .map((layer) => layer.id);
  if (!placeLayers.length) return "";
  let features = state.map3d.queryRenderedFeatures(
    [[x - 220, y - 220], [x + 220, y + 220]],
    { layers: placeLayers },
  );
  if (!features.some((feature) => feature.properties?.name_en || feature.properties?.name)) {
    features = state.map3d.querySourceFeatures("carto", { sourceLayer: "place" })
      .filter((feature) => ["city", "town", "village", "hamlet"].includes(
        String(feature.properties?.class || feature.properties?.type || "").toLowerCase(),
      ));
  }
  const origin = state.map3d.project(position);
  let nearest = null;
  let nearestDistance = Infinity;
  for (const feature of features) {
    const coordinates = feature.geometry?.coordinates;
    const name = feature.properties?.name_en || feature.properties?.name;
    if (!Array.isArray(coordinates) || !name) continue;
    const projected = state.map3d.project(coordinates);
    const distance = Math.hypot(projected.x - origin.x, projected.y - origin.y);
    if (distance < nearestDistance) {
      nearest = String(name);
      nearestDistance = distance;
    }
  }
  return nearest || "";
}

function visible3dReports() {
  const threshold = Number(document.getElementById("rain-threshold").value);
  const localReports = state.lsrReports.filter((report) => state.lsrTypes.has(report.kind)
    && (report.kind !== "rain" || (Number.isFinite(report.amount) && report.amount >= threshold)));
  return state.mpingVisible ? localReports.concat(state.mpingReports) : localReports;
}

function build3dLayers() {
  if (!state.data?.layers?.[state.selected] || !window.deck) return [];
  const surface = surface3dData(state.selected);
  let maximumProbability = 5;
  for (const point of surface) maximumProbability = Math.max(maximumProbability, point.probability);
  const referenceHeight = maximumProbability * SURFACE_HEIGHT_METERS_PER_PERCENT + OBSERVATION_CLEARANCE_METERS;
  const beforeId = first3dLabelLayer();
  const pointRadius = state.separated3dPoints ? SEPARATED_POINT_RADIUS_PIXELS : COMPACT_POINT_RADIUS_PIXELS;
  const shared = beforeId ? { beforeId } : {};
  const layers = [new deck.ColumnLayer({
    ...shared,
    id: `forecast-surface-${state.data.date}-${state.selected}`,
    data: surface,
    diskResolution: 8,
    radius: pointRadius,
    radiusUnits: "pixels",
    extruded: true,
    filled: true,
    wireframe: false,
    opacity: state.fillOpacity,
    pickable: true,
    getPosition: (point) => point.position,
    getElevation: (point) => point.probability * SURFACE_HEIGHT_METERS_PER_PERCENT,
    getFillColor: (point) => continuousRiskColor(point.probability),
    transitions: { getElevation: 350 },
  })];

  for (const key of state.contours) {
    const source = state.data.contours?.[key];
    if (!source) continue;
    for (const threshold of THRESHOLDS) {
      const paths = (source[String(threshold)] || []).map((line) => ({
        path: line.map(([lat, lon]) => [lon, lat, referenceHeight + threshold * 180]),
        key,
        threshold,
      }));
      if (!paths.length) continue;
      layers.push(new deck.PathLayer({
        ...shared,
        id: `contour-3d-${key}-${threshold}`,
        data: paths,
        getPath: (item) => item.path,
        getColor: colorRgba(RISK_COLORS[threshold]),
        getWidth: key === "pp" ? 5200 : 4200,
        widthUnits: "meters",
        widthMinPixels: key === "pp" ? 3 : 2,
        jointRounded: true,
        capRounded: true,
        getDashArray: PRODUCT_META[key]?.dash?.split(/\s+/).map(Number) || [0, 0],
        dashJustified: true,
        extensions: [new deck.PathStyleExtension({ dash: true })],
        pickable: true,
      }));
    }
  }

  const observations = [];
  for (const key of state.observations) {
    const source = state.data.observations?.[key];
    const meta = OBSERVATION_META[key] || { label: source?.label || key, color: "#fff" };
    for (const [lat, lon] of source?.points || []) observations.push({ position: [lon, lat], meta });
  }
  if (observations.length) layers.push(new deck.ColumnLayer({
    ...shared,
    id: "verification-observations-3d",
    data: observations,
    diskResolution: 10,
    radius: pointRadius,
    radiusUnits: "pixels",
    extruded: true,
    filled: true,
    wireframe: false,
    opacity: state.fillOpacity,
    pickable: true,
    getPosition: (item) => item.position,
    getElevation: referenceHeight,
    getFillColor: (item) => colorRgba(item.meta.color),
  }));

  const reports = visible3dReports();
  if (reports.length) layers.push(new deck.ColumnLayer({
    ...shared,
    id: "local-storm-reports-3d",
    data: reports,
    diskResolution: 12,
    radius: pointRadius,
    radiusUnits: "pixels",
    extruded: true,
    filled: true,
    wireframe: false,
    opacity: state.fillOpacity,
    pickable: true,
    getPosition: (report) => [report.lon, report.lat],
    getElevation: referenceHeight + 10000,
    getFillColor: (report) => colorRgba(LSR_META[report.kind].color),
  }));
  if (state.showExpansionRings) {
    const ringHeight = referenceHeight + 22000;
    const rings = observations.map((item) => ({
      position: [item.position[0], item.position[1], ringHeight],
      meta: item.meta,
      expansionRing: true,
    }));
    for (const report of reports) {
      if (report.provider === "mping") continue;
      rings.push({
        position: [report.lon, report.lat, ringHeight],
        meta: LSR_META[report.kind],
        expansionRing: true,
      });
    }
    if (rings.length) layers.push(new deck.ScatterplotLayer({
      ...shared,
      id: "forty-km-expansion-rings-3d",
      data: rings,
      radiusUnits: "meters",
      lineWidthUnits: "pixels",
      stroked: true,
      filled: true,
      pickable: true,
      getPosition: (item) => item.position,
      getRadius: EXPANSION_RADIUS_METERS,
      getLineColor: (item) => colorRgba(item.meta.color, 230),
      getFillColor: (item) => colorRgba(item.meta.color, 10),
      getLineWidth: 1.5,
      lineWidthMinPixels: 1.25,
    }));
  }
  return layers;
}

function render3d() {
  if (state.viewMode !== "3d" || !state.deckOverlay || !state.map3d?.isStyleLoaded()) return;
  state.deckOverlay.setProps({
    layers: build3dLayers(),
    getTooltip: ({ object, x, y }) => {
      if (!object) return null;
      if (object.expansionRing) return `${object.meta.label} · 40-km expansion`;
      if (object.probability) {
        const city = nearestVisibleCity(object.position, x, y);
        const value = state.selected === "wpc"
          ? `${PRODUCT_META.wpc.short}: ${object.wpcRange} category`
          : `${PRODUCT_META[state.selected].short}: ${object.probability.toFixed(1)}%`;
        return city ? `${value}\nNearest city: ${city}` : value;
      }
      if (object.threshold) return `${PRODUCT_META[object.key]?.short || object.key}: >${object.threshold}% contour`;
      if (object.kind) return `${LSR_META[object.kind]?.label || object.type}${Number.isFinite(object.amount) ? ` · ${object.amount.toFixed(2)} in` : ""}`;
      if (object.meta) return object.meta.label;
      return null;
    },
  });
  add3dStateLines();
}

function schedule3dRender() {
  if (state.viewMode !== "3d") return;
  cancelAnimationFrame(state.render3dFrame);
  state.render3dFrame = requestAnimationFrame(render3d);
}

function initialize3dMap() {
  if (state.map3d) return;
  if (!window.maplibregl || !window.deck?.MapboxOverlay) throw new Error("The 3D map libraries did not load");
  const center = map.getCenter();
  state.map3d = new maplibregl.Map({
    container: "map-3d",
    style: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    center: [center.lng, center.lat],
    zoom: map.getZoom(),
    pitch: 20,
    bearing: 0,
    minPitch: 20,
    maxPitch: 20,
    minZoom: 3,
    maxZoom: 9,
    dragRotate: false,
    touchPitch: false,
    pitchWithRotate: false,
    antialias: true,
    attributionControl: true,
  });
  state.map3d.touchZoomRotate.disableRotation();
  state.map3d.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
  state.deckOverlay = new deck.MapboxOverlay({ interleaved: true, layers: [] });
  state.map3d.addControl(state.deckOverlay);
  state.map3d.on("load", () => {
    add3dStateLines();
    render3d();
  });
  state.map3d.on("error", (event) => {
    console.error("3D map error", event.error || event);
    if (state.viewMode === "3d") {
      document.getElementById("product-message").textContent = "The 3D basemap could not be loaded. The standard 2D view remains available.";
    }
  });
}

function setViewMode(mode) {
  if (mode === state.viewMode) return;
  const map2dElement = document.getElementById("map");
  const map3dElement = document.getElementById("map-3d");
  if (mode === "3d") {
    map3dElement.hidden = false;
    try {
      initialize3dMap();
    } catch (error) {
      map3dElement.hidden = true;
      document.getElementById("product-message").textContent = "This browser could not start the 3D map. The standard 2D view remains available.";
      console.error(error);
      return;
    }
    const center = map.getCenter();
    state.map3d.jumpTo({ center: [center.lng, center.lat], zoom: map.getZoom(), pitch: 20, bearing: 0 });
    state.viewMode = "3d";
    map2dElement.hidden = true;
    state.map3d.resize();
    render3d();
  } else {
    const center = state.map3d.getCenter();
    map.setView([center.lat, center.lng], state.map3d.getZoom(), { animate: false });
    state.viewMode = "2d";
    map3dElement.hidden = true;
    map2dElement.hidden = false;
    map.invalidateSize();
    renderFilledLayer();
    renderContours();
    renderObservations();
    renderLsrs();
  }
  document.getElementById("height-legend").hidden = mode !== "3d";
  document.getElementById("point-gap-control").hidden = mode !== "3d";
  document.getElementById("opacity-control-label").textContent = mode === "3d" ? "3D point opacity" : "Forecast opacity";
  for (const candidate of ["2d", "3d"]) {
    const button = document.getElementById(`view-${candidate}`);
    const active = candidate === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  }
  if (state.data?.date) history.replaceState(null, "", `?date=${state.data.date}${mode === "3d" ? "&view=3d" : ""}`);
}

function riskColor(encodedValue) {
  if (encodedValue >= 700) return RISK_COLORS[70];
  if (encodedValue >= 400) return RISK_COLORS[40];
  if (encodedValue >= 150) return RISK_COLORS[15];
  if (encodedValue >= 50) return RISK_COLORS[5];
  return null;
}

function riskLabel(encodedValue) {
  if (encodedValue >= 700) return ">70%";
  if (encodedValue >= 400) return ">40%";
  if (encodedValue >= 150) return ">15%";
  if (encodedValue >= 50) return ">5%";
  return "<5%";
}

function mapPath(entry) {
  return entry.map_href || `archive/${entry.date}/map.json`;
}

function showLoading(message = "Loading map data…") {
  const loading = document.getElementById("loading");
  loading.textContent = message;
  loading.hidden = false;
}

function hideLoading() {
  document.getElementById("loading").hidden = true;
}

function setMessage(key) {
  const radius = { ml_r40: "40 km (25 mi)", ml_r60: "60 km (37 mi)", ml_r75: "75 km (47 mi)", ml_r100: "100 km (62 mi)" }[key];
  let prediction = "";
  if (radius) prediction = ` It predicts the probability that observed rainfall will exceed Flash Flood Guidance within ${radius} of a point.`;
  if (key === "ml_lpmm") prediction = " It combines all four ML configurations using a 300-km radius of influence while retaining local probability extremes.";
  if (key === "wpc") prediction = " It predicts the probability of rainfall exceeding Flash Flood Guidance within 40 km (25 mi) of a point.";
  if (key === "pp") prediction = " It shows an observation-based, idealized placement of risk after the valid period—not a forecast.";
  document.getElementById("product-message").textContent = `${PRODUCT_META[key]?.note || ""}${prediction}`;
}

function renderFilledLayer() {
  if (!state.data || !state.data.layers[state.selected]) return;
  if (state.viewMode === "3d") {
    setMessage(state.selected);
    schedule3dRender();
    return;
  }
  if (state.fillLayer) map.removeLayer(state.fillLayer);

  const values = state.data.layers[state.selected].values;
  const lat = state.data.grid.lat;
  const lon = state.data.grid.lon;
  const group = L.layerGroup();
  const radius = Math.max(2.2, Math.min(4.2, 2.2 + (map.getZoom() - 4) * 0.35));

  for (let index = 0; index < values.length; index += 1) {
    const color = riskColor(values[index]);
    if (!color) continue;
    L.circleMarker([lat[index], lon[index]], {
      pane: "forecastPane",
      renderer: canvasRenderer,
      radius,
      stroke: false,
      fill: true,
      fillColor: color,
      fillOpacity: state.fillOpacity,
      interactive: false,
    }).addTo(group);
  }

  state.fillLayer = group.addTo(map);
  setMessage(state.selected);
}

function renderContours() {
  if (state.viewMode === "3d") {
    schedule3dRender();
    return;
  }
  if (state.contourLayer) map.removeLayer(state.contourLayer);
  const group = L.layerGroup();

  for (const key of state.contours) {
    const layerContours = state.data?.contours?.[key];
    if (!layerContours) continue;
    for (const threshold of THRESHOLDS) {
      const lines = layerContours[String(threshold)] || [];
      for (const line of lines) {
        L.polyline(line, {
          pane: "contourPane",
          color: "#080a0c",
          weight: key === "pp" ? 6.4 : 5.8,
          opacity: 0.9,
          interactive: false,
        }).addTo(group);
        const polyline = L.polyline(line, {
          pane: "contourPane",
          color: RISK_COLORS[threshold],
          weight: key === "pp" ? 3.8 : 3.3,
          opacity: 1,
          dashArray: PRODUCT_META[key]?.dash,
          interactive: true,
        });
        polyline.bindTooltip(`${PRODUCT_META[key]?.short || key} · >${threshold}%`, {
          sticky: true,
          direction: "top",
        });
        polyline.addTo(group);
      }
    }
  }
  state.contourLayer = group.addTo(map);
}

function renderObservations() {
  if (state.viewMode === "3d") {
    schedule3dRender();
    return;
  }
  if (state.observationLayer) map.removeLayer(state.observationLayer);
  const group = L.layerGroup();
  for (const key of state.observations) {
    const source = state.data?.observations?.[key];
    if (!source) continue;
    const meta = OBSERVATION_META[key] || { label: source.label || key, color: "#fff" };
    for (const point of source.points || []) {
      if (state.showExpansionRings) {
        L.circle(point, {
          pane: "observationPane",
          radius: EXPANSION_RADIUS_METERS,
          color: meta.color,
          weight: 1.5,
          opacity: 0.85,
          fill: true,
          fillColor: meta.color,
          fillOpacity: 0.025,
          interactive: false,
        }).addTo(group);
      }
      L.circleMarker(point, {
        pane: "observationPane",
        radius: 4.2,
        color: "#07090b",
        weight: 1.5,
        fillColor: meta.color,
        fillOpacity: 1,
      }).bindTooltip(meta.label, { direction: "top" }).addTo(group);
    }
  }
  state.observationLayer = group.addTo(map);
}

function lsrPopup(report) {
  const container = document.createElement("div");
  container.className = "lsr-popup";
  const heading = document.createElement("strong");
  const formattedAmount = Number.isFinite(report.amount) ? Number(report.amount.toFixed(2)).toString() : "";
  const amount = report.kind === "rain" && formattedAmount ? ` · ${formattedAmount} in` : "";
  heading.textContent = `${LSR_META[report.kind]?.label || report.type}${amount}`;
  container.append(heading);
  const lines = [
    report.valid ? new Date(report.valid).toLocaleString() : "",
    [report.city, report.county, report.state].filter(Boolean).join(", "),
    report.provider === "mping" ? "Source: mPING citizen report" : (report.source ? `Source: ${report.source}` : ""),
    report.remark || "",
  ].filter(Boolean);
  for (const text of lines) {
    const line = document.createElement("div");
    line.textContent = text;
    container.append(line);
  }
  return container;
}

function renderLsrs() {
  if (state.viewMode === "3d") {
    schedule3dRender();
    return;
  }
  if (state.lsrLayer) map.removeLayer(state.lsrLayer);
  const threshold = Number(document.getElementById("rain-threshold").value);
  const group = L.layerGroup();
  const reports = state.mpingVisible ? state.lsrReports.concat(state.mpingReports) : state.lsrReports;
  for (const report of reports) {
    if (report.provider !== "mping" && !state.lsrTypes.has(report.kind)) continue;
    if (report.kind === "rain" && (!Number.isFinite(report.amount) || report.amount < threshold)) continue;
    const meta = LSR_META[report.kind];
    if (state.showExpansionRings && report.provider !== "mping") {
      L.circle([report.lat, report.lon], {
        pane: "lsrPane",
        radius: EXPANSION_RADIUS_METERS,
        color: meta.color,
        weight: 1.5,
        opacity: 0.85,
        fill: true,
        fillColor: meta.color,
        fillOpacity: 0.025,
        interactive: false,
      }).addTo(group);
    }
    L.circleMarker([report.lat, report.lon], {
      pane: "lsrPane",
      radius: report.kind === "rain" ? 5 : 6,
      color: "#050607",
      weight: 2,
      fillColor: meta.color,
      fillOpacity: 1,
    }).bindPopup(lsrPopup(report), { maxWidth: 330 }).addTo(group);
  }
  state.lsrLayer = group.addTo(map);
}

function forecastWindow(date) {
  const start = new Date(`${date.slice(0, 4)}-${date.slice(4, 6)}-${date.slice(6, 8)}T12:00:00Z`);
  const end = new Date(start);
  end.setUTCDate(end.getUTCDate() + 1);
  return { start, end };
}

function parseLsrFeature(feature) {
  const properties = feature?.properties || {};
  const coordinates = feature?.geometry?.coordinates || [];
  const type = String(properties.typetext || "").toUpperCase();
  const kind = type === "FLASH FLOOD" ? "flash_flood"
    : type === "FLOOD" ? "flood"
      : ["RAIN", "HEAVY RAIN"].includes(type) ? "rain" : null;
  const lon = Number(coordinates[0]);
  const lat = Number(coordinates[1]);
  if (!kind || !Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  const rawAmount = properties.magf ?? properties.magnitude;
  const numericAmount = rawAmount === null || rawAmount === "" ? NaN : Number(rawAmount);
  const unit = String(properties.unit || "").toLowerCase();
  const amount = Number.isFinite(numericAmount) && unit.includes("mm") ? numericAmount / 25.4 : numericAmount;
  return {
    kind,
    type,
    lat,
    lon,
    amount: Number.isFinite(amount) ? amount : null,
    valid: properties.valid || "",
    city: properties.city || "",
    county: properties.county || "",
    state: properties.state || properties.st || "",
    source: properties.source || "",
    remark: properties.remark || "",
  };
}

async function fetchLsrs(date, scheduleRefresh = false) {
  const request = ++state.lsrRequest;
  const window = forecastWindow(date);
  const start = window.start.toISOString().slice(0, 16) + "Z";
  const params = new URLSearchParams({
    west: "-105.1", east: "-80.4", south: "30", north: "50.1",
    sts: start, ets: window.end.toISOString().slice(0, 16) + "Z",
  });
  const status = document.getElementById("lsr-status");
  state.lsrReports = [];
  renderLsrs();
  status.textContent = "Loading NWS local storm reports via Iowa Environmental Mesonet…";
  try {
    const response = await fetch(`https://mesonet.agron.iastate.edu/geojson/lsr.geojson?${params}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`IEM LSR request failed (${response.status})`);
    const data = await response.json();
    if (request !== state.lsrRequest) return;
    state.lsrReports = (data.features || []).map(parseLsrFeature).filter((report) => {
      if (!report) return false;
      if (report.kind !== "rain") return true;
      const valid = Date.parse(report.valid);
      return Number.isFinite(valid) && valid >= window.start.getTime() && valid < window.end.getTime();
    });
    renderLsrs();
    const counts = Object.fromEntries(Object.keys(LSR_META).map((key) => [key, state.lsrReports.filter((report) => report.kind === key).length]));
    status.textContent = `Preliminary: ${counts.flash_flood} flash flood, ${counts.flood} flood, ${counts.rain} rain reports. Updated ${new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}.`;
  } catch (error) {
    if (request !== state.lsrRequest) return;
    state.lsrReports = [];
    renderLsrs();
    status.textContent = "Local storm reports are temporarily unavailable.";
    console.error(error);
  }
  clearTimeout(state.lsrTimer);
  if (scheduleRefresh) state.lsrTimer = setTimeout(() => fetchLsrs(date, true), LSR_REFRESH_MS);
}

async function fetchMping(date) {
  const request = ++state.mpingRequest;
  const status = document.getElementById("mping-status");
  state.mpingReports = [];
  renderLsrs();
  status.textContent = "Loading mPING flood reports…";
  try {
    const response = await fetch(`archive/${date}/mping.json?v=${Date.now()}`, { cache: "no-store" });
    if (response.status === 404) {
      status.textContent = "mPING flood reports are not available for this valid period.";
      return;
    }
    if (!response.ok) throw new Error(`mPING report file unavailable (${response.status})`);
    const data = await response.json();
    if (request !== state.mpingRequest) return;
    state.mpingReports = (data.reports || []).map((report) => ({
      kind: "mping_flood",
      provider: "mping",
      type: report.description || "Flood impact",
      lat: Number(report.lat),
      lon: Number(report.lon),
      valid: report.valid || "",
      remark: report.description || "",
    })).filter((report) => Number.isFinite(report.lat) && Number.isFinite(report.lon));
    renderLsrs();
    status.textContent = `${state.mpingReports.length} mPING flood impact report${state.mpingReports.length === 1 ? "" : "s"} during this valid period.`;
  } catch (error) {
    if (request !== state.mpingRequest) return;
    state.mpingReports = [];
    renderLsrs();
    status.textContent = "mPING flood reports are temporarily unavailable.";
    console.error(error);
  }
}

function showProductInfo(key) {
  const meta = PRODUCT_META[key];
  if (!meta) return;
  document.getElementById("product-dialog-title").textContent = meta.title;
  document.getElementById("product-dialog-content").innerHTML = `<p class="lead">${meta.note}</p><p>${meta.detail}</p>`;
  document.getElementById("product-dialog").showModal();
}

function buildLayerControls() {
  const productContainer = document.getElementById("product-options");
  const contourContainer = document.getElementById("contour-options");
  productContainer.replaceChildren();
  contourContainer.replaceChildren();

  for (const key of PRODUCT_ORDER) {
    const available = Boolean(state.data?.layers?.[key]);
    const meta = PRODUCT_META[key];

    const productRow = document.createElement("div");
    productRow.className = "product-row";
    const productButton = document.createElement("button");
    productButton.type = "button";
    productButton.className = `product-choice${state.selected === key ? " active" : ""}`;
    productButton.textContent = meta.short;
    productButton.disabled = !available;
    productButton.addEventListener("click", () => {
      state.selected = key;
      buildLayerControls();
      renderFilledLayer();
    });
    const infoButton = document.createElement("button");
    infoButton.type = "button";
    infoButton.className = "info-mini";
    infoButton.textContent = "i";
    infoButton.setAttribute("aria-label", `About ${meta.title}`);
    infoButton.addEventListener("click", () => showProductInfo(key));
    productRow.append(productButton, infoButton);
    productContainer.append(productRow);

    const contourRow = document.createElement("div");
    contourRow.className = "contour-row";
    const contourLabel = document.createElement("label");
    contourLabel.className = "contour-choice";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = state.contours.has(key);
    checkbox.disabled = !available;
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) state.contours.add(key);
      else state.contours.delete(key);
      renderContours();
    });
    contourLabel.append(checkbox, document.createTextNode(meta.short));
    const contourInfo = infoButton.cloneNode(true);
    contourInfo.addEventListener("click", () => showProductInfo(key));
    contourRow.append(contourLabel, contourInfo);
    contourContainer.append(contourRow);
  }

  const observationSection = document.getElementById("observation-section");
  const observationContainer = document.getElementById("observation-options");
  observationContainer.replaceChildren();
  const availableObservations = Object.entries(state.data?.observations || {}).filter(([, source]) => source?.points?.length);
  observationSection.hidden = availableObservations.length === 0;
  for (const [key, source] of availableObservations) {
    const meta = OBSERVATION_META[key] || { label: source.label || key, color: "#fff" };
    const label = document.createElement("label");
    label.className = "observation-choice";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = state.observations.has(key);
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) state.observations.add(key);
      else state.observations.delete(key);
      renderObservations();
    });
    const swatch = document.createElement("i");
    swatch.style.setProperty("--observation-color", meta.color);
    label.append(checkbox, swatch, document.createTextNode(`${meta.label} (${source.points.length})`));
    observationContainer.append(label);
  }

  const hasVerification = Boolean(state.data?.layers?.pp);
  document.getElementById("verification-availability").textContent = hasVerification
    ? "Practically Perfect verification is available for this valid period."
    : "Practically Perfect verification typically arrives around 11:10 AM CT the following day.";
}

function updateDateUI(entry) {
  document.getElementById("valid-period").textContent = `Valid ${state.data.valid_period_label}`;
  const staticHref = entry?.plot_href || `archive/${state.data.date}/latest.png`;
  document.getElementById("current-png-link").href = `${staticHref}?v=${encodeURIComponent(entry?.site_updated_utc || state.data.generated_utc)}`;
  const verificationLink = document.getElementById("current-verification-link");
  if (entry?.verification_available && entry.verification_plot_href) {
    verificationLink.href = `${entry.verification_plot_href}?v=${encodeURIComponent(entry.verification_updated_utc || entry.site_updated_utc || state.data.generated_utc)}`;
    verificationLink.textContent = entry.verification_embedded_in_forecast ? "Combined PNG" : "Verification PNG";
    verificationLink.hidden = false;
  } else {
    verificationLink.hidden = true;
    verificationLink.removeAttribute("href");
  }
  document.getElementById("date-select").value = state.data.date;
}

async function loadDate(date, fit = false) {
  const entry = state.archive.find((item) => String(item.date) === String(date));
  showLoading(`Loading ${date}…`);
  try {
    const response = await fetch(`${mapPath(entry || { date })}?v=${encodeURIComponent(entry?.map_updated_utc || entry?.site_updated_utc || Date.now())}`);
    if (!response.ok) throw new Error(`Map data unavailable (${response.status})`);
    state.data = await response.json();
    state.surface3dCache.clear();
    if (!state.data.layers[state.selected]) {
      state.selected = state.data.layers.ml_r60 ? "ml_r60" : Object.keys(state.data.layers)[0];
    }
    state.contours = new Set([...state.contours].filter((key) => state.data.layers[key]));
    state.observations = new Set([...state.observations].filter((key) => state.data.observations?.[key]));
    buildLayerControls();
    renderFilledLayer();
    renderContours();
    renderObservations();
    updateDateUI(entry);
    const isLatest = String(state.archive[0]?.date) === String(state.data.date);
    clearTimeout(state.lsrTimer);
    fetchLsrs(state.data.date, isLatest);
    fetchMping(state.data.date);
    if (fit) map.fitBounds([[30, -105], [50, -80.5]], { padding: [15, 15] });
    history.replaceState(null, "", `?date=${state.data.date}${state.viewMode === "3d" ? "&view=3d" : ""}`);
  } catch (error) {
    document.getElementById("product-message").textContent = `Interactive data are unavailable for ${date}. Use the PNG link or archive.`;
    console.error(error);
  } finally {
    hideLoading();
  }
}

function populateDates() {
  const select = document.getElementById("date-select");
  select.replaceChildren();
  for (const entry of state.archive) {
    const option = document.createElement("option");
    option.value = entry.date;
    option.textContent = `${String(entry.date).slice(0, 4)}-${String(entry.date).slice(4, 6)}-${String(entry.date).slice(6, 8)}`;
    option.disabled = entry.map_available === false;
    select.append(option);
  }
  select.addEventListener("change", () => loadDate(select.value));
}

function populateArchive() {
  const rows = document.getElementById("archive-rows");
  rows.replaceChildren();
  for (const entry of state.archive) {
    const row = document.createElement("tr");
    const dateCell = document.createElement("td");
    const validCell = document.createElement("td");
    const mapCell = document.createElement("td");
    const staticCell = document.createElement("td");
    const verificationCell = document.createElement("td");
    dateCell.textContent = entry.date;
    validCell.textContent = entry.valid_period_label || "—";

    const loadButton = document.createElement("button");
    loadButton.type = "button";
    loadButton.className = "archive-load";
    loadButton.textContent = entry.map_available === false ? "Unavailable" : "Load map";
    loadButton.disabled = entry.map_available === false;
    loadButton.addEventListener("click", () => {
      document.getElementById("archive-dialog").close();
      loadDate(entry.date);
    });
    mapCell.append(loadButton);

    if (entry.plot_href) {
      const link = document.createElement("a");
      link.href = entry.plot_href;
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = "Open PNG";
      staticCell.append(link);
    } else {
      staticCell.textContent = "—";
    }

    if (entry.verification_available && entry.verification_plot_href) {
      const link = document.createElement("a");
      link.href = entry.verification_plot_href;
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = entry.verification_embedded_in_forecast ? "Combined PNG" : "Open PNG";
      link.title = entry.verification_embedded_in_forecast
        ? "Forecast and Practically Perfect verification in one image"
        : "Open Practically Perfect verification image";
      verificationCell.append(link);
    } else {
      verificationCell.textContent = "Pending";
      verificationCell.className = "pending-cell";
    }
    row.append(dateCell, validCell, mapCell, staticCell, verificationCell);
    rows.append(row);
  }
}

function setupDialogs() {
  document.getElementById("about-button").addEventListener("click", () => document.getElementById("about-dialog").showModal());
  document.getElementById("archive-button").addEventListener("click", () => document.getElementById("archive-dialog").showModal());
  document.querySelectorAll(".dialog-close").forEach((button) => {
    button.addEventListener("click", () => button.closest("dialog").close());
  });
  document.querySelectorAll("dialog").forEach((dialog) => {
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) dialog.close();
    });
  });

  try {
    if (!localStorage.getItem("ml-flood-map-intro-seen")) {
      document.getElementById("about-dialog").showModal();
      localStorage.setItem("ml-flood-map-intro-seen", "1");
    }
  } catch (_) {
    document.getElementById("about-dialog").showModal();
  }
}

function setupResponsiveControls() {
  const mobile = window.matchMedia("(max-width: 900px)");
  const layerContent = document.getElementById("layer-panel-content");
  const layerToggle = document.getElementById("collapse-layers");
  const actions = document.querySelector(".top-actions");
  const actionsToggle = document.getElementById("mobile-actions-toggle");

  if (mobile.matches) {
    layerContent.hidden = true;
    layerToggle.textContent = "+";
    layerToggle.setAttribute("aria-label", "Expand layer controls");
  }
  actionsToggle.addEventListener("click", () => {
    const open = actions.classList.toggle("mobile-open");
    actionsToggle.setAttribute("aria-expanded", String(open));
    actionsToggle.textContent = open ? "Close" : "Menu";
    actionsToggle.setAttribute("aria-label", open ? "Close map actions" : "Open map actions");
  });
  actions.querySelectorAll("a, button:not(#mobile-actions-toggle)").forEach((control) => {
    control.addEventListener("click", () => {
      if (!mobile.matches) return;
      actions.classList.remove("mobile-open");
      actionsToggle.setAttribute("aria-expanded", "false");
      actionsToggle.textContent = "Menu";
    });
  });
}

document.getElementById("collapse-layers").addEventListener("click", (event) => {
  const content = document.getElementById("layer-panel-content");
  content.hidden = !content.hidden;
  event.currentTarget.textContent = content.hidden ? "+" : "−";
  event.currentTarget.setAttribute("aria-label", content.hidden ? "Expand layer controls" : "Collapse layer controls");
});

document.getElementById("view-2d").addEventListener("click", () => setViewMode("2d"));
document.getElementById("view-3d").addEventListener("click", () => setViewMode("3d"));
document.getElementById("point-gap-toggle").addEventListener("change", (event) => {
  state.separated3dPoints = event.currentTarget.checked;
  schedule3dRender();
});
document.getElementById("mping-flood-toggle").addEventListener("change", (event) => {
  state.mpingVisible = event.currentTarget.checked;
  renderLsrs();
});
document.getElementById("expansion-ring-toggle").addEventListener("change", (event) => {
  state.showExpansionRings = event.currentTarget.checked;
  if (state.viewMode === "3d") schedule3dRender();
  else {
    renderObservations();
    renderLsrs();
  }
});

const opacityInput = document.getElementById("fill-opacity");
const opacityOutput = document.getElementById("fill-opacity-value");
opacityInput.addEventListener("input", () => {
  state.fillOpacity = Number(opacityInput.value) / 100;
  opacityOutput.value = `${opacityInput.value}%`;
  renderFilledLayer();
});

document.querySelectorAll(".lsr-options input").forEach((checkbox) => {
  checkbox.addEventListener("change", () => {
    if (checkbox.checked) state.lsrTypes.add(checkbox.value);
    else state.lsrTypes.delete(checkbox.value);
    renderLsrs();
  });
});
document.getElementById("rain-threshold").addEventListener("change", renderLsrs);

map.on("zoomend", () => {
  renderFilledLayer();
  renderContours();
  renderObservations();
  renderLsrs();
});

async function init() {
  setupDialogs();
  setupResponsiveControls();
  try {
    const response = await fetch(`archive/index.json?v=${Date.now()}`);
    if (!response.ok) throw new Error("Archive index unavailable");
    const archive = await response.json();
    state.archive = Array.isArray(archive) ? archive : archive.entries || [];
    populateDates();
    populateArchive();
    const parameters = new URLSearchParams(location.search);
    const requested = parameters.get("date");
    const initial = state.archive.find((entry) => entry.date === requested && entry.map_available !== false)
      || state.archive.find((entry) => entry.map_available !== false)
      || state.archive[0];
    if (!initial) throw new Error("No forecasts are available");
    await loadDate(initial.date, true);
    if (parameters.get("view") === "3d") setViewMode("3d");
  } catch (error) {
    document.getElementById("product-message").textContent = "Forecast data could not be loaded. Please try again shortly.";
    hideLoading();
    console.error(error);
  }
}

init();
