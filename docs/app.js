"use strict";

const RISK_COLORS = {
  5: "#57c84d",
  15: "#f0df35",
  40: "#e14b3f",
  70: "#d94ad7",
};

const PRODUCT_META = {
  ml_r40: {
    short: "ML r40",
    title: "40-km radius ML",
    note: "40-km radius ML forecasts are typically the most conservative in both risk size and severity.",
    detail: "Predicts rainfall exceeding Flash Flood Guidance within 40 km. Test-set analysis shows that this configuration often produces risk areas that are too small and too weak.",
    dash: null,
  },
  ml_r60: {
    short: "ML r60",
    title: "60-km radius ML",
    note: "60-km radius ML forecasts were the most balanced overall in the test-set analysis.",
    detail: "Predicts rainfall exceeding Flash Flood Guidance within 60 km. This configuration provided the best balance between missed events and overly broad or severe risk areas in the test set.",
    dash: "8 4",
  },
  ml_r75: {
    short: "ML r75",
    title: "75-km radius ML",
    note: "75-km radius ML forecasts were a close second to 60 km for overall balance.",
    detail: "Predicts rainfall exceeding Flash Flood Guidance within 75 km. It performed similarly to 60 km while tending toward somewhat broader and stronger risk areas.",
    dash: "3 5",
  },
  ml_r100: {
    short: "ML r100",
    title: "100-km radius ML",
    note: "100-km radius ML forecasts are typically the most aggressive.",
    detail: "Predicts rainfall exceeding Flash Flood Guidance within 100 km. Test-set analysis shows that risk areas can be too large and that high probabilities can be issued too frequently.",
    dash: "12 5 3 5",
  },
  wpc: {
    short: "WPC ERO",
    title: "WPC Excessive Rainfall Outlook",
    note: "WPC ERO is the official reference forecast shown alongside the experimental ML guidance.",
    detail: "The WPC Excessive Rainfall Outlook is the official categorical reference forecast. Average test-set results indicate better ML skill for Moderate-or-greater risks, but performance varies by event.",
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

const PRODUCT_ORDER = ["ml_r40", "ml_r60", "ml_r75", "ml_r100", "wpc", "pp"];
const THRESHOLDS = [5, 15, 40, 70];

const state = {
  archive: [],
  data: null,
  selected: "ml_r60",
  contours: new Set(),
  fillLayer: null,
  contourLayer: null,
};

const map = L.map("map", {
  zoomControl: false,
  preferCanvas: true,
  minZoom: 3,
  maxZoom: 9,
}).setView([39.5, -92.5], 5);

L.control.zoom({ position: "bottomright" }).addTo(map);
L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
  subdomains: "abcd",
  maxZoom: 20,
  attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
}).addTo(map);

const canvasRenderer = L.canvas({ padding: 0.4, tolerance: 3 });

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
  document.getElementById("product-message").textContent = PRODUCT_META[key]?.note || "";
}

function renderFilledLayer() {
  if (!state.data || !state.data.layers[state.selected]) return;
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
      renderer: canvasRenderer,
      radius,
      stroke: false,
      fill: true,
      fillColor: color,
      fillOpacity: 0.78,
      interactive: false,
    }).addTo(group);
  }

  state.fillLayer = group.addTo(map);
  setMessage(state.selected);
}

function renderContours() {
  if (state.contourLayer) map.removeLayer(state.contourLayer);
  const group = L.layerGroup();

  for (const key of state.contours) {
    const layerContours = state.data?.contours?.[key];
    if (!layerContours) continue;
    for (const threshold of THRESHOLDS) {
      const lines = layerContours[String(threshold)] || [];
      for (const line of lines) {
        const polyline = L.polyline(line, {
          color: RISK_COLORS[threshold],
          weight: key === "pp" ? 2.8 : 2.1,
          opacity: 0.95,
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

  const hasVerification = Boolean(state.data?.layers?.pp);
  document.getElementById("verification-availability").textContent = hasVerification
    ? "Practically Perfect verification is available for this valid period."
    : "Practically Perfect verification typically arrives around 11:10 AM CT the following day.";
}

function updateDateUI(entry) {
  document.getElementById("valid-period").textContent = `Valid ${state.data.valid_period_label}`;
  const staticHref = entry?.plot_href || `archive/${state.data.date}/latest.png`;
  document.getElementById("current-png-link").href = `${staticHref}?v=${encodeURIComponent(entry?.site_updated_utc || state.data.generated_utc)}`;
  document.getElementById("date-select").value = state.data.date;
}

async function loadDate(date, fit = false) {
  const entry = state.archive.find((item) => String(item.date) === String(date));
  showLoading(`Loading ${date}…`);
  try {
    const response = await fetch(`${mapPath(entry || { date })}?v=${encodeURIComponent(entry?.map_updated_utc || entry?.site_updated_utc || Date.now())}`);
    if (!response.ok) throw new Error(`Map data unavailable (${response.status})`);
    state.data = await response.json();
    if (!state.data.layers[state.selected]) {
      state.selected = state.data.layers.ml_r60 ? "ml_r60" : Object.keys(state.data.layers)[0];
    }
    state.contours = new Set([...state.contours].filter((key) => state.data.layers[key]));
    buildLayerControls();
    renderFilledLayer();
    renderContours();
    updateDateUI(entry);
    if (fit) map.fitBounds([[30, -105], [50, -80.5]], { padding: [15, 15] });
    history.replaceState(null, "", `?date=${state.data.date}`);
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
    row.append(dateCell, validCell, mapCell, staticCell);
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

document.getElementById("collapse-layers").addEventListener("click", (event) => {
  const content = document.getElementById("layer-panel-content");
  content.hidden = !content.hidden;
  event.currentTarget.textContent = content.hidden ? "+" : "−";
});

map.on("zoomend", () => {
  renderFilledLayer();
  renderContours();
});

async function init() {
  setupDialogs();
  try {
    const response = await fetch(`archive/index.json?v=${Date.now()}`);
    if (!response.ok) throw new Error("Archive index unavailable");
    const archive = await response.json();
    state.archive = Array.isArray(archive) ? archive : archive.entries || [];
    populateDates();
    populateArchive();
    const requested = new URLSearchParams(location.search).get("date");
    const initial = state.archive.find((entry) => entry.date === requested && entry.map_available !== false)
      || state.archive.find((entry) => entry.map_available !== false)
      || state.archive[0];
    if (!initial) throw new Error("No forecasts are available");
    await loadDate(initial.date, true);
  } catch (error) {
    document.getElementById("product-message").textContent = "Forecast data could not be loaded. Please try again shortly.";
    hideLoading();
    console.error(error);
  }
}

init();
