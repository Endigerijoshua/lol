/* SIH26162 — Leaflet dashboard: FIRMS fire points × OSM industrial zones ×
   WRI power plants, with NASA GIBS satellite imagery as a toggleable base layer
   and a switch between individual detections and clustered thermal sites.

   Filter model: every fire belongs to exactly ONE category (persistent → flare →
   industrial → mining → forest → other_natural). The sidebar list and the map
   markers are both driven by that single assignment, so a filter tab shows the
   same fires in both. Map markers are grouped per category in their own
   L.layerGroup; switching tabs only shows/hides whole groups (no per-marker
   re-filtering, no data re-fetch). */
"use strict";

const FIRES_URL = "/api/flagged-fires";
const ZONES_URL = "/api/industrial-zones";
const SITES_URL = "/api/thermal-sites";
const POWER_PLANTS_URL = "/api/power-plants";
const FLARES_URL = "/api/flares";
const MINING_URL = "/api/mining-zones";

/* Ember→amber→pale severity ramp (matches the CSS custom properties). */
const CONFIDENCE_COLORS = {
  h: "#e5484d",
  n: "#f59e0b",
  l: "#fde68a",
};
const CONFIDENCE_LABELS = { h: "high", n: "nominal", l: "low" };

const CATEGORY_ORDER = [
  "persistent",
  "flare",
  "industrial",
  "mining",
  "forest",
  "other_natural",
];

const CATEGORY_TITLES = {
  persistent: "Persistent Thermal Sources",
  flare: "Gas-Flare Fires",
  industrial: "Industrial Fires & Power Plants",
  mining: "Mining & Quarry Fires",
  forest: "Forest & Vegetation Fires",
  other_natural: "Crop / Agricultural Burning & Natural Hotspots",
};

/* Category accent colors for the map markers (mirror of the CSS palette). */
const CATEGORY_COLORS = {
  persistent: "#b4532a",
  flare: "#94a3b8",
  industrial: "#475569",
  mining: "#64748b",
  forest: "#64748b",
  other_natural: "#94a3b8",
};

const SITE_COLOR = "#64748b";
const POWER_PLANT_COLOR = "#64748b";
const FLARE_COLOR = "#94a3b8";
const MINING_COLOR = "#475569";

// GIBS WAITS ~1 day to publish true-color; use yesterday so tiles always exist.
const GIBS_DATE = new Date(Date.now() - 24 * 3600 * 1000)
  .toISOString()
  .slice(0, 10);
const GIBS_BASE_URL =
  "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/{layer}/default/" +
  GIBS_DATE +
  "/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg";

const map = L.map("map").setView([22, 79], 5);

const osmTile = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "&copy; OpenStreetMap contributors",
  maxZoom: 18,
}).addTo(map);

const gibsMODIS = L.tileLayer(
  GIBS_BASE_URL.replace(
    "{layer}",
    "MODIS_Terra_CorrectedReflectance_TrueColor",
  ),
  { maxNativeZoom: 9, maxZoom: 18, attribution: "Imagery &copy; NASA GIBS" },
);
const gibsVIIRS = L.tileLayer(
  GIBS_BASE_URL.replace("{layer}", "VIIRS_SNPP_CorrectedReflectance_TrueColor"),
  { maxNativeZoom: 9, maxZoom: 18, attribution: "Imagery &copy; NASA GIBS" },
);

const zoneLayer = L.geoJSON(null, {
  style: {
    color: "#64748b",
    weight: 1,
    fillColor: "#64748b",
    fillOpacity: 0.15,
  },
  onEachFeature: (feature, layer) => {
    const props = feature.properties || {};
    const rows = [
      ["Name", props.name || "—"],
      ["Landuse", props.landuse || "—"],
      ["Industrial", props.industrial || "—"],
      ["OSM id", props.osm_id],
    ];
    layer.bindPopup(rows.map(([k, v]) => `<b>${k}:</b> ${v}`).join("<br>"));
  },
});

const powerPlantLayer = L.geoJSON(null, {
  pointToLayer: (feature, latlng) =>
    L.circleMarker(latlng, {
      radius: 4,
      color: "#334155",
      weight: 1,
      fillColor: POWER_PLANT_COLOR,
      fillOpacity: 0.8,
    }),
  onEachFeature: (feature, layer) => {
    const props = feature.properties || {};
    const rows = [
      ["Plant", props.name || "—"],
      ["Capacity", props.capacity_mw != null ? `${props.capacity_mw} MW` : "—"],
      ["Fuel", props.primary_fuel || "—"],
      ["Source", props.source || "—"],
    ];
    layer.bindPopup(rows.map(([k, v]) => `<b>${k}:</b> ${v}`).join("<br>"));
  },
});

const flareLayer = L.geoJSON(null, {
  pointToLayer: (feature, latlng) =>
    L.circleMarker(latlng, {
      radius: 3,
      color: "#334155",
      weight: 1,
      fillColor: FLARE_COLOR,
      fillOpacity: 0.7,
    }),
  onEachFeature: (feature, layer) => {
    const props = feature.properties || {};
    const rows = [
      ["Flare site", props.name || "—"],
      ["Country", props.country || "—"],
      ["Catalog year", props.year != null ? props.year : "—"],
      ["Source", props.source || "—"],
    ];
    layer.bindPopup(rows.map(([k, v]) => `<b>${k}:</b> ${v}`).join("<br>"));
  },
});

const miningLayer = L.geoJSON(null, {
  style: {
    color: MINING_COLOR,
    weight: 1,
    fillColor: MINING_COLOR,
    fillOpacity: 0.18,
    dashArray: "3 3",
  },
  onEachFeature: (featureMap, layer) => {
    const props = featureMap.properties || {};
    const rows = [
      ["Name", props.name || "—"],
      ["Type", props.landuse === "quarry" ? "quarry" : props.man_made === "mineshaft" ? "mine shaft" : "mining"],
      ["OSM id", props.osm_id],
    ];
    layer.bindPopup(rows.map(([k, v]) => `<b>${k}:</b> ${v}`).join("<br>"));
  },
});

L.control
  .layers(
    {
      OpenStreetMap: osmTile,
      "NASA GIBS true-color (MODIS)": gibsMODIS,
      "NASA GIBS true-color (VIIRS)": gibsVIIRS,
    },
    null,
    { collapsed: true, position: "topright" },
  )
  .addTo(map);

/* One Leaflet layer group per fire category. Toggling a tab = show/hide the
   whole group, never touching individual markers. */
const fireGroups = {};
for (const cat of CATEGORY_ORDER) {
  fireGroups[cat] = L.layerGroup();
  fireGroups[cat].addTo(map);
}

const fireMarkers = new Map();
const siteLayer = L.layerGroup();

const listEl = document.getElementById("list");
const summaryEl = document.getElementById("summary");
const detectionsBtn = document.getElementById("view-detections");
const sitesBtn = document.getElementById("view-sites");

const cache = { firesFC: null, zonesFC: null, sitesFC: null, powerFC: null, flaresFC: null, miningFC: null };
let currentView = "detections";
let selectedCategories = new Set();

/* Skeleton loaders shown while the initial / refresh fetch is in flight. */
function skeletonListHtml() {
  const patterns = [
    ["46%", "92%", "64%"],
    ["38%", "84%", "72%", "46%"],
    ["52%", "88%", "58%"],
    ["44%", "90%", "68%", "42%"],
    ["40%", "86%", "74%"],
  ];
  return patterns
    .map((widths) => {
      const [verdictW, ...rest] = widths;
      const lines = rest
        .map((w) => `<div class="skeleton s-line" style="width:${w}"></div>`)
        .join("");
      return (
        `<div class="entry entry-skeleton">` +
        `<div class="verdict">` +
        `<div class="skeleton s-badge"></div>` +
        `<div class="skeleton s-line s-verdict" style="width:${verdictW}"></div>` +
        `</div>${lines}</div>`
      );
    })
    .join("");
}

function showSkeletonLoaders(on) {
  document.querySelectorAll(".badge-count").forEach((el) => {
    el.classList.toggle("skeleton", on);
    if (on) el.textContent = "";
  });
  if (on) listEl.innerHTML = skeletonListHtml();
}

function clearSkeletonLoaders() {
  document.querySelectorAll(".badge-count").forEach((el) => {
    el.classList.remove("skeleton");
  });
}

function confidenceColor(confidence) {
  return CONFIDENCE_COLORS[confidence] || "#8892a6";
}

function confidenceLabel(confidence) {
  return CONFIDENCE_LABELS[confidence] || confidence || "unknown";
}

function formatDistance(meters) {
  if (meters == null || !isFinite(meters)) return "—";
  return meters >= 1000
    ? `${(meters / 1000).toFixed(1)} km`
    : `${Math.round(meters)} m`;
}

function formatTime(acqDate, acqTime) {
  const time = String(acqTime).padStart(4, "0");
  return `${acqDate} ${time.slice(0, 2)}:${time.slice(2)} UTC`;
}

/* Single category per fire; the sidebar and map both use this. */
function fireCategory(props) {
  if (props.persistent_thermal_source) return "persistent";
  if (props.gas_flare) return "flare";
  const t = props.fire_type_rule || "other_natural";
  if (t === "industrial" || props.near_industrial) return "industrial";
  if (t === "mining" || props.near_mining) return "mining";
  if (t === "forest") return "forest";
  return "other_natural";
}

function fireKvHtml(props) {
  const rows = [
    ["Confidence", confidenceLabel(props.confidence)],
    ["Brightness", props.bright_ti4 != null ? `${props.bright_ti4} K` : "—"],
    ["FRP", props.frp != null ? `${props.frp} MW` : "—"],
    ["Observed", formatTime(props.acq_date, props.acq_time)],
    ["Satellite", props.satellite || "—"],
    ["Day/night", props.daynight || "—"],
    ["Nearest industrial", formatDistance(props.distance_m)],
    ["Match source", props.industrial_match_source || "—"],
  ];
  if (props.near_power_plant) {
    rows.push(["Power plant", props.power_plant_name || "—"]);
    rows.push(["Plant distance", formatDistance(props.power_plant_distance_m)]);
  }
  if (props.fire_type_rule === "mining" || props.near_mining) {
    rows.push(["Mining zone", props.mining_site_name || "—"]);
    rows.push(["Mining distance", formatDistance(props.distance_to_mining)]);
  }
  if (props.gas_flare) {
    rows.push(["Gas flare (VNF)", props.flare_site_name || "yes"]);
    rows.push(["Flare distance", formatDistance(props.distance_to_flare)]);
  }
  if (props.fire_type_ml) rows.push(["ML type", props.fire_type_ml]);
  if (props.fire_type_ml_confidence != null) {
    rows.push(["ML confidence", Number(props.fire_type_ml_confidence).toFixed(2)]);
  }
  if (props.persistent_thermal_source) {
    rows.push(["Occurrences", `${props.occurrence_count} days / 14`]);
  }
  return rows.map(([k, v]) => `<div><b>${k}:</b> ${v}</div>`).join("");
}

function fireVerdictHtml(props) {
  const icon = props.summary_icon || "\u{1F525}";
  const headline = props.summary_headline || "Fire";
  const cat = fireCategory(props);
  const detail = props.summary_detail ? `<div class="detail">${props.summary_detail}</div>` : "";
  const persistent = props.summary_persistent
    ? `<div class="persist">${props.summary_persistent}</div>`
    : "";
  const reason = props.explanation
    ? `<div class="reason">${props.explanation}</div>`
    : "";
  return (
    `<div class="verdict"><span class="badge badge-${cat}">${icon}</span>${headline}</div>` +
    detail +
    persistent +
    reason +
    `<details><summary>Details</summary><div class="kv">${fireKvHtml(props)}</div></details>`
  );
}

function firePopupContent(prop) {
  return `<div class="fb">${fireVerdictHtml(prop)}</div>`;
}

function sitePopupContent(prop) {
  const rows = [
    ["Members", `${prop.member_count} recurring detection(s)`],
    ["Total occurrences", `${prop.total_occurrences} days / 14`],
    ["Nearest zone", prop.near_industrial_zone || "—"],
    ["Zone distance", formatDistance(prop.distance_m)],
  ];
  const body = rows.map(([k, v]) => `<div><b>${k}:</b> ${v}</div>`).join("");
  return `<p style="margin:0 0 4px;font-weight:600;color:${SITE_COLOR}">${prop.site_name}</p>${body}`;
}

function markerStyle(props) {
  const isPersistent = props.persistent_thermal_source;
  const cat = fireCategory(props);
  const color = isPersistent
    ? CATEGORY_COLORS.persistent
    : CATEGORY_COLORS[cat] || CATEGORY_COLORS.other_natural;
  return {
    radius: isPersistent ? 10 : (cat === "industrial" || cat === "flare" ? 7 : 5),
    color: color,
    weight: isPersistent ? 2 : 1.5,
    dashArray: isPersistent ? "4 4" : null,
    fillColor: confidenceColor(props.confidence),
    fillOpacity: 0.85,
  };
}

function renderFireMarkers(features) {
  for (const g of Object.values(fireGroups)) g.clearLayers();
  fireMarkers.clear();
  for (const feature of features) {
    const [lon, lat] = feature.geometry.coordinates;
    const props = feature.properties || {};
    const cat = fireCategory(props);
    const marker = L.circleMarker([lat, lon], markerStyle(props))
      .bindPopup(firePopupContent(props))
      .addTo(fireGroups[cat]);
    if (props.persistent_thermal_source) {
      const el = marker.getElement();
      if (el) el.classList.add("marker-persistent");
    }
    fireMarkers.set(feature.id, marker);
  }
  applyFireFilter();
}

function renderSiteMarkers(sitesFC) {
  siteLayer.clearLayers();
  for (const feature of sitesFC.features || []) {
    const [lon, lat] = feature.geometry.coordinates;
    L.circleMarker([lat, lon], {
      radius: 12,
      color: "#334155",
      weight: 2,
      fillColor: SITE_COLOR,
      fillOpacity: 0.8,
    })
      .bindPopup(sitePopupContent(feature.properties))
      .addTo(siteLayer);
  }
}

function renderZones(featureCollection) {
  zoneLayer.clearLayers();
  zoneLayer.addData(featureCollection);
}

function renderPowerPlants(featureCollection) {
  powerPlantLayer.clearLayers();
  powerPlantLayer.addData(featureCollection);
}

function renderFlareSites(featureCollection) {
  flareLayer.clearLayers();
  flareLayer.addData(featureCollection);
}

function renderMiningZones(featureCollection) {
  miningLayer.clearLayers();
  miningLayer.addData(featureCollection);
}

function sectionTitle(label, count) {
  const div = document.createElement("div");
  div.className = "section-title";
  div.style.textAlign = "left";
  div.innerHTML = `${label} <span class="n">(${count})</span>`;
  return div;
}

function highlightMarker(featureId, on) {
  const marker = fireMarkers.get(featureId);
  if (!marker) return;
  const el = marker.getElement && marker.getElement();
  if (el) el.classList.toggle("marker-hover", on);
}

function buildEntry(feature) {
  const props = feature.properties;
  const cat = fireCategory(props);
  const entry = document.createElement("div");
  entry.className = `entry entry-${cat}`;
  entry.insertAdjacentHTML("beforeend", fireVerdictHtml(props));

  entry.addEventListener("click", () => {
    const [lon, lat] = feature.geometry.coordinates;
    highlightMarker(feature.id, true);
    map.flyTo([lat, lon], 12);
    const marker = fireMarkers.get(feature.id);
    if (marker) marker.openPopup();
    setTimeout(() => highlightMarker(feature.id, false), 900);
  });
  entry.addEventListener("mouseenter", () => highlightMarker(feature.id, true));
  entry.addEventListener("mouseleave", () => highlightMarker(feature.id, false));
  return entry;
}

function buildSiteEntry(feature) {
  const props = feature.properties;
  const entry = document.createElement("div");
  entry.className = "entry";
  entry.style.borderLeftColor = SITE_COLOR;

  const head = document.createElement("div");
  head.className = "head";
  const title = document.createElement("span");
  title.textContent = props.site_name;
  const count = document.createElement("span");
  count.className = "dist";
  count.textContent = `${props.member_count} members`;
  head.append(title, count);
  entry.appendChild(head);

  const dl = document.createElement("dl");
  const rows = [
    ["Total occurrences", `${props.total_occurrences} days / 14`],
    ["Nearest zone", props.near_industrial_zone || "—"],
    ["Zone distance", formatDistance(props.distance_m)],
  ];
  for (const [k, v] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    dd.textContent = v;
    dl.append(dt, dd);
  }
  entry.appendChild(dl);

  entry.addEventListener("click", () => {
    const [lon, lat] = feature.geometry.coordinates;
    map.flyTo([lat, lon], 13);
  });
  return entry;
}

function animateCount(el, target) {
  if (!el) return;
  const from = parseInt(el.textContent, 10) || 0;
  if (from === target) {
    el.textContent = String(target);
    return;
  }
  if (el._raf) cancelAnimationFrame(el._raf);
  const t0 = performance.now();
  const dur = 420;
  const frame = (t) => {
    const p = Math.min(1, (t - t0) / dur);
    el.textContent = Math.round(from + (target - from) * (1 - Math.pow(1 - p, 3)));
    if (p < 1) el._raf = requestAnimationFrame(frame);
  };
  el._raf = requestAnimationFrame(frame);
}

function fadeInList() {
  listEl.classList.remove("list-fade");
  void listEl.offsetWidth;
  listEl.classList.add("list-fade");
}

function triggerMapSweep() {
  const mapEl = document.getElementById("map");
  mapEl.classList.remove("map-sweep");
  void mapEl.offsetWidth;
  mapEl.classList.add("map-sweep");
}

/* Show/hide whole Leaflet layer groups per the active filter checkboxes. */
function applyFireFilter() {
  if (currentView !== "detections") return;
  for (const [cat, group] of Object.entries(fireGroups)) {
    if (selectedCategories.size === 0) {
      hideLayer(group);
    } else if (selectedCategories.has(cat)) {
      showLayer(group);
    } else {
      hideLayer(group);
    }
  }

  // Reference layers (only shown when their corresponding category is selected)
  if (selectedCategories.has("industrial")) {
    showLayer(zoneLayer);
    showLayer(powerPlantLayer);
  } else {
    hideLayer(zoneLayer);
    hideLayer(powerPlantLayer);
  }

  if (selectedCategories.has("flare")) {
    showLayer(flareLayer);
  } else {
    hideLayer(flareLayer);
  }

  if (selectedCategories.has("mining")) {
    showLayer(miningLayer);
  } else {
    hideLayer(miningLayer);
  }
}

function renderDetectionsSidebar(features) {
  const buckets = {};
  for (const cat of CATEGORY_ORDER) buckets[cat] = [];

  for (const f of features) {
    const cat = fireCategory(f.properties || {});
    buckets[cat].push(f);
  }

  const sorters = {
    persistent: (a, b) =>
      (b.properties.occurrence_count || 0) - (a.properties.occurrence_count || 0) ||
      (a.properties.distance_m || 99999) - (b.properties.distance_m || 99999),
    flare: (a, b) => (a.properties.distance_to_flare || 99999) - (b.properties.distance_to_flare || 99999),
    industrial: (a, b) => (a.properties.distance_m || 99999) - (b.properties.distance_m || 99999),
    mining: (a, b) => (a.properties.distance_to_mining || 99999) - (b.properties.distance_to_mining || 99999),
    forest: (a, b) => (a.properties.vegetation_distance_m || 99999) - (b.properties.vegetation_distance_m || 99999),
    other_natural: (a, b) => (b.properties.frp || 0) - (a.properties.frp || 0),
  };
  for (const cat of CATEGORY_ORDER) buckets[cat].sort(sorters[cat]);

  const countElId = (cat) =>
    ({ industrial: "ind", other_natural: "nat", persistent: "persist" })[cat] || cat;

  // Real data is here — replace the skeleton badges with live count-ups.
  clearSkeletonLoaders();

  // Animated stat badge counts
  animateCount(document.getElementById("count-all"), features.length);
  for (const cat of CATEGORY_ORDER) {
    animateCount(document.getElementById(`count-${countElId(cat)}`), buckets[cat].length);
  }

  listEl.innerHTML = "";

  const renderSection = (cat) => {
    const items = buckets[cat];
    if (items.length === 0) return;
    listEl.appendChild(sectionTitle(CATEGORY_TITLES[cat], items.length));
    for (const f of items) listEl.appendChild(buildEntry(f));
  };

  if (selectedCategories.size === 0) {
    // Show nothing when no categories selected
  } else {
    for (const cat of CATEGORY_ORDER) {
      if (selectedCategories.has(cat)) {
        renderSection(cat);
      }
    }
  }

  if (listEl.children.length === 0) {
    const el = document.createElement("div");
    el.className = "empty";
    el.textContent = selectedCategories.size === 0
      ? "Select a category to view detections."
      : "No detections matching this filter.";
    listEl.appendChild(el);
  }

  fadeInList();

  return {
    total: features.length,
    persistent: buckets.persistent.length,
    flare: buckets.flare.length,
    industrial: buckets.industrial.length,
    mining: buckets.mining.length,
    forest: buckets.forest.length,
    natural: buckets.other_natural.length,
  };
}

function setSummaryCounts(counts) {
  summaryEl.innerHTML =
    `<span class="big" id="s-total">0</span> live hotspots &middot; ` +
    `<span id="s-ind">0</span> \u{1F3ED} ind &middot; ` +
    `<span id="s-mining">0</span> \u{26CF}\uFE0F mining &middot; ` +
    `<span id="s-forest">0</span> \u{1F332} forest &middot; ` +
    `<span id="s-nat">0</span> \u{1F33E} agri &middot; ` +
    `<span id="s-persist">0</span> \u{1F534} persist`;
  animateCount(summaryEl.querySelector("#s-total"), counts.total);
  animateCount(summaryEl.querySelector("#s-ind"), counts.industrial);
  animateCount(summaryEl.querySelector("#s-mining"), counts.mining);
  animateCount(summaryEl.querySelector("#s-forest"), counts.forest);
  animateCount(summaryEl.querySelector("#s-nat"), counts.natural);
  animateCount(summaryEl.querySelector("#s-persist"), counts.persistent);
}

function renderSitesSidebar(sitesFC) {
  const sites = sitesFC.features || [];
  const meta = sitesFC.meta || {};
  listEl.innerHTML = "";
  if (sites.length === 0) {
    const el = document.createElement("div");
    el.className = "empty";
    el.textContent = "No thermal sites — persistent sources are isolated, or none yet.";
    listEl.appendChild(el);
    return;
  }
  listEl.appendChild(sectionTitle(`Thermal sites (${sites.length})`));
  for (const feature of sites) listEl.appendChild(buildSiteEntry(feature));
  fadeInList();
  return meta;
}

function showError(message) {
  listEl.innerHTML = "";
  clearSkeletonLoaders();
  const el = document.createElement("div");
  el.className = "empty";
  el.textContent = message;
  listEl.appendChild(el);
}

function setViewButtons(view) {
  detectionsBtn.classList.toggle("active", view === "detections");
  sitesBtn.classList.toggle("active", view === "sites");
}

function showLayer(layer) {
  if (!map.hasLayer(layer)) layer.addTo(map);
}

function hideLayer(layer) {
  if (map.hasLayer(layer)) map.removeLayer(layer);
}

async function loadDetections(force = false) {
  currentView = "detections";
  setViewButtons("detections");
  hideLayer(siteLayer);
  summaryEl.textContent = "Fetching live data…";
  const willFetch = force || !cache.firesFC;
  if (willFetch) showSkeletonLoaders(true);
  try {
    if (willFetch) {
      // Fires are the primary data — render them as soon as they arrive so a
      // slow or failing reference layer can never blank the whole dashboard.
      cache.firesFC = await fetchJson(FIRES_URL);
    }
    const features = cache.firesFC.features || [];
    renderFireMarkers(features);
    const counts = renderDetectionsSidebar(features);
    setSummaryCounts(counts);
    const ts = document.getElementById("live-ts");
    if (ts) ts.textContent = `FIRMS feed · ${new Date().toUTCString().slice(17, 25)} UTC`;

    if (willFetch) {
      // Reference layers stream in independently; a rejection is logged and
      // skipped rather than failing the whole load (fires stay on the map).
      const refResults = await Promise.allSettled([
        fetchJson(ZONES_URL),
        fetchJson(POWER_PLANTS_URL),
        fetchJson(FLARES_URL),
        fetchJson(MINING_URL),
      ]);
      const [zones, power, flares, mining] = refResults;
      if (zones.status === "fulfilled") {
        cache.zonesFC = zones.value;
        renderZones(cache.zonesFC);
      } else {
        console.warn("industrial zones unavailable:", zones.reason);
      }
      if (power.status === "fulfilled") {
        cache.powerFC = power.value;
        renderPowerPlants(cache.powerFC);
      } else {
        console.warn("power plants unavailable:", power.reason);
      }
      if (flares.status === "fulfilled") {
        cache.flaresFC = flares.value;
        renderFlareSites(cache.flaresFC);
      } else {
        console.warn("flare sites unavailable:", flares.reason);
      }
      if (mining.status === "fulfilled") {
        cache.miningFC = mining.value;
        renderMiningZones(cache.miningFC);
      } else {
        console.warn("mining zones unavailable:", mining.reason);
      }
    } else {
      // Cached path: no network; just re-render whatever reference layers we
      // already hold (they may have failed to load earlier).
      if (cache.zonesFC) renderZones(cache.zonesFC);
      if (cache.powerFC) renderPowerPlants(cache.powerFC);
      if (cache.flaresFC) renderFlareSites(cache.flaresFC);
      if (cache.miningFC) renderMiningZones(cache.miningFC);
    }
    applyFireFilter();
  } catch (err) {
    // Only a fires-endpoint failure lands here (reference layers degrade
    // gracefully above), so this message always means the core view really
    // could not load.
    console.error(err);
    showError(`Failed to load detections: ${err.message}`);
  }
}

async function loadSites(force = false) {
  currentView = "sites";
  setViewButtons("sites");
  for (const g of Object.values(fireGroups)) hideLayer(g);
  hideLayer(zoneLayer);
  hideLayer(powerPlantLayer);
  hideLayer(flareLayer);
  hideLayer(miningLayer);
  summaryEl.textContent = "Clustering persistent sources…";
  try {
    if (force || !cache.sitesFC) {
      cache.sitesFC = await fetchJson(SITES_URL);
      if (cache.zonesFC) renderZones(cache.zonesFC);
    }
    renderSiteMarkers(cache.sitesFC);
    showLayer(siteLayer);
    const meta = renderSitesSidebar(cache.sitesFC);
    const sites = cache.sitesFC.features || [];
    summaryEl.textContent =
      `${sites.length} thermal site(s) from ${(meta && meta.persistent_count) || 0} ` +
      `persistent recurrences (${(meta && meta.unclustered) || 0} isolated)`;
  } catch (err) {
    console.error(err);
    showError(`Failed to load thermal sites: ${err.message}`);
  }
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    let detail = "";
    try {
      const data = await response.json();
      if (data && data.detail) {
        detail = `: ${data.detail}`;
      }
    } catch {
      // Not JSON or empty body
    }
    throw new Error(`${url} -> HTTP ${response.status}${detail}`);
  }
  return response.json();
}

// Category filter checkboxes: filter the sidebar AND show/hide the map layer groups.

// Update "All" checkbox state based on selection
function updateFilterState() {
  const allCheckbox = document.getElementById("filter-all");

  if (selectedCategories.size === CATEGORY_ORDER.length) {
    if (allCheckbox) {
      allCheckbox.checked = true;
      allCheckbox.indeterminate = false;
    }
  } else if (selectedCategories.size === 0) {
    if (allCheckbox) {
      allCheckbox.checked = false;
      allCheckbox.indeterminate = false;
    }
  } else {
    if (allCheckbox) {
      allCheckbox.checked = false;
      allCheckbox.indeterminate = true;
    }
  }
}

// Initialize filter state on page load
updateFilterState();

// Filter collapse toggle
const filterHeader = document.getElementById("filter-header");
const filterGroup = document.getElementById("filter-group");
if (filterHeader && filterGroup) {
  filterHeader.addEventListener("click", () => {
    const isExpanded = filterHeader.getAttribute("aria-expanded") === "true";
    if (isExpanded) {
      filterHeader.setAttribute("aria-expanded", "false");
      filterGroup.classList.add("collapsed");
    } else {
      filterHeader.setAttribute("aria-expanded", "true");
      filterGroup.classList.remove("collapsed");
    }
  });
}

document.querySelectorAll("#category-filters input[type='checkbox']").forEach((checkbox) => {
  checkbox.addEventListener("change", () => {
    const category = checkbox.dataset.category;
    if (category === "all") {
      if (checkbox.checked) {
        selectedCategories = new Set(CATEGORY_ORDER);
        document.querySelectorAll("#category-filters input[type='checkbox']:not([data-category='all'])").forEach((cb) => {
          cb.checked = true;
        });
      } else {
        selectedCategories.clear();
        document.querySelectorAll("#category-filters input[type='checkbox']:not([data-category='all'])").forEach((cb) => {
          cb.checked = false;
        });
      }
    } else {
      if (checkbox.checked) {
        selectedCategories.add(category);
      } else {
        selectedCategories.delete(category);
      }
    }
    updateFilterState();
    if (currentView === "detections") {
      triggerMapSweep();
      applyFireFilter();
      if (cache.firesFC && cache.firesFC.features) {
        renderDetectionsSidebar(cache.firesFC.features);
      }
    }
  });
});

detectionsBtn.addEventListener("click", () => loadDetections());
sitesBtn.addEventListener("click", () => loadSites());
document.getElementById("reload").addEventListener("click", () => {
  if (currentView === "sites") {
    cache.sitesFC = null;
    loadSites(true);
  } else {
    cache.firesFC = null;
    loadDetections(true);
  }
});

loadDetections();