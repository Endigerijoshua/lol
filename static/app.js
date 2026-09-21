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
  "unregistered",
  "persistent",
  "flare",
  "industrial",
  "mining",
  "forest",
  "other_natural",
];

const CATEGORY_TITLES = {
  unregistered: "Unregistered Persistent Sources",
  persistent: "Persistent Thermal Sources",
  flare: "Gas-Flare Fires",
  industrial: "Industrial Fires & Power Plants",
  mining: "Mining & Quarry Fires",
  forest: "Forest & Vegetation Fires",
  other_natural: "Crop / Agricultural Burning & Natural Hotspots",
};

/* Category accent colors for the map markers (mirror of the CSS palette). */
const CATEGORY_COLORS = {
  unregistered: "#c14953",
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

/* Directional Risk Indicator (heuristic, opt-in overlay). */
const RISK_TIER_COLORS = { high: "#c14953", medium: "#d48b3a", low: "#8b96a3" };
let riskEnabled = false;
const riskLayers = new Map(); // feature.id -> [coneLayer, arrowMarker]

// GIBS publishes with ~1 d of lag and a given date can have a missing granule
// over India (observed 404s from the WMTS for some dates). Instead of pinning
// "yesterday", probe the last 7 days and pick the newest date whose India-region
// tiles at a representative zoom (z5) actually return 200; null if none do.
function probeTile(url) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(true);
    img.onerror = () => resolve(false);
    img.src = url;
  });
}

function pickGIBSDate() {
  const probes = async (date) => {
    const layers = [
      "MODIS_Terra_CorrectedReflectance_TrueColor",
      "VIIRS_SNPP_CorrectedReflectance_TrueColor",
    ];
    const results = [];
    for (const layer of layers) {
      for (const tile of ["5/13/23", "5/12/22"]) {
        results.push(
          probeTile(
            "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/" +
              layer +
              "/default/" +
              date +
              "/GoogleMapsCompatible_Level9/" +
              tile +
              ".jpg",
          ),
        );
      }
    }
    return (await Promise.all(results)).every(Boolean);
  };
  return (async function loop() {
    for (let i = 1; i <= 7; i++) {
      const date = new Date(Date.now() - i * 24 * 3600 * 1000)
        .toISOString()
        .slice(0, 10);
      if (await probes(date)) return date;
    }
    return null;
  })();
}

const GIBS_TILE_URL = (layer, date) =>
  "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/" +
  layer +
  "/default/" +
  date +
  "/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg";

const map = L.map("map").setView([22, 79], 5);

const osmTile = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "&copy; OpenStreetMap contributors",
  maxZoom: 18,
}).addTo(map);

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

// Default base layer is OpenStreetMap (added above). GIBS satellite layers are
// strictly opt-in: they are only listed here in the toggle (never auto-selected).
// Because GIBS lags ~1 d and a date can lack an India granule (404 tiles → black
// gaps), pick the newest date with verified India coverage, then register both
// GIBS layers against it. If none is found the toggle is simply omitted.
const layerControl = L.control
  .layers(
    {
      OpenStreetMap: osmTile,
    },
    null,
    { collapsed: true, position: "topright" },
  )
  .addTo(map);

(async () => {
  const date = await pickGIBSDate();
if (!date) {
     return null;
   }
  const attribution = "Imagery &copy; NASA GIBS";
  layerControl.addBaseLayer(
    L.tileLayer(
      GIBS_TILE_URL("MODIS_Terra_CorrectedReflectance_TrueColor", date),
      { maxNativeZoom: 9, maxZoom: 18, attribution },
    ),
    "NASA GIBS true-color (MODIS)",
  );
  layerControl.addBaseLayer(
    L.tileLayer(
      GIBS_TILE_URL("VIIRS_SNPP_CorrectedReflectance_TrueColor", date),
      { maxNativeZoom: 9, maxZoom: 18, attribution },
    ),
    "NASA GIBS true-color (VIIRS)",
  );
})();

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

function formatFrp(frp) {
  if (frp == null || !isFinite(frp)) return "—";
  return `${Number(frp).toFixed(1)} MW`;
}

/* ---- Directional Risk Indicator helpers ---- */

function cardinal(deg) {
  if (deg == null || !isFinite(deg)) return "—";
  const dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
  return dirs[Math.round((((Number(deg) % 360) + 360) % 360) / 45) % 8];
}

function fmtDeg(deg) {
  if (deg == null || !isFinite(deg)) return "—";
  return `${Math.round(Number(deg))}° ${cardinal(deg)}`;
}

function fmtKmh(kmh) {
  if (kmh == null || !isFinite(kmh)) return "—";
  return `${Math.round(Number(kmh))} km/h`;
}

function riskTierColor(tier) {
  return RISK_TIER_COLORS[tier] || RISK_TIER_COLORS.low;
}

/* Rotated wind arrow (points in the estimated spread direction) + speed. */
function windArrowSvg(spreadDeg, speedKmh) {
  const num =
    speedKmh == null || !isFinite(speedKmh)
      ? ""
      : `<div class="wind-num">${Math.round(Number(speedKmh))}</div>`;
  return (
    `<div class="wind-arrow" title="Directional Risk Indicator \u00b7 wind ` +
    `${fmtKmh(speedKmh)} (arrow = estimated spread direction, heuristic)">` +
    `<svg width="26" height="26" viewBox="0 0 26 26" style="display:block;` +
    `transform:rotate(${spreadDeg}deg);transform-origin:50% 50%">` +
    `<path d="M13 1 L21 15 L15.5 15 L15.5 25 L10.5 25 L10.5 15 L5 15 Z" fill="#334155"/>` +
    `</svg>${num}</div>`
  );
}

/* Human-friendly risk note shown in every card + popup. */
function riskBlockHtml(props) {
  const r = props.directional_risk;
  if (!r) return "";
  const w = r.weather || {};
  const tip =
    "Estimated spread direction (heuristic) \u2014 based on current wind, " +
    "nearby vegetation density and temperature/humidity dryness. This is " +
    "NOT a validated fire-behavior model.";
  return (
    `<div class="risknote" title="${tip}">` +
    `<span class="risk-title">\u{1F9ED} Directional Risk Indicator</span>` +
    ` \u00b7 spread toward ${fmtDeg(r.spread_direction_deg)} (heuristic)<br>` +
    `<span class="risk-meta">risk ${Number(r.risk_score).toFixed(2)} ` +
    `(${r.risk_tier || "low"}) \u00b7 wind ${fmtKmh(w.wind_speed_kmh)} from ` +
    `${fmtDeg(w.wind_direction_deg)}</span>` +
    `</div>`
  );
}

/* Draw the cone + wind arrow for one fire into its category group. */
function renderRiskForFire(feature, group) {
  const props = feature.properties || {};
  const risk = props.directional_risk;
  if (!risk) return;
  const coords = feature.geometry && feature.geometry.coordinates;
  if (!coords || !risk.geometry) return;
  const [lon, lat] = coords;
  const w = risk.weather || {};

  const cone = L.geoJSON(null, {
    className: "risk-cones",
    style: {
      color: riskTierColor(risk.risk_tier),
      weight: 1,
      fillColor: riskTierColor(risk.risk_tier),
      fillOpacity: 0.2,
      dashArray: "4 4",
    },
    interactive: false,
  });
  cone.addData({
    type: "FeatureCollection",
    features: [{ type: "Feature", geometry: risk.geometry, properties: {} }],
  });
  cone.addTo(group);

  const arrow = L.marker([lat, lon], {
    icon: L.divIcon({
      className: "wind-arrow-icon",
      html: windArrowSvg(risk.spread_direction_deg || 0, w.wind_speed_kmh),
      iconSize: [30, 34],
      iconAnchor: [15, 15],
    }),
    keyboard: false,
    zIndexOffset: 900,
  }).bindPopup(firePopupContent(props));
  arrow.addTo(group);

  riskLayers.set(feature.id, [cone, arrow]);
}

function resetRiskLayers() {
  for (const layers of riskLayers.values()) {
    for (const layer of layers) layer.remove();
  }
  riskLayers.clear();
}

/* Toggle the risk overlay without touching markers/sidebar. */
function syncRiskLayers() {
  if (!cache.firesFC) return;
  resetRiskLayers();
  if (!riskEnabled) return;
  for (const feature of cache.firesFC.features || []) {
    const cat = fireCategory(feature.properties || {});
    renderRiskForFire(feature, fireGroups[cat]);
  }
}

/* FIRMS `acq_time` is a UTC HHMM integer (2400 = midnight, next day). */
function acqTimestamp(acqDate, acqTime) {
  if (!acqDate) return null;
  const raw = String(acqTime == null ? "0" : acqTime);
  const t = raw.padStart(4, "0");
  let h = parseInt(t.slice(0, 2), 10) || 0;
  const m = parseInt(t.slice(2, 4), 10) || 0;
  const ts = new Date(acqDate + "T00:00:00Z");
  if (h === 24) {
    h = 0;
    ts.setUTCDate(ts.getUTCDate() + 1);
  }
  ts.setUTCHours(h, m, 0, 0);
  return ts.getTime();
}

const FRESH_MS = 30 * 60 * 1000; // under 30 min = "fresh" accent

function timeSinceText(ms) {
  if (ms == null) return "—";
  const sec = Math.max(0, Math.floor((Date.now() - ms) / 1000));
  if (sec < 60) return "just now";
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min} min ago`;
  const hr = Math.floor(min / 60);
  if (hr < 48) return `${hr} hour${hr === 1 ? "" : "s"} ago`;
  const d = Math.floor(hr / 24);
  return `${d} day${d === 1 ? "" : "s"} ago`;
}

/* Re-compute every relative age on screen (sidebar + open popup) live. */
function refreshAges() {
  document.querySelectorAll(".age").forEach((el) => {
    const ts = acqTimestamp(el.dataset.acqDate, el.dataset.acqTime);
    if (ts == null) {
      el.textContent = "—";
      el.classList.remove("fresh");
      return;
    }
    el.textContent = timeSinceText(ts);
    el.classList.toggle("fresh", Date.now() - ts < FRESH_MS);
  });
}

/* Single category per fire; the sidebar and map both use this. */
function fireCategory(props) {
  if (props.unregistered_persistent) return "unregistered";
  if (props.persistent_thermal_source) return "persistent";
  if (props.gas_flare) return "flare";
  const t = props.fire_type_rule || "other_natural";
  if (t === "industrial" || props.near_industrial) return "industrial";
  if (t === "mining" || props.near_mining) return "mining";
  // Backend invariant: near_vegetation ⇒ forest (unless industrial/mining won).
  if (t === "forest" || props.near_vegetation) return "forest";
  return "other_natural";
}

function fireKvHtml(props) {
  const rows = [
    ["Confidence", confidenceLabel(props.confidence)],
    ["Brightness", props.bright_ti4 != null ? `${props.bright_ti4} K` : "—"],
    ["FRP", formatFrp(props.frp)],
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
  if (props.directional_risk) {
    const r = props.directional_risk;
    const w = r.weather || {};
    rows.push([
      "Directional Risk Indicator",
      "Estimated spread direction (heuristic) — wind/vegetation/dryness blend, not a validated fire-behavior model.",
    ]);
    rows.push([
      "Risk (heuristic)",
      `${Number(r.risk_score).toFixed(2)} ${r.risk_tier || "low"}`,
    ]);
    rows.push(["Spread toward", fmtDeg(r.spread_direction_deg)]);
    rows.push(["Cone length", formatDistance(r.cone_length_m)]);
    rows.push([
      "Cone half-angle",
      r.cone_half_angle_deg != null ? `${r.cone_half_angle_deg}°` : "—",
    ]);
    rows.push([
      "Wind",
      `${fmtKmh(w.wind_speed_kmh)} from ${fmtDeg(w.wind_direction_deg)}`,
    ]);
    rows.push([
      "Temperature",
      w.temperature_c != null ? `${w.temperature_c}°C` : "—",
    ]);
    rows.push(["Humidity", w.humidity_pct != null ? `${w.humidity_pct}%` : "—"]);
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
  const unreg = props.summary_unregistered
    ? `<div class="unreg">\u{1F6A8} ${props.summary_unregistered}</div>`
    : "";
  const reason = props.explanation
    ? `<div class="reason">${props.explanation}</div>`
    : "";
  const frpEl =
    props.frp != null && isFinite(props.frp)
      ? `<span class="frp">FRP <b>${formatFrp(props.frp)}</b></span>`
      : "";
  const ageTs = acqTimestamp(props.acq_date, props.acq_time);
  const ageEl =
    ageTs == null
      ? ""
      : `<span class="age${Date.now() - ageTs < FRESH_MS ? " fresh" : ""}" ` +
        `data-acq-date="${props.acq_date}" data-acq-time="${props.acq_time}">` +
        `${timeSinceText(ageTs)}</span>`;
  const meta = frpEl || ageEl ? `<div class="meta">${frpEl}${ageEl}</div>` : "";
  const risk = riskBlockHtml(props);
  return (
    `<div class="verdict"><span class="badge badge-${cat}">${icon}</span>${headline}</div>` +
    detail +
    meta +
    risk +
    persistent +
    unreg +
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
  const isUnregistered = props.unregistered_persistent;
  const isPersistent = props.persistent_thermal_source;
  const cat = fireCategory(props);
  const color = isUnregistered
    ? CATEGORY_COLORS.unregistered
    : isPersistent
      ? CATEGORY_COLORS.persistent
      : CATEGORY_COLORS[cat] || CATEGORY_COLORS.other_natural;
  const big = isUnregistered || isPersistent;
  return {
    radius: big ? 10 : (cat === "industrial" || cat === "flare" ? 7 : 5),
    color: color,
    weight: isPersistent ? 2 : 1.5,
    dashArray: big ? "4 4" : null,
    fillColor: confidenceColor(props.confidence),
    fillOpacity: 0.85,
  };
}

function renderFireMarkers(features) {
  for (const g of Object.values(fireGroups)) g.clearLayers();
  fireMarkers.clear();
  resetRiskLayers();
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
  if (riskEnabled) {
    for (const feature of features) {
      const cat = fireCategory(feature.properties || {});
      renderRiskForFire(feature, fireGroups[cat]);
    }
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

/* Show/hide sidebar category sections per the active filter checkboxes.
   Sections are built once and only re-classed, so a checkbox toggle touches
   zero DOM nodes — no rebuild, no count-up replay, no list fade. */
function applySidebarFilter() {
  if (currentView !== "detections") return;
  let visible = 0;
  for (const section of listEl.querySelectorAll(".cat-section")) {
    const show =
      selectedCategories.size > 0 && selectedCategories.has(section.dataset.cat);
    section.classList.toggle("hidden", !show);
    if (show) visible += 1;
  }
  let emptyEl = document.getElementById("list-empty");
  if (visible === 0) {
    if (!emptyEl) {
      emptyEl = document.createElement("div");
      emptyEl.id = "list-empty";
      emptyEl.className = "empty";
      listEl.appendChild(emptyEl);
    }
    emptyEl.textContent =
      selectedCategories.size === 0
        ? "Select a category to view detections."
        : "No detections matching this filter.";
  } else if (emptyEl) {
    emptyEl.remove();
  }
}

function renderDetectionsSidebar(features) {
  const buckets = {};
  for (const cat of CATEGORY_ORDER) buckets[cat] = [];

  for (const f of features) {
    const cat = fireCategory(f.properties || {});
    buckets[cat].push(f);
  }

  const priority = {
    persistent: (a, b) =>
      (b.properties.occurrence_count || 0) - (a.properties.occurrence_count || 0) ||
      (a.properties.distance_m || 99999) - (b.properties.distance_m || 99999),
    unregistered: (a, b) =>
      (b.properties.occurrence_count || 0) - (a.properties.occurrence_count || 0) ||
      (b.properties.frp || 0) - (a.properties.frp || 0),
    flare: (a, b) => (a.properties.distance_to_flare || 99999) - (b.properties.distance_to_flare || 99999),
    industrial: (a, b) => (a.properties.distance_m || 99999) - (b.properties.distance_m || 99999),
    mining: (a, b) => (a.properties.distance_to_mining || 99999) - (b.properties.distance_to_mining || 99999),
    forest: (a, b) => (a.properties.vegetation_distance_m || 99999) - (b.properties.vegetation_distance_m || 99999),
    other_natural: (a, b) => (b.properties.frp || 0) - (a.properties.frp || 0),
  };

  function bucketSorter(cat) {
    return priority[cat] || priority.other_natural;
  }
  for (const cat of CATEGORY_ORDER) buckets[cat].sort(bucketSorter(cat));

  const countElId = (cat) =>
    ({ industrial: "ind", other_natural: "nat", persistent: "persist", unregistered: "unreg" })[cat] || cat;

  // Real data is here — replace the skeleton badges with live count-ups.
  clearSkeletonLoaders();

  // Animated stat badge counts
  animateCount(document.getElementById("count-all"), features.length);
  for (const cat of CATEGORY_ORDER) {
    animateCount(document.getElementById(`count-${countElId(cat)}`), buckets[cat].length);
  }

  listEl.innerHTML = "";

  /* Build every category section once. Filter toggles only flip a `hidden`
     class on these sections (applySidebarFilter) — they never re-create the
     DOM, re-run the badge count-ups or re-play the list fade. */
  for (const cat of CATEGORY_ORDER) {
    const items = buckets[cat];
    if (items.length === 0) continue;
    const section = document.createElement("div");
    section.className = "cat-section";
    section.dataset.cat = cat;
    section.appendChild(sectionTitle(CATEGORY_TITLES[cat], items.length));
    for (const f of items) section.appendChild(buildEntry(f));
    listEl.appendChild(section);
  }
  applySidebarFilter();

  fadeInList();

  return {
    total: features.length,
    unregistered: buckets.unregistered.length,
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
    `<span id="s-unreg">0</span> \u{1F6A8} unreg &middot; ` +
    `<span id="s-ind">0</span> \u{1F3ED} ind &middot; ` +
    `<span id="s-mining">0</span> \u{26CF}\uFE0F mining &middot; ` +
    `<span id="s-forest">0</span> \u{1F332} forest &middot; ` +
    `<span id="s-nat">0</span> \u{1F33E} agri &middot; ` +
    `<span id="s-persist">0</span> \u{1F534} persist`;
  animateCount(summaryEl.querySelector("#s-total"), counts.total);
  animateCount(summaryEl.querySelector("#s-unreg"), counts.unregistered || 0);
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

/* Render markers + sidebar + counts together so filters stay in sync. */
function renderCurrentDetections() {
  if (!cache.firesFC) return;
  const features = cache.firesFC.features || [];
  renderFireMarkers(features);
  renderZones(cache.zonesFC);
  renderPowerPlants(cache.powerFC);
  renderFlareSites(cache.flaresFC);
  renderMiningZones(cache.miningFC);
  applyFireFilter();
  // Sweep animation only on a fresh data load, never on a filter toggle.
  triggerMapSweep();
  const counts = renderDetectionsSidebar(features);
  setSummaryCounts(counts);
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
      [cache.firesFC, cache.zonesFC, cache.powerFC, cache.flaresFC, cache.miningFC] =
        await Promise.all([
          fetchJson(FIRES_URL),
          fetchJson(ZONES_URL),
          fetchJson(POWER_PLANTS_URL),
          fetchJson(FLARES_URL),
          fetchJson(MINING_URL),
        ]);
    }
    renderCurrentDetections();
    const ts = document.getElementById("live-ts");
    if (ts) ts.textContent = `FIRMS feed · ${new Date().toUTCString().slice(17, 25)} UTC`;
  } catch (err) {
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
      applyFireFilter();
      applySidebarFilter();
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

/* Directional Risk Indicator overlay: opt-in, never on by default. */
const riskToggle = document.getElementById("risk-toggle");
if (riskToggle) {
  riskToggle.addEventListener("change", (e) => {
    riskEnabled = e.target.checked;
    if (currentView === "detections") syncRiskLayers();
  });
}

loadDetections();

/* Live update of the relative "time since detection" every 45 s, no reload. */
setInterval(refreshAges, 45 * 1000);