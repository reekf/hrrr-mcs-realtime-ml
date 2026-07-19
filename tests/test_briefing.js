"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const briefing = require("../docs/briefing.js");

const grid = {
  lat: [40, 41],
  lon: [-93, -92],
};

const nearest = briefing.nearestGridPoint(grid, 40.01, -93.01);
assert.strictEqual(nearest.index, 0);
assert(briefing.nearestGridPoint(grid, 0, 0) === null);

const boundaryExpected = [
  [5, "Marginal"],
  [15, "Slight"],
  [40, "Moderate"],
  [70, "High"],
];
for (const [value, label] of boundaryExpected) {
  assert.strictEqual(briefing.riskCategory(value).label, label);
}

const data = {
  layers: {
    ml_r40: { values: [100] },
    ml_r60: { values: [110] },
    ml_r75: { values: [120] },
    ml_r100: { values: [130] },
    ml_r60v2: { values: [700] },
  },
};
const agreement = briefing.agreementSummary(data, 0);
assert.strictEqual(agreement.count, 4);
assert.strictEqual(agreement.qualitative, "High");
assert.strictEqual(agreement.exceedanceCounts[5], 4);
assert.strictEqual(agreement.exceedanceCounts[15], 0);
assert.strictEqual(briefing.agreementSummary({ layers: {} }, 0), null);

const decoded = briefing.decodePredictor(
  { values: [500], scale_min: 10, scale_max: 20 },
  0,
);
assert.strictEqual(decoded.value, 15);
assert.strictEqual(decoded.percentilePosition, 50);

const copy = briefing.copyBriefingText({
  latitude: 41.59,
  longitude: -93.62,
  validPeriod: "2026-07-17 12Z to 2026-07-18 12Z",
  ensembleMean: 24,
  agreement,
  wpcCategory: "Marginal",
  activeWarning: "No",
});
assert(copy.includes("XGBoosted Flash Flood Predictions"));
assert(copy.includes("not an official NWS forecast"));

const realtimeMap = JSON.parse(fs.readFileSync(
  path.join(__dirname, "../docs/archive/20260717/map.json"),
  "utf8",
));
const realNearest = briefing.nearestGridPoint(
  realtimeMap.grid,
  realtimeMap.grid.lat[0],
  realtimeMap.grid.lon[0],
);
assert.strictEqual(realNearest.index, 0);
assert(briefing.agreementSummary(realtimeMap, realNearest.index));
assert(Number.isFinite(briefing.probabilityPercent(realtimeMap.layers.pp, realNearest.index)));

console.log("Location Briefing unit tests passed.");
