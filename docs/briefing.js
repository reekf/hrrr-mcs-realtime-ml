(function briefingModule(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.XGBFFPBriefing = api;
}(typeof globalThis !== "undefined" ? globalThis : this, function buildBriefingApi() {
  "use strict";

  const STANDARD_ML_PRODUCTS = ["ml_r40", "ml_r60", "ml_r75", "ml_r100"];
  const THRESHOLDS_PERCENT = [5, 15, 40, 70];

  function haversineKm(lat1, lon1, lat2, lon2) {
    const radians = Math.PI / 180;
    const deltaLat = (lat2 - lat1) * radians;
    const deltaLon = (lon2 - lon1) * radians;
    const a = Math.sin(deltaLat / 2) ** 2
      + Math.cos(lat1 * radians) * Math.cos(lat2 * radians) * Math.sin(deltaLon / 2) ** 2;
    return 6371.0088 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  }

  function nearestGridPoint(grid, latitude, longitude, maxDistanceKm = 100) {
    const latitudes = grid?.lat;
    const longitudes = grid?.lon;
    if (!Array.isArray(latitudes) || !Array.isArray(longitudes) || latitudes.length !== longitudes.length) {
      return null;
    }
    let nearestIndex = -1;
    let nearestDistance = Infinity;
    for (let index = 0; index < latitudes.length; index += 1) {
      const distance = haversineKm(latitude, longitude, Number(latitudes[index]), Number(longitudes[index]));
      if (distance < nearestDistance) {
        nearestIndex = index;
        nearestDistance = distance;
      }
    }
    if (nearestIndex < 0 || nearestDistance > maxDistanceKm) return null;
    return {
      index: nearestIndex,
      distanceKm: nearestDistance,
      latitude: Number(latitudes[nearestIndex]),
      longitude: Number(longitudes[nearestIndex]),
    };
  }

  function probabilityPercent(layer, index) {
    const value = layer?.values?.[index];
    return Number.isFinite(Number(value)) ? Number(value) / 10 : null;
  }

  function riskCategory(probability) {
    if (!Number.isFinite(probability)) return { label: "Not available", rank: -1, range: "" };
    if (probability >= 70) return { label: "High", rank: 4, range: "≥70%" };
    if (probability >= 40) return { label: "Moderate", rank: 3, range: "40–70%" };
    if (probability >= 15) return { label: "Slight", rank: 2, range: "15–40%" };
    if (probability >= 5) return { label: "Marginal", rank: 1, range: "5–15%" };
    return { label: "Below Marginal", rank: 0, range: "<5%" };
  }

  function standardProbabilities(data, index) {
    return STANDARD_ML_PRODUCTS
      .map((key) => ({ key, probability: probabilityPercent(data?.layers?.[key], index) }))
      .filter((entry) => Number.isFinite(entry.probability));
  }

  function agreementSummary(data, index) {
    const entries = standardProbabilities(data, index);
    if (!entries.length) return null;
    const values = entries.map((entry) => entry.probability);
    const minimum = Math.min(...values);
    const maximum = Math.max(...values);
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
    const categories = values.map((value) => riskCategory(value).rank);
    const categorySpread = Math.max(...categories) - Math.min(...categories);
    const range = maximum - minimum;
    let qualitative = "Low";
    if (categorySpread === 0 && range <= 10) qualitative = "High";
    else if (categorySpread <= 1 || range <= 20) qualitative = "Moderate";
    return {
      entries,
      count: values.length,
      minimum,
      maximum,
      mean,
      range,
      standardDeviation: Math.sqrt(variance),
      qualitative,
      categorySpread,
      exceedanceCounts: Object.fromEntries(
        THRESHOLDS_PERCENT.map((threshold) => [
          threshold,
          values.filter((value) => value >= threshold).length,
        ]),
      ),
    };
  }

  function decodePredictor(predictor, index) {
    const encoded = Number(predictor?.values?.[index]);
    const low = Number(predictor?.scale_min);
    const high = Number(predictor?.scale_max);
    if (![encoded, low, high].every(Number.isFinite)) return null;
    const normalized = Math.max(0, Math.min(1000, encoded));
    return {
      value: low + (high - low) * normalized / 1000,
      percentilePosition: normalized / 10,
    };
  }

  function formatProbability(value) {
    return Number.isFinite(value) ? `${value.toFixed(1)}%` : "Not available";
  }

  function copyBriefingText(context) {
    const agreement = context.agreement;
    const lines = [
      "Experimental XGBoosted Flash Flood Predictions Briefing",
      `Location: ${Math.abs(context.latitude).toFixed(2)}°${context.latitude >= 0 ? "N" : "S"}, ${Math.abs(context.longitude).toFixed(2)}°${context.longitude >= 0 ? "E" : "W"}`,
      `Valid: ${context.validPeriod || "Not available"}`,
      "",
      `ML ensemble mean: ${formatProbability(context.ensembleMean)}`,
    ];
    if (agreement) {
      lines.push(
        `40/60/75/100-km range: ${agreement.minimum.toFixed(1)}–${agreement.maximum.toFixed(1)}%`,
        `${agreement.exceedanceCounts[15]} of ${agreement.count} standard ML configurations meet or exceed 15%`,
      );
    }
    lines.push(
      `WPC outlook: ${context.wpcCategory || "Not available"}`,
      `Active NWS flood warning: ${context.activeWarning || "Not available"}`,
      "",
      "Experimental machine-learning guidance; not an official NWS forecast, watch, or warning.",
    );
    return lines.join("\n");
  }

  return {
    STANDARD_ML_PRODUCTS,
    THRESHOLDS_PERCENT,
    agreementSummary,
    copyBriefingText,
    decodePredictor,
    formatProbability,
    haversineKm,
    nearestGridPoint,
    probabilityPercent,
    riskCategory,
    standardProbabilities,
  };
}));
