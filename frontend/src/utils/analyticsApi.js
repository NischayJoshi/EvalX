/**
 * Analytics API Service
 * =====================
 * 
 * API functions for the Advanced Analytics Dashboard.
 * Separate from main api.js to avoid conflicts.
 */

import api from "./api";

// =============================================================================
// ORGANIZER ANALYTICS API
// =============================================================================

/**
 * Get AI evaluation calibration metrics for an event.
 * @param {string} eventId - Event ID
 * @param {string} [roundFilter] - Optional round filter (ppt, repo, viva)
 */
export const getCalibrationMetrics = async (eventId, roundFilter = null) => {
  const params = roundFilter ? { round_filter: roundFilter } : {};
  const res = await api.get(`/org/analytics/${eventId}/calibration`, { params });
  return res.data;
};

/**
 * Get theme-wise performance analysis for an event.
 * @param {string} eventId - Event ID
 */
export const getThemeAnalysis = async (eventId) => {
  const res = await api.get(`/org/analytics/${eventId}/themes`);
  return res.data;
};

/**
 * Get historical trends across events.
 * @param {string} [scope='organizer'] - 'organizer' or 'global'
 * @param {string} [currentEventId] - Optional current event for comparison
 */
export const getHistoricalTrends = async (scope = "organizer", currentEventId = null) => {
  const params = { scope };
  if (currentEventId) params.current_event_id = currentEventId;
  const res = await api.get(`/org/analytics/trends`, { params });
  return res.data;
};

/**
 * Get submission timing patterns for an event.
 * @param {string} eventId - Event ID
 */
export const getSubmissionPatterns = async (eventId) => {
  const res = await api.get(`/org/analytics/${eventId}/patterns`);
  return res.data;
};

/**
 * Get scoring anomalies for an event.
 * @param {string} eventId - Event ID
 * @param {number} [threshold=2.0] - Z-score threshold
 */
export const getScoringAnomalies = async (eventId, threshold = 2.0) => {
  const res = await api.get(`/org/analytics/${eventId}/anomalies`, {
    params: { z_score_threshold: threshold },
  });
  return res.data;
};

/**
 * Get available export columns.
 */
export const getExportColumns = async () => {
  const res = await api.get(`/org/analytics/export/columns`);
  return res.data;
};

/**
 * Export evaluation data.
 * @param {string} eventId - Event ID
 * @param {Object} options - Export options
 */
export const exportEvaluationData = async (eventId, options = {}) => {
  const { columns, format = "csv", roundFilter, minScore, maxScore } = options;
  const params = { format };
  if (columns?.length) params.columns = columns.join(",");
  if (roundFilter) params.round_filter = roundFilter;
  if (minScore !== undefined) params.min_score = minScore;
  if (maxScore !== undefined) params.max_score = maxScore;

  const res = await api.get(`/org/analytics/${eventId}/export`, { params });
  return res.data;
};

/**
 * Download export file.
 * @param {string} eventId - Event ID
 * @param {string[]} [columns] - Columns to export
 * @param {string} [format='csv'] - Export format
 */
export const downloadExport = async (eventId, columns = null, format = "csv") => {
  const params = { format };
  if (columns?.length) params.columns = columns.join(",");

  const res = await api.get(`/org/analytics/${eventId}/export/download`, {
    params,
    responseType: "blob",
  });

  // Create download link
  const url = window.URL.createObjectURL(new Blob([res.data]));
  const link = document.createElement("a");
  link.href = url;
  link.setAttribute("download", `evalx_export_${eventId}.${format}`);
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
};

// =============================================================================
// PARTICIPANT ANALYTICS API
// =============================================================================

/**
 * Get skill radar chart data for participant's team.
 * @param {string} eventId - Event ID
 */
export const getSkillRadar = async (eventId) => {
  const res = await api.get(`/dev/analytics/${eventId}/radar`);
  return res.data;
};

/**
 * Get peer comparison metrics.
 * @param {string} eventId - Event ID
 */
export const getPeerComparison = async (eventId) => {
  const res = await api.get(`/dev/analytics/${eventId}/comparison`);
  return res.data;
};

/**
 * Get progress timeline across all events.
 */
export const getProgressTimeline = async () => {
  const res = await api.get(`/dev/analytics/progress`);
  return res.data;
};

// =============================================================================
// HEALTH CHECK
// =============================================================================

/**
 * Check analytics module health.
 */
export const checkAnalyticsHealth = async () => {
  const res = await api.get(`/analytics/health`);
  return res.data;
};

export default {
  // Organizer
  getCalibrationMetrics,
  getThemeAnalysis,
  getHistoricalTrends,
  getSubmissionPatterns,
  getScoringAnomalies,
  getExportColumns,
  exportEvaluationData,
  downloadExport,
  // Participant
  getSkillRadar,
  getPeerComparison,
  getProgressTimeline,
  // Health
  checkAnalyticsHealth,
};
