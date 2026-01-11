/**
 * Analytics Custom Hooks
 * ======================
 * 
 * React hooks for fetching and managing analytics data.
 */

import { useState, useEffect, useCallback } from "react";
import {
  getCalibrationMetrics,
  getThemeAnalysis,
  getHistoricalTrends,
  getSubmissionPatterns,
  getScoringAnomalies,
  getSkillRadar,
  getPeerComparison,
  getProgressTimeline,
} from "@/utils/analyticsApi";

/**
 * Generic fetch hook with loading and error states.
 */
const useAsyncData = (fetchFn, deps = []) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const refetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchFn();
      setData(result?.data || result);
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || "An error occurred");
    } finally {
      setLoading(false);
    }
  }, [fetchFn]);

  useEffect(() => {
    refetch();
  }, deps);

  return { data, loading, error, refetch };
};

// =============================================================================
// ORGANIZER HOOKS
// =============================================================================

/**
 * Hook for calibration metrics.
 * @param {string} eventId - Event ID
 * @param {string} [roundFilter] - Optional round filter
 */
export const useCalibrationMetrics = (eventId, roundFilter = null) => {
  return useAsyncData(
    () => getCalibrationMetrics(eventId, roundFilter),
    [eventId, roundFilter]
  );
};

/**
 * Hook for theme analysis.
 * @param {string} eventId - Event ID
 */
export const useThemeAnalysis = (eventId) => {
  return useAsyncData(() => getThemeAnalysis(eventId), [eventId]);
};

/**
 * Hook for historical trends.
 * @param {string} [scope='organizer'] - Scope of analysis
 * @param {string} [currentEventId] - Current event for comparison
 */
export const useHistoricalTrends = (scope = "organizer", currentEventId = null) => {
  return useAsyncData(
    () => getHistoricalTrends(scope, currentEventId),
    [scope, currentEventId]
  );
};

/**
 * Hook for submission patterns.
 * @param {string} eventId - Event ID
 */
export const useSubmissionPatterns = (eventId) => {
  return useAsyncData(() => getSubmissionPatterns(eventId), [eventId]);
};

/**
 * Hook for scoring anomalies.
 * @param {string} eventId - Event ID
 * @param {number} [threshold=2.0] - Z-score threshold
 */
export const useScoringAnomalies = (eventId, threshold = 2.0) => {
  return useAsyncData(
    () => getScoringAnomalies(eventId, threshold),
    [eventId, threshold]
  );
};

// =============================================================================
// PARTICIPANT HOOKS
// =============================================================================

/**
 * Hook for skill radar data.
 * @param {string} eventId - Event ID
 */
export const useSkillRadar = (eventId) => {
  return useAsyncData(() => getSkillRadar(eventId), [eventId]);
};

/**
 * Hook for peer comparison.
 * @param {string} eventId - Event ID
 */
export const usePeerComparison = (eventId) => {
  return useAsyncData(() => getPeerComparison(eventId), [eventId]);
};

/**
 * Hook for progress timeline.
 */
export const useProgressTimeline = () => {
  return useAsyncData(() => getProgressTimeline(), []);
};

// =============================================================================
// COMBINED HOOKS
// =============================================================================

/**
 * Hook for all organizer analytics data.
 * @param {string} eventId - Event ID
 */
export const useOrganizerAnalytics = (eventId) => {
  const calibration = useCalibrationMetrics(eventId);
  const themes = useThemeAnalysis(eventId);
  const patterns = useSubmissionPatterns(eventId);
  const anomalies = useScoringAnomalies(eventId);

  const loading =
    calibration.loading || themes.loading || patterns.loading || anomalies.loading;

  const error =
    calibration.error || themes.error || patterns.error || anomalies.error;

  const refetchAll = () => {
    calibration.refetch();
    themes.refetch();
    patterns.refetch();
    anomalies.refetch();
  };

  return {
    calibration: calibration.data,
    themes: themes.data,
    patterns: patterns.data,
    anomalies: anomalies.data,
    loading,
    error,
    refetchAll,
  };
};

/**
 * Hook for all participant analytics data.
 * @param {string} eventId - Event ID
 */
export const useParticipantAnalytics = (eventId) => {
  const radar = useSkillRadar(eventId);
  const comparison = usePeerComparison(eventId);
  const progress = useProgressTimeline();

  const loading = radar.loading || comparison.loading || progress.loading;
  const error = radar.error || comparison.error || progress.error;

  const refetchAll = () => {
    radar.refetch();
    comparison.refetch();
    progress.refetch();
  };

  return {
    radar: radar.data,
    comparison: comparison.data,
    progress: progress.data,
    loading,
    error,
    refetchAll,
  };
};

export default {
  useCalibrationMetrics,
  useThemeAnalysis,
  useHistoricalTrends,
  useSubmissionPatterns,
  useScoringAnomalies,
  useSkillRadar,
  usePeerComparison,
  useProgressTimeline,
  useOrganizerAnalytics,
  useParticipantAnalytics,
};
