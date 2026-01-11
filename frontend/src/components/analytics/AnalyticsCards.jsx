/**
 * Analytics Card Components
 * =========================
 * 
 * Reusable card components for displaying analytics metrics.
 */

import { motion } from "framer-motion";
import {
  FiActivity,
  FiAlertTriangle,
  FiAward,
  FiBarChart2,
  FiClock,
  FiDownload,
  FiTrendingUp,
  FiTrendingDown,
  FiMinus,
  FiUsers,
  FiTarget,
  FiZap,
} from "react-icons/fi";

// =============================================================================
// STAT CARD
// =============================================================================

/**
 * Statistics Card Component
 * Displays a single metric with icon and optional trend.
 */
export function StatCard({
  title,
  value,
  subtitle,
  icon: Icon = FiActivity,
  trend,
  trendValue,
  variant = "default",
  className = "",
}) {
  const variants = {
    default: "bg-white dark:bg-gray-800",
    primary: "bg-indigo-50 dark:bg-indigo-900/20",
    success: "bg-green-50 dark:bg-green-900/20",
    warning: "bg-amber-50 dark:bg-amber-900/20",
    danger: "bg-red-50 dark:bg-red-900/20",
  };

  const iconVariants = {
    default: "text-gray-500 bg-gray-100 dark:bg-gray-700",
    primary: "text-indigo-600 bg-indigo-100 dark:bg-indigo-800",
    success: "text-green-600 bg-green-100 dark:bg-green-800",
    warning: "text-amber-600 bg-amber-100 dark:bg-amber-800",
    danger: "text-red-600 bg-red-100 dark:bg-red-800",
  };

  const TrendIcon =
    trend === "up" ? FiTrendingUp : trend === "down" ? FiTrendingDown : FiMinus;

  const trendColor =
    trend === "up"
      ? "text-green-600"
      : trend === "down"
      ? "text-red-600"
      : "text-gray-500";

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`rounded-xl p-5 shadow-sm border border-gray-200 dark:border-gray-700 ${variants[variant]} ${className}`}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">{title}</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">{value}</p>
          {subtitle && (
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">{subtitle}</p>
          )}
          {trend && trendValue && (
            <div className={`flex items-center gap-1 mt-2 ${trendColor}`}>
              <TrendIcon size={14} />
              <span className="text-sm font-medium">{trendValue}</span>
            </div>
          )}
        </div>
        <div className={`p-3 rounded-lg ${iconVariants[variant]}`}>
          <Icon size={24} />
        </div>
      </div>
    </motion.div>
  );
}

// =============================================================================
// CALIBRATION CARD
// =============================================================================

/**
 * Calibration Metrics Card
 * Displays AI evaluation consistency metrics.
 */
export function CalibrationCard({ data, className = "" }) {
  if (!data) {
    return (
      <div className={`p-6 rounded-xl bg-white dark:bg-gray-800 shadow-sm ${className}`}>
        <p className="text-gray-400">Loading calibration data...</p>
      </div>
    );
  }

  const getConsistencyLevel = (stdDev) => {
    if (stdDev < 10) return { label: "Excellent", color: "text-green-600", bg: "bg-green-100" };
    if (stdDev < 15) return { label: "Good", color: "text-blue-600", bg: "bg-blue-100" };
    if (stdDev < 20) return { label: "Moderate", color: "text-amber-600", bg: "bg-amber-100" };
    return { label: "High Variance", color: "text-red-600", bg: "bg-red-100" };
  };

  const consistency = getConsistencyLevel(data.std_deviation);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`p-6 rounded-xl bg-white dark:bg-gray-800 shadow-sm border border-gray-200 dark:border-gray-700 ${className}`}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Calibration Metrics
        </h3>
        <span
          className={`px-3 py-1 rounded-full text-sm font-medium ${consistency.bg} ${consistency.color}`}
        >
          {consistency.label}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-700/50">
          <p className="text-sm text-gray-500 dark:text-gray-400">Mean Score</p>
          <p className="text-xl font-bold text-gray-900 dark:text-white">
            {data.mean_score?.toFixed(1)}
          </p>
        </div>
        <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-700/50">
          <p className="text-sm text-gray-500 dark:text-gray-400">Median Score</p>
          <p className="text-xl font-bold text-gray-900 dark:text-white">
            {data.median_score?.toFixed(1)}
          </p>
        </div>
        <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-700/50">
          <p className="text-sm text-gray-500 dark:text-gray-400">Std Deviation</p>
          <p className="text-xl font-bold text-gray-900 dark:text-white">
            {data.std_deviation?.toFixed(2)}
          </p>
        </div>
        <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-700/50">
          <p className="text-sm text-gray-500 dark:text-gray-400">Total Submissions</p>
          <p className="text-xl font-bold text-gray-900 dark:text-white">
            {data.total_submissions}
          </p>
        </div>
      </div>

      {data.anomalies_detected > 0 && (
        <div className="mt-4 p-3 rounded-lg bg-amber-50 dark:bg-amber-900/20 flex items-center gap-2">
          <FiAlertTriangle className="text-amber-600" />
          <span className="text-sm text-amber-700 dark:text-amber-400">
            {data.anomalies_detected} scoring anomalies detected
          </span>
        </div>
      )}
    </motion.div>
  );
}

// =============================================================================
// ANOMALY LIST
// =============================================================================

/**
 * Scoring Anomalies List
 * Displays flagged submissions with unusual scores.
 */
export function AnomalyList({ anomalies = [], className = "" }) {
  if (!anomalies?.length) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className={`p-6 rounded-xl bg-white dark:bg-gray-800 shadow-sm border border-gray-200 dark:border-gray-700 ${className}`}
      >
        <div className="flex items-center gap-2 mb-4">
          <FiAlertTriangle className="text-amber-500" />
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Scoring Anomalies
          </h3>
        </div>
        <p className="text-gray-500 dark:text-gray-400 text-center py-8">
          No anomalies detected - scores are consistent! 🎉
        </p>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className={`p-6 rounded-xl bg-white dark:bg-gray-800 shadow-sm border border-gray-200 dark:border-gray-700 ${className}`}
    >
      <div className="flex items-center gap-2 mb-4">
        <FiAlertTriangle className="text-amber-500" />
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Scoring Anomalies ({anomalies.length})
        </h3>
      </div>

      <div className="space-y-3 max-h-64 overflow-y-auto">
        {anomalies.map((anomaly, i) => (
          <motion.div
            key={anomaly.submission_id || i}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.05 }}
            className="p-3 rounded-lg bg-gray-50 dark:bg-gray-700/50 flex items-center justify-between"
          >
            <div>
              <p className="font-medium text-gray-900 dark:text-white">
                {anomaly.team_name || `Submission ${i + 1}`}
              </p>
              <p className="text-sm text-gray-500">
                Score: {anomaly.score?.toFixed(1)} | Z-Score: {anomaly.z_score?.toFixed(2)}
              </p>
            </div>
            <span
              className={`px-2 py-1 rounded text-xs font-medium ${
                anomaly.z_score > 0
                  ? "bg-green-100 text-green-700"
                  : "bg-red-100 text-red-700"
              }`}
            >
              {anomaly.z_score > 0 ? "High" : "Low"}
            </span>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

// =============================================================================
// PEER COMPARISON CARD
// =============================================================================

/**
 * Peer Comparison Card
 * Shows team's ranking among peers.
 */
export function PeerComparisonCard({ data, className = "" }) {
  if (!data) {
    return (
      <div className={`p-6 rounded-xl bg-white dark:bg-gray-800 shadow-sm ${className}`}>
        <p className="text-gray-400">Loading comparison data...</p>
      </div>
    );
  }

  const isAboveAverage = data.team_score > data.event_avg;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`p-6 rounded-xl bg-white dark:bg-gray-800 shadow-sm border border-gray-200 dark:border-gray-700 ${className}`}
    >
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
        Peer Comparison
      </h3>

      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-gray-500 dark:text-gray-400">Your Score</span>
          <span className="text-2xl font-bold text-indigo-600">
            {data.team_score?.toFixed(1)}
          </span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-gray-500 dark:text-gray-400">Event Average</span>
          <span className="text-lg font-medium text-gray-700 dark:text-gray-300">
            {data.event_avg?.toFixed(1)}
          </span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-gray-500 dark:text-gray-400">Difference</span>
          <span
            className={`text-lg font-medium flex items-center gap-1 ${
              isAboveAverage ? "text-green-600" : "text-red-600"
            }`}
          >
            {isAboveAverage ? <FiTrendingUp /> : <FiTrendingDown />}
            {isAboveAverage ? "+" : ""}
            {data.score_difference?.toFixed(1)}
          </span>
        </div>

        <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <span className="text-gray-500 dark:text-gray-400">Rank</span>
            <span className="flex items-center gap-2">
              <FiAward className="text-amber-500" />
              <span className="text-xl font-bold text-gray-900 dark:text-white">
                #{data.rank}
              </span>
              <span className="text-gray-400">/ {data.total_teams}</span>
            </span>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

// =============================================================================
// EXPORT BUTTON
// =============================================================================

/**
 * Export Data Button
 * Triggers data export download.
 */
export function ExportButton({ onClick, loading = false, className = "" }) {
  return (
    <motion.button
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      disabled={loading}
      className={`flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600 text-white font-medium hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors ${className}`}
    >
      <FiDownload className={loading ? "animate-bounce" : ""} />
      {loading ? "Exporting..." : "Export Data"}
    </motion.button>
  );
}

// =============================================================================
// LOADING SKELETON
// =============================================================================

/**
 * Analytics Loading Skeleton
 * Placeholder while data loads.
 */
export function AnalyticsSkeleton({ className = "" }) {
  return (
    <div className={`animate-pulse ${className}`}>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="h-28 bg-gray-200 dark:bg-gray-700 rounded-xl" />
        ))}
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="h-64 bg-gray-200 dark:bg-gray-700 rounded-xl" />
        <div className="h-64 bg-gray-200 dark:bg-gray-700 rounded-xl" />
      </div>
    </div>
  );
}

export default {
  StatCard,
  CalibrationCard,
  AnomalyList,
  PeerComparisonCard,
  ExportButton,
  AnalyticsSkeleton,
};
