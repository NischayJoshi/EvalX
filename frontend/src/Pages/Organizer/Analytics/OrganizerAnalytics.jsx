/**
 * Organizer Analytics Dashboard
 * =============================
 * 
 * Comprehensive analytics view for hackathon organizers.
 * Displays calibration metrics, theme analysis, submission patterns,
 * and anomaly detection.
 */

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useParams, useNavigate } from "react-router-dom";
import {
  FiArrowLeft,
  FiBarChart2,
  FiCalendar,
  FiClock,
  FiDownload,
  FiFilter,
  FiPieChart,
  FiRefreshCw,
  FiTarget,
  FiTrendingUp,
  FiUsers,
  FiZap,
} from "react-icons/fi";

import {
  StatCard,
  CalibrationCard,
  AnomalyList,
  ExportButton,
  AnalyticsSkeleton,
} from "@/components/analytics";

import {
  SubmissionHeatmap,
  TrendChart,
  ThemeBarChart,
  ScoreDistribution,
} from "@/components/analytics";

import {
  useCalibrationMetrics,
  useThemeAnalysis,
  useSubmissionPatterns,
  useScoringAnomalies,
  useHistoricalTrends,
} from "@/hooks/useAnalytics";

import { downloadExport, getExportColumns } from "@/utils/analyticsApi";
import api from "@/utils/api";

export default function OrganizerAnalytics() {
  const { eventId } = useParams();
  const navigate = useNavigate();

  // State
  const [event, setEvent] = useState(null);
  const [activeTab, setActiveTab] = useState("overview");
  const [roundFilter, setRoundFilter] = useState(null);
  const [trendScope, setTrendScope] = useState("organizer");
  const [exportLoading, setExportLoading] = useState(false);
  const [exportColumns, setExportColumns] = useState([]);

  // Fetch event details
  useEffect(() => {
    const fetchEvent = async () => {
      try {
        const res = await api.get(`/org/event/${eventId}`);
        setEvent(res.data.data);
      } catch (err) {
        console.error("Failed to fetch event:", err);
      }
    };
    if (eventId) fetchEvent();
  }, [eventId]);

  // Fetch export columns
  useEffect(() => {
    const fetchColumns = async () => {
      try {
        const res = await getExportColumns();
        setExportColumns(res.data?.available_columns || []);
      } catch (err) {
        console.error("Failed to fetch export columns:", err);
      }
    };
    fetchColumns();
  }, []);

  // Analytics hooks
  const calibration = useCalibrationMetrics(eventId, roundFilter);
  const themes = useThemeAnalysis(eventId);
  const patterns = useSubmissionPatterns(eventId);
  const anomalies = useScoringAnomalies(eventId);
  const trends = useHistoricalTrends(trendScope, eventId);

  // Loading state
  const isLoading =
    calibration.loading || themes.loading || patterns.loading;

  // Handle export
  const handleExport = async (format = "csv") => {
    setExportLoading(true);
    try {
      await downloadExport(eventId, null, format);
    } catch (err) {
      console.error("Export failed:", err);
    } finally {
      setExportLoading(false);
    }
  };

  // Refresh all data
  const handleRefresh = () => {
    calibration.refetch();
    themes.refetch();
    patterns.refetch();
    anomalies.refetch();
    trends.refetch();
  };

  // Tabs configuration
  const tabs = [
    { id: "overview", label: "Overview", icon: <FiBarChart2 size={18} /> },
    { id: "themes", label: "Themes", icon: <FiPieChart size={18} /> },
    { id: "patterns", label: "Patterns", icon: <FiClock size={18} /> },
    { id: "trends", label: "Trends", icon: <FiTrendingUp size={18} /> },
    { id: "anomalies", label: "Anomalies", icon: <FiZap size={18} /> },
  ];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={() => navigate(-1)}
                className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              >
                <FiArrowLeft size={20} />
              </button>
              <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                  Analytics Dashboard
                </h1>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {event?.name || "Loading..."}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={handleRefresh}
                className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                title="Refresh data"
              >
                <FiRefreshCw size={20} className={isLoading ? "animate-spin" : ""} />
              </button>
              <ExportButton onClick={() => handleExport("csv")} loading={exportLoading} />
            </div>
          </div>

          {/* Tabs */}
          <div className="flex gap-1 mt-4 overflow-x-auto pb-2">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-sm whitespace-nowrap transition-colors ${
                  activeTab === tab.id
                    ? "bg-indigo-100 text-indigo-700 dark:bg-indigo-900/50 dark:text-indigo-300"
                    : "text-gray-600 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700"
                }`}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {isLoading ? (
          <AnalyticsSkeleton />
        ) : (
          <AnimatePresence mode="wait">
            {/* Overview Tab */}
            {activeTab === "overview" && (
              <motion.div
                key="overview"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-6"
              >
                {/* Stats Grid */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  <StatCard
                    title="Total Submissions"
                    value={calibration.data?.total_submissions || 0}
                    icon={FiUsers}
                    variant="primary"
                  />
                  <StatCard
                    title="Average Score"
                    value={calibration.data?.mean_score?.toFixed(1) || "0"}
                    subtitle={`Median: ${calibration.data?.median_score?.toFixed(1) || "0"}`}
                    icon={FiTarget}
                    variant="success"
                  />
                  <StatCard
                    title="Themes Detected"
                    value={themes.data?.themes_detected || 0}
                    subtitle={themes.data?.top_theme ? `Top: ${themes.data.top_theme}` : ""}
                    icon={FiPieChart}
                    variant="default"
                  />
                  <StatCard
                    title="Anomalies"
                    value={anomalies.data?.count || 0}
                    subtitle="Unusual scores flagged"
                    icon={FiZap}
                    variant={anomalies.data?.count > 0 ? "warning" : "default"}
                  />
                </div>

                {/* Charts Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <CalibrationCard data={calibration.data} />
                  
                  <div className="p-6 rounded-xl bg-white dark:bg-gray-800 shadow-sm border border-gray-200 dark:border-gray-700">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                      Score Distribution
                    </h3>
                    <ScoreDistribution data={calibration.data} />
                  </div>
                </div>

                {/* Theme Overview */}
                {themes.data?.themes?.length > 0 && (
                  <div className="p-6 rounded-xl bg-white dark:bg-gray-800 shadow-sm border border-gray-200 dark:border-gray-700">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                      Theme Performance Overview
                    </h3>
                    <ThemeBarChart data={themes.data} />
                  </div>
                )}
              </motion.div>
            )}

            {/* Themes Tab */}
            {activeTab === "themes" && (
              <motion.div
                key="themes"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-6"
              >
                <div className="p-6 rounded-xl bg-white dark:bg-gray-800 shadow-sm border border-gray-200 dark:border-gray-700">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                      Theme-wise Analysis
                    </h3>
                    <span className="text-sm text-gray-500">
                      {themes.data?.themes_detected || 0} themes detected
                    </span>
                  </div>
                  <ThemeBarChart data={themes.data} className="mb-6" />

                  {/* Theme Details */}
                  {themes.data?.themes?.length > 0 && (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-6">
                      {themes.data.themes.map((theme, i) => (
                        <motion.div
                          key={theme.theme_name}
                          initial={{ opacity: 0, scale: 0.95 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: i * 0.1 }}
                          className="p-4 rounded-lg bg-gray-50 dark:bg-gray-700/50"
                        >
                          <h4 className="font-semibold text-gray-900 dark:text-white">
                            {theme.theme_name}
                          </h4>
                          <div className="mt-2 space-y-1 text-sm text-gray-600 dark:text-gray-400">
                            <p>Teams: {theme.submission_count}</p>
                            <p>Avg Score: {theme.avg_score?.toFixed(1)}</p>
                            <p>Range: {theme.min_score?.toFixed(0)} - {theme.max_score?.toFixed(0)}</p>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  )}
                </div>
              </motion.div>
            )}

            {/* Patterns Tab */}
            {activeTab === "patterns" && (
              <motion.div
                key="patterns"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-6"
              >
                {/* Timing Stats */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <StatCard
                    title="Early Submissions"
                    value={`${((patterns.data?.early_submission_ratio || 0) * 100).toFixed(0)}%`}
                    subtitle="> 24h before deadline"
                    icon={FiClock}
                    variant="success"
                  />
                  <StatCard
                    title="Last-Minute"
                    value={`${((patterns.data?.late_submission_ratio || 0) * 100).toFixed(0)}%`}
                    subtitle="< 1h before deadline"
                    icon={FiZap}
                    variant="warning"
                  />
                  <StatCard
                    title="Peak Activity"
                    value={patterns.data?.peak_hour !== undefined ? `${patterns.data.peak_hour}:00` : "N/A"}
                    subtitle={patterns.data?.peak_day || ""}
                    icon={FiCalendar}
                    variant="primary"
                  />
                </div>

                {/* Heatmap */}
                <div className="p-6 rounded-xl bg-white dark:bg-gray-800 shadow-sm border border-gray-200 dark:border-gray-700">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                    Submission Activity Heatmap
                  </h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                    Darker cells indicate more submissions during that time period.
                  </p>
                  <SubmissionHeatmap data={patterns.data} />
                </div>
              </motion.div>
            )}

            {/* Trends Tab */}
            {activeTab === "trends" && (
              <motion.div
                key="trends"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-6"
              >
                {/* Scope Toggle */}
                <div className="flex items-center gap-4">
                  <span className="text-sm text-gray-500">Compare with:</span>
                  <div className="flex rounded-lg bg-gray-100 dark:bg-gray-800 p-1">
                    <button
                      onClick={() => setTrendScope("organizer")}
                      className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                        trendScope === "organizer"
                          ? "bg-white dark:bg-gray-700 shadow-sm text-indigo-600"
                          : "text-gray-600 dark:text-gray-400"
                      }`}
                    >
                      My Events
                    </button>
                    <button
                      onClick={() => setTrendScope("global")}
                      className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                        trendScope === "global"
                          ? "bg-white dark:bg-gray-700 shadow-sm text-indigo-600"
                          : "text-gray-600 dark:text-gray-400"
                      }`}
                    >
                      All Events
                    </button>
                  </div>
                </div>

                {/* Trend Chart */}
                <div className="p-6 rounded-xl bg-white dark:bg-gray-800 shadow-sm border border-gray-200 dark:border-gray-700">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                    Historical Performance Trends
                  </h3>
                  {trends.loading ? (
                    <div className="h-48 flex items-center justify-center text-gray-400">
                      Loading trends...
                    </div>
                  ) : trends.data?.events?.length > 0 ? (
                    <TrendChart data={trends.data} height={250} />
                  ) : (
                    <div className="h-48 flex items-center justify-center text-gray-400">
                      Not enough events for trend analysis
                    </div>
                  )}
                </div>

                {/* Trend Summary */}
                {trends.data && (
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <StatCard
                      title="Events Analyzed"
                      value={trends.data.total_events || 0}
                      icon={FiCalendar}
                    />
                    <StatCard
                      title="Average Score"
                      value={trends.data.average_score?.toFixed(1) || "N/A"}
                      icon={FiTarget}
                    />
                    <StatCard
                      title="Trend"
                      value={trends.data.trend || "Stable"}
                      icon={FiTrendingUp}
                      trend={
                        trends.data.trend === "improving"
                          ? "up"
                          : trends.data.trend === "declining"
                          ? "down"
                          : undefined
                      }
                    />
                  </div>
                )}
              </motion.div>
            )}

            {/* Anomalies Tab */}
            {activeTab === "anomalies" && (
              <motion.div
                key="anomalies"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-6"
              >
                <div className="flex items-center justify-between">
                  <p className="text-gray-600 dark:text-gray-400">
                    Submissions with scores significantly different from the average (Z-score &gt; 2.0)
                  </p>
                </div>
                <AnomalyList anomalies={anomalies.data?.anomalies} />
              </motion.div>
            )}
          </AnimatePresence>
        )}
      </main>
    </div>
  );
}
