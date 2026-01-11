/**
 * Participant Analytics Dashboard
 * ================================
 * 
 * Analytics view for hackathon participants (developers).
 * Displays skill radar, peer comparison, and progress timeline.
 */

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useParams, useNavigate } from "react-router-dom";
import {
  FiArrowLeft,
  FiAward,
  FiBarChart2,
  FiRefreshCw,
  FiTarget,
  FiTrendingUp,
  FiUser,
  FiZap,
} from "react-icons/fi";

import {
  StatCard,
  PeerComparisonCard,
  AnalyticsSkeleton,
} from "@/components/analytics";

import {
  RadarChart,
  TrendChart,
  PercentileGauge,
} from "@/components/analytics";

import {
  useSkillRadar,
  usePeerComparison,
  useProgressTimeline,
} from "@/hooks/useAnalytics";

import api from "@/utils/api";

export default function ParticipantAnalytics() {
  const { eventId } = useParams();
  const navigate = useNavigate();

  // State
  const [event, setEvent] = useState(null);
  const [activeTab, setActiveTab] = useState("overview");

  // Fetch event details
  useEffect(() => {
    const fetchEvent = async () => {
      try {
        const res = await api.get(`/dev/event/${eventId}`);
        setEvent(res.data.data);
      } catch (err) {
        console.error("Failed to fetch event:", err);
      }
    };
    if (eventId) fetchEvent();
  }, [eventId]);

  // Analytics hooks
  const radar = useSkillRadar(eventId);
  const comparison = usePeerComparison(eventId);
  const progress = useProgressTimeline();

  // Loading state
  const isLoading = radar.loading || comparison.loading;

  // Refresh all data
  const handleRefresh = () => {
    radar.refetch();
    comparison.refetch();
    progress.refetch();
  };

  // Tabs configuration
  const tabs = [
    { id: "overview", label: "Overview", icon: <FiBarChart2 size={18} /> },
    { id: "skills", label: "Skills", icon: <FiTarget size={18} /> },
    { id: "progress", label: "Progress", icon: <FiTrendingUp size={18} /> },
  ];

  // Calculate overall grade
  const getGrade = (score) => {
    if (score >= 90) return { grade: "A", color: "text-green-600" };
    if (score >= 80) return { grade: "B", color: "text-blue-600" };
    if (score >= 70) return { grade: "C", color: "text-amber-600" };
    if (score >= 60) return { grade: "D", color: "text-orange-600" };
    return { grade: "F", color: "text-red-600" };
  };

  const overallScore = radar.data?.overall_score || comparison.data?.team_score || 0;
  const gradeInfo = getGrade(overallScore);

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
                  My Analytics
                </h1>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {event?.name || "Loading..."}
                </p>
              </div>
            </div>

            <button
              onClick={handleRefresh}
              className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              title="Refresh data"
            >
              <FiRefreshCw size={20} className={isLoading ? "animate-spin" : ""} />
            </button>
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
                {/* Hero Stats */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {/* Overall Score Card */}
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="lg:col-span-1 p-6 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 text-white shadow-lg"
                  >
                    <h3 className="text-lg font-medium opacity-90">Overall Score</h3>
                    <div className="mt-4 flex items-end gap-4">
                      <span className="text-5xl font-bold">
                        {overallScore.toFixed(1)}
                      </span>
                      <span className={`text-3xl font-bold mb-1 ${gradeInfo.color} bg-white/20 px-3 py-1 rounded-lg`}>
                        {gradeInfo.grade}
                      </span>
                    </div>
                    <p className="mt-4 text-sm opacity-80">
                      {radar.data?.team_name || "Your Team"}
                    </p>
                  </motion.div>

                  {/* Percentile Gauge */}
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.1 }}
                    className="p-6 rounded-xl bg-white dark:bg-gray-800 shadow-sm border border-gray-200 dark:border-gray-700 flex flex-col items-center justify-center"
                  >
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                      Your Ranking
                    </h3>
                    <PercentileGauge
                      percentile={comparison.data?.percentile}
                      rank={comparison.data?.rank}
                      totalTeams={comparison.data?.total_teams}
                    />
                  </motion.div>

                  {/* Quick Stats */}
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.2 }}
                    className="space-y-4"
                  >
                    <StatCard
                      title="Event Average"
                      value={comparison.data?.event_avg?.toFixed(1) || "N/A"}
                      icon={FiBarChart2}
                      variant="default"
                    />
                    <StatCard
                      title="Above Average"
                      value={comparison.data?.score_difference > 0 
                        ? `+${comparison.data.score_difference.toFixed(1)}` 
                        : comparison.data?.score_difference?.toFixed(1) || "0"}
                      icon={comparison.data?.score_difference > 0 ? FiTrendingUp : FiTarget}
                      variant={comparison.data?.score_difference > 0 ? "success" : "warning"}
                    />
                  </motion.div>
                </div>

                {/* Skill Radar Preview */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="p-6 rounded-xl bg-white dark:bg-gray-800 shadow-sm border border-gray-200 dark:border-gray-700">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                      Skill Overview
                    </h3>
                    <div className="flex justify-center">
                      <RadarChart data={radar.data} size={280} />
                    </div>
                  </div>

                  <PeerComparisonCard data={comparison.data} />
                </div>
              </motion.div>
            )}

            {/* Skills Tab */}
            {activeTab === "skills" && (
              <motion.div
                key="skills"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-6"
              >
                {/* Large Radar Chart */}
                <div className="p-6 rounded-xl bg-white dark:bg-gray-800 shadow-sm border border-gray-200 dark:border-gray-700">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
                    Skill Radar Analysis
                  </h3>
                  <div className="flex justify-center">
                    <RadarChart data={radar.data} size={350} />
                  </div>
                </div>

                {/* Skill Breakdown */}
                {radar.data?.dimensions?.length > 0 && (
                  <div className="p-6 rounded-xl bg-white dark:bg-gray-800 shadow-sm border border-gray-200 dark:border-gray-700">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                      Skill Breakdown
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {radar.data.dimensions.map((dim, i) => {
                        const score = dim.score;
                        const maxScore = dim.max_score || 100;
                        const percentage = (score / maxScore) * 100;

                        return (
                          <motion.div
                            key={dim.dimension_name || dim.name}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: i * 0.1 }}
                            className="p-4 rounded-lg bg-gray-50 dark:bg-gray-700/50"
                          >
                            <div className="flex justify-between items-center mb-2">
                              <span className="font-medium text-gray-900 dark:text-white">
                                {dim.dimension_name || dim.name}
                              </span>
                              <span className="text-indigo-600 font-bold">
                                {score.toFixed(0)}
                              </span>
                            </div>
                            <div className="h-2 bg-gray-200 dark:bg-gray-600 rounded-full overflow-hidden">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${percentage}%` }}
                                transition={{ duration: 0.5, delay: i * 0.1 }}
                                className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full"
                              />
                            </div>
                          </motion.div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Improvement Tips */}
                <div className="p-6 rounded-xl bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800">
                  <h3 className="text-lg font-semibold text-indigo-900 dark:text-indigo-200 mb-3">
                    💡 Improvement Tips
                  </h3>
                  {radar.data?.dimensions && (
                    <ul className="space-y-2 text-indigo-800 dark:text-indigo-300">
                      {radar.data.dimensions
                        .sort((a, b) => a.score - b.score)
                        .slice(0, 2)
                        .map((dim) => (
                          <li key={dim.dimension_name || dim.name}>
                            Focus on improving <strong>{dim.dimension_name || dim.name}</strong> (current: {dim.score.toFixed(0)})
                          </li>
                        ))}
                    </ul>
                  )}
                </div>
              </motion.div>
            )}

            {/* Progress Tab */}
            {activeTab === "progress" && (
              <motion.div
                key="progress"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-6"
              >
                {/* Progress Summary */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <StatCard
                    title="Events Participated"
                    value={progress.data?.total_events_participated || 0}
                    icon={FiAward}
                    variant="primary"
                  />
                  <StatCard
                    title="Average Score"
                    value={progress.data?.average_score?.toFixed(1) || "N/A"}
                    icon={FiTarget}
                  />
                  <StatCard
                    title="Best Score"
                    value={progress.data?.best_score?.toFixed(1) || "N/A"}
                    icon={FiZap}
                    variant="success"
                  />
                </div>

                {/* Progress Chart */}
                <div className="p-6 rounded-xl bg-white dark:bg-gray-800 shadow-sm border border-gray-200 dark:border-gray-700">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                      Performance Over Time
                    </h3>
                    {progress.data?.improvement_trend && (
                      <span
                        className={`px-3 py-1 rounded-full text-sm font-medium ${
                          progress.data.improvement_trend === "improving"
                            ? "bg-green-100 text-green-700"
                            : progress.data.improvement_trend === "declining"
                            ? "bg-red-100 text-red-700"
                            : "bg-gray-100 text-gray-700"
                        }`}
                      >
                        {progress.data.improvement_trend === "improving" ? "📈 Improving" :
                         progress.data.improvement_trend === "declining" ? "📉 Declining" :
                         "➡️ Stable"}
                      </span>
                    )}
                  </div>

                  {progress.loading ? (
                    <div className="h-64 flex items-center justify-center text-gray-400">
                      Loading progress data...
                    </div>
                  ) : progress.data?.events?.length > 0 ? (
                    <TrendChart data={progress.data} height={250} />
                  ) : (
                    <div className="h-64 flex flex-col items-center justify-center text-gray-400">
                      <FiUser size={48} className="mb-4 opacity-50" />
                      <p>Participate in more events to see your progress!</p>
                    </div>
                  )}
                </div>

                {/* Percentile History */}
                {progress.data?.percentile_trend?.length > 0 && (
                  <div className="p-6 rounded-xl bg-white dark:bg-gray-800 shadow-sm border border-gray-200 dark:border-gray-700">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                      Percentile History
                    </h3>
                    <div className="flex items-center gap-4 overflow-x-auto pb-2">
                      {progress.data.percentile_trend.map((p, i) => (
                        <motion.div
                          key={i}
                          initial={{ opacity: 0, scale: 0.8 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: i * 0.1 }}
                          className="flex-shrink-0 w-16 text-center"
                        >
                          <div
                            className="w-12 h-12 mx-auto rounded-full flex items-center justify-center text-sm font-bold"
                            style={{
                              background: `conic-gradient(#6366f1 ${p}%, #e5e7eb ${p}%)`,
                            }}
                          >
                            <span className="bg-white dark:bg-gray-800 w-9 h-9 rounded-full flex items-center justify-center">
                              {p.toFixed(0)}
                            </span>
                          </div>
                          <p className="text-xs text-gray-500 mt-1">Event {i + 1}</p>
                        </motion.div>
                      ))}
                    </div>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        )}
      </main>
    </div>
  );
}
