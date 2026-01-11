/**
 * Analytics Charts Components
 * ===========================
 * 
 * Reusable chart components for analytics dashboard.
 * Uses lightweight Canvas-based rendering.
 */

import { useEffect, useRef } from "react";
import { motion } from "framer-motion";

// =============================================================================
// RADAR CHART
// =============================================================================

/**
 * Skill Radar Chart Component
 * Displays 6-dimensional skill assessment.
 */
export function RadarChart({ data, size = 300, className = "" }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !data?.dimensions?.length) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const centerX = size / 2;
    const centerY = size / 2;
    const radius = size * 0.35;

    // Clear canvas
    ctx.clearRect(0, 0, size, size);

    const dimensions = data.dimensions;
    const numDimensions = dimensions.length;
    const angleStep = (Math.PI * 2) / numDimensions;

    // Draw grid circles
    ctx.strokeStyle = "#e5e7eb";
    ctx.lineWidth = 1;
    for (let i = 1; i <= 5; i++) {
      ctx.beginPath();
      ctx.arc(centerX, centerY, (radius * i) / 5, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Draw axis lines and labels
    ctx.fillStyle = "#6b7280";
    ctx.font = "12px Inter, sans-serif";
    ctx.textAlign = "center";

    dimensions.forEach((dim, i) => {
      const angle = angleStep * i - Math.PI / 2;
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;

      // Axis line
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(x, y);
      ctx.strokeStyle = "#d1d5db";
      ctx.stroke();

      // Label
      const labelX = centerX + Math.cos(angle) * (radius + 25);
      const labelY = centerY + Math.sin(angle) * (radius + 25);
      ctx.fillText(dim.dimension_name || dim.name, labelX, labelY + 4);
    });

    // Draw data polygon
    ctx.beginPath();
    dimensions.forEach((dim, i) => {
      const angle = angleStep * i - Math.PI / 2;
      const value = (dim.score / (dim.max_score || 100)) * radius;
      const x = centerX + Math.cos(angle) * value;
      const y = centerY + Math.sin(angle) * value;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.closePath();

    // Fill
    ctx.fillStyle = "rgba(99, 102, 241, 0.3)";
    ctx.fill();

    // Stroke
    ctx.strokeStyle = "#6366f1";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw data points
    dimensions.forEach((dim, i) => {
      const angle = angleStep * i - Math.PI / 2;
      const value = (dim.score / (dim.max_score || 100)) * radius;
      const x = centerX + Math.cos(angle) * value;
      const y = centerY + Math.sin(angle) * value;

      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = "#6366f1";
      ctx.fill();
    });
  }, [data, size]);

  if (!data?.dimensions?.length) {
    return (
      <div className={`flex items-center justify-center h-[${size}px] text-gray-400 ${className}`}>
        No data available
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className={className}
    >
      <canvas ref={canvasRef} width={size} height={size} />
    </motion.div>
  );
}

// =============================================================================
// HEATMAP
// =============================================================================

const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const HOURS = Array.from({ length: 24 }, (_, i) => i);

/**
 * Submission Heatmap Component
 * Shows submission frequency by hour and day.
 */
export function SubmissionHeatmap({ data, className = "" }) {
  if (!data?.heatmap?.length) {
    return (
      <div className={`flex items-center justify-center h-48 text-gray-400 ${className}`}>
        No submission data available
      </div>
    );
  }

  // Find max count for color scaling
  const maxCount = Math.max(...data.heatmap.map((cell) => cell.count), 1);

  // Create a lookup map
  const cellMap = new Map();
  data.heatmap.forEach((cell) => {
    const key = `${cell.day}-${cell.hour}`;
    cellMap.set(key, cell.count);
  });

  const getColor = (count) => {
    if (count === 0) return "bg-gray-100 dark:bg-gray-800";
    const intensity = Math.ceil((count / maxCount) * 4);
    const colors = [
      "bg-indigo-100 dark:bg-indigo-900/30",
      "bg-indigo-200 dark:bg-indigo-800/50",
      "bg-indigo-400 dark:bg-indigo-600/70",
      "bg-indigo-600 dark:bg-indigo-500",
    ];
    return colors[Math.min(intensity - 1, 3)];
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className={`overflow-x-auto ${className}`}
    >
      <div className="min-w-[600px]">
        {/* Hour labels */}
        <div className="flex ml-12 mb-1">
          {HOURS.filter((h) => h % 3 === 0).map((hour) => (
            <div
              key={hour}
              className="text-xs text-gray-500"
              style={{ width: "36px", marginLeft: hour === 0 ? 0 : "72px" }}
            >
              {hour}:00
            </div>
          ))}
        </div>

        {/* Grid */}
        {DAYS.map((day, dayIndex) => (
          <div key={day} className="flex items-center">
            <div className="w-10 text-xs text-gray-500 text-right pr-2">{day}</div>
            <div className="flex gap-0.5">
              {HOURS.map((hour) => {
                const count = cellMap.get(`${dayIndex + 1}-${hour}`) || 0;
                return (
                  <div
                    key={`${day}-${hour}`}
                    className={`w-5 h-5 rounded-sm ${getColor(count)} transition-colors`}
                    title={`${day} ${hour}:00 - ${count} submissions`}
                  />
                );
              })}
            </div>
          </div>
        ))}

        {/* Legend */}
        <div className="flex items-center justify-end mt-3 gap-2">
          <span className="text-xs text-gray-500">Less</span>
          <div className="flex gap-0.5">
            <div className="w-4 h-4 rounded-sm bg-gray-100 dark:bg-gray-800" />
            <div className="w-4 h-4 rounded-sm bg-indigo-100 dark:bg-indigo-900/30" />
            <div className="w-4 h-4 rounded-sm bg-indigo-200 dark:bg-indigo-800/50" />
            <div className="w-4 h-4 rounded-sm bg-indigo-400 dark:bg-indigo-600/70" />
            <div className="w-4 h-4 rounded-sm bg-indigo-600 dark:bg-indigo-500" />
          </div>
          <span className="text-xs text-gray-500">More</span>
        </div>
      </div>
    </motion.div>
  );
}

// =============================================================================
// TREND LINE CHART
// =============================================================================

/**
 * Trend Line Chart Component
 * Shows historical performance trends.
 */
export function TrendChart({ data, height = 200, className = "" }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !data?.events?.length) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const padding = 40;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    // Clear
    ctx.clearRect(0, 0, width, height);

    const events = data.events;
    const scores = events.map((e) => e.avg_score || e.score || 0);
    const maxScore = Math.max(...scores, 100);
    const minScore = Math.min(...scores, 0);
    const range = maxScore - minScore || 1;

    // Draw grid lines
    ctx.strokeStyle = "#e5e7eb";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padding + (chartHeight * i) / 4;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();

      // Y-axis labels
      const value = maxScore - (range * i) / 4;
      ctx.fillStyle = "#9ca3af";
      ctx.font = "11px Inter, sans-serif";
      ctx.textAlign = "right";
      ctx.fillText(value.toFixed(0), padding - 8, y + 4);
    }

    // Draw trend line
    ctx.beginPath();
    ctx.strokeStyle = "#6366f1";
    ctx.lineWidth = 2;

    events.forEach((event, i) => {
      const x = padding + (chartWidth * i) / (events.length - 1 || 1);
      const score = event.avg_score || event.score || 0;
      const y = padding + chartHeight - ((score - minScore) / range) * chartHeight;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Draw points and labels
    events.forEach((event, i) => {
      const x = padding + (chartWidth * i) / (events.length - 1 || 1);
      const score = event.avg_score || event.score || 0;
      const y = padding + chartHeight - ((score - minScore) / range) * chartHeight;

      // Point
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fillStyle = "#6366f1";
      ctx.fill();

      // X-axis label
      ctx.fillStyle = "#9ca3af";
      ctx.font = "10px Inter, sans-serif";
      ctx.textAlign = "center";
      const label = event.event_name || `Event ${i + 1}`;
      ctx.fillText(label.substring(0, 10), x, height - 10);
    });
  }, [data, height]);

  if (!data?.events?.length) {
    return (
      <div className={`flex items-center justify-center h-[${height}px] text-gray-400 ${className}`}>
        No trend data available
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className={className}
    >
      <canvas ref={canvasRef} width={600} height={height} className="w-full" />
    </motion.div>
  );
}

// =============================================================================
// BAR CHART (for theme comparison)
// =============================================================================

/**
 * Horizontal Bar Chart Component
 * Shows theme-wise performance comparison.
 */
export function ThemeBarChart({ data, className = "" }) {
  if (!data?.themes?.length) {
    return (
      <div className={`flex items-center justify-center h-48 text-gray-400 ${className}`}>
        No theme data available
      </div>
    );
  }

  const maxScore = Math.max(...data.themes.map((t) => t.avg_score), 100);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className={`space-y-3 ${className}`}
    >
      {data.themes.map((theme, i) => (
        <motion.div
          key={theme.theme_name}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.1 }}
          className="space-y-1"
        >
          <div className="flex justify-between text-sm">
            <span className="font-medium text-gray-700 dark:text-gray-300">
              {theme.theme_name}
            </span>
            <span className="text-gray-500">
              {theme.avg_score.toFixed(1)} ({theme.submission_count} teams)
            </span>
          </div>
          <div className="h-6 bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${(theme.avg_score / maxScore) * 100}%` }}
              transition={{ duration: 0.5, delay: i * 0.1 }}
              className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full"
            />
          </div>
        </motion.div>
      ))}
    </motion.div>
  );
}

// =============================================================================
// SCORE DISTRIBUTION
// =============================================================================

/**
 * Score Distribution Chart
 * Shows histogram of score ranges.
 */
export function ScoreDistribution({ data, className = "" }) {
  if (!data?.score_distribution) {
    return (
      <div className={`flex items-center justify-center h-48 text-gray-400 ${className}`}>
        No distribution data available
      </div>
    );
  }

  const distribution = data.score_distribution;
  const ranges = ["0-20", "21-40", "41-60", "61-80", "81-100"];
  const maxCount = Math.max(...Object.values(distribution), 1);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className={`flex items-end justify-around h-48 ${className}`}
    >
      {ranges.map((range, i) => {
        const count = distribution[range] || 0;
        const heightPercent = (count / maxCount) * 100;

        return (
          <div key={range} className="flex flex-col items-center gap-2">
            <span className="text-xs text-gray-500">{count}</span>
            <motion.div
              initial={{ height: 0 }}
              animate={{ height: `${Math.max(heightPercent, 5)}%` }}
              transition={{ duration: 0.5, delay: i * 0.1 }}
              className="w-12 bg-gradient-to-t from-indigo-600 to-indigo-400 rounded-t-lg"
              style={{ minHeight: "8px" }}
            />
            <span className="text-xs text-gray-600 dark:text-gray-400">{range}</span>
          </div>
        );
      })}
    </motion.div>
  );
}

// =============================================================================
// PERCENTILE GAUGE
// =============================================================================

/**
 * Percentile Gauge Component
 * Shows team's percentile ranking.
 */
export function PercentileGauge({ percentile, rank, totalTeams, className = "" }) {
  const circumference = 2 * Math.PI * 45;
  const strokeDashoffset = circumference - (percentile / 100) * circumference;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`flex flex-col items-center ${className}`}
    >
      <div className="relative w-32 h-32">
        <svg className="w-full h-full transform -rotate-90">
          {/* Background circle */}
          <circle
            cx="64"
            cy="64"
            r="45"
            stroke="#e5e7eb"
            strokeWidth="10"
            fill="none"
          />
          {/* Progress circle */}
          <motion.circle
            cx="64"
            cy="64"
            r="45"
            stroke="url(#percentileGradient)"
            strokeWidth="10"
            fill="none"
            strokeLinecap="round"
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset }}
            transition={{ duration: 1, ease: "easeOut" }}
            style={{ strokeDasharray: circumference }}
          />
          <defs>
            <linearGradient id="percentileGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#6366f1" />
              <stop offset="100%" stopColor="#a855f7" />
            </linearGradient>
          </defs>
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-3xl font-bold text-indigo-600">{percentile?.toFixed(0)}%</span>
          <span className="text-xs text-gray-500">percentile</span>
        </div>
      </div>
      {rank && totalTeams && (
        <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
          Rank <span className="font-semibold">{rank}</span> of {totalTeams} teams
        </p>
      )}
    </motion.div>
  );
}

export default {
  RadarChart,
  SubmissionHeatmap,
  TrendChart,
  ThemeBarChart,
  ScoreDistribution,
  PercentileGauge,
};
