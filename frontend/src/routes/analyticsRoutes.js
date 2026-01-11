/**
 * Analytics Routes Configuration
 * ==============================
 * 
 * Route configuration for analytics dashboard pages.
 * Import these routes in your main router to enable analytics features.
 * 
 * Usage in App.jsx or router config:
 * 
 * import { analyticsRoutes } from './routes/analyticsRoutes';
 * 
 * // Add to your routes:
 * ...analyticsRoutes
 */

import { lazy } from "react";

// Lazy load analytics components for code splitting
const OrganizerAnalytics = lazy(() => 
  import("@/Pages/Organizer/Analytics/OrganizerAnalytics")
);
const ParticipantAnalytics = lazy(() => 
  import("@/Pages/Developer/Analytics/ParticipantAnalytics")
);

/**
 * Analytics routes configuration
 * Add these to your React Router configuration
 */
export const analyticsRoutes = [
  {
    path: "/org/analytics/:eventId",
    element: OrganizerAnalytics,
    name: "Organizer Analytics",
    description: "Analytics dashboard for organizers",
    requiresAuth: true,
    requiredRole: "Organizer",
  },
  {
    path: "/dev/analytics/:eventId",
    element: ParticipantAnalytics,
    name: "Participant Analytics",
    description: "Analytics dashboard for participants",
    requiresAuth: true,
    requiredRole: "Developer",
  },
];

/**
 * Helper to get route by name
 */
export const getAnalyticsRoute = (name) => {
  return analyticsRoutes.find((r) => r.name === name);
};

/**
 * Build analytics URL for navigation
 */
export const buildAnalyticsUrl = {
  organizer: (eventId) => `/org/analytics/${eventId}`,
  participant: (eventId) => `/dev/analytics/${eventId}`,
};

export default analyticsRoutes;
