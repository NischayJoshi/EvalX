import { useEffect, useState, useMemo } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { FiArrowLeft } from "react-icons/fi";

import EventInfo from "./Events/EventInfo.jsx";
import EventTeams from "./Events/EventTeams.jsx";
import EventSubmissions from "./Events/EventSubmissions.jsx";
import EventLeaderboard from "./Events/EventLeaderboard.jsx";

import api from "@/utils/api";

export default function EventDetails() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [loadingEvent, setLoadingEvent] = useState(true);
  const [event, setEvent] = useState(null);

  const [loadingResponses, setLoadingResponses] = useState(true);
  const [responses, setResponses] = useState(null);

  const [activeTab, setActiveTab] = useState("info");
  const [activeRoundId, setActiveRoundId] = useState(null);
  const [savingScore, setSavingScore] = useState(false);

  const [error, setError] = useState(null);

  // Load event
  useEffect(() => {
    api
      .get(`/org/events/${id}`)
      .then((res) => setEvent(res.data.data))
      .catch(() => setError("Failed to load event"))
      .finally(() => setLoadingEvent(false));
  }, [id]);

  // Load responses
  useEffect(() => {
    api
      .get(`/org/events/${id}/responses`)
      .then((res) => {
        const data = res.data.data;
        setResponses(data);

        if (data.rounds?.length) setActiveRoundId(data.rounds[0].id);
      })
      .finally(() => setLoadingResponses(false));
  }, [id]);

  const activeRound = useMemo(() => {
    return responses?.rounds?.find((r) => r.id === activeRoundId) || null;
  }, [responses, activeRoundId]);

  const sortedLeaderboard =
    activeRound?.submissions
      ?.slice()
      .sort((a, b) => (b.score || 0) - (a.score || 0)) || [];

  if (loadingEvent)
    return (
      <div className="p-12 w-full flex justify-center text-neutral-400">
        Loading event…
      </div>
    );

  if (error || !event)
    return (
      <div className="p-12 text-red-500 text-center">
        {error || "Event not found"}
      </div>
    );

  return (
    <div className="w-full">

      {/* --- HEADER (FULL WIDTH) --- */}
      <div className="w-full bg-white border-b border-neutral-200 px-6 py-4 flex items-center gap-3 sticky top-0 z-20">
        <button
          onClick={() => navigate(-1)}
          className="p-2 rounded-xl border border-neutral-200 bg-white shadow-sm hover:shadow transition"
        >
          <FiArrowLeft size={18} />
        </button>

        <h1 className="text-lg font-semibold tracking-tight">{event.name}</h1>
      </div>

      {/* --- TABS (FULL WIDTH) --- */}
      <div className="
        w-full sticky top-[64px] z-10 bg-neutral-50 
        border-b border-neutral-200 
        px-6 pt-3 pb-2 backdrop-blur-md
      ">
        <div className="flex gap-6 max-w-6xl mx-auto">
          {[
            { id: "info", label: "Event Info" },
            { id: "teams", label: "Teams" },
            { id: "submissions", label: "Submissions" },
            { id: "leaderboard", label: "Leaderboard" },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`
                pb-2 text-sm font-medium transition-all
                ${
                  activeTab === tab.id
                    ? "border-b-2 border-neutral-900 text-neutral-900"
                    : "text-neutral-500 hover:text-neutral-800"
                }
              `}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* --- CONTENT (CENTERED 80%) --- */}
      <div className="w-full flex justify-center px-4">
        <div className="w-full max-w-6xl py-6">

          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.22 }}
            >
              {activeTab === "info" && (
                <EventInfo event={event} responses={responses} />
              )}

              {activeTab === "teams" && (
                <EventTeams event={event} responses={responses} />
              )}

              {activeTab === "submissions" && (
                <EventSubmissions
                  responses={responses}
                  loadingResponses={loadingResponses}
                  activeRoundId={activeRoundId}
                  setActiveRoundId={setActiveRoundId}
                  handleScoreChangeLocal={(roundId, teamId, score) => {
                    setResponses((prev) => {
                      const updated = prev.rounds.map((r) =>
                        r.id !== roundId
                          ? r
                          : {
                              ...r,
                              submissions: r.submissions.map((s) =>
                                s.teamId === teamId ? { ...s, score } : s
                              ),
                            }
                      );
                      return { ...prev, rounds: updated };
                    });
                  }}
                  handleSaveScore={async (roundId, teamId, score) => {
                    setSavingScore(true);
                    await api.patch(
                      `/api/org/events/${id}/rounds/${roundId}/submissions/${teamId}`,
                      { score }
                    );
                    setSavingScore(false);
                  }}
                />
              )}

              {activeTab === "leaderboard" && (
                <EventLeaderboard
                  responses={responses}
                  loadingResponses={loadingResponses}
                  sortedLeaderboard={sortedLeaderboard}
                  activeRoundId={activeRoundId}
                  setActiveRoundId={setActiveRoundId}
                />
              )}
            </motion.div>
          </AnimatePresence>

        </div>
      </div>
    </div>
  );
}
