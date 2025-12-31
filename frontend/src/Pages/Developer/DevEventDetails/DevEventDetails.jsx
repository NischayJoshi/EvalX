import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useParams, useNavigate } from "react-router-dom";

import useEventDetails from "./useEventDetails";

import EventInfo from "./Tabs/EventInfo";
import EventTeams from "./Tabs/EventTeams";
import EventSubmissions from "./Tabs/EventSubmissions";
import EventLeaderboard from "./Tabs/EventLeaderboard";

import CreateTeamModal from "./Modals/CreateTeamModal";
import JoinTeamModal from "./Modals/JoinTeamModal";

export default function DevEventDetails() {
  const { id } = useParams();
  const navigate = useNavigate();

  const {
    event,
    submissions,
    loadingEvent,
    loadingSubs,
    error,
    activeRoundId,
    setActiveRoundId,
    sortedLeaderboard,
    refreshTeams,
    createTeam,
    joinTeam,
    getCurrentUserId,
    getOpenTeams,
    deleteTeam,
    addMember,
    removeMember,
    inviteMember,
    acceptRequest,
    rejectRequest,
  } = useEventDetails(id);

  const [activeTab, setActiveTab] = useState("info");
  const [showCreateTeam, setShowCreateTeam] = useState(false);
  const [showJoinTeam, setShowJoinTeam] = useState(false);

  if (loadingEvent)
    return (
      <div className="p-12 w-full flex justify-center text-neutral-400">Loading event…</div>
    );
  if (error || !event)
    return (
      <div className="p-12 text-red-500 text-center">{error || "Event not found"}</div>
    );

  return (
    <div className="w-full">
      <div className="w-full bg-white border-b border-neutral-200 px-6 py-4 flex items-center gap-3 sticky top-0 z-20">
        <button onClick={() => navigate(-1)} className="p-2 rounded-xl border border-neutral-200 bg-white shadow-sm hover:shadow transition">
          ←
        </button>
        <h1 className="text-lg font-semibold tracking-tight">{event.name}</h1>
      </div>

      <div className="w-full sticky top-[64px] z-10 bg-neutral-50 border-b border-neutral-200 px-6 pt-3 pb-2">
        <div className="flex gap-6 max-w-6xl mx-auto">
          {[{ id: "info", label: "Event Info" }, { id: "teams", label: "Teams" }, { id: "submissions", label: "Submissions" }, { id: "leaderboard", label: "Leaderboard" }].map((t) => (
            <button key={t.id} onClick={() => setActiveTab(t.id)} className={`pb-2 text-sm transition font-medium ${activeTab === t.id ? "border-b-2 border-neutral-900 text-neutral-900" : "text-neutral-500 hover:text-neutral-800"}`}>
              {t.label}
            </button>
          ))}
        </div>
      </div>

      <div className="w-full flex justify-center px-4">
        <div className="w-full max-w-6xl py-6">
          <AnimatePresence mode="wait">
            <motion.div key={activeTab} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.22 }}>
              {activeTab === "info" && (
                <EventInfo event={event} onCreateTeam={() => setShowCreateTeam(true)} onJoinTeam={() => setShowJoinTeam(true)} teamRefreshTrigger={refreshTeams} getOpenTeams={getOpenTeams} />
              )}

              {activeTab === "teams" && (
                <EventTeams
                  event={event}
                  onCreateTeam={() => setShowCreateTeam(true)}
                  onJoinTeam={() => setShowJoinTeam(true)}
                />
              )}

              {activeTab === "submissions" && (
                <EventSubmissions
                  rounds={event?.rounds || []}
                  loading={loadingSubs}
                  activeRoundId={activeRoundId}
                  setActiveRoundId={setActiveRoundId}
                  eventId={event._id || event.id}
                />
              )}

              {activeTab === "leaderboard" && (
                <EventLeaderboard
  eventId={event._id || event.id}        // <-- THIS WAS MISSING
  activeRoundId={activeRoundId}
  setActiveRoundId={setActiveRoundId}
/>
              )}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>

      {showCreateTeam && (
        <CreateTeamModal
          onClose={() => setShowCreateTeam(false)}
          onCreate={async (name) => {
            try {
              await createTeam(name);
              setShowCreateTeam(false);
            } catch (err) {
              alert(err?.response?.data?.detail || "Failed to create team");
            }
          }}
        />
      )}

      {showJoinTeam && (
        <JoinTeamModal
          onClose={() => setShowJoinTeam(false)}
          onJoin={async (teamId) => {
            try {
              await joinTeam(teamId);
              setShowJoinTeam(false);
              alert("Request sent to team leader");
            } catch (err) {
              alert(err?.response?.data?.detail || "Failed to send join request");
            }
          }}
        />
      )}
    </div>
  );
}
