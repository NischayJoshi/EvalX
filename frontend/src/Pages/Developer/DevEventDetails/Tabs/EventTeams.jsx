import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FiChevronDown, FiChevronUp, FiTrash2, FiUserMinus } from "react-icons/fi";
import api from "@/utils/api";

export default function EventTeams({ event, onCreateTeam, onJoinTeam }) {
  const [team, setTeam] = useState(null);
  const [loading, setLoading] = useState(true);
  const [openRequests, setOpenRequests] = useState(false);

  const fetchTeam = async () => {
    setLoading(true);
    try {
      const res = await api.get(`/team/events/${event._id || event.id}/my-team`);
      setTeam(res.data.data);
    } catch {
      setTeam(null);
    } finally {
      setLoading(false);
    }
  };

  const removeMember = async (userId) => {
    await api.post(
      `/team/events/${event._id || event.id}/teams/${team._id}/members/remove`,
      new URLSearchParams({ userId })
    );
    fetchTeam();
  };

  const acceptRequest = async (requestId) => {
    await api.post(
      `/team/events/${event._id || event.id}/teams/${team._id}/requests/${requestId}/accept`
    );
    fetchTeam();
  };

  const rejectRequest = async (requestId) => {
    await api.post(
      `/team/events/${event._id || event.id}/teams/${team._id}/requests/${requestId}/reject`
    );
    fetchTeam();
  };

  const deleteTeam = async () => {
    await api.delete(`/team/events/${event._id || event.id}/teams/${team._id}`);
    fetchTeam();
  };

  useEffect(() => {
    if (event) fetchTeam();
  }, [event]);

  if (loading)
    return (
      <div className="p-4 text-sm rounded-xl bg-gray-200 animate-pulse text-gray-600 shadow">
        Checking team status…
      </div>
    );

  if (!team)
    return (
      <motion.div
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex gap-3"
      >
        <button
          onClick={onCreateTeam}
          className="px-5 py-2.5 rounded-xl bg-black text-white text-sm font-medium shadow hover:scale-[1.02] hover:bg-neutral-900 transition-all"
        >
          Create Team
        </button>
        <button
          onClick={onJoinTeam}
          className="px-5 py-2.5 rounded-xl border border-neutral-400 bg-white text-sm hover:border-neutral-600 hover:bg-neutral-50 transition-all"
        >
          Join Team
        </button>
      </motion.div>
    );

  const isLeader = team.leaderId === team.members[0]?.userId;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="bg-white p-6 border border-neutral-300 rounded-2xl shadow-xl space-y-6 max-w-2xl backdrop-blur-sm"
    >
      {/* HEADER */}
      <div className="flex justify-between items-start">
        <div>
          <h3 className="text-neutral-900 font-semibold text-lg">{team.teamName}</h3>
          <p className="text-xs text-neutral-500 mt-1">Team ID: {team._id}</p>
        </div>

        {isLeader && (
          <button
            onClick={deleteTeam}
            className="flex items-center gap-1 px-3 py-1.5 text-xs bg-neutral-900 text-white rounded-lg hover:bg-neutral-800 hover:scale-[1.04] transition-all shadow"
          >
            <FiTrash2 size={14} /> Delete
          </button>
        )}
      </div>

      {/* MEMBERS */}
      <div>
        <p className="text-sm font-semibold text-neutral-800">Members</p>
        <div className="mt-3 space-y-2">
          {team.members.map((m) => (
            <motion.div
              key={m.userId}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex justify-between items-center p-3 bg-neutral-50 border border-neutral-200 rounded-xl hover:bg-neutral-100 hover:shadow transition-all"
            >
              <span className="text-sm text-neutral-800">
                {m.firstName} {m.lastName}{" "}
                <span className="text-neutral-500 text-xs">({m.email})</span>
              </span>

              {isLeader && m.userId !== team.leaderId && (
                <button
                  onClick={() => removeMember(m.userId)}
                  className="text-neutral-900 hover:text-black flex items-center gap-1 text-xs font-medium hover:scale-[1.05] transition-all"
                >
                  <FiUserMinus size={14} /> Remove
                </button>
              )}
            </motion.div>
          ))}
        </div>
      </div>

      {/* REQUESTS */}
      {isLeader && (
        <div className="pt-4 border-t border-neutral-300">
          <button
            onClick={() => setOpenRequests(!openRequests)}
            className="flex items-center justify-between w-full text-sm font-semibold text-neutral-900"
          >
            Join Requests
            {openRequests ? (
              <FiChevronUp className="text-neutral-500" />
            ) : (
              <FiChevronDown className="text-neutral-500" />
            )}
          </button>

          <AnimatePresence>
            {openRequests && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.25 }}
                className="mt-3 space-y-3"
              >
                {team.requests.length === 0 && (
                  <p className="text-xs text-neutral-500">No pending requests</p>
                )}

                {team.requests.map((r) => (
                  <motion.div
                    key={r.requestId}
                    initial={{ opacity: 0, y: 3 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex justify-between items-center p-3 bg-neutral-50 border border-neutral-200 rounded-xl hover:bg-neutral-100 hover:shadow transition-all"
                  >
                    <div>
                      <p className="text-sm text-neutral-900">
                        {r.firstName} {r.lastName}
                      </p>
                      <p className="text-xs text-neutral-500">{r.email}</p>
                    </div>

                    <div className="flex gap-2">
                      <button
                        onClick={() => acceptRequest(r.requestId)}
                        className="px-3 py-1.5 text-xs rounded-lg bg-neutral-900 text-white hover:bg-black hover:scale-[1.05] transition-all shadow"
                      >
                        Accept
                      </button>
                      <button
                        onClick={() => rejectRequest(r.requestId)}
                        className="px-3 py-1.5 text-xs rounded-lg bg-neutral-700 text-white hover:bg-neutral-800 hover:scale-[1.05] transition-all shadow"
                      >
                        Reject
                      </button>
                    </div>
                  </motion.div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </motion.div>
  );
}
