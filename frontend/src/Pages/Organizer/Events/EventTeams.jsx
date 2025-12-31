import React, { useEffect, useState, useMemo, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  FiUsers,
  FiChevronDown,
  FiChevronUp,
  FiFolder,
  FiSearch,
  FiTrash2,
} from "react-icons/fi";
import api from "@/utils/api";

export default function EventTeams({ eventId: propEventId, event }) {
  const eventId = propEventId || event?._id;
  console.log(event._id)

  const [teams, setTeams] = useState([]);
  const [rounds, setRounds] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [openTeam, setOpenTeam] = useState(null);
  const [query, setQuery] = useState("");

  const [confirmOpen, setConfirmOpen] = useState(false);
  const [teamToDelete, setTeamToDelete] = useState(null);
  const [deleting, setDeleting] = useState(false);

  // ---------------- FETCH TEAMS (AXIOS VERSION) ----------------
  const fetchTeams = useCallback(async () => {
    if (!eventId) return;

    setLoading(true);
    setError(null);

    try {
      const res = await api.get(`/org/get-teams/${eventId}`);

      if (!res.data?.success)
        throw new Error(res.data.message || "Failed to fetch teams");

      setTeams(res.data.data || []);
    } catch (err) {
      setError(
        err.response?.data?.message || err.message || "Unknown error occurred"
      );
    } finally {
      setLoading(false);
    }
  }, [eventId]);

  useEffect(() => {
    fetchTeams();
  }, [fetchTeams]);

  // ---------------- SUBMISSION COUNT MAP ----------------
  const submissionCountMap = useMemo(() => {
    const map = {};

    rounds.forEach((r) => {
      (r.submissions || []).forEach((s) => {
        map[s.teamId] = (map[s.teamId] || 0) + 1;
      });
    });

    teams.forEach((t) => {
      if (typeof t.submissionCount === "number")
        map[t.teamId] = t.submissionCount;
    });

    return map;
  }, [rounds, teams]);

  // ---------------- SEARCH FILTER ----------------
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return teams;

    return teams.filter((t) => {
      const leader = t.leader?.name || "";
      const email = t.leader?.email || "";

      const memberMatch = (t.members || []).some((m) =>
        `${m.name} ${m.email}`.toLowerCase().includes(q)
      );

      return (
        t.teamName.toLowerCase().includes(q) ||
        leader.toLowerCase().includes(q) ||
        email.toLowerCase().includes(q) ||
        memberMatch
      );
    });
  }, [teams, query]);

  // ---------------- DELETE TEAM ----------------
  function openConfirm(team) {
    setTeamToDelete(team);
    setConfirmOpen(true);
  }

  async function doDeleteTeam(teamId) {
    if (!teamId) return;

    setDeleting(true);
    setError(null);

    const prev = teams;
    setTeams((x) => x.filter((t) => t.teamId !== teamId));

    try {
      const res = await api.delete(`/org/delete-team/${teamId}`);

      if (!res.data?.success) {
        setTeams(prev);
        throw new Error(res.data.message || "Delete failed");
      }

      setConfirmOpen(false);
      setTeamToDelete(null);
    } catch (err) {
      setTeams(prev);
      setError(
        err.response?.data?.message || err.message || "Error deleting team"
      );
    } finally {
      setDeleting(false);
    }
  }

  // Esc key
  useEffect(() => {
    function onKey(e) {
      if (e.key === "Escape") {
        setConfirmOpen(false);
        setTeamToDelete(null);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // ----------------------------------------------------------
  // -------------------------- UI ----------------------------
  // ----------------------------------------------------------

  return (
    <div className="space-y-4">
      {/* Search + Refresh */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-3 w-full md:max-w-md">
          <div className="relative flex items-center w-full">
            <FiSearch className="absolute left-3 text-neutral-400" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search teams or members..."
              className="pl-10 pr-3 py-2 w-full rounded-lg border border-neutral-200 bg-white text-sm placeholder:text-neutral-400 focus:outline-none focus:ring-2 focus:ring-indigo-200"
            />
          </div>

          <div className="text-xs text-neutral-500 px-2 py-1 rounded-md bg-neutral-50">
            {teams.length} teams
          </div>
        </div>

        <button
          className="text-sm px-3 py-2 rounded-lg border bg-white hover:shadow-sm"
          onClick={fetchTeams}
        >
          Refresh
        </button>
      </div>

      {/* States */}
      {loading ? (
        <div className="text-sm text-neutral-400 animate-pulse">
          Loading teams…
        </div>
      ) : error ? (
        <div className="p-4 bg-red-50 border border-red-100 rounded-md text-sm text-red-700">
          <div className="flex justify-between">
            <span>{error}</span>
            <button className="underline" onClick={fetchTeams}>
              Retry
            </button>
          </div>
        </div>
      ) : filtered.length === 0 ? (
        <div className="w-full flex flex-col items-center py-16 text-neutral-500">
          <FiUsers size={40} className="opacity-40" />
          <p className="mt-3 text-sm">No teams found.</p>
        </div>
      ) : null}

      {/* Teams */}
      <div className="space-y-4">
        <AnimatePresence>
          {filtered.map((team) => {
            const isOpen = openTeam === team.teamId;

            return (
              <motion.div
                key={team.teamId}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 6 }}
                transition={{ duration: 0.2 }}
                className="bg-white rounded-xl border border-neutral-200 p-5 shadow-sm hover:shadow-md transition"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h3 className="font-semibold text-neutral-900 text-base truncate">
                      {team.teamName}
                    </h3>

                    <div className="mt-1 text-xs text-neutral-600">
                      <div className="flex items-center gap-3">
                        <span className="flex items-center gap-1">
                          <FiUsers size={14} />
                          {team.members?.length || 0} members
                        </span>

                        <span className="flex items-center gap-1">
                          <FiFolder size={14} />
                          {submissionCountMap[team.teamId] || 0} submissions
                        </span>
                      </div>

                      <p className="mt-2 text-sm text-neutral-700">
                        Leader:{" "}
                        <span className="font-medium">
                          {team.leader?.name || "—"}
                        </span>
                        <span className="ml-2 text-neutral-500">
                          {team.leader?.email || ""}
                        </span>
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    {/* Expand */}
                    <button
                      className="p-2 rounded-lg hover:bg-neutral-100"
                      onClick={() =>
                        setOpenTeam((p) =>
                          p === team.teamId ? null : team.teamId
                        )
                      }
                    >
                      {isOpen ? <FiChevronUp /> : <FiChevronDown />}
                    </button>

                    {/* Delete */}
                    <button
                      className="px-3 py-1.5 text-sm bg-red-50 text-red-600 border border-red-200 rounded-lg hover:bg-red-100 flex items-center gap-2"
                      onClick={() => openConfirm(team)}
                    >
                      <FiTrash2 size={14} />
                      Delete
                    </button>
                  </div>
                </div>

                {/* Members expand */}
                <AnimatePresence>
                  {isOpen && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                      className="mt-4 border-t border-neutral-200 pt-3"
                    >
                      <p className="text-xs text-neutral-500 font-medium mb-2">
                        Team Members
                      </p>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                        {team.members?.map((m) => (
                          <div
                            key={m.userId}
                            className="px-3 py-2 bg-neutral-50 rounded-lg border text-sm"
                          >
                            <div className="font-medium">{m.name}</div>
                            <div className="text-xs text-neutral-500">
                              {m.email}
                            </div>
                          </div>
                        ))}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>

      {/* Delete Modal */}
      <AnimatePresence>
        {confirmOpen && teamToDelete && (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div
              className="absolute inset-0 bg-black/40"
              onClick={() => !deleting && setConfirmOpen(false)}
            />

            <motion.div
              initial={{ y: 12, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              exit={{ y: 12, opacity: 0 }}
              className="bg-white rounded-2xl p-6 max-w-md w-full shadow-xl border"
            >
              <h4 className="text-lg font-semibold">Confirm Delete</h4>
              <p className="mt-2 text-sm text-neutral-600">
                This action cannot be undone. Delete this team?
              </p>

              <div className="mt-4 bg-neutral-50 border rounded-lg p-4">
                <p className="text-sm">
                  Team:{" "}
                  <span className="font-medium">
                    {teamToDelete.teamName}
                  </span>
                </p>
                <p className="text-sm mt-1">
                  Leader:{" "}
                  <span className="font-medium">
                    {teamToDelete.leader?.name}
                  </span>
                </p>
                <p className="text-xs mt-1 text-neutral-600">
                  ID: {teamToDelete.teamId}
                </p>
              </div>

              {error && (
                <div className="text-sm text-red-600 mt-3">{error}</div>
              )}

              <div className="mt-6 flex justify-end gap-3">
                <button
                  className="px-4 py-2 text-sm rounded-lg border"
                  onClick={() =>
                    !deleting && (setConfirmOpen(false), setTeamToDelete(null))
                  }
                >
                  Cancel
                </button>

                <button
                  className="px-4 py-2 text-sm rounded-lg bg-red-600 text-white disabled:opacity-60"
                  disabled={deleting}
                  onClick={() => doDeleteTeam(teamToDelete.teamId)}
                >
                  {deleting ? "Deleting..." : "Delete"}
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
