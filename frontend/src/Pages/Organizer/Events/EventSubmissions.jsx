import { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FiSearch, FiChevronDown, FiChevronUp } from "react-icons/fi";
import { FaGithub } from "react-icons/fa";

export default function EventSubmissions({
  responses,
  loadingResponses,
  activeRoundId,
  setActiveRoundId,
  handleScoreChangeLocal,
}) {
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState("team"); // team | score
  const [sortDir, setSortDir] = useState("asc"); // asc | desc

  if (loadingResponses) return <div>Loading submissions...</div>;
  if (!responses) return <div>No submissions yet.</div>;

  const activeRound =
    responses.rounds?.find((r) => r.id === activeRoundId) || null;

  // --------------------------
  // FILTER + SEARCH + SORT
  // --------------------------
  const filtered = useMemo(() => {
    if (!activeRound) return [];

    let rows = [...activeRound.submissions];

    // search filter
    if (search.trim()) {
      rows = rows.filter((s) =>
        s.teamName.toLowerCase().includes(search.toLowerCase())
      );
    }

    // sorting
    rows.sort((a, b) => {
      let valA = sortBy === "team" ? a.teamName : a.score ?? 0;
      let valB = sortBy === "team" ? b.teamName : b.score ?? 0;

      if (typeof valA === "string") valA = valA.toLowerCase();
      if (typeof valB === "string") valB = valB.toLowerCase();

      if (sortDir === "asc") return valA > valB ? 1 : -1;
      return valA < valB ? 1 : -1;
    });

    return rows;
  }, [activeRound, search, sortBy, sortDir]);

  const toggleSort = (key) => {
    if (sortBy === key) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortBy(key);
      setSortDir("asc");
    }
  };

  return (
    <div className="space-y-5">

      {/* ------------------------ */}
      {/* ROUND SELECTOR */}
      {/* ------------------------ */}
      <div className="flex gap-2 overflow-x-auto pb-2">
        {responses.rounds.map((round) => (
          <button
            key={round.id}
            onClick={() => setActiveRoundId(round.id)}
            className={`px-4 py-1.5 rounded-xl text-xs font-medium transition-all ${
              activeRoundId === round.id
                ? "bg-neutral-900 text-white shadow"
                : "bg-white border text-neutral-700 border-neutral-300 hover:border-neutral-500"
            }`}
          >
            {round.name || round.id.toUpperCase()}
          </button>
        ))}
      </div>

      {/* ------------------------ */}
      {/* SEARCH + SORT BAR */}
      {/* ------------------------ */}
      <div className="flex items-center justify-between">
        {/* search */}
        <div className="relative flex-1">
          <FiSearch className="absolute left-3 top-2.5 text-neutral-400" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search team…"
            className="w-full pl-10 pr-3 py-2 bg-white border border-neutral-300 rounded-xl text-sm focus:border-neutral-700 transition"
          />
        </div>

        {/* sorting */}
        <div className="flex gap-2 ml-3">
          <button
            onClick={() => toggleSort("team")}
            className={`px-3 py-2 text-xs rounded-lg border ${
              sortBy === "team"
                ? "bg-neutral-900 text-white border-neutral-900"
                : "bg-white border-neutral-300"
            }`}
          >
            Team {sortBy === "team" ? (sortDir === "asc" ? "↑" : "↓") : ""}
          </button>

          <button
            onClick={() => toggleSort("score")}
            className={`px-3 py-2 text-xs rounded-lg border ${
              sortBy === "score"
                ? "bg-neutral-900 text-white border-neutral-900"
                : "bg-white border-neutral-300"
            }`}
          >
            Score {sortBy === "score" ? (sortDir === "asc" ? "↑" : "↓") : ""}
          </button>
        </div>
      </div>

      {/* ------------------------ */}
      {/* TABLE */}
      {/* ------------------------ */}
      {!filtered.length ? (
        <div className="text-neutral-500 text-sm py-10 text-center">
          No submissions for this round.
        </div>
      ) : (
        <div className="border rounded-xl overflow-hidden shadow-sm">
          <div className="grid grid-cols-4 bg-neutral-50 text-xs font-medium px-4 py-2 border-b">
            <span>Team</span>
            <span>Submission</span>
            <span>Links</span>
            <span>Score</span>
          </div>

          <AnimatePresence>
            {filtered.map((s) => (
              <motion.div
                key={s.teamId}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                className="grid grid-cols-4 px-4 py-3 text-xs border-b hover:bg-neutral-50 transition"
              >
                {/* Team */}
                <span className="font-medium text-neutral-800">{s.teamName}</span>

                {/* Submission */}
                <span className="flex items-center">
                  {activeRoundId === "ppt" && (
                    <button
                      className="underline text-neutral-700"
                      onClick={() => window.open(s.submissionUrl, "_blank")}
                    >
                      View File
                    </button>
                  )}

                  {activeRoundId === "repo" && (
                    <span className="text-neutral-700">Repo + Video</span>
                  )}

                  {activeRoundId === "viva" && (
                    <span className="text-neutral-700">Interview</span>
                  )}
                </span>

                {/* LINKS */}
                <div className="flex items-center gap-3">
                  {s.repo && (
                    <a
                      href={s.repo}
                      target="_blank"
                      rel="noreferrer"
                      className="text-neutral-800 hover:text-neutral-900"
                    >
                      <FaGithub size={18} />
                    </a>
                  )}

                  {s.video && (
                    <a
                      href={s.video}
                      target="_blank"
                      rel="noreferrer"
                      className="text-xs underline text-neutral-700 hover:text-neutral-900"
                    >
                      Video
                    </a>
                  )}
                </div>

                {/* Score */}
                <input
                  type="number"
                  className="w-20 border border-neutral-300 rounded-lg px-2 py-1 text-xs"
                  value={s.score ?? ""}
                  onChange={(e) =>
                    handleScoreChangeLocal(
                      activeRound.id,
                      s.teamId,
                      Number(e.target.value)
                    )
                  }
                />
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}
