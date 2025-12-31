import { useEffect, useState, useCallback } from "react";
import api from "@/utils/api";

export default function EventLeaderboard({
  eventId,
  activeRoundId,
  setActiveRoundId
}) {
  const [loading, setLoading] = useState(true);
  const [sortedLeaderboard, setSortedLeaderboard] = useState([]);

  // Fetch Leaderboard
  const getData = useCallback(async () => {
    try {
      setLoading(true);
      const res = await api.get(`/team/events/${eventId}/leaderboard`);

      if (!res?.data?.data) {
        setSortedLeaderboard([]);
        setLoading(false);
        return;
      }

      const d = res.data.data;

      let selected = [];

      if (activeRoundId === "ppt") selected = d.ppt_leaderboard || [];
      else if (activeRoundId === "repo") selected = d.repo_leaderboard || [];
      else selected = d.overall_leaderboard || [];

      // normalize fields
      const formatted = selected.map((x) => ({
        teamName: x.teamName,
        teamId: x.teamId,
        score: x.score ?? x.totalScore ?? 0,
      }));

      // sort by score desc
      formatted.sort((a, b) => b.score - a.score);

      setSortedLeaderboard(formatted);
      setLoading(false);
    } catch (err) {
      console.log("LEADERBOARD ERROR →", err);
      setSortedLeaderboard([]);
      setLoading(false);
    }
  }, [eventId, activeRoundId]);

  useEffect(() => {
    getData();
  }, [getData]);

  // ---------------------- UI START ----------------------

  if (loading)
    return (
      <div className="text-sm text-neutral-400 animate-pulse">
        Loading leaderboard…
      </div>
    );

  if (!sortedLeaderboard.length)
    return (
      <div className="py-12 text-center text-neutral-500">
        No leaderboard data
      </div>
    );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold tracking-wide text-neutral-900">
          Leaderboard
        </h2>

        {/* Round Switch */}
        <div className="flex items-center gap-2">
          {["ppt", "repo", "overall"].map((r) => (
            <button
              key={r}
              onClick={() => setActiveRoundId(r)}
              className={`
                px-3 py-1.5 rounded-lg text-sm font-medium
                ${
                  activeRoundId === r
                    ? "bg-black text-white"
                    : "bg-neutral-200 text-neutral-700 hover:bg-neutral-300"
                }
              `}
            >
              {r.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Leaderboard Card */}
      <div className="rounded-xl border border-neutral-200 bg-white shadow-sm">
        <ol className="divide-y divide-neutral-100">
          {sortedLeaderboard.map((s, idx) => (
            <li
              key={s.teamId + "_" + idx}
              className="p-4 flex items-center justify-between hover:bg-neutral-50 transition-all"
            >
              {/* Left */}
              <div className="flex items-center gap-4">
                {/* Rank Badge */}
                <div
                  className={`
                    w-10 h-10 flex items-center justify-center rounded-full text-sm font-bold
                    ${
                      idx === 0
                        ? "bg-gradient-to-br from-black to-neutral-700 text-white"
                        : idx === 1
                        ? "bg-neutral-200 text-neutral-700"
                        : idx === 2
                        ? "bg-neutral-300 text-neutral-700"
                        : "bg-neutral-100 text-neutral-600"
                    }
                  `}
                >
                  {idx + 1}
                </div>

                {/* Team */}
                <div>
                  <div className="font-semibold text-neutral-900 text-sm">
                    {s.teamName}
                  </div>
                </div>
              </div>

              {/* Score */}
              <div className="text-right">
                <div className="text-base font-semibold text-neutral-900">
                  {s.score}
                </div>
              </div>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
