export default function EventLeaderboard({
  responses,
  loadingResponses,
  sortedLeaderboard,
  activeRoundId,
  setActiveRoundId,
}) {
  if (loadingResponses) return <div>Loading leaderboard...</div>;
  if (!responses) return <div>No data</div>;

  return (
    <div className="space-y-4">

      {/* Round Tabs */}
      <div className="flex gap-2">
        {responses.rounds.map((round) => (
          <button
            key={round.id}
            onClick={() => setActiveRoundId(round.id)}
            className={`px-4 py-1.5 rounded-full text-xs border ${
              activeRoundId === round.id
                ? "bg-neutral-900 text-white border-neutral-900"
                : "bg-white border-neutral-300 text-neutral-700"
            }`}
          >
            {round.id}
          </button>
        ))}
      </div>

      {/* Leaderboard */}
      <div className="border rounded-xl overflow-hidden">
        <div className="grid grid-cols-3 bg-neutral-50 px-4 py-2 text-xs font-medium border-b">
          <span>#</span>
          <span>Team</span>
          <span>Score</span>
        </div>

        {sortedLeaderboard.map((s, i) => (
          <div
            key={s.teamId}
            className="grid grid-cols-3 px-4 py-2 text-xs border-b"
          >
            <span className="font-semibold">{i + 1}</span>
            <span>{s.teamName}</span>
            <span className="font-semibold">{s.score ?? 0}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
