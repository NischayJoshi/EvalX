export default function EventInfo({ event, onEdit, onCreateTeam, onJoinTeam, teamRefresh = 0 }) {
  const [myTeam, setMyTeam] = useState(null);
  const [loadingTeam, setLoadingTeam] = useState(true);

  useEffect(() => { loadTeam(); }, [event && (event.id || event._id), teamRefresh]);

  const loadTeam = async () => {
    if (!event) return;
    setLoadingTeam(true);
    try {
      const eid = event.id || event._id;
      const res = await api.get(`/team/events/${eid}/my-team`);
      setMyTeam(res.data.data);
    } catch {
      setMyTeam(null);
    } finally {
      setLoadingTeam(false);
    }
  };

  const roundsCount = event.rounds?.length || 0;

  return (
    <div className="space-y-6">
      {event.banner && (
        <div className="relative rounded-2xl overflow-hidden shadow-lg border border-neutral-200">
          <img src={event.banner} className="w-full h-56 object-cover" />
          <div className="absolute inset-0 bg-gradient-to-t from-black/40 to-transparent" />
          {event.logo && (
            <div className="absolute bottom-4 left-4 w-16 h-16 rounded-xl overflow-hidden border-2 border-white shadow-xl">
              <img src={event.logo} className="w-full h-full object-cover" />
            </div>
          )}
        </div>
      )}

      <div className="flex justify-between items-start">
        <div>
          <h2 className="text-xl font-semibold text-neutral-900">{event.name}</h2>
          <p className="text-sm text-neutral-600 mt-1">{event.summary}</p>
        </div>

        <div className="flex items-center gap-3">
          {loadingTeam ? (
            <div className="px-4 py-2 bg-neutral-200 rounded-xl text-sm animate-pulse">Checking…</div>
          ) : myTeam ? (
            <div className="px-4 py-2 bg-green-600 text-white rounded-xl text-sm shadow">Registered</div>
          ) : (
            <div className="flex gap-2">
              <button onClick={onCreateTeam} className="px-4 py-2 rounded-xl bg-black text-white text-sm hover:bg-neutral-800">Create Team</button>
              <button onClick={onJoinTeam} className="px-4 py-2 rounded-xl border border-neutral-300 bg-white text-sm hover:border-neutral-900">Join Team</button>
            </div>
          )}

          {/* <button onClick={onEdit} className="flex items-center gap-2 px-4 py-2 rounded-xl border border-neutral-300 bg-white text-sm hover:border-neutral-900"><FiEdit3 size={16} />Edit</button> */}
        </div>
      </div>

      <div className="bg-white border rounded-2xl p-6 shadow-sm">
        <h3 className="font-semibold text-sm">Description</h3>
        <p className="text-sm text-neutral-600 mt-2 leading-relaxed">{event.description}</p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Prize" value={`₹${event.prize}`} icon={<LuTrophy size={20} className="text-yellow-600" />} />
        <StatCard label="Rounds" value={roundsCount} icon={<FiLayers size={20} className="text-blue-600" />} />
        <StatCard label="Max Teams" value={event.maxTeams} icon={<FiUsers size={20} className="text-green-600" />} />
        <StatCard label="Team Size" value={`${event.minMembers} - ${event.maxMembers}`} icon={<PiUsersThree size={20} className="text-purple-600" />} />
      </div>

      {myTeam && (
        <div className="bg-white border rounded-2xl p-6 shadow-sm">
          <h3 className="font-semibold text-neutral-900 text-sm mb-3">Your Team</h3>
          <p className="text-sm"><b>Name:</b> {myTeam.teamName || myTeam.name}</p>
          <p className="text-sm mt-2 font-semibold">Members:</p>
          <ul className="text-sm text-neutral-700 mt-1 list-disc ml-5">
            {myTeam.members.map((m, idx) => (
              <li key={idx}>{m.firstName} {m.lastName} ({m.email})</li>
            ))}
          </ul>
          <p className="text-xs text-neutral-500 mt-3">Team ID: {myTeam.teamId || myTeam._id}</p>
        </div>
      )}
    </div>
  );
}

function StatCard({ label, value, icon }) {
  return (
    <div className="bg-white rounded-xl border border-neutral-200 p-4 shadow-sm">
      <div className="flex justify-between items-center">
        <p className="text-xs text-neutral-500">{label}</p>
        {icon}
      </div>
      <p className="mt-1 text-lg font-semibold">{value}</p>
    </div>
  );
}