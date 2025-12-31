import { useEffect, useState, useMemo } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { FiArrowLeft, FiEdit3, FiUsers, FiLayers, FiChevronDown, FiChevronUp, FiFolder } from "react-icons/fi";
import { LuTrophy } from "react-icons/lu";
import { PiUsersThree } from "react-icons/pi";
import api from "@/utils/api";

export default function DevEventDetails() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [event, setEvent] = useState(null);
  const [loadingEvent, setLoadingEvent] = useState(true);

  const [submissionsData, setSubmissionsData] = useState(null);
  const [loadingSubs, setLoadingSubs] = useState(true);

  const [showCreateTeam, setShowCreateTeam] = useState(false);
  const [showJoinTeam, setShowJoinTeam] = useState(false);
  const [createTeamName, setCreateTeamName] = useState("");
  const [joinTeamId, setJoinTeamId] = useState("");
  const [teamRefresh, setTeamRefresh] = useState(0);

  const [activeTab, setActiveTab] = useState("info");
  const [activeRoundId, setActiveRoundId] = useState(null);

  const [error, setError] = useState(null);

  const[submitData,setSubmitData]=useState(null);

  console.log("testing")

  const fetchSubmission = async()=>{
    try{
      const res = await api.get(`/team/events/${id}/my-submissions`, { withCredentials: true });
      setSubmitData(res);
      console.log(res)
    } catch (err) {
      console.log("SUBMISSION_FETCH_ERROR", err);
      console.log("erroir aagya")
    }
  }

  useEffect(() => {
    fetchSubmission();
    console.log(submitData)
  }, [id]);

  useEffect(() => {
    api.get(`/team/events/${id}`).then((res) => setEvent(res.data.data)).catch(() => setError("Failed to load event")).finally(() => setLoadingEvent(false));
  }, [id]);

  useEffect(() => {
    api.get(`/team/events/${id}/submissions`).then((res) => {
      const data = res.data.data;
      setSubmissionsData(data);
      if (data.rounds?.length) setActiveRoundId(data.rounds[0].id);
    }).finally(() => setLoadingSubs(false));
  }, [id]);

  useEffect(() => {
    if (!id) return;
    setLoadingSubs(true);
    api.get(`/team/events/${id}/submissions`).then((res) => setSubmissionsData(res.data.data)).finally(() => setLoadingSubs(false));
  }, [id, teamRefresh]);

  const activeRound = useMemo(() => submissionsData?.rounds?.find((r) => r.id === activeRoundId) || null, [submissionsData, activeRoundId]);

  const sortedLeaderboard = activeRound?.submissions?.slice().sort((a, b) => (b.score || 0) - (a.score || 0)) || [];

  const handleCreateTeam = async () => {
    if (!createTeamName.trim()) return;
    try {
      const fd = new FormData();
      fd.append("teamName", createTeamName.trim());
      await api.post(`/team/events/${id}/teams/create`, fd);
      setShowCreateTeam(false);
      setCreateTeamName("");
      setTeamRefresh((t) => t + 1);
    } catch (err) {
      alert(err?.response?.data?.detail || "Failed to create team");
    }
  };

  const handleJoinTeam = async (teamIdParam) => {
    const tid = teamIdParam || joinTeamId;
    if (!tid) return;
    try {
      await api.post(`/team/events/${id}/teams/${tid}/requests/send`);
      setShowJoinTeam(false);
      setJoinTeamId("");
      setTeamRefresh((t) => t + 1);
      alert("Request sent to team leader");
    } catch (err) {
      alert(err?.response?.data?.detail || "Failed to send join request");
    }
  };

  if (loadingEvent) return <div className="p-12 w-full flex justify-center text-neutral-400">Loading event…</div>;
  if (error || !event) return <div className="p-12 text-red-500 text-center">{error || "Event not found"}</div>;

  return (
    <div className="w-full">
      <div className="w-full bg-white border-b border-neutral-200 px-6 py-4 flex items-center gap-3 sticky top-0 z-20">
        <button onClick={() => navigate(-1)} className="p-2 rounded-xl border border-neutral-200 bg-white shadow-sm hover:shadow transition">
          <FiArrowLeft size={18} />
        </button>
        <h1 className="text-lg font-semibold tracking-tight">{event.name}</h1>
      </div>

      <div className="w-full sticky top-[64px] z-10 bg-neutral-50 border-b border-neutral-200 px-6 pt-3 pb-2">
        <div className="flex gap-6 max-w-6xl mx-auto">
          {[{ id: "info", label: "Event Info" },{ id: "teams", label: "Teams" },{ id: "submissions", label: "Submissions" },{ id: "leaderboard", label: "Leaderboard" }].map((t) => (
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
                <EventInfo event={event} onEdit={() => console.log("edit")} onCreateTeam={() => setShowCreateTeam(true)} onJoinTeam={() => setShowJoinTeam(true)} teamRefresh={teamRefresh} />
              )}

              {activeTab === "teams" && (
                <EventTeams event={event} responses={submissionsData} onJoin={handleJoinTeam} onTeamChange={() => setTeamRefresh((t) => t + 1)} />
              )}

              {activeTab === "submissions" && (
                <EventSubmissions rounds={submissionsData?.rounds || []} loading={loadingSubs} activeRoundId={activeRoundId} setActiveRoundId={setActiveRoundId} />
              )}

              {activeTab === "leaderboard" && (
                <EventLeaderboard loading={loadingSubs} sortedLeaderboard={sortedLeaderboard} activeRoundId={activeRoundId} setActiveRoundId={setActiveRoundId} />
              )}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>

      {showCreateTeam && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-40">
          <div className="bg-white rounded-xl p-6 w-full max-w-md">
            <h3 className="text-lg font-semibold mb-3">Create Team</h3>
            <input value={createTeamName} onChange={(e) => setCreateTeamName(e.target.value)} className="w-full rounded-xl border border-neutral-300 px-3 py-2 text-sm shadow-sm mb-4" placeholder="Team name" />
            <div className="flex justify-end gap-3">
              <button onClick={() => setShowCreateTeam(false)} className="px-4 py-2 rounded-xl border">Cancel</button>
              <button onClick={handleCreateTeam} className="px-4 py-2 rounded-xl bg-black text-white">Create</button>
            </div>
          </div>
        </div>
      )}

      {showJoinTeam && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-40">
          <div className="bg-white rounded-xl p-6 w-full max-w-md">
            <h3 className="text-lg font-semibold mb-3">Join Team</h3>
            <input value={joinTeamId} onChange={(e) => setJoinTeamId(e.target.value)} className="w-full rounded-xl border border-neutral-300 px-3 py-2 text-sm shadow-sm mb-4" placeholder="Team ID" />
            <div className="flex justify-end gap-3">
              <button onClick={() => setShowJoinTeam(false)} className="px-4 py-2 rounded-xl border">Cancel</button>
              <button onClick={() => handleJoinTeam()} className="px-4 py-2 rounded-xl bg-black text-white">Join</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function EventInfo({ event, onEdit, onCreateTeam, onJoinTeam, teamRefresh = 0 }) {
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

          <button onClick={onEdit} className="flex items-center gap-2 px-4 py-2 rounded-xl border border-neutral-300 bg-white text-sm hover:border-neutral-900"><FiEdit3 size={16} />Edit</button>
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

function EventTeams({ event, responses, onJoin, onTeamChange }) {
  const [openTeamList, setOpenTeamList] = useState(null);
  const [currentUserId, setCurrentUserId] = useState(null);
  const [addUserMap, setAddUserMap] = useState({});
  const [inviteEmailMap, setInviteEmailMap] = useState({});
  const [openTeam, setOpenTeam] = useState(null);

  const teams = responses?.teams || [];
  const rounds = responses?.rounds || [];

  useEffect(() => { api.get('/team/get-user').then((res) => setCurrentUserId(res.data.data.id)).catch(() => setCurrentUserId(null)); }, []);

  useEffect(() => { if (!event) return; api.get(`/team/events/${event.id || event._id}/teams/open`).then((res) => setOpenTeamList(res.data.data)).catch(() => setOpenTeamList([])); }, [event, responses, onTeamChange]);

  const submissionCount = {};
  (rounds || []).forEach((r) => { (r.submissions || []).forEach((s) => { submissionCount[s.teamId] = (submissionCount[s.teamId] || 0) + 1; }); });

  if (!responses) return <div className="text-sm text-neutral-400 animate-pulse">Loading teams…</div>;
  if (teams.length === 0) return (
    <div className="w-full flex flex-col items-center py-16 text-neutral-500">
      <FiUsers size={40} className="opacity-40" />
      <p className="mt-3 text-sm">No teams registered yet.</p>
    </div>
  );

  return (
    <div className="space-y-4">
      <div className="flex gap-3 items-center">
        <h4 className="text-sm font-semibold">Open teams</h4>
        <p className="text-xs text-neutral-500">Teams with available slots</p>
      </div>

      {(openTeamList || []).map((team) => {
        const id = team._id || team.teamId || team.id;
        const name = team.teamName || team.name || 'Team';
        const membersCount = (team.members || []).length;
        return (
          <div key={id} className="bg-white rounded-xl border border-neutral-200 p-4">
            <div className="flex justify-between items-center">
              <div>
                <div className="font-semibold">{name}</div>
                <div className="text-xs text-neutral-600 mt-1">{membersCount} members • ID: {team.teamId || id}</div>
              </div>
              <div className="flex items-center gap-2">
                <button onClick={() => onJoin && onJoin(id)} className="px-3 py-1 text-xs rounded-lg border border-neutral-300 bg-white hover:bg-neutral-100">Request to join</button>
              </div>
            </div>
          </div>
        );
      })}

      {teams.map((team) => {
        const id = team._id || team.teamId || team.id;
        const name = team.teamName || team.name || 'Team';
        const isLeader = currentUserId && (team.leaderId === currentUserId || (team.members || [])[0]?.userId === currentUserId);
        const isMember = currentUserId && (team.members || []).some((m) => m.userId === currentUserId);
        const hasRequested = (team.requests || []).some((r) => r.userId === currentUserId && r.status === 'pending');
        const isOpen = openTeam === id;

        return (
          <motion.div key={id} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }} className="bg-white rounded-xl border border-neutral-200 p-5 shadow-sm hover:shadow-md transition-all">
            <div className="flex justify-between items-center">
              <div>
                <h3 className="font-semibold text-neutral-900 text-base">{name}</h3>
                <div className="flex items-center gap-4 mt-1 text-xs text-neutral-600">
                  <span className="flex items-center gap-1"><FiUsers size={14} /> {team.members?.length || 0} members</span>
                  <span className="flex items-center gap-1"><FiFolder size={14} /> {submissionCount[id] || 0} submissions</span>
                </div>
              </div>

              <div className="flex items-center gap-2">
                {!isMember && !hasRequested && <button onClick={() => onJoin && onJoin(id)} className="px-3 py-1 text-xs rounded-lg border border-neutral-300 bg-white hover:bg-neutral-100">Request</button>}
                {hasRequested && <div className="px-3 py-1 text-xs rounded-lg bg-yellow-50 text-yellow-800 border border-yellow-100">Requested</div>}
                {isMember && <div className="px-3 py-1 text-xs rounded-lg bg-green-50 text-green-800 border border-green-100">Member</div>}
                {isLeader && <button onClick={async () => { if (!confirm('Delete this team? This cannot be undone.')) return; try { await api.delete(`/team/events/${event.id || event._id}/teams/${id}`); onTeamChange && onTeamChange(); } catch (err) { alert(err?.response?.data?.detail || 'Failed to delete team'); } }} className="px-3 py-1 text-xs rounded-lg border border-red-300 bg-white hover:bg-red-50 text-red-600">Delete</button>}
                <button className="p-2 rounded-lg hover:bg-neutral-100 transition" onClick={() => setOpenTeam((prev) => (prev === id ? null : id))}>{isOpen ? <FiChevronUp size={18} /> : <FiChevronDown size={18} />}</button>
              </div>
            </div>

            <AnimatePresence>
              {isOpen && (
                <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.25 }} className="mt-4 border-t border-neutral-200 pt-3">
                  <p className="text-xs font-medium text-neutral-500 mb-2">Team Members</p>

                  <div className="space-y-2">
                    {(team.members || []).map((m, idx) => {
                      const label = `${m.firstName || ''} ${m.lastName || ''} (${m.email || ''})`;
                      const memberId = m.userId;
                      return (
                        <div key={idx} className="px-3 py-2 bg-neutral-50 rounded-lg border border-neutral-200 text-sm text-neutral-700 flex justify-between items-center gap-3">
                          <div>{label}</div>
                          <div className="flex items-center gap-2">
                            {(isLeader || (currentUserId && currentUserId === memberId)) && (
                              <button onClick={async () => { try { await api.post(`/team/events/${event.id || event._id}/teams/${id}/members/remove`, new URLSearchParams({ userId: memberId })); onTeamChange && onTeamChange(); } catch (err) { alert(err?.response?.data?.detail || 'Failed to remove member'); } }} className="text-xs px-2 py-1 rounded bg-red-50 text-red-600 border border-red-100">{currentUserId === memberId ? 'Leave' : 'Remove'}</button>
                            )}
                          </div>
                        </div>
                      );
                    })}

                    {isLeader && (
                      <>
                        <div className="mt-3 grid grid-cols-1 gap-2">
                          <div className="flex gap-2">
                            <input value={addUserMap[id] || ''} onChange={(e) => setAddUserMap((s) => ({ ...s, [id]: e.target.value }))} placeholder="Add userId" className="flex-1 rounded-xl border border-neutral-300 px-3 py-2 text-sm" />
                            <button onClick={async () => { const uid = addUserMap[id]; if (!uid) return alert('Enter userId to add'); try { await api.post(`/team/events/${event.id || event._id}/teams/${id}/members/add`, new URLSearchParams({ userId: uid })); setAddUserMap((s) => ({ ...s, [id]: '' })); onTeamChange && onTeamChange(); } catch (err) { alert(err?.response?.data?.detail || 'Failed to add member'); } }} className="px-3 py-2 rounded-lg bg-black text-white text-sm">Add</button>
                          </div>

                          <div className="flex gap-2">
                            <input value={inviteEmailMap[id] || ''} onChange={(e) => setInviteEmailMap((s) => ({ ...s, [id]: e.target.value }))} placeholder="Invite by email" className="flex-1 rounded-xl border border-neutral-300 px-3 py-2 text-sm" />
                            <button onClick={async () => { const email = inviteEmailMap[id]; if (!email) return alert('Enter email to invite'); try { await api.post(`/team/events/${event.id || event._id}/teams/${id}/invite`, new URLSearchParams({ email })); setInviteEmailMap((s) => ({ ...s, [id]: '' })); onTeamChange && onTeamChange(); } catch (err) { alert(err?.response?.data?.detail || 'Failed to invite member'); } }} className="px-3 py-2 rounded-lg bg-white border border-neutral-300 text-sm">Invite</button>
                          </div>
                        </div>

                        <div className="mt-4">
                          <p className="text-xs font-medium text-neutral-500 mb-2">Pending Requests</p>
                          {(team.requests || []).filter(r => r.status === 'pending').map((r) => (
                            <div key={r.requestId} className="flex items-center justify-between gap-3 px-3 py-2 bg-neutral-50 rounded-lg border border-neutral-200 mb-2">
                              <div className="text-sm">{r.firstName} {r.lastName} ({r.email})</div>
                              <div className="flex items-center gap-2">
                                <button onClick={async () => { try { await api.post(`/team/events/${event.id || event._id}/teams/${id}/requests/${r.requestId}/accept`); onTeamChange && onTeamChange(); } catch (err) { alert(err?.response?.data?.detail || 'Failed to accept request'); } }} className="px-3 py-1 text-xs rounded-lg bg-green-50 text-green-700 border border-green-100">Accept</button>
                                <button onClick={async () => { try { await api.post(`/team/events/${event.id || event._id}/teams/${id}/requests/${r.requestId}/reject`); onTeamChange && onTeamChange(); } catch (err) { alert(err?.response?.data?.detail || 'Failed to reject request'); } }} className="px-3 py-1 text-xs rounded-lg bg-red-50 text-red-700 border border-red-100">Reject</button>
                              </div>
                            </div>
                          ))}
                          {!(team.requests || []).filter(r=>r.status==='pending').length && <div className="text-xs text-neutral-400">No pending requests</div>}
                        </div>
                      </>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        );
      })}
    </div>
  );
}

function EventSubmissions() {
  return <div className="text-sm text-neutral-400">Submissions UI place --</div>;
}

function EventLeaderboard() {
  return <div className="text-sm text-neutral-400">Leaderboard UI placeholder</div>;
}
