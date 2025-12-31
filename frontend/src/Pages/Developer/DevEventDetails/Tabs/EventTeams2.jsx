import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FiUsers, FiFolder, FiChevronDown, FiChevronUp } from "react-icons/fi";

export default function EventTeams({
  event,
  responses,
  onJoin,
  onTeamChange,
  getCurrentUserId,
  getOpenTeams,
  deleteTeam,
  addMember,
  removeMember,
  inviteMember,
  acceptRequest,
  rejectRequest,
}) {
  const [openTeamList, setOpenTeamList] = useState([]);
  const [currentUserId, setCurrentUserId] = useState(null);
  const [addUserMap, setAddUserMap] = useState({});
  const [inviteEmailMap, setInviteEmailMap] = useState({});
  const [openTeam, setOpenTeam] = useState(null);

  const teams = responses?.teams || [];
  const rounds = responses?.rounds || [];

  useEffect(() => {
    (async () => {
      try {
        const id = await getCurrentUserId();
        setCurrentUserId(id);
      } catch {
        setCurrentUserId(null);
      }
    })();
  }, [getCurrentUserId]);

  useEffect(() => {
    (async () => {
      try {
        const list = await getOpenTeams();
        setOpenTeamList(list || []);
      } catch {
        setOpenTeamList([]);
      }
    })();
  }, [event, responses, onTeamChange, getOpenTeams]);

  const submissionCount = {};
  (rounds || []).forEach((r) => {
    (r.submissions || []).forEach((s) => {
      submissionCount[s.teamId || s.team || s.team_id] = (submissionCount[s.teamId || s.team || s.team_id] || 0) + 1;
    });
  });

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
                {isLeader && <button onClick={async () => { if (!confirm('Delete this team? This cannot be undone.')) return; try { await deleteTeam(id); onTeamChange && onTeamChange(); } catch (err) { alert(err?.response?.data?.detail || 'Failed to delete team'); } }} className="px-3 py-1 text-xs rounded-lg border border-red-300 bg-white hover:bg-red-50 text-red-600">Delete</button>}
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
                              <button onClick={async () => { try { await removeMember(id, memberId); onTeamChange && onTeamChange(); } catch (err) { alert(err?.response?.data?.detail || 'Failed to remove member'); } }} className="text-xs px-2 py-1 rounded bg-red-50 text-red-600 border border-red-100">{currentUserId === memberId ? 'Leave' : 'Remove'}</button>
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
                            <button onClick={async () => { const uid = addUserMap[id]; if (!uid) return alert('Enter userId to add'); try { await addMember(id, uid); setAddUserMap((s) => ({ ...s, [id]: '' })); onTeamChange && onTeamChange(); } catch (err) { alert(err?.response?.data?.detail || 'Failed to add member'); } }} className="px-3 py-2 rounded-lg bg-black text-white text-sm">Add</button>
                          </div>

                          <div className="flex gap-2">
                            <input value={inviteEmailMap[id] || ''} onChange={(e) => setInviteEmailMap((s) => ({ ...s, [id]: e.target.value }))} placeholder="Invite by email" className="flex-1 rounded-xl border border-neutral-300 px-3 py-2 text-sm" />
                            <button onClick={async () => { const email = inviteEmailMap[id]; if (!email) return alert('Enter email to invite'); try { await inviteMember(id, email); setInviteEmailMap((s) => ({ ...s, [id]: '' })); onTeamChange && onTeamChange(); } catch (err) { alert(err?.response?.data?.detail || 'Failed to invite member'); } }} className="px-3 py-2 rounded-lg bg-white border border-neutral-300 text-sm">Invite</button>
                          </div>
                        </div>

                        <div className="mt-4">
                          <p className="text-xs font-medium text-neutral-500 mb-2">Pending Requests</p>
                          {(team.requests || []).filter(r => r.status === 'pending').map((r) => (
                            <div key={r.requestId || r.id} className="flex items-center justify-between gap-3 px-3 py-2 bg-neutral-50 rounded-lg border border-neutral-200 mb-2">
                              <div className="text-sm">{r.firstName} {r.lastName} ({r.email})</div>
                              <div className="flex items-center gap-2">
                                <button onClick={async () => { try { await acceptRequest(id, r.requestId || r.id); onTeamChange && onTeamChange(); } catch (err) { alert(err?.response?.data?.detail || 'Failed to accept request'); } }} className="px-3 py-1 text-xs rounded-lg bg-green-50 text-green-700 border border-green-100">Accept</button>
                                <button onClick={async () => { try { await rejectRequest(id, r.requestId || r.id); onTeamChange && onTeamChange(); } catch (err) { alert(err?.response?.data?.detail || 'Failed to reject request'); } }} className="px-3 py-1 text-xs rounded-lg bg-red-50 text-red-700 border border-red-100">Reject</button>
                              </div>
                            </div>
                          ))}
                          {!(team.requests || []).filter(r => r.status === 'pending').length && <div className="text-xs text-neutral-400">No pending requests</div>}
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
