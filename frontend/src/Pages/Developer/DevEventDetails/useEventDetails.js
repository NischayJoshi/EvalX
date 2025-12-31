import { useEffect, useState, useMemo } from "react";
import api from "@/utils/api";

export default function useEventDetails(id) {
  const [event, setEvent] = useState(null);
  const [submissions, setSubmissions] = useState(null);
  const [loadingEvent, setLoadingEvent] = useState(true);
  const [loadingSubs, setLoadingSubs] = useState(true);
  const [error, setError] = useState(null);

  const [activeRoundId, setActiveRoundId] = useState(null);
  const [teamRefresh, setTeamRefresh] = useState(0);

  useEffect(() => {
    if (!id) return;
    setLoadingEvent(true);
    api.get(`/team/events/${id}`)
      .then((res) => setEvent(res.data.data))
      .catch(() => setError("Failed to load event"))
      .finally(() => setLoadingEvent(false));
  }, [id]);

  useEffect(() => {
    if (!id) return;
    setLoadingSubs(true);
    api.get(`/team/events/${id}/submissions`)
      .then((res) => {
        const data = res.data.data;
        setSubmissions(data);
        if (data?.rounds?.length) setActiveRoundId(data.rounds[0].id || data.rounds[0]._id);
      })
      .catch(() => {
        setSubmissions(null);
      })
      .finally(() => setLoadingSubs(false));
  }, [id, teamRefresh]);

  const sortedLeaderboard = useMemo(() => {
    const r = submissions?.rounds?.find((r) => (r.id || r._id) === activeRoundId) || submissions?.rounds?.find((r) => r.id === activeRoundId) || null;
    return r?.submissions?.slice().sort((a, b) => (b.score || 0) - (a.score || 0)) || [];
  }, [submissions, activeRoundId]);

  const refreshTeams = () => setTeamRefresh((t) => t + 1);

  const createTeam = async (teamName) => {
    if (!teamName || !teamName.trim()) throw new Error("Invalid team name");
    const fd = new FormData();
    fd.append("teamName", teamName.trim());
    await api.post(`/team/events/${id}/teams/create`, fd);
    refreshTeams();
  };

  const joinTeam = async (teamId) => {
    if (!teamId) throw new Error("Invalid team id");
    await api.post(`/team/events/${id}/teams/${teamId}/requests/send`);
    refreshTeams();
  };

  const getCurrentUserId = async () => {
    try {
      const res = await api.get("/team/get-user");
      return res.data.data?.id || null;
    } catch {
      return null;
    }
  };

  const getOpenTeams = async () => {
    try {
      const res = await api.get(`/team/events/${id}/teams/open`);
      return res.data.data || [];
    } catch {
      return [];
    }
  };

  const deleteTeam = async (teamId) => {
    await api.delete(`/team/events/${id}/teams/${teamId}`);
    refreshTeams();
  };

  const addMember = async (teamId, userId) => {
    await api.post(`/team/events/${id}/teams/${teamId}/members/add`, new URLSearchParams({ userId }));
    refreshTeams();
  };

  const removeMember = async (teamId, userId) => {
    await api.post(`/team/events/${id}/teams/${teamId}/members/remove`, new URLSearchParams({ userId }));
    refreshTeams();
  };

  const inviteMember = async (teamId, email) => {
    await api.post(`/team/events/${id}/teams/${teamId}/invite`, new URLSearchParams({ email }));
    refreshTeams();
  };

  const acceptRequest = async (teamId, requestId) => {
    await api.post(`/team/events/${id}/teams/${teamId}/requests/${requestId}/accept`);
    refreshTeams();
  };

  const rejectRequest = async (teamId, requestId) => {
    await api.post(`/team/events/${id}/teams/${teamId}/requests/${requestId}/reject`);
    refreshTeams();
  };

  return {
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
  };
}
