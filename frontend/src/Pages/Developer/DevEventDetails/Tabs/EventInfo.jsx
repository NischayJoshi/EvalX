import { useEffect, useState } from "react";
import { FiEdit3, FiLayers, FiUsers } from "react-icons/fi";
import { LuTrophy } from "react-icons/lu";
import { PiUsersThree } from "react-icons/pi";
import StatCard from "./StatCard";
import api from "@/utils/api";

export default function EventInfo({
  event,
  onEdit,
  onCreateTeam,
  onJoinTeam,
  teamRefreshTrigger = () => {},
}) {
  const [team, setTeam] = useState(null);
  const [loading, setLoading] = useState(true);

  const checkRegistration = async (eventId) => {
    setLoading(true);
    try {
      const res = await api.get(`/v1/connect/is-registered/${eventId}`);
      setTeam(res.data.team); // team object includes isActive
      console.log(res.data)
    } catch (err) {
      if (err.response?.status === 403) {
        setTeam(null);
      } else {
        console.error(err);
      }
    } finally {
      setLoading(false);
    }
    
  };

  useEffect(() => {
    if (event && (event.id || event._id)) {
      checkRegistration(event.id || event._id);
    }
    console.log(team)
  }, [event, teamRefreshTrigger]);

  const roundsCount = event.rounds?.length || 0;

  return (
    <div className="space-y-6">
      {/* Banner */}
      {event.banner && (
        <div className="relative rounded-2xl overflow-hidden shadow-lg border border-neutral-200">
          <img src={event.banner} alt="banner" className="w-full h-56 object-cover" />
          <div className="absolute inset-0 bg-gradient-to-t from-black/40 to-transparent" />
          {event.logo && (
            <div className="absolute bottom-4 left-4 w-16 h-16 rounded-xl overflow-hidden border-2 border-white shadow-xl">
              <img src={event.logo} alt="logo" className="w-full h-full object-cover" />
            </div>
          )}
        </div>
      )}

      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h2 className="text-xl font-semibold text-neutral-900">{event.name}</h2>
          <p className="text-sm text-neutral-600 mt-1">{event.summary}</p>
        </div>

        <div className="flex items-center gap-3">
          {loading ? (
            <div className="px-4 py-2 bg-neutral-200 rounded-xl text-sm animate-pulse">
              Checking…
            </div>
          ) : team ? (
            <div
              className={`px-4 py-2 rounded-xl text-sm shadow text-white ${
                team.isActive ? "bg-green-600" : "bg-yellow-500"
              }`}
            >
              {team.isActive ? "Registered" : "Pending Approval"}
            </div>
          ) : (
            <div className="flex gap-2">
              <button
                onClick={onCreateTeam}
                className="px-4 py-2 rounded-xl bg-black text-white text-sm hover:bg-neutral-800"
              >
                Create Team
              </button>
              <button
                onClick={onJoinTeam}
                className="px-4 py-2 rounded-xl border border-neutral-300 bg-white text-sm hover:border-neutral-900"
              >
                Join Team
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Description */}
      <div className="bg-white border rounded-2xl p-6 shadow-sm">
        <h3 className="font-semibold text-sm">Description</h3>
        <p className="text-sm text-neutral-600 mt-2 leading-relaxed">{event.description}</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          label="Prize"
          value={`₹${event.prize || 0}`}
          icon={<LuTrophy size={20} className="text-yellow-600" />}
        />
        <StatCard
          label="Rounds"
          value={roundsCount}
          icon={<FiLayers size={20} className="text-blue-600" />}
        />
        <StatCard
          label="Max Teams"
          value={event.maxTeams || "—"}
          icon={<FiUsers size={20} className="text-green-600" />}
        />
        <StatCard
          label="Team Size"
          value={`${event.minMembers || 1} - ${event.maxMembers || 1}`}
          icon={<PiUsersThree size={20} className="text-purple-600" />}
        />
      </div>

      {/* Team Box */}
      {team && (
        <div className="bg-white border rounded-2xl p-6 shadow-sm">
          <h3 className="font-semibold text-neutral-900 text-sm mb-3">Your Team</h3>

          <p className="text-sm"><b>Name:</b> {team.teamName || team.name}</p>

          <p className="text-sm mt-2 font-semibold">Members:</p>
          <ul className="text-sm text-neutral-700 mt-1 list-disc ml-5">
            {team.members?.map((m, idx) => (
              <li key={idx}>
                {m.firstName} {m.lastName} ({m.email})
              </li>
            ))}
          </ul>

          <p className="text-xs text-neutral-500 mt-3">Team ID: {team._id}</p>
        </div>
      )}
    </div>
  );
}
