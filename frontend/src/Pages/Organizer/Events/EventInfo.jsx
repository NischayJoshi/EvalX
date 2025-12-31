import {
  FiEdit3,
  FiCalendar,
  FiUsers,
  FiLayers,
} from "react-icons/fi";
import { LuTrophy } from "react-icons/lu";
import { PiUsersThree } from "react-icons/pi";

export default function EventInfo({ event, responses, onEdit }) {
  return (
    <div className="space-y-6">

      {/* Banner */}
      {event.banner && (
        <div className="relative rounded-2xl overflow-hidden shadow-lg border border-neutral-200">
          <img
            src={event.banner}
            className="w-full h-56 object-cover"
          />

          <div className="absolute inset-0 bg-gradient-to-t from-black/40 to-transparent"></div>

          {event.logo && (
            <div className="absolute bottom-4 left-4 w-16 h-16 rounded-xl overflow-hidden border-2 border-white shadow-xl">
              <img
                src={event.logo}
                className="w-full h-full object-cover"
              />
            </div>
          )}
        </div>
      )}

      {/* Header + Edit Button */}
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-xl font-semibold text-neutral-900 tracking-tight">
            {event.name}
          </h2>

          <p className="text-sm text-neutral-600 mt-1 leading-relaxed">
            {event.summary}
          </p>

          <div className="mt-3 flex flex-wrap gap-2 text-xs">
            <span className="px-3 py-1 rounded-full bg-neutral-100 border text-neutral-700">
              {event.date}
            </span>
            <span className="px-3 py-1 rounded-full bg-neutral-100 border text-neutral-700">
              {event.rounds?.length} Rounds
            </span>
            <span className="px-3 py-1 rounded-full bg-neutral-100 border text-neutral-700">
              {event.maxTeams} Teams
            </span>
          </div>
        </div>

        <button
          onClick={onEdit}
          className="
            flex items-center gap-2 px-4 py-2 
            rounded-xl border border-neutral-300 bg-white 
            text-sm hover:border-neutral-900 hover:text-neutral-900 
            transition shadow-sm
          "
        >
          <FiEdit3 size={16} /> Edit
        </button>
      </div>

      {/* Description Card */}
      <div className="bg-white rounded-2xl border border-neutral-200 p-6 shadow-sm">
        <h3 className="font-semibold text-neutral-900 text-sm">Description</h3>
        <p className="text-sm text-neutral-600 leading-relaxed mt-2">
          {event.description}
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          label="Prize"
          value={`₹${event.prize}`}
          icon={<LuTrophy size={20} className="text-yellow-600" />}
        />
        <StatCard
          label="Rounds"
          value={event.rounds.length}
          icon={<FiLayers size={20} className="text-blue-600" />}
        />
        <StatCard
          label="Teams Registered"
          value={responses?.teamsCount || 0}
          icon={<FiUsers size={20} className="text-green-600" />}
        />
        <StatCard
          label="Total Submissions"
          value={
            responses?.rounds?.reduce(
              (sum, r) => sum + (r.submissions?.length || 0),
              0
            ) || 0
          }
          icon={<PiUsersThree size={20} className="text-purple-600" />}
        />
      </div>

      {/* Additional Details */}
      <div className="bg-white rounded-2xl border border-neutral-200 p-6 shadow-sm">
        <h3 className="text-sm font-semibold text-neutral-900 mb-3">
          Additional Details
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-y-3 text-sm text-neutral-700">
          <Detail label="Event Date" value={event.date} />
          <Detail
            label="Team Size"
            value={`${event.minMembers} – ${event.maxMembers}`}
          />
          <Detail
            label="Max Teams Allowed"
            value={event.maxTeams}
          />
          <Detail
            label="Total Rounds"
            value={event.rounds.length}
          />
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value, icon }) {
  return (
    <div
      className="
        bg-white rounded-xl border border-neutral-200 p-4 
        shadow-sm hover:shadow-md transition cursor-default
      "
    >
      <div className="flex items-center justify-between">
        <p className="text-xs text-neutral-500">{label}</p>
        {icon}
      </div>
      <p className="text-lg font-semibold mt-1">{value}</p>
    </div>
  );
}

function Detail({ label, value }) {
  return (
    <div>
      <p className="text-xs text-neutral-500">{label}</p>
      <p className="font-medium">{value}</p>
    </div>
  );
}
