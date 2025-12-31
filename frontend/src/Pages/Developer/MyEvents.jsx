import { motion, AnimatePresence } from "framer-motion";
import {
  FiStar,
  FiCalendar,
  FiUsers,
  FiLayers,
} from "react-icons/fi";
import { LuTrophy } from "react-icons/lu";
import { PiUsersThree } from "react-icons/pi";
import { useNavigate } from "react-router-dom";

function highlightText(text, query) {
  if (!query) return text;
  const lower = text.toLowerCase();
  const idx = lower.indexOf(query.toLowerCase());
  if (idx === -1) return text;

  return (
    <>
      {text.slice(0, idx)}
      <span className="bg-yellow-100 rounded px-1">
        {text.slice(idx, idx + query.length)}
      </span>
      {text.slice(idx + query.length)}
    </>
  );
}

function AnalyticsCard({ label, value }) {
  return (
    <div className="bg-white border border-neutral-200 rounded-2xl p-4 shadow-sm hover:shadow-md transition">
      <p className="text-xs text-neutral-500">{label}</p>
      <p className="text-xl font-semibold mt-1">{value}</p>
    </div>
  );
}

export default function MyEvents({
  events,
  setEvents,
  search,
  setSearch,
  sort,
  setSort,
  page,
  setPage,
  pageSize,
  eventsLoading,
  analytics,
}) {
  const navigate = useNavigate();

  const sorted = [...events].sort((a, b) => {
    if (sort === "date-desc") return new Date(b.date) - new Date(a.date);
    if (sort === "date-asc") return new Date(a.date) - new Date(b.date);
    if (sort === "alpha-asc") return a.name.localeCompare(b.name);
    if (sort === "alpha-desc") return b.name.localeCompare(a.name);
    return 0;
  });

  const filtered = sorted.filter((e) =>
    e.name
  );

  const pinned = filtered.filter((e) => e.pinned);
  const others = filtered.filter((e) => !e.pinned);
  const finalList = [...pinned, ...others];

  const totalPages = Math.max(1, Math.ceil(finalList.length / pageSize));
  const currentPage = Math.min(page, totalPages);

  const paginated = finalList.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize
  );

  const togglePin = (id) => {
    setEvents((prev) =>
      prev.map((e) =>
        (e.id || e._id) === id ? { ...e, pinned: !e.pinned } : e
      )
    );
  };

  return (
    <>
      <h2 className="text-2xl font-semibold mb-6">My Events</h2>

      {/* Analytics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
        <AnalyticsCard label="Total" value={analytics.total} />
        <AnalyticsCard label="Upcoming" value={analytics.upcoming} />
        <AnalyticsCard label="Past" value={analytics.past} />
        <AnalyticsCard label="Pinned" value={analytics.pinned} />
      </div>

      {/* Search + Sort */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
        <input
          type="text"
          placeholder="Search events..."
          className="w-full px-4 py-2 rounded-xl border border-neutral-300 bg-white text-sm outline-none focus:border-neutral-900 transition"
          value={search}
          onChange={(e) => {
            setSearch(e.target.value);
            setPage(1);
          }}
        />

        <select
          className="px-4 py-2 rounded-xl border border-neutral-300 bg-white text-sm focus:border-neutral-900 transition"
          value={sort}
          onChange={(e) => setSort(e.target.value)}
        >
          <option value="date-desc">Date: Newest First</option>
          <option value="date-asc">Date: Oldest First</option>
          <option value="alpha-asc">Name: A → Z</option>
          <option value="alpha-desc">Name: Z → A</option>
        </select>
      </div>

      {/* Loader */}
      {eventsLoading && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          {[1, 2, 3, 4].map((i) => (
            <div
              key={i}
              className="bg-white rounded-2xl border border-neutral-200 p-6 animate-pulse"
            >
              <div className="h-38 w-full bg-neutral-200 rounded mb-3" />
              <div className="h-4 w-32 bg-neutral-200 rounded mb-3" />
              <div className="h-3 w-24 bg-neutral-200 rounded" />
            </div>
          ))}
        </div>
      )}

      {/* No events */}
      {!eventsLoading && finalList.length === 0 && (
        <div className="flex flex-col items-center justify-center py-16 text-neutral-500 text-sm">
          <div className="w-24 h-24 rounded-full bg-neutral-200 animate-pulse" />
          <p className="mt-4">No events found.</p>
        </div>
      )}

      {/* Events */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
        <AnimatePresence>
          {!eventsLoading &&
            paginated.map((event) => {
              const id = event.id || event._id;

              return (
                <motion.div
                  key={id}
                  initial={{ opacity: 0, y: 14 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -14 }}
                  transition={{ duration: 0.28 }}
                  className="
                    bg-white rounded-2xl shadow-sm hover:shadow-xl 
                    transition-all overflow-hidden cursor-pointer group
                    border border-neutral-200
                  "
                  onClick={() => navigate(`/dashboard/id/${id}`)}
                >
                  {/* Banner */}
                  <div className="relative w-full h-44 bg-neutral-100 overflow-hidden">
                    <img
                      src={event.banner}
                      alt="banner"
                      className="w-full h-full object-cover group-hover:scale-105 transition duration-500"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/30 to-transparent"></div>

                    {/* Floating Logo */}
                    {event.logo && (
                      <div
                        className="
                          absolute -bottom-6 left-5
                          w-14 h-14 rounded-xl overflow-hidden shadow-lg
                          border border-white bg-white
                        "
                      >
                        <img
                          src={event.logo}
                          className="w-full h-full object-cover"
                        />
                      </div>
                    )}
                  </div>

                  {/* Content */}
                  <div className="pt-8 px-5 pb-5">
                    {/* Title */}
                    <h3 className="font-semibold text-neutral-900 text-lg leading-tight">
                      {highlightText(event.name, search)}
                    </h3>

                    {/* Summary */}
                    <p className="text-sm text-neutral-600 mt-1 line-clamp-2">
                      {event.summary}
                    </p>

                    {/* Description */}
                    <p className="text-xs text-neutral-500 mt-1 line-clamp-1">
                      {event.description}
                    </p>

                    {/* Stats */}
                    <div className="mt-4 flex flex-wrap gap-3 text-[12px] text-neutral-700">
                      <div className="flex items-center gap-1.5">
                        <FiCalendar size={14} className="text-neutral-500" />
                        {event.date}
                      </div>

                      <div className="flex items-center gap-1.5">
                        <LuTrophy size={14} className="text-neutral-500" />
                        ₹{event.prize}
                      </div>

                      <div className="flex items-center gap-1.5">
                        <FiLayers size={14} className="text-neutral-500" />
                        {event.rounds?.length || 0} Rounds
                      </div>

                      <div className="flex items-center gap-1.5">
                        <FiUsers size={14} className="text-neutral-500" />
                        {event.maxTeams} Teams
                      </div>

                      <div className="flex items-center gap-1.5">
                        <PiUsersThree size={14} className="text-neutral-500" />
                        {event.minMembers}-{event.maxMembers} Members
                      </div>
                    </div>

                    {/* CTA + Pin */}
                    <div className="mt-5 flex justify-between items-center">
                      <button
                        type="button"
                        className="
                          px-4 py-1.5 text-xs rounded-lg border border-neutral-300 
                          hover:border-neutral-900 hover:text-neutral-900 
                          transition-all
                        "
                      >
                        View details →
                      </button>

                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          togglePin(id);
                        }}
                        className="text-neutral-400 hover:text-yellow-400 transition"
                      >
                        <FiStar
                          size={20}
                          className={
                            event.pinned ? "fill-yellow-400 text-yellow-400" : ""
                          }
                        />
                      </button>
                    </div>
                  </div>
                </motion.div>
              );
            })}
        </AnimatePresence>
      </div>

      {/* Pagination */}
      {!eventsLoading && finalList.length > pageSize && (
        <div className="flex items-center justify-between mt-6 text-sm text-neutral-600">
          <span>
            Page {currentPage} of {totalPages}
          </span>

          <div className="flex gap-2">
            <button
              disabled={currentPage === 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              className="px-3 py-1 rounded-lg border border-neutral-300 bg-white disabled:opacity-40"
            >
              Prev
            </button>

            <button
              disabled={currentPage === totalPages}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              className="px-3 py-1 rounded-lg border border-neutral-300 bg-white disabled:opacity-40"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </>
  );
}
