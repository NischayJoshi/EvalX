import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  FiPlusCircle,
  FiFolder,
  FiUser,
  FiMessageSquare,
  FiLayers,
  FiLogOut,
  FiRss,
} from "react-icons/fi";

import MyEvents from "./MyEvents";
import api from "@/utils/api";
import { useNavigate } from "react-router";

export default function DeveloperDashborad() {
  const [active, setActive] = useState("all-events");

  const [name, setName] = useState("");
  const [user, setUser] = useState(null);
  const navigate = useNavigate();

  // Fetch User
  const getUser = async () => {
    try {
      const res = await api.get("/org/profile", { withCredentials: true });
      setUser(res.data.data);
      setName(res.data.data.firstName);
      let role = res.data.data.role;
      console.log(role)
      if (role !== "Developer") {
        navigate("/org/dashboard");
      }
    } catch (error) {
      console.log("USER_FETCH_ERROR:", error);
    }
  };

  // Fetch Events
  const [events, setEvents] = useState([]);
  const [eventsLoading, setEventsLoading] = useState(true);

  const fetchEvents = async () => {
    try {
      setEventsLoading(true);
      const res = await api.get("/dev/my-events", { withCredentials: true });
      const data = res.data.data || [];

      const cleaned = data.map((e) => ({
        ...e,
        id: e._id,
        pinned: e.pinned || false,
      }));

      setEvents(cleaned);
    } catch (error) {
      console.log("EVENTS_FETCH_ERROR:", error);
    } finally {
      setEventsLoading(false);
    }
  };

  useEffect(() => {
    getUser();
    fetchEvents();
  }, []);

  const [search, setSearch] = useState("");
  const [sort, setSort] = useState("date-desc");
  const [page, setPage] = useState(1);
  const pageSize = 4;

  const tabs = [
    { id: "all-events", label: "All Events", icon: <FiFolder size={18} /> },
    { id: "registered-events", label: "Registered Events", icon: <FiLayers size={18} /> },
    { id: "messages", label: "Messages", icon: <FiMessageSquare size={18} /> },
    { id: "feed", label: "Feed", icon: <FiRss size={18} /> },
    { id: "profile", label: "Profile", icon: <FiUser size={18} /> },
  ];

  const slideAnim = {
    initial: { opacity: 0, y: 10 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.25 },
  };

  const analytics = (() => {
    const total = events.length;
    const upcoming = events.filter((e) => new Date(e.date) >= new Date()).length;
    const past = events.filter((e) => new Date(e.date) < new Date()).length;
    const pinned = events.filter((e) => e.pinned).length;
    return { total, upcoming, past, pinned };
  })();

  return (
    <div className="min-h-screen bg-neutral-100">

      {/* NAVBAR — Premium */}
      <nav className="w-full bg-white border-b border-neutral-200 px-8 py-4 flex items-center justify-between shadow-sm">
        <div className="flex items-center gap-3">
          <FiLayers size={26} className="text-neutral-800" />
          <h1 className="text-xl font-semibold tracking-tight text-neutral-900">
            EvalX 
          </h1>
        </div>

        <div className="flex items-center gap-6">
          <span className="text-sm text-neutral-700">
            Hi, {name || "Organizer"}
          </span>
          <button
            onClick={() => {
              localStorage.clear();
              window.location.href = "/";
            }}
            className="text-sm flex items-center gap-2 text-neutral-600 hover:text-black transition font-medium"
          >
            <FiLogOut size={16} /> Logout
          </button>
        </div>
      </nav>

      {/* TABS — Clean, No Scroll */}
      <div className="w-full bg-white border-b border-neutral-200 shadow-sm">
        <div className="max-w-6xl mx-auto flex gap-10 px-8">

          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setActive(t.id)}
              className={`
                py-5 text-sm flex items-center gap-2 relative transition-all
                ${
                  active === t.id
                    ? "text-black font-semibold"
                    : "text-neutral-500 hover:text-neutral-800"
                }
              `}
            >
              {t.icon}
              {t.label}

              {active === t.id && (
                <motion.div
                  layoutId="underline"
                  className="absolute left-0 right-0 -bottom-[2px] h-[2px] bg-black rounded-full"
                />
              )}
            </button>
          ))}

        </div>
      </div>

      {/* CONTENT */}
      <main className="p-8 flex justify-center">
        <div className="w-full max-w-6xl">
          <AnimatePresence mode="wait">
            <motion.div
              key={active}
              {...slideAnim}
              className="bg-white rounded-xl shadow-sm p-6 border border-neutral-200"
            >

              {active === "all-events" && (
                <MyEvents
                  events={events}
                  setEvents={setEvents}
                  search={search}
                  setSearch={setSearch}
                  sort={sort}
                  setSort={setSort}
                  page={page}
                  setPage={setPage}
                  pageSize={pageSize}
                  eventsLoading={eventsLoading}
                  analytics={analytics}
                />
              )}

              {active === "registered-events" && (
                <>
                  <h2 className="text-xl font-semibold">Registered Events</h2>
                  <p className="text-neutral-600 text-sm mt-1">
                    Events you have registered for.
                  </p>
                </>
              )}

              {active === "messages" && (
                <>
                  <h2 className="text-xl font-semibold">Messages</h2>
                  <p className="text-neutral-600 text-sm mt-1">
                    Chat system coming soon.
                  </p>
                </>
              )}

              {active === "feed" && (
                <>
                  <h2 className="text-xl font-semibold">Feed</h2>
                  <p className="text-neutral-600 text-sm mt-1">
                    Updates & announcements will appear here.
                  </p>
                </>
              )}

              {active === "profile" && (
                <>
                  <h2 className="text-xl font-semibold mb-4">Profile</h2>
                  {user ? (
                    <div className="text-sm space-y-1">
                      <p><b>Name:</b> {user.firstname} {user.lastname}</p>
                      <p><b>Email:</b> {user.email}</p>
                    </div>
                  ) : (
                    <p className="text-sm text-neutral-600">Loading profile...</p>
                  )}
                </>
              )}

            </motion.div>
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}
