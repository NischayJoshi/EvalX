import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  FiPlusCircle,
  FiFolder,
  FiUser,
  FiLayers,
  FiTool,
  FiLogOut,
} from "react-icons/fi";

import MyEvents from "./MyEvents";
import CreateEventPanel from "./CreateEventPanel";
import api from "@/utils/api";

export default function OrganizerDashboard() {
  const [active, setActive] = useState("events");

  const [name, setName] = useState("");
  const [user, setUser] = useState(null);

  // -----------------------------
  // FIX: Load user profile
  // -----------------------------
  const getUser = async () => {
    try {
      const res = await api.get("/org/profile", { withCredentials: true });
      setUser(res.data.data);
      setName(res.data.data.firstName);
      console.log(res.data.data.firstName)
    } catch (error) {
      console.log("USER_FETCH_ERROR:", error);
    }
  };

  useEffect(() => {
    getUser();
    console.log(name);
    fetchEvents();
  }, []);

  // load name from localStorage if exists
  // useEffect(() => {
  //   const stored = localStorage.getItem("name");
  //   if (stored && !name) setName(stored);
  // }, [name]);

  // -----------------------------
  // Events (UNCHANGED as requested)
  // -----------------------------
  const [events, setEvents] = useState([]);
const [eventsLoading, setEventsLoading] = useState(true);


  const [search, setSearch] = useState("");
  const [sort, setSort] = useState("date-desc");
  const [page, setPage] = useState(1);
  const pageSize = 4;



  useEffect(() => {
    const timeout = setTimeout(() => setEventsLoading(false), 500);
    return () => clearTimeout(timeout);
  }, []);

  const fetchEvents = async () => {
  try {
    setEventsLoading(true);
    const res = await api.get("/org/my-events", { withCredentials: true });
    const data = res.data.data || [];

    // Add pinned=false by default if not present
    const cleaned = data.map(e => ({
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


  const tabs = [
    { id: "events", label: "My Events", icon: <FiFolder size={18} /> },
    { id: "create-event", label: "Create Event", icon: <FiPlusCircle size={18} /> },
    { id: "ai-tools", label: "AI Tools", icon: <FiTool size={18} /> },
    { id: "templates", label: "Templates & Rubrics", icon: <FiLayers size={18} /> },
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
      {/* Navbar */}
      <nav className="w-full bg-white border-b border-neutral-200 px-8 py-4 flex items-center justify-between shadow-sm">
        <div className="flex items-center gap-3">
          <FiLayers size={26} />
          <h1 className="text-xl font-semibold tracking-tight">EvalX Dashboard</h1>
        </div>

        <div className="flex items-center gap-6">
          <span className="text-sm text-neutral-700">Hi, {name || "Organizer"}</span>
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

      {/* Tabs */}
      <div className="w-full bg-white px-8 flex gap-8 overflow-x-auto border-b border-neutral-200">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setActive(t.id)}
            className={`py-5 text-sm flex items-center gap-2 border-b-2 transition-all
              ${
                active === t.id
                  ? "border-neutral-900 text-neutral-900 font-semibold"
                  : "border-transparent text-neutral-500 hover:text-neutral-800"
              }`}
          >
            {t.icon}
            {t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <main className="p-8 flex justify-center">
        <div className="w-full max-w-6xl">
          <AnimatePresence mode="wait">
            <motion.div key={active} {...slideAnim} className="bg-white rounded-xl shadow-sm p-6 border border-neutral-200">
              
              {active === "events" && (
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

              {active === "create-event" && (
                <CreateEventPanel
                  onCreate={(evt) => {
                    setEvents((prev) => [
                      ...prev,
                      {
                        id: Date.now(),
                        name: evt.name,
                        date: evt.date || new Date().toISOString().slice(0, 10),
                        pinned: false,
                      },
                    ]);
                    setActive("events");
                  }}
                />
              )}

              {active === "ai-tools" && (
                <>
                  <h2 className="text-2xl font-semibold mb-4">AI Tools</h2>
                  <p className="text-neutral-600 text-sm">
                    This will later connect to your PPT analyzer, repo analyzer, and AI interview engine.
                  </p>
                </>
              )}

              {active === "templates" && (
                <>
                  <h2 className="text-2xl font-semibold mb-4">Templates & Rubrics</h2>
                  <p className="text-neutral-600 text-sm">Store scoring rubrics and event presets here.</p>
                </>
              )}

              {active === "profile" && (
                <>
                  <h2 className="text-2xl font-semibold mb-4">Profile</h2>
                  <p className="text-neutral-600 text-sm">Organizer details will be loaded from backend.</p>

                  {user && (
                    <div className="text-sm mt-2">
                      <p><b>Name:</b> {user.firstname} {user.lastname}</p>
                      <p><b>Email:</b> {user.email}</p>
                    </div>
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
