import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  FiFileText,
  FiSearch,
  FiMessageSquare,
  FiShield,
  FiTarget,
  FiAward,
  FiUsers,
  FiZap,
  FiTrendingUp,
  FiArrowRight,
  FiPlayCircle,
  FiLayers
} from "react-icons/fi";
import { useNavigate } from "react-router";

function Hero() {
  const icons = [
    <FiFileText size={22} />,
    <FiSearch size={22} />,
    <FiMessageSquare size={22} />,
    <FiShield size={22} />,
    <FiTarget size={22} />,
    <FiAward size={22} />,
    <FiUsers size={22} />,
    <FiZap size={22} />,
    <FiTrendingUp size={22} />
  ];

  const features = [
    { title: "AI-Powered PPT Evaluation", description: "Slide-by-slide analysis, clarity and missing section detection." },
    { title: "Repo + Video Intelligence", description: "Security scans, code smells, commit behavior, video authenticity." },
    { title: "AI Interview Panel", description: "Context-aware questions generated from your actual project." },
    { title: "Fraud Detection", description: "Detect copied repos, bulk commits and AI-generated content." },
    { title: "Progress Tracking", description: "PPT, code and video improvement checklists per round." },
    { title: "AI Leaderboard", description: "Fairness-driven scoring with explanations." },
    { title: "Team Matchmaking", description: "Skill-based teammate recommendations." },
    { title: "Auto Event Builder", description: "Generate problem statements, rubrics, rules and emails in seconds." },
    { title: "Startup Mentor", description: "Turn your project into a startup with guided support." }
  ];

  const [active, setActive] = useState(0);
  const navigate = useNavigate();

  useEffect(() => {
    const interval = setInterval(() => {
      setActive(prev => (prev + 1) % features.length);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-white text-neutral-900">
      <header className="sticky top-0 z-30 border-b border-neutral-200 bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-4">
          <div className="flex items-center gap-2">
            <FiLayers size={24} />
            <span className="text-xl font-semibold">EvalX</span>
          </div>

          <nav className="hidden items-center gap-8 text-sm text-neutral-700 md:flex">
            <a href="#features" className="hover:text-black">
              Features
            </a>
            <a href="#how" className="hover:text-black">
              How It Works
            </a>
            <a href="#solutions" className="hover:text-black">
              Solutions
            </a>
          </nav>

          <div className="hidden gap-3 md:flex">
            <button className="rounded-full border border-neutral-300 px-4 py-1.5 text-sm hover:bg-neutral-100" onClick={() => navigate('/login')}>
              Log In
            </button>
            <button className="rounded-full bg-black px-4 py-1.5 text-sm text-white hover:opacity-85" onClick={() => navigate('/signup')}>
              Create Account
            </button>
          </div>

          <div className="flex items-center gap-2 md:hidden">
            <button className="rounded-full border border-neutral-300 px-3 py-1 text-xs" onClick={() => navigate('/login')}>
              Log In
            </button>
            <button className="rounded-full bg-black px-3 py-1 text-xs text-white" onClick={() => navigate('/signup')}>
              Sign Up
            </button>
          </div>
        </div>
      </header>

      <section className="border-b border-neutral-200">
        <div className="mx-auto max-w-6xl px-4 py-16 sm:py-20 lg:px-8">
          <div className="grid grid-cols-1 gap-16 lg:grid-cols-[1.3fr_0.9fr]">
            <motion.div
              initial={{ opacity: 0, y: 25 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="space-y-7"
            >
              <div className="inline-flex items-center gap-2 rounded-full border border-neutral-300 bg-white px-3 py-1 text-xs font-medium">
                <span className="h-2 w-2 rounded-full bg-black"></span>
                AI Judge + Mentor System
              </div>

              <h1 className="text-3xl font-semibold leading-tight sm:text-4xl md:text-5xl">
                The Future of Hackathon
                <span className="block text-neutral-500">Evaluation & Mentoring</span>
              </h1>

              <p className="max-w-xl text-neutral-600 text-lg">
                EvalX automates PPT reviews, code audits, video checks, interviews, and fraud detection — making
                events fair, fast, and scalable.
              </p>

              <div className="flex flex-col gap-3 pt-4 sm:flex-row sm:flex-wrap">
                <motion.button
                  whileHover={{ scale: 1.03 }}
                  whileTap={{ scale: 0.98 }}
                  className="flex items-center gap-2 rounded-full bg-black px-6 py-3 text-sm font-medium text-white"
                >
                  Launch EvalX <FiArrowRight />
                </motion.button>

                <button className="flex items-center gap-2 rounded-full border border-neutral-300 px-6 py-3 text-sm font-medium hover:bg-neutral-100">
                  Interactive Demo <FiPlayCircle />
                </button>

                <button className="rounded-full border border-neutral-300 px-6 py-3 text-sm font-medium hover:bg-neutral-100">
                  See All Features
                </button>
              </div>

              <div className="flex flex-wrap gap-8 pt-8">
                <div>
                  <p className="text-2xl font-bold">500+</p>
                  <p className="text-neutral-600 text-sm">Events Evaluated</p>
                </div>
                <div>
                  <p className="text-2xl font-bold">50K+</p>
                  <p className="text-neutral-600 text-sm">Participants</p>
                </div>
                <div>
                  <p className="text-2xl font-bold">98%</p>
                  <p className="text-neutral-600 text-sm">Fairness Boost</p>
                </div>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
              className="flex justify-center pr-2 lg:justify-end lg:pr-4"
            >
              <div className="w-full max-w-xs rounded-3xl border border-neutral-200 bg-white p-6 shadow-[0_6px_30px_rgba(0,0,0,0.05)] sm:max-w-sm">
                <p className="mb-4 text-xs font-semibold text-neutral-700">Live AI Spotlight</p>

                <motion.div
                  key={active}
                  initial={{ opacity: 0.2, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4 }}
                  className="flex h-40 flex-col justify-center rounded-xl border border-neutral-200 bg-neutral-50 p-5"
                >
                  <div className="flex items-center gap-3">
                    <div className="rounded-xl bg-white p-2 shadow-sm">
                      {icons[active]}
                    </div>
                    <h3 className="text-sm font-semibold">{features[active].title}</h3>
                  </div>

                  <p className="mt-3 text-xs text-neutral-600 line-clamp-3">{features[active].description}</p>
                </motion.div>

                <div className="mt-5 flex flex-wrap gap-2">
                  {[0, 1, 2].map(i => (
                    <span
                      key={i}
                      className="flex items-center gap-1 rounded-full border border-neutral-200 bg-neutral-100 px-3 py-1 text-[11px]"
                    >
                      {icons[i]} {features[i].title.split(" ")[0]}
                    </span>
                  ))}
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      <section id="features" className="py-16">
        <div className="mx-auto max-w-6xl px-4 lg:px-8">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center text-3xl font-semibold"
          >
            Explore EvalX Features
          </motion.h2>
          <p className="mt-2 text-center text-neutral-600">Swipe horizontally to explore all capabilities</p>

          <div className="mt-10 overflow-x-auto pb-2 scrollbar-none">
            <div className="flex min-w-max gap-6">
              {features.map((f, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.05 }}
                  className="w-72 flex-shrink-0 rounded-2xl border border-neutral-200 bg-white p-6 shadow-sm transition hover:-translate-y-1 hover:shadow-md"
                >
                  <div className="mb-3 w-fit rounded-xl bg-neutral-100 p-3">{icons[i]}</div>
                  <h3 className="mb-2 text-sm font-semibold">{f.title}</h3>
                  <p className="text-sm text-neutral-600">{f.description}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section id="usecases" className="bg-neutral-50 py-16">
        <div className="mx-auto max-w-6xl px-4 lg:px-8">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center text-3xl font-semibold"
          >
            Built For Everyone In The Event
          </motion.h2>

          <div className="mt-10 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {[
              { title: "Organizers", desc: "Automate evaluation, manage rounds, reduce manual workload." },
              { title: "Judges", desc: "Get AI context, fair scoring support, and clear breakdowns." },
              { title: "Participants", desc: "Get feedback each round and understand how to improve." }
            ].map((item, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1 * i }}
                className="rounded-2xl border border-neutral-200 bg-white p-6 shadow-sm"
              >
                <h3 className="text-lg font-semibold">{item.title}</h3>
                <p className="mt-2 text-sm text-neutral-600">{item.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <section className="border-y border-neutral-200 bg-white py-14">
        <div className="mx-auto max-w-6xl px-4 lg:px-8">
          <p className="text-center text-xs font-medium uppercase tracking-[0.2em] text-neutral-500">
            Trusted by Teams
          </p>
          <div className="mt-6 flex flex-wrap items-center justify-center gap-8 opacity-70">
            <div className="h-8 w-24 rounded bg-neutral-200" />
            <div className="h-8 w-24 rounded bg-neutral-200" />
            <div className="h-8 w-24 rounded bg-neutral-200" />
            <div className="h-8 w-24 rounded bg-neutral-200" />
          </div>
        </div>
      </section>

      <section id="testimonials" className="bg-white py-16">
        <div className="mx-auto max-w-6xl px-4 lg:px-8">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center text-3xl font-semibold"
          >
            What People Say
          </motion.h2>

          <div className="mt-10 grid gap-6 md:grid-cols-3">
            {[
              "EvalX cut our judging time in half without losing quality.",
              "Participants finally understand why they scored what they scored.",
              "Fraud detection saved us from multiple copied projects."
            ].map((t, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1 * i }}
                className="rounded-2xl border border-neutral-200 bg-white p-6 shadow-sm"
              >
                <p className="text-sm text-neutral-700">{t}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <section className="bg-neutral-50 py-16">
        <div className="mx-auto max-w-6xl px-4 text-center lg:px-8">
          <h2 className="text-3xl font-semibold">Ready to modernize your next event?</h2>
          <p className="mt-3 text-sm text-neutral-600">
            Use EvalX to evaluate smarter, faster, and more fairly.
          </p>
          <button className="mt-6 rounded-full bg-black px-8 py-3 text-sm font-medium text-white hover:opacity-90">
            Get Started Free
          </button>
        </div>
      </section>

      <footer className="border-t border-neutral-200 bg-white py-12">
        <div className="mx-auto grid max-w-6xl grid-cols-1 gap-10 px-4 sm:grid-cols-3 lg:grid-cols-4">
          <div>
            <div className="mb-3 flex items-center gap-2">
              <FiLayers size={22} />
              <span className="text-lg font-semibold">EvalX</span>
            </div>
            <p className="text-sm text-neutral-600">
              AI-first evaluation engine for modern hackathons.
            </p>
          </div>

          <div>
            <h4 className="mb-3 text-sm font-semibold">Product</h4>
            <ul className="space-y-2 text-sm text-neutral-600">
              <li>
                <a href="#features" className="hover:text-black">
                  Features
                </a>
              </li>
              <li>
                <a href="#how" className="hover:text-black">
                  Architecture
                </a>
              </li>
              <li>
                <a href="#solutions" className="hover:text-black">
                  API
                </a>
              </li>
            </ul>
          </div>

          <div>
            <h4 className="mb-3 text-sm font-semibold">Company</h4>
            <ul className="space-y-2 text-sm text-neutral-600">
              <li>
                <a href="#" className="hover:text-black">
                  About
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-black">
                  Jobs
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-black">
                  Contact
                </a>
              </li>
            </ul>
          </div>

          <div>
            <h4 className="mb-3 text-sm font-semibold">Legal</h4>
            <ul className="space-y-2 text-sm text-neutral-600">
              <li>
                <a href="#" className="hover:text-black">
                  Privacy
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-black">
                  Terms
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-black">
                  Security
                </a>
              </li>
            </ul>
          </div>
        </div>

        <p className="mt-10 text-center text-xs text-neutral-500">
          © 2025 EvalX. All rights reserved.
        </p>
      </footer>
    </div>
  );
}

export default Hero;
