import { useEffect, useState, useCallback } from "react";
import { FiCheckCircle } from "react-icons/fi";
import api from "@/utils/api";
import ReactMarkdown from "react-markdown";

export default function TeamSubmissions({ eventId }) {
  const [subs, setSubs] = useState({});
  const [pptFile, setPptFile] = useState(null);
  const [repo, setRepo] = useState("");
  const [video, setVideo] = useState("");
  const [loading, setLoading] = useState({ ppt: false, repo: false });
  const [showSlides, setShowSlides] = useState(false);

  const loadSubs = useCallback(async () => {
    try {
      const res = await api.get(`/team/events/${eventId}/my-submissions`, {
        withCredentials: true,
      });
      setSubs(res.data.data || {});
      console.log("first")
    } catch (err) {
      console.log("SUBS_FETCH_ERROR", err);
      console.log("second")
    }
  }, [eventId]);

  useEffect(() => {
    console.log("dekh ra")
    loadSubs();
  }, [loadSubs]);

  console.log("hp")

  const submitForm = async (route, key, value, resetFn) => {
    try {
      setLoading((p) => ({ ...p, [key]: true }));
      const fd = new FormData();

      if (key === "ppt") fd.append("file", value);
      else {
        fd.append("repo", repo);
        fd.append("video", video);
      }

      await api.post(route, fd, { withCredentials: true });
      await loadSubs();
      resetFn(null);
    } catch (err) {
      console.log(`${key.toUpperCase()}_SUBMIT_ERROR`, err);
    } finally {
      setLoading((p) => ({ ...p, [key]: false }));
    }
  };

  const pptSubmitted = !!subs.ppt;
  const repoSubmitted = !!subs.repo;

  const ppt = subs.ppt;
  const ai = ppt?.aiResult;

  return (
    <div className="space-y-8">

      {/* ---------------- PPT SUBMISSION / RESULTS ---------------- */}
      <section className="border rounded-xl p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold">Round 1: PPT Submission</h3>

          {pptSubmitted && (
            <div className="flex items-center gap-1 text-xs text-emerald-600">
              <FiCheckCircle size={14} />
              <span>Submitted</span>
            </div>
          )}
        </div>

        {/* PPT ALREADY SUBMITTED → SHOW AI RESULTS */}
        {pptSubmitted ? (
          <div className="space-y-4">

            {/* Overall Score */}
            <div className="p-3 bg-neutral-100 rounded-lg border">
              <h4 className="text-sm font-semibold mb-1">Overall Score</h4>
              <p className="text-xl font-bold text-neutral-800">
                {ai?.score?.overall_score?.toFixed(2) ?? "N/A"} / 100
              </p>
            </div>

            {/* Mentor Summary (Markdown) */}
            {ai?.mentor_summary && (
              <div className="p-3 bg-neutral-100 rounded-lg border">
                <h4 className="text-sm font-semibold mb-2">Summary</h4>
                <div className="prose prose-sm max-w-none text-neutral-800">
                  <ReactMarkdown>{ai.mentor_summary}</ReactMarkdown>
                </div>
              </div>
            )}

            {/* Slide Breakdown */}
            {ai?.slides && (
              <div className="p-3 bg-neutral-100 rounded-lg border">
                <button
                  className="text-sm font-medium underline"
                  onClick={() => setShowSlides((s) => !s)}
                >
                  {showSlides ? "Hide Slide Breakdown" : "Show Slide Breakdown"}
                </button>

                {showSlides && (
                  <div className="mt-3 space-y-3">
                    {ai.slides.map((slide) => (
                      <div
                        key={slide.slide_number}
                        className="border p-3 rounded-lg bg-white"
                      >
                        <h4 className="text-sm font-semibold mb-1">
                          Slide {slide.slide_number}
                        </h4>

                        <p className="text-xs text-neutral-700 mb-1">
                          Score:{" "}
                          <span className="font-semibold">
                            {slide.score?.toFixed(2)}
                          </span>
                        </p>

                        <p className="text-xs text-neutral-600 mb-1">
                          Missing: {slide.analysis?.missing_elements?.join(", ") || "None"}
                        </p>

                        <p className="text-xs text-neutral-600">
                          Suggestions:
                          <br />
                          {slide.analysis?.suggestions?.length > 0
                            ? slide.analysis.suggestions.map((s, i) => (
                                <li key={i} className="ml-4 list-disc">
                                  {s}
                                </li>
                              ))
                            : "No suggestions"}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        ) : (
          // ---------------- PPT UPLOAD FORM ----------------
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (!pptFile) return;
              submitForm(
                `/events/${eventId}/submit/ppt`,
                "ppt",
                pptFile,
                setPptFile
              );
            }}
            className="flex items-center gap-3 text-xs"
          >
            <input
              type="file"
              accept=".ppt,.pptx,.pdf"
              onChange={(e) => setPptFile(e.target.files[0] || null)}
              className="text-xs"
            />
            <button
              type="submit"
              disabled={!pptFile || loading.ppt}
              className="px-3 py-1 rounded-lg border text-xs bg-black text-white disabled:bg-neutral-300"
            >
              {loading.ppt ? "Uploading..." : "Submit PPT"}
            </button>
          </form>
        )}
      </section>

      {/* ---------------- REPO + VIDEO ---------------- */}
      <section className="border rounded-xl p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold">Round 2: Repo & Video</h3>

          {repoSubmitted && (
            <div className="flex items-center gap-1 text-xs text-emerald-600">
              <FiCheckCircle size={14} />
              <span>Submitted</span>
            </div>
          )}
        </div>

        {repoSubmitted ? (
          <p className="text-xs text-neutral-600">
            Repo & video already submitted. Upload disabled.
          </p>
        ) : (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (!repo || !video) return;
              submitForm(
                `/events/${eventId}/submit/repo`,
                "repo",
                { repo, video },
                () => {
                  setRepo("");
                  setVideo("");
                }
              );
            }}
            className="space-y-3 text-xs"
          >
            <div className="flex flex-col gap-1">
              <label className="font-medium">GitHub Repository URL</label>
              <input
                type="url"
                value={repo}
                onChange={(e) => setRepo(e.target.value)}
                className="border rounded-lg px-2 py-1 text-xs"
                placeholder="https://github.com/username/repo"
              />
            </div>

            <div className="flex flex-col gap-1">
              <label className="font-medium">Demo Video URL</label>
              <input
                type="url"
                value={video}
                onChange={(e) => setVideo(e.target.value)}
                className="border rounded-lg px-2 py-1 text-xs"
                placeholder="https://www.youtube.com/..."
              />
            </div>

            <button
              type="submit"
              disabled={!repo || !video || loading.repo}
              className="px-3 py-1 rounded-lg border text-xs bg-black text-white disabled:bg-neutral-300"
            >
              {loading.repo ? "Submitting..." : "Submit Repo & Video"}
            </button>
          </form>
        )}
      </section>
    </div>
  );
}
