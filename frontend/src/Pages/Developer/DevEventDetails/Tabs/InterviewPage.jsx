import { useState, useRef, useEffect } from "react";

const API_BASE = "http://localhost:8000/api/interview";

export default function InterviewPage() {
  const [file, setFile] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [phase, setPhase] = useState("setup");
  const [introAnswer, setIntroAnswer] = useState("");
  const [projectAnswer, setProjectAnswer] = useState("");
  const [pdfQuestions, setPdfQuestions] = useState([]);
  const [pdfAnswers, setPdfAnswers] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [score, setScore] = useState(null);
  const [detail, setDetail] = useState("");
  const containerRef = useRef(null);

  const allQuestions = [
    "Please introduce yourself.",
    "Tell me about your project."
  ].concat(pdfQuestions);

  useEffect(() => {
    function handleFullscreenChange() {
      if (!document.fullscreenElement && phase === "interview") {
        setPhase("finished");
      }
    }
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, [phase]);

  function handleFileChange(e) {
    const f = e.target.files[0];
    if (f && f.type === "application/pdf") {
      setFile(f);
    } else {
      setFile(null);
      e.target.value = "";
    }
  }

  async function handleStart() {
    if (!file) return;
    try {
      setLoading(true);
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`${API_BASE}/start`, {
        method: "POST",
        body: formData
      });
      if (!res.ok) {
        const err = await res.json().catch(() => null);
        throw new Error(err?.detail || "Failed to start interview");
      }
      const data = await res.json();
      setSessionId(data.session_id);
      setPdfQuestions(data.pdf_questions || []);
      setPdfAnswers(new Array((data.pdf_questions || []).length).fill(""));
      setCurrentIndex(0);
      if (containerRef.current && containerRef.current.requestFullscreen) {
        await containerRef.current.requestFullscreen();
      }
      setPhase("interview");
    } catch (err) {
      console.error(err);
      alert(err.message);
      setLoading(false);
      return;
    }
    setLoading(false);
  }

  function getCurrentQuestion() {
    return allQuestions[currentIndex] || "";
  }

  function getCurrentAnswer() {
    if (currentIndex === 0) return introAnswer;
    if (currentIndex === 1) return projectAnswer;
    const idx = currentIndex - 2;
    return pdfAnswers[idx] || "";
  }

  function setCurrentAnswer(val) {
    if (currentIndex === 0) {
      setIntroAnswer(val);
    } else if (currentIndex === 1) {
      setProjectAnswer(val);
    } else {
      const idx = currentIndex - 2;
      const next = [...pdfAnswers];
      next[idx] = val;
      setPdfAnswers(next);
    }
  }

  function handleNext() {
    if (currentIndex < allQuestions.length - 1) {
      setCurrentIndex(currentIndex + 1);
    }
  }

  async function handleEnd() {
    if (!sessionId) {
      if (document.fullscreenElement && document.exitFullscreen) {
        await document.exitFullscreen();
      }
      setPhase("finished");
      return;
    }
    try {
      setLoading(true);
      const payload = {
        session_id: sessionId,
        intro_answer: introAnswer,
        project_answer: projectAnswer,
        pdf_answers: pdfAnswers
      };
      const res = await fetch(`${API_BASE}/score`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!res.ok) {
        const err = await res.json().catch(() => null);
        throw new Error(err?.detail || "Failed to score interview");
      }
      const data = await res.json();
      setScore(data.score);
      setDetail(data.detail);
    } catch (err) {
      console.error(err);
      alert(err.message);
    }
    if (document.fullscreenElement && document.exitFullscreen) {
      await document.exitFullscreen();
    }
    setPhase("finished");
    setLoading(false);
  }

  if (phase === "finished") {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-950 text-slate-50">
        <div className="w-full max-w-md border border-slate-800 rounded-xl p-6 bg-slate-900">
          <h1 className="text-2xl font-semibold mb-4">Interview Result</h1>
          {score !== null && (
            <p className="text-xl mb-2">Score: <span className="font-bold">{score}/100</span></p>
          )}
          <p className="text-sm text-slate-300 mb-6">{detail}</p>
          <button
            className="w-full py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-sm font-medium"
            onClick={() => {
              setPhase("setup");
              setFile(null);
              setSessionId(null);
              setIntroAnswer("");
              setProjectAnswer("");
              setPdfQuestions([]);
              setPdfAnswers([]);
              setCurrentIndex(0);
              setScore(null);
              setDetail("");
            }}
          >
            Start New Interview
          </button>
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="min-h-screen flex items-center justify-center bg-slate-950 text-slate-50">
      {phase === "setup" && (
        <div className="w-full max-w-lg border border-slate-800 rounded-xl p-6 bg-slate-900">
          <h1 className="text-2xl font-semibold mb-4">AI Interview</h1>
          <p className="text-sm text-slate-300 mb-4">
            Upload your project report PDF and start the interview. Once started, the screen will go fullscreen like an exam until you end the interview.
          </p>
          <div className="mb-4">
            <label className="block text-sm mb-1">Project Report (PDF)</label>
            <input
              type="file"
              accept="application/pdf"
              onChange={handleFileChange}
              className="w-full text-sm bg-slate-800 border border-slate-700 rounded-md p-2"
            />
          </div>
          <button
            disabled={!file || loading}
            onClick={handleStart}
            className={`w-full py-2 rounded-lg text-sm font-medium ${
              !file || loading
                ? "bg-slate-700 text-slate-400 cursor-not-allowed"
                : "bg-emerald-600 hover:bg-emerald-500"
            }`}
          >
            {loading ? "Starting..." : "Start Interview"}
          </button>
        </div>
      )}
      {phase === "interview" && (
        <div className="w-full h-screen flex flex-col bg-slate-950 px-6 py-4">
          <div className="flex justify-between items-center mb-4">
            <div>
              <h1 className="text-xl font-semibold">AI Interview</h1>
              <p className="text-xs text-slate-400">
                Question {currentIndex + 1} of {allQuestions.length}
              </p>
            </div>
            <button
              onClick={handleEnd}
              disabled={loading}
              className="px-4 py-2 rounded-lg bg-red-600 hover:bg-red-500 text-sm font-medium"
            >
              {loading ? "Ending..." : "End Interview"}
            </button>
          </div>
          <div className="flex-1 flex flex-col">
            <div className="mb-4">
              <div className="text-sm text-slate-400 mb-1">Question</div>
              <div className="text-lg font-medium">{getCurrentQuestion()}</div>
            </div>
            <textarea
              className="flex-1 w-full text-sm bg-slate-900 border border-slate-700 rounded-md p-3 resize-none focus:outline-none focus:border-indigo-500"
              value={getCurrentAnswer()}
              onChange={e => setCurrentAnswer(e.target.value)}
              placeholder="Type your answer here..."
            />
            <div className="mt-4 flex justify-end">
              {currentIndex < allQuestions.length - 1 && (
                <button
                  onClick={handleNext}
                  className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-sm font-medium"
                >
                  Next Question
                </button>
              )}
              {currentIndex === allQuestions.length - 1 && (
                <button
                  onClick={handleEnd}
                  className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-sm font-medium"
                >
                  Submit And End
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
