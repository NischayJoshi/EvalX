import { useEffect, useState, useRef, useCallback } from "react";
import { useLocation, useParams, useNavigate } from "react-router-dom";
import { FiMic, FiSquare, FiCheckCircle } from "react-icons/fi";
import { motion, AnimatePresence } from "framer-motion";
import api from "@/utils/api";

export default function InterviewRoom() {
  const { sessionId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();

  const initial = location.state || {};
  const [question, setQuestion] = useState(initial.firstQuestion || "");
  const [questionIndex, setQuestionIndex] = useState(initial.questionIndex || 0);
  const [totalQuestions, setTotalQuestions] = useState(initial.totalQuestions || 5);

  const [recording, setRecording] = useState(false);
  const [loading, setLoading] = useState(false);

  const [transcript, setTranscript] = useState("");
  const [feedback, setFeedback] = useState("");
  const [lastScore, setLastScore] = useState(null);
  const [totalScore, setTotalScore] = useState(0);
  const [answeredCount, setAnsweredCount] = useState(0);
  const [done, setDone] = useState(false);
  const [feedbackAudioUrl, setFeedbackAudioUrl] = useState(null);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const streamRef = useRef(null);
  const canvasRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const sourceRef = useRef(null);
  const animationRef = useRef(null);

  const eventId = initial.eventId;
  const teamId = initial.teamId;
  const teamName = initial.teamName;

  const clearWaveform = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  const stopVisualization = () => {
    if (animationRef.current) cancelAnimationFrame(animationRef.current);
    animationRef.current = null;
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    analyserRef.current = null;
    sourceRef.current = null;
    clearWaveform();
  };

  const startVisualization = (stream) => {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 128;
    const source = audioCtx.createMediaStreamSource(stream);
    source.connect(analyser);

    audioContextRef.current = audioCtx;
    analyserRef.current = analyser;
    sourceRef.current = source;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const draw = () => {
      animationRef.current = requestAnimationFrame(draw);
      analyser.getByteFrequencyData(dataArray);

      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);

      const barWidth = (width / bufferLength) * 1.5;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 255;
        const barHeight = v * height;
        const opacity = 0.25 + v * 0.6;
        ctx.fillStyle = `rgba(15,23,42,${opacity})`;
        if (ctx.roundRect) {
          ctx.roundRect(x, height - barHeight, barWidth, barHeight, 4);
        } else {
          ctx.fillRect(x, height - barHeight, barWidth, barHeight);
        }
        ctx.fill();
        x += barWidth + 1;
      }
    };

    draw();
  };

  const fetchSession = useCallback(async () => {
    if (!sessionId) return;
    try {
      const res = await api.get(`/interview/session/${sessionId}`);
      const data = res.data;
      setQuestion(data.question || "");
      setQuestionIndex(data.questionIndex || 0);
      setTotalQuestions(data.totalQuestions || 5);
      setAnsweredCount(data.answeredCount || 0);
      setTotalScore(data.totalScore || 0);
      setDone(data.done || false);
    } catch {
      navigate(-1);
    }
  }, [sessionId, navigate]);

  useEffect(() => {
    if (!question && sessionId) {
      fetchSession();
    }
  }, [question, sessionId, fetchSession]);

  const fetchTts = async (text) => {
    try {
      const res = await api.post(
        "/interview/tts",
        { text },
        { responseType: "blob" }
      );
      const blob = res.data;
      return URL.createObjectURL(blob);
    } catch {
      return null;
    }
  };

  useEffect(() => {
    let audio;
    (async () => {
      if (!question || done) return;
      const textForTts = `Question ${questionIndex + 1}. ${question}`;
      const url = await fetchTts(textForTts);
      if (!url) return;
      audio = new Audio(url);
      audio.play().catch(() => {});
    })();
    return () => {
      if (audio) {
        audio.pause();
        audio = null;
      }
    };
  }, [question, questionIndex, done]);

  const cleanupMedia = () => {
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
    } catch {}
    stopVisualization();
    setRecording(false);
  };

  useEffect(() => {
    return () => {
      cleanupMedia();
    };
  }, []);

  const sendAudio = async (blob) => {
    if (!sessionId) return;
    setLoading(true);
    try {
      const fd = new FormData();
      fd.append("file", blob);
      fd.append("sessionId", sessionId);
      fd.append("questionIndex", String(questionIndex));
      if (eventId) fd.append("eventId", eventId);
      if (teamId) fd.append("teamId", teamId);
      if (teamName) fd.append("teamName", teamName);

      const res = await api.post("/interview/answer-audio", fd, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const data = res.data;

      setTranscript(data.transcript || "");
      setFeedback(data.feedback || "");
      setLastScore(data.score ?? null);
      setTotalScore(data.totalScore || 0);
      setAnsweredCount(data.answeredCount || 0);
      setDone(data.done || false);
      setTotalQuestions(data.totalQuestions || totalQuestions);

      if (data.feedback) {
        const speakText = `Score ${data.score} out of 10. ${data.feedback}`;
        const url = await fetchTts(speakText);
        // if (url) {
        //   setFeedbackAudioUrl(url);
        //   const a = new Audio(url);
        //   a.play().catch(() => {});
        // }
      }

      if (data.nextQuestion) {
        setQuestion(data.nextQuestion);
        setQuestionIndex(data.nextIndex || questionIndex + 1);
      } else {
        setQuestion("");
      }
    } catch (e) {
      alert(e?.response?.data?.detail || "Audio processing failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleToggleRecord = async () => {
    if (done || loading) return;

    if (recording) {
      try {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
          mediaRecorderRef.current.stop();
        }
      } catch {}
      setRecording(false);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        sendAudio(blob)
          .catch(() => {})
          .finally(() => {
            cleanupMedia();
          });
      };

      startVisualization(stream);
      recorder.start();
      setRecording(true);
      setTranscript("");
      setFeedback("");
      setLastScore(null);
    } catch {
      alert("Mic access denied.");
    }
  };

  const handleExit = () => {
    cleanupMedia();
    navigate(-1);
  };

  const progressPercent =
    totalQuestions > 0 ? Math.min(100, (answeredCount / totalQuestions) * 100) : 0;

  return (
    <div className="fixed inset-0 bg-neutral-950/85 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="w-full max-w-5xl h-[85vh] bg-white rounded-3xl shadow-[0_24px_80px_rgba(0,0,0,0.45)] flex overflow-hidden border border-neutral-200/70">
        <div className="w-2/3 p-8 md:p-10 flex flex-col relative">
          <div className="flex items-center justify-between pb-6 border-b border-neutral-200/70">
            <div>
              <p className="text-[11px] uppercase tracking-[0.2em] text-neutral-400 mb-1">
                Interview Session
              </p>
              <h2 className="text-2xl md:text-3xl font-semibold text-neutral-900">
                AI Technical Viva
              </h2>
              <p className="text-[11px] text-neutral-500 mt-1">
                Question {Math.min(questionIndex + 1, totalQuestions)} of {totalQuestions}
              </p>
            </div>

            <div className="flex flex-col items-end gap-1">
              {teamName ? (
                <span className="text-[11px] font-medium text-neutral-600">
                  Team: <span className="text-neutral-900">{teamName}</span>
                </span>
              ) : null}
              <button
                onClick={handleExit}
                className="text-[11px] px-3 py-1.5 rounded-full border border-neutral-300 text-neutral-600 hover:bg-neutral-50 transition"
              >
                Exit
              </button>
            </div>
          </div>

          <div className="pt-8 flex-1 flex flex-col gap-8">
            <div>
              {!done ? (
                <>
                  <p className="text-[11px] font-medium text-neutral-500 mb-2 tracking-[0.18em] uppercase">
                    Question
                  </p>
                  <p className="text-lg md:text-xl font-medium text-neutral-900 leading-relaxed">
                    {question || "Loading question..."}
                  </p>
                </>
              ) : (
                <div className="flex items-center justify-between gap-4">
                  <div className="flex items-center gap-3">
                    <div className="flex items-center justify-center w-9 h-9 rounded-full bg-emerald-50">
                      <FiCheckCircle className="w-5 h-5 text-emerald-600" />
                    </div>
                    <div>
                      <p className="text-base font-semibold text-neutral-900">
                        Interview completed
                      </p>
                      <p className="text-xs text-neutral-600">
                        You answered {answeredCount} questions.
                      </p>
                    </div>
                  </div>

                  <div className="text-right">
                    <p className="text-[11px] text-neutral-500 mb-1">Final Score</p>
                    <p className="text-3xl font-semibold text-emerald-700">
                      {totalScore}
                      <span className="text-[11px] text-neutral-400">
                        {" "}
                        / {totalQuestions * 10}
                      </span>
                    </p>
                  </div>
                </div>
              )}
            </div>

            <div className="flex flex-col gap-3">
              <p className="text-[11px] font-medium text-neutral-500 tracking-[0.18em] uppercase">
                Transcript
              </p>
              <div className="min-h-[96px] bg-neutral-50 border border-neutral-200 rounded-2xl p-4 text-sm text-neutral-800 leading-relaxed">
                {loading ? (
                  <span className="text-xs text-neutral-400">
                    Processing your answer…
                  </span>
                ) : transcript ? (
                  <p>{transcript}</p>
                ) : (
                  <span className="text-xs text-neutral-400">
                    Press the mic and speak your answer clearly.
                  </span>
                )}
              </div>

              <div className="flex items-center justify-between mt-1">
                <div>
                  {lastScore !== null && !done ? (
                    <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full bg-emerald-50 text-emerald-700 text-[11px] font-medium">
                      Score: {lastScore}/10
                    </span>
                  ) : null}
                </div>

                {done ? (
                  <button
                    onClick={handleExit}
                    className="text-[11px] px-3 py-1.5 rounded-full bg-neutral-900 text-white hover:bg-black transition"
                  >
                    Close session
                  </button>
                ) : null}
              </div>
            </div>
          </div>

          {feedback ? (
            <div className="mt-4 rounded-2xl border border-amber-100 bg-amber-50/80 p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-[11px] font-medium text-amber-800 mb-1 tracking-[0.16em] uppercase">
                    Feedback
                  </p>
                  <p className="text-xs text-amber-900 leading-relaxed">{feedback}</p>
                </div>
                {feedbackAudioUrl ? (
                  <button
                    onClick={() => {
                      const a = new Audio(feedbackAudioUrl);
                      a.play().catch(() => {});
                    }}
                    className="text-[11px] px-3 py-1.5 rounded-full border border-amber-200 text-amber-800 hover:bg-amber-100 transition"
                  >
                    🔊 Hear again
                  </button>
                ) : null}
              </div>
            </div>
          ) : null}
        </div>

        <div className="w-1/3 relative bg-gradient-to-b from-neutral-900 via-neutral-950 to-black text-white p-6 md:p-8 flex flex-col items-center justify-between">
          <div className="absolute inset-0 opacity-40 pointer-events-none">
            <div className="absolute -top-10 -right-10 w-40 h-40 bg-emerald-500/20 blur-3xl" />
            <div className="absolute bottom-0 -left-10 w-52 h-52 bg-sky-500/20 blur-3xl" />
          </div>

          <div className="relative w-full z-10">
            <div className="flex items-center justify-between mb-3">
              <p className="text-[11px] font-medium tracking-[0.18em] uppercase text-neutral-400">
                Progress
              </p>
              <p className="text-[11px] text-neutral-400">
                {answeredCount}/{totalQuestions} answered
              </p>
            </div>

            <div className="w-full h-1.5 bg-neutral-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-emerald-400 rounded-full transition-all duration-300"
                style={{ width: `${progressPercent}%` }}
              />
            </div>

            <div className="mt-4 flex items-center justify-between text-[11px] text-neutral-400">
              <span>Total Score</span>
              <span>
                {totalScore} / {totalQuestions * 10}
              </span>
            </div>
          </div>

          <div className="relative flex flex-col items-center gap-4 z-10">
            <div className="relative flex items-center justify-center">
              <AnimatePresence>
                {recording && (
                  <motion.div
                    className="absolute w-32 h-32 rounded-full border border-emerald-400/60"
                    initial={{ scale: 0.9, opacity: 0.3 }}
                    animate={{ scale: 1.35, opacity: 0 }}
                    exit={{ opacity: 0 }}
                    transition={{
                      duration: 1.1,
                      repeat: Infinity,
                      ease: "easeOut",
                    }}
                  />
                )}
              </AnimatePresence>

              <button
                onClick={handleToggleRecord}
                disabled={loading}
                className={`relative w-24 h-24 md:w-28 md:h-28 rounded-full flex items-center justify-center shadow-[0_20px_60px_rgba(0,0,0,0.7)] transition 
                  ${recording ? "bg-red-500 hover:bg-red-600" : "bg-white hover:bg-neutral-100"} 
                  disabled:opacity-60`}
              >
                {recording ? (
                  <FiSquare className="w-8 h-8 text-white" />
                ) : (
                  <FiMic className="w-8 h-8 text-neutral-900" />
                )}
              </button>
            </div>

            <p className="text-[11px] text-neutral-300 text-center max-w-xs">
              {recording
                ? "Recording… tap to stop and let AI evaluate your answer."
                : "Tap the mic, answer naturally. AI will transcribe, score and speak feedback."}
            </p>
          </div>

          <div className="relative w-full z-10">
            <p className="text-[11px] font-medium tracking-[0.18em] uppercase text-neutral-400 mb-2">
              Live waveform
            </p>
            <div className="w-full h-24 rounded-2xl bg-neutral-900/80 border border-neutral-800/80 overflow-hidden flex items-center">
              <canvas ref={canvasRef} className="w-full h-full" width={400} height={96} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
