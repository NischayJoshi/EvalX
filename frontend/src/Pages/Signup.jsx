import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { FiGithub } from "react-icons/fi";
import { FcGoogle } from "react-icons/fc";
import api from "@/utils/api";

export default function Signup() {
  const [step, setStep] = useState(1);
  const [loading, setLoading] = useState(false);

  const [errorMsg, setError] = useState("");
  const [successMsg, setSuccess] = useState("");

  const [form, setForm] = useState({
    firstName: "",
    lastName: "",
    email: "",
    phone: "",
    date_of_birth: "",
    gender: "",
    role: "",
    password: "",
    agreement: false,
  });

  const update = (field, value) => setForm({ ...form, [field]: value });

  const next = () => {
    setError("");
    // validation per step
    if (step === 1 && (!form.firstName || !form.lastName)) {
      setError("Please enter your full name.");
      return;
    }
    if (step === 2 && (!form.email || !form.phone)) {
      setError("Email & phone number are required.");
      return;
    }
    if (step === 3 && (!form.gender || !form.role || !form.date_of_birth)) {
      setError("Please complete all fields.");
      return;
    }
    if (step === 4 && form.password.length < 6) {
      setError("Password must be at least 6 characters.");
      return;
    }
    setStep(step + 1);
  };

  const prev = () => {
    setError("");
    setStep(step - 1);
  };

  const slide = {
    initial: { opacity: 0, x: 40 },
    animate: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: -40 },
    transition: { duration: 0.35 }
  };

  // FINAL SUBMIT
  const handleSubmit = async () => {
    setError("");
    setSuccess("");

    if (!form.agreement) {
      setError("You must accept the terms.");
      return;
    }

    setLoading(true);

    const payload = {
      username: `${form.firstName}${form.lastName}`.toLowerCase(),
      firstName: form.firstName,
      lastName: form.lastName,
      email: form.email,
      phone: form.phone,
      date_of_birth: form.date_of_birth,
      gender: form.gender,
      role: form.role,
      password: form.password,
    };

    try {
      const res = await api.post("/auth/signup", payload);

      if (res.data?.access_token) {
        localStorage.setItem("token", res.data.access_token);
      }

      setSuccess("Account created successfully! Redirecting...");

      setLoading(false);

      setTimeout(() => {
        window.location.href = "/dashboard";
      }, 900);

    } catch (err) {
      const msg = err?.response?.data?.detail || "Signup failed.";
      setError(msg);
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-neutral-50 flex items-center justify-center px-4">

      <motion.div
        initial={{ opacity: 0, y: 25 }}
        animate={{ opacity: 1, y: 0 }}
        className="
          w-full max-w-lg
          rounded-3xl
          px-10 py-10
          bg-white
          border border-neutral-100
          shadow-[0_2px_12px_rgba(0,0,0,0.04),0_24px_48px_rgba(0,0,0,0.06)]
        "
      >

        {/* HEADER */}
        <div className="text-center mb-8">
          <h2 className="text-3xl font-semibold tracking-tight">
            Create Your EvalX Account
          </h2>

          <div className="flex justify-center gap-2 mt-3">
            {[1, 2, 3, 4, 5].map(i => (
              <div
                key={i}
                className={`h-1.5 w-1.5 rounded-full transition-all ${
                  i <= step ? "bg-neutral-900" : "bg-neutral-300"
                }`}
              ></div>
            ))}
          </div>

          <p className="text-neutral-500 text-sm mt-3">
            Step {step} of 5
          </p>
        </div>

        {/* ERROR */}
        {errorMsg && (
          <motion.div
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-4 p-3 text-sm rounded-xl bg-red-50 text-red-600 border border-red-200"
          >
            {errorMsg}
          </motion.div>
        )}

        {/* SUCCESS */}
        {successMsg && (
          <motion.div
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-4 p-3 text-sm rounded-xl bg-green-50 text-green-600 border border-green-200"
          >
            {successMsg}
          </motion.div>
        )}

        {/* FORM STEPS */}
        <AnimatePresence mode="wait">
          
          {/* STEP 1 */}
          {step === 1 && (
            <motion.div key="s1" {...slide}>
              <div className="grid grid-cols-2 gap-5">
                <div>
                  <label className="text-sm font-medium">First Name</label>
                  <Input
                    placeholder="Arju"
                    className="mt-1 rounded-xl"
                    value={form.firstName}
                    onChange={(e) => update("firstName", e.target.value)}
                  />
                </div>

                <div>
                  <label className="text-sm font-medium">Last Name</label>
                  <Input
                    placeholder="Srivastava"
                    className="mt-1 rounded-xl"
                    value={form.lastName}
                    onChange={(e) => update("lastName", e.target.value)}
                  />
                </div>
              </div>

              <Button
                className="w-full mt-8 bg-neutral-900 text-white rounded-xl py-3 text-sm"
                onClick={next}
              >
                Continue
              </Button>
            </motion.div>
          )}

          {/* STEP 2 */}
          {step === 2 && (
            <motion.div key="s2" {...slide}>
              <div>
                <label className="text-sm font-medium">Email</label>
                <Input
                  type="email"
                  placeholder="you@example.com"
                  className="mt-1 rounded-xl"
                  value={form.email}
                  onChange={(e) => update("email", e.target.value)}
                />
              </div>

              <div className="mt-4">
                <label className="text-sm font-medium">Phone</label>
                <Input
                  type="tel"
                  placeholder="+91 99999 99999"
                  className="mt-1 rounded-xl"
                  value={form.phone}
                  onChange={(e) => update("phone", e.target.value)}
                />
              </div>

              <div className="flex gap-4 mt-8">
                <Button
                  variant="outline"
                  className="w-1/2 rounded-xl border-neutral-300 py-3"
                  onClick={prev}
                >
                  Back
                </Button>

                <Button
                  className="w-1/2 bg-neutral-900 text-white rounded-xl py-3"
                  onClick={next}
                >
                  Continue
                </Button>
              </div>
            </motion.div>
          )}

          {/* STEP 3 */}
          {step === 3 && (
            <motion.div key="s3" {...slide}>
              
              <div>
                <label className="text-sm font-medium">Date of Birth</label>
                <Input
                  type="date"
                  className="mt-1 rounded-xl"
                  value={form.date_of_birth}
                  onChange={(e) => update("date_of_birth", e.target.value)}
                />
              </div>

              <div className="mt-4">
                <label className="text-sm font-medium">Gender</label>
                <div className="mt-2 flex items-center gap-6">
                  {["male", "female", "other"].map((g) => (
                    <label key={g} className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        name="gender"
                        checked={form.gender === g}
                        onChange={() => update("gender", g)}
                        className="h-4 w-4"
                      />
                      <span className="text-sm capitalize">{g}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="mt-6">
                <label className="text-sm font-medium">Role</label>

                <div className="mt-3 grid grid-cols-2 gap-4">
                  <div
                    onClick={() => update("role", "Developer")}
                    className={`
                      cursor-pointer rounded-xl border p-4 text-center transition 
                      ${form.role === "Developer" 
                        ? "border-neutral-900 bg-neutral-900 text-white" 
                        : "border-neutral-200 hover:bg-neutral-100"}
                    `}
                  >
                    Developer
                  </div>

                  <div
                    onClick={() => update("role", "Organiser")}
                    className={`
                      cursor-pointer rounded-xl border p-4 text-center transition 
                      ${form.role === "Organiser" 
                        ? "border-neutral-900 bg-neutral-900 text-white" 
                        : "border-neutral-200 hover:bg-neutral-100"}
                    `}
                  >
                    Organiser
                  </div>
                </div>
              </div>

              <div className="flex gap-4 mt-8">
                <Button variant="outline" className="w-1/2 rounded-xl" onClick={prev}>
                  Back
                </Button>
                <Button className="w-1/2 bg-neutral-900 text-white rounded-xl" onClick={next}>
                  Continue
                </Button>
              </div>

            </motion.div>
          )}

          {/* STEP 4 */}
          {step === 4 && (
            <motion.div key="s4" {...slide}>
              <label className="text-sm font-medium">Password</label>
              <Input
                type="password"
                placeholder="••••••••"
                className="mt-1 rounded-xl"
                value={form.password}
                onChange={(e) => update("password", e.target.value)}
              />

              <div className="flex gap-4 mt-8">
                <Button variant="outline" className="w-1/2 rounded-xl" onClick={prev}>
                  Back
                </Button>
                <Button className="w-1/2 bg-neutral-900 text-white rounded-xl" onClick={next}>
                  Continue
                </Button>
              </div>
            </motion.div>
          )}

          {/* STEP 5 */}
          {step === 5 && (
            <motion.div key="s5" {...slide}>
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={form.agreement}
                  onChange={(e) => update("agreement", e.target.checked)}
                  className="h-4 w-4"
                />
                <span className="text-sm text-neutral-700">
                  I agree to the Terms & Privacy Policy
                </span>
              </label>

              <Button
                disabled={!form.agreement || loading}
                onClick={handleSubmit}
                className="w-full mt-8 bg-neutral-900 text-white rounded-xl py-3 disabled:opacity-50"
              >
                {loading ? "Creating account..." : "Create Account"}
              </Button>

              <Button
                variant="outline"
                className="w-full mt-3 border-neutral-300 rounded-xl py-3"
                onClick={prev}
              >
                Back
              </Button>
            </motion.div>
          )}

        </AnimatePresence>

        {/* DIVIDER */}
        <div className="my-8 flex items-center">
          <div className="flex-1 h-[1px] bg-neutral-200"></div>
          <span className="px-3 text-neutral-500 text-sm">OR</span>
          <div className="flex-1 h-[1px] bg-neutral-200"></div>
        </div>

        {/* SOCIAL */}
        <div className="space-y-3">
          <Button variant="outline" className="w-full rounded-xl py-3 border-neutral-300 flex items-center justify-center gap-2">
            <FcGoogle size={20} /> Continue with Google
          </Button>

          <Button variant="outline" className="w-full rounded-xl py-3 border-neutral-300 flex items-center justify-center gap-2">
            <FiGithub size={20} /> Continue with GitHub
          </Button>
        </div>

        <p className="text-center text-sm mt-6 text-neutral-600">
          Already have an account?{" "}
          <a href="/login" className="font-medium hover:underline">
            Log In
          </a>
        </p>

      </motion.div>

    </div>
  );
}
