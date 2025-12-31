import { useState } from "react";
import { motion } from "framer-motion";

export default function CreateTeamModal({ onClose, onCreate }) {
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async () => {
    if (!name.trim()) return;
    try {
      setLoading(true);
      await onCreate(name.trim());
      setName("");
    } catch (err) {
      alert(err?.response?.data?.detail || "Failed to create team");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-40">
      <motion.div initial={{ scale: 0.98, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.98, opacity: 0 }} transition={{ duration: 0.12 }} className="bg-white rounded-xl p-6 w-full max-w-md">
        <h3 className="text-lg font-semibold mb-3">Create Team</h3>
        <input value={name} onChange={(e) => setName(e.target.value)} className="w-full rounded-xl border border-neutral-300 px-3 py-2 text-sm shadow-sm mb-4" placeholder="Team name" />
        <div className="flex justify-end gap-3">
          <button onClick={onClose} className="px-4 py-2 rounded-xl border">Cancel</button>
          <button onClick={submit} disabled={loading} className="px-4 py-2 rounded-xl bg-black text-white">{loading ? "Creating…" : "Create"}</button>
        </div>
      </motion.div>
    </div>
  );
}
