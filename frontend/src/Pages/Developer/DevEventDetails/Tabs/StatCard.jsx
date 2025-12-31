export default function StatCard({ label, value, icon }) {
  return (
    <div className="bg-white rounded-xl border border-neutral-200 p-4 shadow-sm">
      <div className="flex justify-between items-center">
        <p className="text-xs text-neutral-500">{label}</p>
        <div>{icon}</div>
      </div>
      <p className="mt-1 text-lg font-semibold">{value}</p>
    </div>
  );
}
