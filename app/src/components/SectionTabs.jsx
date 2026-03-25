// Reusable tab bar for tour stop sections
// Used at both top and bottom of each stop so users don't lose track of available tabs

export default function SectionTabs({ tabs, active, onSelect, color = 'orange' }) {
  const colorMap = {
    orange: 'bg-orange-600',
    yellow: 'bg-yellow-600',
    violet: 'bg-violet-600',
    pink: 'bg-pink-600',
    emerald: 'bg-emerald-600',
    cyan: 'bg-cyan-600',
  }
  const activeClass = colorMap[color] || colorMap.orange

  return (
    <div className="flex gap-1 bg-slate-800 rounded-lg p-1 w-fit">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onSelect(tab.id)}
          className={`px-4 py-2 text-sm rounded-md transition-colors ${
            active === tab.id
              ? `${activeClass} text-white font-semibold`
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  )
}
