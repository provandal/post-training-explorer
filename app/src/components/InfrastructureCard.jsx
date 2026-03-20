// Infrastructure resource profile card shown at each post-training stop
// Connects ML technique to infrastructure implications for the SNIA audience

export default function InfrastructureCard({ data, comparison = null }) {
  if (!data) return null

  const fields = [
    { key: 'gpuMemoryGB', label: 'GPU Memory', format: v => typeof v === 'number' ? `${v} GB` : v, icon: '&#x1F4BB;' },
    { key: 'trainingTimeMinutes', label: 'Training Time', format: v => typeof v === 'number' ? `${v} min` : v, icon: '&#x23F1;' },
    { key: 'checkpointSizeMB', label: 'Checkpoint Size', format: v => typeof v === 'number' ? `${v} MB` : v, icon: '&#x1F4BE;' },
    { key: 'peakGPUUtilization', label: 'Peak GPU Util', format: v => typeof v === 'number' ? `${v}%` : v, icon: '&#x1F525;' },
    { key: 'modelsInMemory', label: 'Models in Memory', format: v => `${v}`, icon: '&#x1F9E0;' },
    { key: 'storageIOPattern', label: 'Storage I/O', format: v => v, icon: '&#x1F4CA;' },
  ]

  return (
    <div className="border border-slate-700/50 rounded-lg bg-slate-800/30 p-4">
      <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-400 mb-3">
        Infrastructure Profile
      </h4>

      {/* Main metrics */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        {fields.map(({ key, label, format }) => {
          if (data[key] === undefined) return null
          const val = format(data[key])
          const compVal = comparison && comparison[key] !== undefined ? format(comparison[key]) : null

          return (
            <div key={key} className="flex flex-col">
              <span className="text-xs text-slate-500">{label}</span>
              <div className="flex items-baseline gap-2">
                <span className="text-sm font-semibold text-slate-200">{val}</span>
                {compVal && (
                  <span className="text-xs text-slate-500 line-through">{compVal}</span>
                )}
              </div>
            </div>
          )
        })}
      </div>

      {/* Notes */}
      {data.note && (
        <p className="text-xs text-slate-400 italic border-t border-slate-700/50 pt-2 mt-2">
          {data.note}
        </p>
      )}

      {/* RLHF comparison callout (for DPO) */}
      {data.vsRLHF && (
        <div className="mt-3 p-3 rounded bg-red-950/20 border border-red-800/30">
          <p className="text-xs font-semibold text-red-400 mb-1">vs. RLHF (PPO)</p>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-slate-500">GPU Memory: </span>
              <span className="text-red-300">{data.vsRLHF.rlhfGPUMemoryGB} GB</span>
            </div>
            <div>
              <span className="text-slate-500">Training Time: </span>
              <span className="text-red-300">{data.vsRLHF.rlhfTrainingTimeMinutes} min</span>
            </div>
            <div>
              <span className="text-slate-500">Models in Memory: </span>
              <span className="text-red-300">{data.vsRLHF.rlhfModelsInMemory}</span>
            </div>
            <div>
              <span className="text-slate-500">DPO Models: </span>
              <span className="text-green-400">{data.vsRLHF.dpoModelsInMemory}</span>
            </div>
          </div>
          <p className="text-xs text-slate-500 mt-1">{data.vsRLHF.note}</p>
        </div>
      )}

      {/* Scaling note */}
      <p className="text-xs text-slate-600 mt-2">
        Numbers shown for SmolLM2-360M. For 7B models, multiply GPU memory ~20x and training time ~15x.
      </p>
    </div>
  )
}
