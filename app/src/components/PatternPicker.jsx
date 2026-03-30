import useStore from '../store'
import { isLoaded, getTestPrompts, formatPromptMetrics } from '../data/loadArtifacts'

/**
 * Dropdown selector for the 20 precomputed test prompts.
 * When real training data is available, users can switch between test prompts
 * to see precomputed outputs across all model variants. Zero latency — just array lookup.
 *
 * Props:
 *  - onChange(promptId) — optional callback when selection changes
 *  - compact — smaller variant for embedding in demo tabs
 */
export default function PatternPicker({ onChange, compact = false }) {
  const selectedPromptId = useStore((s) => s.selectedPromptId)
  const setSelectedPromptId = useStore((s) => s.setSelectedPromptId)

  if (!isLoaded()) return null

  const testPrompts = getTestPrompts()
  if (!testPrompts.length) return null

  const handleChange = (e) => {
    const id = parseInt(e.target.value, 10)
    setSelectedPromptId(id)
    onChange?.(id)
  }

  const selected = testPrompts.find((p) => p.id === selectedPromptId) ?? testPrompts[0]

  return (
    <div className={compact ? 'mb-3' : 'mb-4'}>
      <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-1.5">
        Test Pattern
      </label>
      <div className="flex items-center gap-2">
        <select
          value={selectedPromptId}
          onChange={handleChange}
          className="bg-slate-800 border border-slate-600 text-slate-200 text-sm rounded-md px-3 py-1.5 focus:outline-none focus:border-cyan-600 w-full max-w-md"
        >
          {testPrompts.map((p) => (
            <option key={p.id} value={p.id}>
              #{p.id + 1} — {p.true_label}
            </option>
          ))}
        </select>
      </div>

      {/* Show selected pattern metrics */}
      {!compact && selected && (
        <div className="mt-2 bg-slate-800 border border-slate-700/50 rounded-lg p-2.5 font-mono text-xs text-slate-300">
          {formatPromptMetrics(selected)}
        </div>
      )}
    </div>
  )
}
