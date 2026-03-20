// Renders a model's text output with a label and optional styling
// Used throughout the app for side-by-side comparisons

export default function ModelOutput({ label, text, variant = 'default', isCorrect = null, className = '' }) {
  const variantStyles = {
    default: 'border-slate-600 bg-slate-800/50',
    base: 'border-red-800/50 bg-red-950/20',
    sft: 'border-violet-800/50 bg-violet-950/20',
    dpo: 'border-pink-800/50 bg-pink-950/20',
    grpo: 'border-emerald-800/50 bg-emerald-950/20',
    chosen: 'border-green-600/50 bg-green-950/20',
    rejected: 'border-red-600/50 bg-red-950/20',
    rag: 'border-yellow-700/50 bg-yellow-950/20',
  }

  const labelColors = {
    default: 'text-slate-400',
    base: 'text-red-400',
    sft: 'text-violet-400',
    dpo: 'text-pink-400',
    grpo: 'text-emerald-400',
    chosen: 'text-green-400',
    rejected: 'text-red-400',
    rag: 'text-yellow-400',
  }

  return (
    <div className={`border rounded-lg p-4 ${variantStyles[variant]} ${className}`}>
      <div className="flex items-center justify-between mb-2">
        <span className={`text-xs font-semibold uppercase tracking-wide ${labelColors[variant]}`}>
          {label}
        </span>
        {isCorrect !== null && (
          <span className={`text-xs px-2 py-0.5 rounded-full ${isCorrect ? 'bg-green-900/50 text-green-400' : 'bg-red-900/50 text-red-400'}`}>
            {isCorrect ? 'Correct' : 'Incorrect'}
          </span>
        )}
      </div>
      <pre className="text-sm text-slate-200 whitespace-pre-wrap font-mono leading-relaxed">
        {text}
      </pre>
    </div>
  )
}
