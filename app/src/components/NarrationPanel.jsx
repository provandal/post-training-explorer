import useStore from '../store'
import tourSteps from '../data/tourSteps'

export default function NarrationPanel() {
  const currentStep = useStore((s) => s.currentStep)
  const nextStep = useStore((s) => s.nextStep)
  const prevStep = useStore((s) => s.prevStep)

  const step = tourSteps[currentStep]
  const isFirst = currentStep === 0
  const isLast = currentStep === tourSteps.length - 1
  const total = tourSteps.length

  // Color based on quadrant
  const quadrantColors = {
    prompt: 'border-orange-500/50 bg-orange-950/20',
    rag: 'border-yellow-500/50 bg-yellow-950/20',
    posttraining: 'border-slate-400/50 bg-slate-800/30',
    alloptions: 'border-cyan-500/50 bg-cyan-950/20',
  }
  const borderClass = step.quadrant
    ? quadrantColors[step.quadrant]
    : 'border-blue-500/50 bg-blue-950/20'

  return (
    <div className={`border-t-2 ${borderClass} px-6 py-4`}>
      {/* Progress bar */}
      <div className="flex items-center gap-3 mb-3">
        <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-500 rounded-full transition-all duration-500"
            style={{ width: `${((currentStep + 1) / total) * 100}%` }}
          />
        </div>
        <span className="text-xs text-slate-500 whitespace-nowrap">
          {currentStep + 1} / {total}
        </span>
      </div>

      {/* Title */}
      <h3 className="text-lg font-semibold text-white mb-2">{step.title}</h3>

      {/* Narration text */}
      <p className="text-slate-300 text-sm leading-relaxed mb-4 max-w-4xl">{step.narration}</p>

      {/* Navigation */}
      <div className="flex items-center justify-between">
        <button
          onClick={prevStep}
          disabled={isFirst}
          className="px-4 py-2 text-sm rounded-md bg-slate-700 hover:bg-slate-600 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        >
          &larr; Back
        </button>

        {step.quadrant && (
          <div className="flex gap-1">
            {tourSteps.map((s, i) => (
              <div
                key={i}
                className={`w-2 h-2 rounded-full transition-colors ${
                  i === currentStep
                    ? 'bg-blue-500'
                    : i < currentStep
                      ? 'bg-slate-500'
                      : 'bg-slate-700'
                }`}
              />
            ))}
          </div>
        )}

        <button
          onClick={nextStep}
          className="px-4 py-2 text-sm rounded-md bg-blue-600 hover:bg-blue-500 font-medium transition-colors"
        >
          {isLast ? 'Explore Freely' : 'Next \u2192'}
        </button>
      </div>
    </div>
  )
}
