import { useState, useEffect } from 'react'

// Animated step-by-step pipeline diagram
// Shows data flowing through a series of processing stages

export default function AnimatedPipeline({ steps, autoPlay = false, speed = 800 }) {
  const [activeStep, setActiveStep] = useState(-1)

  useEffect(() => {
    if (autoPlay) setActiveStep(0)
  }, [autoPlay])

  useEffect(() => {
    if (activeStep >= 0 && activeStep < steps.length - 1) {
      const timer = setTimeout(() => setActiveStep(activeStep + 1), speed)
      return () => clearTimeout(timer)
    }
  }, [activeStep, steps.length, speed])

  return (
    <div className="py-3">
      <div className="flex items-stretch gap-0 overflow-x-auto">
        {steps.map((step, i) => {
          const isActive = i <= activeStep
          const isCurrent = i === activeStep
          const isLast = i === steps.length - 1

          return (
            <div key={i} className="flex items-center flex-shrink-0">
              {/* Step box */}
              <div
                className={`relative px-4 py-3 rounded-lg border-2 transition-all duration-500 min-w-[130px] ${
                  isCurrent
                    ? 'border-blue-500 bg-blue-950/40 shadow-lg shadow-blue-500/20 scale-105'
                    : isActive
                    ? 'border-slate-500 bg-slate-800/50'
                    : 'border-slate-700/30 bg-slate-800/20 opacity-40'
                }`}
              >
                {/* Icon */}
                <div className={`text-2xl mb-1 transition-all duration-500 ${isCurrent ? 'scale-110' : ''}`}>
                  {step.icon}
                </div>
                {/* Label */}
                <div className={`text-xs font-semibold transition-colors duration-500 ${
                  isCurrent ? 'text-blue-300' : isActive ? 'text-slate-300' : 'text-slate-600'
                }`}>
                  {step.label}
                </div>
                {/* Description */}
                {step.desc && (
                  <div className={`text-xs mt-1 transition-colors duration-500 ${
                    isCurrent ? 'text-slate-400' : 'text-slate-600'
                  }`}>
                    {step.desc}
                  </div>
                )}
                {/* Pulse ring on current */}
                {isCurrent && (
                  <div className="absolute -inset-1 rounded-lg border-2 border-blue-400 opacity-30 animate-ping" style={{ animationDuration: '2s' }} />
                )}
              </div>

              {/* Arrow between steps */}
              {!isLast && (
                <div className={`flex items-center mx-1 transition-all duration-500 ${
                  i < activeStep ? 'opacity-100' : 'opacity-20'
                }`}>
                  <div className={`w-8 h-0.5 ${i < activeStep ? 'bg-blue-500' : 'bg-slate-700'}`} />
                  <div className={`w-0 h-0 border-t-[5px] border-t-transparent border-b-[5px] border-b-transparent border-l-[8px] ${
                    i < activeStep ? 'border-l-blue-500' : 'border-l-slate-700'
                  }`} />
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Control */}
      {activeStep === -1 && (
        <button
          onClick={() => setActiveStep(0)}
          className="mt-3 px-4 py-1.5 text-sm bg-blue-700 hover:bg-blue-600 rounded-md transition-colors"
        >
          Animate pipeline
        </button>
      )}
      {activeStep >= steps.length - 1 && (
        <button
          onClick={() => setActiveStep(-1)}
          className="mt-3 px-3 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded text-slate-400 transition-colors"
        >
          Replay
        </button>
      )}
    </div>
  )
}
