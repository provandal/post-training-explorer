import useStore from '../store'
import tourSteps from '../data/tourSteps'
import QuadrantMap from './QuadrantMap'
import NarrationPanel from './NarrationPanel'

// Stop components
import Welcome from '../stops/Welcome'
import PromptBasic from '../stops/PromptBasic'
import PromptFewShot from '../stops/PromptFewShot'
import PromptLimitation from '../stops/PromptLimitation'
import RAGSimple from '../stops/RAGSimple'
import RAGLimitation from '../stops/RAGLimitation'
import SFTComparison from '../stops/SFTComparison'
import SFTUnderTheHood from '../stops/SFTUnderTheHood'
import DPOPreferences from '../stops/DPOPreferences'
import DPOUnderTheHood from '../stops/DPOUnderTheHood'
import GRPOGenerations from '../stops/GRPOGenerations'
import GRPOUnderTheHood from '../stops/GRPOUnderTheHood'
import CombinedResults from '../stops/CombinedResults'
import InfrastructureSummary from '../stops/InfrastructureSummary'
import Epilogue from '../stops/Epilogue'

const STOP_COMPONENTS = {
  Welcome,
  PromptBasic,
  PromptFewShot,
  PromptLimitation,
  RAGSimple,
  RAGLimitation,
  SFTComparison,
  SFTUnderTheHood,
  DPOPreferences,
  DPOUnderTheHood,
  GRPOGenerations,
  GRPOUnderTheHood,
  CombinedResults,
  InfrastructureSummary,
  Epilogue,
}

export default function TourView() {
  const currentStep = useStore((s) => s.currentStep)
  const step = tourSteps[currentStep]

  const StopComponent = STOP_COMPONENTS[step.component]

  return (
    <div className="min-h-screen flex flex-col">
      {/* Top bar with mini map */}
      <header className="flex items-center gap-4 px-4 py-2 bg-slate-800/50 border-b border-slate-700/50">
        <QuadrantMap mini />
        <div className="flex-1">
          <h2 className="text-sm font-semibold text-white">{step.title}</h2>
          <p className="text-xs text-slate-400">
            {step.quadrant
              ? `${step.quadrant === 'posttraining' ? 'Post Training' : step.quadrant === 'alloptions' ? 'All Options' : step.quadrant.toUpperCase()}${step.subStop ? ` > ${step.subStop.toUpperCase()}` : ''}`
              : 'Overview'
            }
          </p>
        </div>
        <button
          onClick={() => useStore.getState().setMode('landing')}
          className="text-xs text-slate-500 hover:text-slate-300 px-3 py-1 rounded border border-slate-700 hover:border-slate-500 transition-colors"
        >
          Exit Tour
        </button>
      </header>

      {/* Main content area */}
      <main className="flex-1 overflow-y-auto p-6">
        {StopComponent ? <StopComponent /> : (
          <div className="text-center text-slate-500 py-20">
            Component "{step.component}" not yet implemented
          </div>
        )}
      </main>

      {/* Narration panel at bottom */}
      <NarrationPanel />
    </div>
  )
}
