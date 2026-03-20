import useStore from '../store'
import QuadrantMap from './QuadrantMap'

// Reuse stop components
import PromptBasic from '../stops/PromptBasic'
import RAGSimple from '../stops/RAGSimple'
import SFTComparison from '../stops/SFTComparison'
import SFTUnderTheHood from '../stops/SFTUnderTheHood'
import DPOPreferences from '../stops/DPOPreferences'
import GRPOGenerations from '../stops/GRPOGenerations'
import CombinedResults from '../stops/CombinedResults'
import InfrastructureSummary from '../stops/InfrastructureSummary'

const EXPLORE_VIEWS = {
  prompt: PromptBasic,
  rag: RAGSimple,
  posttraining: SFTComparison,
  alloptions: CombinedResults,
}

export default function ExploreView() {
  const activeQuadrant = useStore((s) => s.activeQuadrant)
  const setActiveQuadrant = useStore((s) => s.setActiveQuadrant)

  const ViewComponent = EXPLORE_VIEWS[activeQuadrant]

  return (
    <div className="min-h-screen flex">
      {/* Left sidebar with full quadrant map */}
      <aside className="w-[500px] flex-shrink-0 bg-slate-800/30 border-r border-slate-700/50 p-4 flex flex-col">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-300">Navigate</h2>
          <button
            onClick={() => useStore.getState().setMode('landing')}
            className="text-xs text-slate-500 hover:text-slate-300 px-2 py-1 rounded border border-slate-700 hover:border-slate-500 transition-colors"
          >
            Home
          </button>
        </div>

        <QuadrantMap interactive />

        {/* Quick links for post-training sub-stops */}
        {activeQuadrant === 'posttraining' && (
          <div className="mt-4 flex gap-2">
            {['sft', 'dpo', 'grpo'].map((sub) => (
              <button
                key={sub}
                onClick={() => setActiveQuadrant('posttraining', sub)}
                className="flex-1 px-3 py-2 text-xs font-semibold rounded bg-slate-700 hover:bg-slate-600 text-slate-300 uppercase tracking-wide transition-colors"
              >
                {sub}
              </button>
            ))}
          </div>
        )}

        {/* Legend */}
        <div className="mt-auto pt-4 border-t border-slate-700/50">
          <p className="text-xs text-slate-500 mb-2">Legend</p>
          <div className="grid grid-cols-2 gap-1 text-xs">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-orange-500" /> Prompt Engineering
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-yellow-500" /> RAG
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-slate-400" /> Post Training
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-cyan-500" /> All Options
            </span>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto p-6">
        {ViewComponent ? (
          <ViewComponent explore />
        ) : (
          <div className="text-center text-slate-500 py-20">
            Select a quadrant from the map to explore
          </div>
        )}
      </main>
    </div>
  )
}
