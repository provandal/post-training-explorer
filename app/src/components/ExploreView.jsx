import { useTranslation } from 'react-i18next'
import useStore from '../store'
import QuadrantMap from './QuadrantMap'
// Reuse stop components
import PromptBasic from '../stops/PromptBasic'
import PromptFewShot from '../stops/PromptFewShot'
import PromptLimitation from '../stops/PromptLimitation'
import RAGSimple from '../stops/RAGSimple'
import RAGLimitation from '../stops/RAGLimitation'
import SFTComparison from '../stops/SFTComparison'
import DPOPreferences from '../stops/DPOPreferences'
import GRPOGenerations from '../stops/GRPOGenerations'
import CombinedResults from '../stops/CombinedResults'
import InfrastructureSummary from '../stops/InfrastructureSummary'
import ContextWindowDeepDive from '../stops/ContextWindowDeepDive'
import TransformersDeepDive from '../stops/TransformersDeepDive'
import LoRADeepDive from '../stops/LoRADeepDive'
import ModelSizeComparison from '../stops/ModelSizeComparison'

// Nested view map: quadrant → { subStop: Component }
// null key = default view when no sub-stop is selected
const EXPLORE_VIEWS = {
  prompt: {
    [null]: PromptBasic,
    basic: PromptBasic,
    fewshot: PromptFewShot,
    limitation: PromptLimitation,
  },
  rag: {
    [null]: RAGSimple,
    simple: RAGSimple,
    limitation: RAGLimitation,
  },
  posttraining: {
    [null]: SFTComparison,
    sft: SFTComparison,
    dpo: DPOPreferences,
    grpo: GRPOGenerations,
  },
  alloptions: {
    [null]: CombinedResults,
    combined: CombinedResults,
    infrastructure: InfrastructureSummary,
    modelsize: ModelSizeComparison,
  },
  context: { [null]: ContextWindowDeepDive },
  transformers: { [null]: TransformersDeepDive },
  lora: { [null]: LoRADeepDive },
}

// Sub-stop labels per quadrant (built with t() inside component)
function getSubStops(t) {
  return {
    prompt: [
      { key: 'basic', label: t('explore.subStops.zeroShot') },
      { key: 'fewshot', label: t('explore.subStops.fewShot') },
      { key: 'limitation', label: t('explore.subStops.limitations') },
    ],
    rag: [
      { key: 'simple', label: t('explore.subStops.howRagWorks') },
      { key: 'limitation', label: t('explore.subStops.ragLimitations') },
    ],
    posttraining: [
      { key: 'sft', label: t('explore.subStops.sft') },
      { key: 'dpo', label: t('explore.subStops.dpo') },
      { key: 'grpo', label: t('explore.subStops.grpo') },
    ],
    alloptions: [
      { key: 'combined', label: t('explore.subStops.combinedResults') },
      { key: 'infrastructure', label: t('explore.subStops.infrastructure') },
      { key: 'modelsize', label: t('explore.subStops.modelSizes') },
    ],
  }
}

function getDeepDives(t) {
  return [
    { key: 'context', label: t('explore.deepDive.contextWindows'), color: 'text-purple-400' },
    {
      key: 'transformers',
      label: t('explore.deepDive.transformersAttention'),
      color: 'text-purple-400',
    },
    { key: 'lora', label: t('explore.deepDive.loraDeepDive'), color: 'text-purple-400' },
  ]
}

export default function ExploreView() {
  const { t } = useTranslation()
  const activeQuadrant = useStore((s) => s.activeQuadrant)
  const activeSubStop = useStore((s) => s.activeSubStop)
  const setActiveQuadrant = useStore((s) => s.setActiveQuadrant)
  const tourReturnStep = useStore((s) => s.tourReturnStep)

  const quadrantViews = EXPLORE_VIEWS[activeQuadrant]
  const ViewComponent = quadrantViews?.[activeSubStop] || quadrantViews?.[null]
  const SUB_STOPS = getSubStops(t)
  const DEEP_DIVES = getDeepDives(t)
  const subStops = SUB_STOPS[activeQuadrant]

  return (
    <div className="min-h-screen flex">
      {/* Left sidebar with full quadrant map */}
      <aside className="w-[500px] flex-shrink-0 bg-slate-800/30 border-r border-slate-700/50 p-4 flex flex-col">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-300">{t('explore.navigate')}</h2>
          <div className="flex items-center gap-2">
            {tourReturnStep !== null && (
              <button
                onClick={() =>
                  useStore.setState({
                    mode: 'tour',
                    currentStep: tourReturnStep,
                    tourReturnStep: null,
                  })
                }
                className="text-xs text-blue-400 hover:text-blue-300 px-2 py-1 rounded border border-blue-700/50 hover:border-blue-500 transition-colors"
              >
                {t('nav.returnToTour')}
              </button>
            )}
            <button
              onClick={() => useStore.getState().setMode('landing')}
              className="text-xs text-slate-500 hover:text-slate-300 px-2 py-1 rounded border border-slate-700 hover:border-slate-500 transition-colors"
            >
              {t('nav.home')}
            </button>
          </div>
        </div>

        <QuadrantMap size="full" interactive />

        {/* Sub-stop navigation for the active quadrant */}
        {subStops && (
          <div className="mt-4">
            <p className="text-xs text-slate-500 mb-2">{t('explore.topics')}</p>
            <div className="flex flex-wrap gap-2">
              {subStops.map((sub) => {
                const isActive =
                  activeSubStop === sub.key || (!activeSubStop && sub === subStops[0])
                return (
                  <button
                    key={sub.key}
                    onClick={() => setActiveQuadrant(activeQuadrant, sub.key)}
                    className={`px-3 py-2 text-xs font-semibold rounded transition-colors ${
                      isActive
                        ? 'bg-blue-600 text-white'
                        : 'bg-slate-700 hover:bg-slate-600 text-slate-300'
                    }`}
                  >
                    {sub.label}
                  </button>
                )
              })}
            </div>
          </div>
        )}

        {/* Bonus deep dives */}
        <div className="mt-4">
          <p className="text-xs text-slate-500 mb-2">{t('explore.deepDives')}</p>
          <div className="flex flex-wrap gap-2">
            {DEEP_DIVES.map((dd) => {
              const isActive = activeQuadrant === dd.key
              return (
                <button
                  key={dd.key}
                  onClick={() => setActiveQuadrant(dd.key)}
                  className={`px-3 py-2 text-xs font-semibold rounded transition-colors ${
                    isActive
                      ? 'bg-purple-600 text-white'
                      : 'bg-slate-700/60 hover:bg-slate-600 text-purple-300'
                  }`}
                >
                  {dd.label}
                </button>
              )
            })}
          </div>
        </div>

        {/* Legend */}
        <div className="mt-auto pt-4 border-t border-slate-700/50">
          <p className="text-xs text-slate-500 mb-2">{t('explore.legend')}</p>
          <div className="grid grid-cols-2 gap-1 text-xs">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-orange-500" />{' '}
              {t('explore.legend.promptEngineering')}
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-yellow-500" /> {t('explore.legend.rag')}
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-slate-400" />{' '}
              {t('explore.legend.postTraining')}
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-cyan-500" /> {t('explore.legend.allOptions')}
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-purple-500" />{' '}
              {t('explore.legend.deepDives')}
            </span>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto p-6">
        {ViewComponent ? (
          <ViewComponent explore />
        ) : (
          <div className="text-center text-slate-500 py-20">{t('explore.selectQuadrant')}</div>
        )}
      </main>
    </div>
  )
}
