import { useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import useStore from '../store'
import getTourSteps from '../data/tourSteps'
import QuadrantMap from './QuadrantMap'
import LanguageSelector from './LanguageSelector'

// Stop components
import Welcome from '../stops/Welcome'
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
import Epilogue from '../stops/Epilogue'

const STOP_COMPONENTS = {
  Welcome,
  PromptBasic,
  PromptFewShot,
  PromptLimitation,
  RAGSimple,
  RAGLimitation,
  SFTComparison,
  DPOPreferences,
  GRPOGenerations,
  CombinedResults,
  InfrastructureSummary,
  Epilogue,
}

export default function TourView() {
  const { t } = useTranslation()
  const currentStep = useStore((s) => s.currentStep)
  const nextStep = useStore((s) => s.nextStep)
  const prevStep = useStore((s) => s.prevStep)
  const tourSteps = getTourSteps(t)
  const step = tourSteps[currentStep]

  const isFirst = currentStep === 0
  const isLast = currentStep === tourSteps.length - 1
  const total = tourSteps.length

  // Scroll to top on step change
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [currentStep])

  // Keyboard navigation: Left/Right arrows move between tour steps
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Don't intercept when the user is typing in an input or textarea
      const tag = e.target.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || e.target.isContentEditable) return

      if (e.key === 'ArrowLeft') {
        useStore.getState().prevStep()
      } else if (e.key === 'ArrowRight') {
        useStore.getState().nextStep()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const StopComponent = STOP_COMPONENTS[step.component]

  const prevTitle = isFirst ? null : tourSteps[currentStep - 1].shortTitle
  const nextTitle = isLast ? null : tourSteps[currentStep + 1].shortTitle

  return (
    <div className="min-h-screen flex flex-col">
      {/* Combined top bar: Home | title | progress | nav */}
      <header className="sticky top-0 z-10 flex items-center gap-3 px-4 py-2 bg-slate-800/95 backdrop-blur border-b border-slate-700/50">
        <button
          onClick={() => useStore.getState().setMode('landing')}
          className="text-xs text-slate-400 hover:text-white px-2.5 py-1.5 rounded border border-slate-700 hover:border-slate-500 transition-colors flex-shrink-0"
        >
          {t('nav.home')}
        </button>

        <div className="flex-1 min-w-0">
          <h2 className="text-sm font-semibold text-white truncate">{step.title}</h2>
        </div>

        {/* Progress dots */}
        <div className="hidden sm:flex items-center gap-0.5 flex-shrink-0">
          {tourSteps.map((_, i) => (
            <div
              key={i}
              className={`w-1.5 h-1.5 rounded-full transition-colors ${
                i === currentStep
                  ? 'bg-blue-500'
                  : i < currentStep
                    ? 'bg-slate-500'
                    : 'bg-slate-700'
              }`}
            />
          ))}
        </div>
        <span className="text-xs text-slate-500 whitespace-nowrap flex-shrink-0">
          {currentStep + 1}/{total}
        </span>

        {/* Nav buttons with chapter names */}
        <button
          onClick={prevStep}
          disabled={isFirst}
          className="text-xs px-3 py-1.5 rounded-md bg-slate-700 hover:bg-slate-600 disabled:opacity-30 disabled:cursor-not-allowed transition-colors flex-shrink-0"
        >
          {prevTitle ? `\u2190 ${prevTitle}` : t('nav.previous')}
        </button>
        <button
          onClick={nextStep}
          className="text-xs px-3 py-1.5 rounded-md bg-blue-600 hover:bg-blue-500 font-medium transition-colors flex-shrink-0"
        >
          {isLast ? t('nav.exploreFreelyNav') : `${nextTitle} \u2192`}
        </button>
        <LanguageSelector />
      </header>

      {/* Main content area */}
      <main className="flex-1 p-6">
        {/* Narration callout */}
        {step.narration && (
          <div className="mb-4 p-3 rounded-lg bg-slate-800/50 border border-slate-700/30 text-sm text-slate-300">
            {step.narration}
          </div>
        )}

        {StopComponent ? (
          <StopComponent />
        ) : (
          <div className="text-center text-slate-500 py-20">
            {t('nav.componentNotImplemented', { component: step.component })}
          </div>
        )}

        {/* QuadrantMap below stop content */}
        <div className="flex justify-center mt-6">
          <QuadrantMap size="medium" />
        </div>
      </main>
    </div>
  )
}
