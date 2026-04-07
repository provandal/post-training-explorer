import { Suspense, useEffect } from 'react'
import useStore from './store'
import useDirection from './hooks/useDirection'
import Landing from './components/Landing'
import TourView from './components/TourView'
import ExploreView from './components/ExploreView'
import TrainView from './components/TrainView'
import ResultsView from './components/ResultsView'

export default function App() {
  const mode = useStore((s) => s.mode)
  const currentStep = useStore((s) => s.currentStep)
  useDirection()

  // Track SPA navigation in GoatCounter
  useEffect(() => {
    const path = mode === 'tour' ? `/tour/step-${currentStep}` : `/${mode}`
    if (window.goatcounter?.count) {
      window.goatcounter.count({ path })
    }
  }, [mode, currentStep])

  return (
    <Suspense fallback={<div className="min-h-screen bg-slate-900" />}>
      <div className="min-h-screen bg-slate-900 text-slate-200">
        {mode === 'landing' && <Landing />}
        {mode === 'tour' && <TourView />}
        {mode === 'explore' && <ExploreView />}
        {mode === 'train' && <TrainView />}
        {mode === 'results' && <ResultsView />}
      </div>
    </Suspense>
  )
}
