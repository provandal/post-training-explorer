import { Suspense } from 'react'
import useStore from './store'
import useDirection from './hooks/useDirection'
import Landing from './components/Landing'
import TourView from './components/TourView'
import ExploreView from './components/ExploreView'
import TrainView from './components/TrainView'
import ResultsView from './components/ResultsView'

export default function App() {
  const mode = useStore((s) => s.mode)
  useDirection()

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
