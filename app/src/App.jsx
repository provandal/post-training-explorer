import useStore from './store'
import Landing from './components/Landing'
import TourView from './components/TourView'
import ExploreView from './components/ExploreView'

export default function App() {
  const mode = useStore((s) => s.mode)

  return (
    <div className="min-h-screen bg-slate-900 text-slate-200">
      {mode === 'landing' && <Landing />}
      {mode === 'tour' && <TourView />}
      {mode === 'explore' && <ExploreView />}
    </div>
  )
}
