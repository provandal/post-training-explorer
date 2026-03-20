import useStore from '../store'
import QuadrantMap from './QuadrantMap'

export default function Landing() {
  const startTour = useStore((s) => s.startTour)
  const startExplore = useStore((s) => s.startExplore)

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 py-10">
      {/* Header */}
      <div className="text-center mb-6">
        <p className="text-sm text-blue-400 font-semibold tracking-wide uppercase mb-2">
          SNIA DSN AI Stack Webinar Series
        </p>
        <h1 className="text-5xl font-extrabold text-white mb-3 tracking-tight">
          Post-Training Explorer
        </h1>
        <p className="text-lg text-slate-400 max-w-xl mx-auto leading-relaxed">
          An interactive guide to the techniques that turn a general-purpose
          language model into one that works for{' '}
          <span className="text-cyan-400 font-semibold">your</span> organization.
        </p>
      </div>

      {/* Quadrant Map — large and centered */}
      <div className="w-full max-w-2xl mx-auto mb-6">
        <QuadrantMap />
      </div>

      {/* Subtitle */}
      <p className="text-slate-500 text-sm mb-8 max-w-md text-center leading-relaxed">
        Follow the zig-zag path from simple prompting to advanced
        reinforcement learning. At each stop, try the technique and see
        what's happening under the covers.
      </p>

      {/* Action Buttons */}
      <div className="flex gap-4 mb-10">
        <button
          onClick={startTour}
          className="px-10 py-3.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-bold text-lg transition-all shadow-lg shadow-blue-600/30 hover:shadow-blue-500/40 hover:scale-[1.02] active:scale-[0.98]"
        >
          Start Guided Tour
        </button>
        <button
          onClick={startExplore}
          className="px-10 py-3.5 bg-slate-700/80 hover:bg-slate-600 text-slate-200 rounded-lg font-bold text-lg transition-all hover:scale-[1.02] active:scale-[0.98]"
        >
          Explore Freely
        </button>
      </div>

      {/* Bottom details */}
      <div className="flex items-center gap-6 text-xs text-slate-600">
        <span>Storage I/O Workload Classification</span>
        <span className="w-1 h-1 rounded-full bg-slate-700" />
        <span>SmolLM2-360M</span>
        <span className="w-1 h-1 rounded-full bg-slate-700" />
        <span>Real models &middot; Real weights &middot; Real inference</span>
      </div>
    </div>
  )
}
