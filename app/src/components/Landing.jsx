import useStore from '../store'

export default function Landing() {
  const startTour = useStore((s) => s.startTour)
  const startExplore = useStore((s) => s.startExplore)
  const startTrain = useStore((s) => s.startTrain)

  const cards = [
    {
      title: 'Guided Tour',
      description:
        'Follow the zig-zag path from simple prompting to advanced reinforcement learning. At each stop, try the technique and see what happens under the covers.',
      action: startTour,
      buttonLabel: 'Start Tour',
      buttonClass: 'bg-blue-600 hover:bg-blue-500 shadow-lg shadow-blue-600/30 hover:shadow-blue-500/40',
    },
    {
      title: 'Explore Freely',
      description:
        'Dive into any technique directly — prompt engineering, RAG, SFT, DPO, or GRPO. Compare results across approaches and dig into deep dives on transformers, LoRA, and context windows.',
      action: startExplore,
      buttonLabel: 'Explore',
      buttonClass: 'bg-slate-600 hover:bg-slate-500 shadow-lg shadow-slate-600/20',
    },
    {
      title: 'Train Your Model',
      description:
        'Run the notebooks yourself in Google Colab. Train SFT, DPO, and GRPO models on a free GPU, then test your model right here in the browser.',
      action: startTrain,
      buttonLabel: 'Get Started',
      buttonClass: 'bg-emerald-600 hover:bg-emerald-500 shadow-lg shadow-emerald-600/30 hover:shadow-emerald-500/40',
    },
  ]

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 py-10">
      {/* Header */}
      <div className="text-center mb-10">
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

      {/* Three cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl w-full mb-10">
        {cards.map((card) => (
          <div
            key={card.title}
            className="bg-slate-800/50 border border-slate-700/50 hover:border-blue-600/50 rounded-xl p-6 flex flex-col transition-colors"
          >
            <h2 className="text-xl font-bold text-white mb-3">{card.title}</h2>
            <p className="text-sm text-slate-400 leading-relaxed flex-1 mb-5">
              {card.description}
            </p>
            <button
              onClick={card.action}
              className={`w-full py-3 text-white rounded-lg font-semibold text-sm transition-all hover:scale-[1.02] active:scale-[0.98] ${card.buttonClass}`}
            >
              {card.buttonLabel}
            </button>
          </div>
        ))}
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
