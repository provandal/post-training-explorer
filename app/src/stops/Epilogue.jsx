import useStore from '../store'

const RESOURCES = [
  {
    category: 'Try It Yourself',
    items: [
      { label: 'Training Notebooks (Colab)', url: '#', description: 'Run SFT, DPO, and GRPO training yourself — free GPU included' },
      { label: 'Storage I/O Dataset', url: '#', description: 'The full labeled dataset used in this demo on HuggingFace' },
      { label: 'Pre-trained Models', url: '#', description: 'All model variants (base, SFT, DPO, GRPO) on HuggingFace' },
    ],
  },
  {
    category: 'Learn More',
    items: [
      { label: 'HuggingFace SmolLM Training Playbook', url: '#', description: 'Comprehensive open-source post-training guide' },
      { label: 'Sebastian Raschka, "The State of LLMs"', url: '#', description: 'Overview of the reasoning model revolution' },
      { label: 'DeepSeek R1 Technical Report', url: '#', description: 'The paper that catalyzed the GRPO/RLVR movement' },
    ],
  },
  {
    category: 'SNIA Resources',
    items: [
      { label: 'Storage in AI Training Workloads', url: '#', description: 'Previous SNIA webinar by Ugur Kaynar' },
      { label: 'SNIA DSN AI Stack Series', url: '#', description: 'Full webinar series on AI infrastructure' },
    ],
  },
]

export default function Epilogue() {
  const startExplore = useStore((s) => s.startExplore)

  return (
    <div className="max-w-3xl mx-auto text-center">
      {/* Headline */}
      <h2 className="text-2xl font-bold text-white mb-2">
        Everything you just saw is real.
      </h2>
      <p className="text-slate-400 mb-8">
        Real models, real weights, real training artifacts. And you can do it yourself.
      </p>

      {/* Explore button */}
      <button
        onClick={startExplore}
        className="px-8 py-3 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-semibold text-lg transition-colors shadow-lg shadow-blue-600/25 mb-8"
      >
        Explore Freely
      </button>

      {/* Resource links */}
      <div className="text-left space-y-6">
        {RESOURCES.map((group) => (
          <div key={group.category}>
            <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-3">
              {group.category}
            </h3>
            <div className="space-y-2">
              {group.items.map((item) => (
                <a
                  key={item.label}
                  href={item.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block p-3 rounded-lg bg-slate-800/50 border border-slate-700/50 hover:border-blue-700/50 hover:bg-slate-800 transition-colors"
                >
                  <span className="text-sm font-medium text-blue-400">{item.label}</span>
                  <span className="block text-xs text-slate-500 mt-0.5">{item.description}</span>
                </a>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Footer */}
      <div className="mt-8 pt-6 border-t border-slate-800">
        <p className="text-xs text-slate-600">
          Post-Training Explorer &middot; SNIA DSN AI Stack Webinar Series &middot; 2026
        </p>
        <p className="text-xs text-slate-700 mt-1">
          Built with React, D3.js, and SmolLM2-360M
        </p>
      </div>
    </div>
  )
}
