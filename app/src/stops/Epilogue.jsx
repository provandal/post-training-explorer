import useStore from '../store'

// Resource links — placeholder '#' URLs will be replaced with real HuggingFace/Colab
// links after Phase 4 (model/dataset/notebook publishing).
const RESOURCES = [
  {
    category: 'Try It Yourself',
    items: [
      {
        label: 'Training Notebooks (Colab)',
        url: 'https://colab.research.google.com/github/provandal/post-training-explorer/blob/main/notebooks/Post_Training_Pipeline.ipynb',
        description: 'Run SFT, DPO, and GRPO training yourself — free GPU included',
      },
      {
        label: 'Training Data (Synthetic)',
        url: 'https://github.com/provandal/post-training-explorer/tree/main/scripts',
        description:
          'Data is generated synthetically at runtime — see the training scripts for workload profiles and generation logic',
      },
      {
        label: 'Pre-trained Models',
        url: '#',
        description:
          'All model variants (base, SFT, DPO, GRPO) on HuggingFace — coming after initial release',
      },
    ],
  },
  {
    category: 'Learn More',
    items: [
      {
        label: 'HuggingFace SmolLM2',
        url: 'https://huggingface.co/HuggingFaceTB/SmolLM2-360M',
        description: 'SmolLM2-360M model card — the base model used in this demo',
      },
      {
        label: 'Sebastian Raschka, "Understanding Reasoning LLMs"',
        url: 'https://magazine.sebastianraschka.com/p/understanding-reasoning-llms',
        description: 'Overview of the reasoning model revolution',
      },
      {
        label: 'DeepSeek R1 Technical Report',
        url: 'https://arxiv.org/abs/2501.12948',
        description: 'The paper that catalyzed the GRPO/RLVR movement',
      },
      {
        label: 'DPO: Direct Preference Optimization',
        url: 'https://arxiv.org/abs/2305.18290',
        description: 'Rafailov et al. — the paper that simplified RLHF',
      },
    ],
  },
  {
    category: 'SNIA Resources',
    items: [
      {
        label: 'SNIA Networking Storage Forum',
        url: 'https://www.snia.org/forums/dsn',
        description: 'SNIA DSN forum — AI infrastructure discussions',
      },
      {
        label: 'SNIA Educational Library',
        url: 'https://www.snia.org/education/education-library',
        description: 'Free technical content from SNIA',
      },
    ],
  },
]

export default function Epilogue() {
  const startExplore = useStore((s) => s.startExplore)
  const startTrain = useStore((s) => s.startTrain)
  const setActiveQuadrant = useStore((s) => s.setActiveQuadrant)
  const setMode = useStore((s) => s.setMode)
  const currentStep = useStore((s) => s.currentStep)

  const goToDeepDive = (key) => {
    useStore.setState({ tourReturnStep: currentStep })
    setActiveQuadrant(key)
    setMode('explore')
  }

  return (
    <div className="max-w-3xl mx-auto text-center">
      {/* Headline */}
      <h2 className="text-2xl font-bold text-white mb-2">Everything you just saw is real.</h2>
      <p className="text-slate-400 mb-8">
        Real models, real weights, real training artifacts. And you can do it yourself.
      </p>

      {/* Action buttons */}
      <div className="flex gap-4 justify-center mb-8">
        <button
          onClick={startExplore}
          className="px-8 py-3 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-semibold text-lg transition-colors shadow-lg shadow-blue-600/25"
        >
          Explore Freely
        </button>
        <button
          onClick={startTrain}
          className="px-8 py-3 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg font-semibold text-lg transition-colors shadow-lg shadow-emerald-600/25"
        >
          Train Your Model
        </button>
      </div>

      {/* Deep Dives */}
      <div className="text-left mb-8">
        <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-3">
          Bonus Deep Dives
        </h3>
        <p className="text-xs text-slate-500 mb-3">
          Want to understand the concepts behind the tour in more detail? These deep dives go
          further.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {[
            {
              key: 'context',
              label: 'Context Windows',
              desc: 'How context is built, managed, and where it breaks down — context rot, attention dilution, and compaction strategies',
            },
            {
              key: 'transformers',
              label: 'Transformers & Attention',
              desc: 'What a Transformer actually is, how Q/K/V attention works, multi-head attention, and how layers build understanding',
            },
            {
              key: 'lora',
              label: 'LoRA',
              desc: 'Why low-rank adaptation works, which parameters get trained, rank decomposition math, and infrastructure implications',
            },
          ].map((dd) => (
            <button
              key={dd.key}
              onClick={() => goToDeepDive(dd.key)}
              className="text-left p-4 rounded-lg bg-purple-950/20 border border-purple-800/30 hover:border-purple-600/50 hover:bg-purple-950/30 transition-colors cursor-pointer"
            >
              <span className="text-sm font-semibold text-purple-400">{dd.label}</span>
              <span className="block text-xs text-slate-500 mt-1">{dd.desc}</span>
            </button>
          ))}
        </div>
      </div>

      {/* AINOS Research Advisor Teaser */}
      <div className="text-left mb-8">
        <h3 className="text-sm font-semibold text-indigo-400 uppercase tracking-wide mb-3">
          What Comes Next: AI-Guided Research
        </h3>
        <div className="p-4 rounded-lg bg-indigo-950/20 border border-indigo-800/30">
          <p className="text-sm text-slate-300 mb-3">
            Everything in this explorer required manual decisions — which technique to try, what
            hyperparameters to use, when to stop training. But what if an AI system could analyze
            your results and propose the next experiment?
          </p>
          <p className="text-sm text-slate-400">
            That's the core idea behind <span className="text-indigo-400 font-medium">AINOS</span> —
            an AI-powered research advisor that examines training artifacts, identifies
            opportunities for improvement, and suggests specific experiments with cost estimates. It
            amplifies engineer judgment rather than replacing it: the AI proposes, you approve, the
            system executes. The infrastructure implications for storage and compute teams are
            significant — automated experiment loops mean predictable, bursty GPU demand and
            structured artifact management at scale.
          </p>
          <p className="text-xs text-indigo-500/70 mt-3 italic">More on this soon.</p>
        </div>
      </div>

      {/* Resource links */}
      <div className="text-left space-y-6">
        {RESOURCES.map((group) => (
          <div key={group.category}>
            <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-3">
              {group.category}
            </h3>
            <div className="space-y-2">
              {group.items.map((item) => {
                const isPlaceholder = item.placeholder || item.url === '#'
                const Tag = isPlaceholder ? 'div' : 'a'
                return (
                  <Tag
                    key={item.label}
                    {...(!isPlaceholder
                      ? { href: item.url, target: '_blank', rel: 'noopener noreferrer' }
                      : {})}
                    className={`block p-3 rounded-lg border transition-colors ${
                      isPlaceholder
                        ? 'bg-slate-800/30 border-slate-700/30 opacity-60'
                        : 'bg-slate-800/50 border-slate-700/50 hover:border-blue-700/50 hover:bg-slate-800 cursor-pointer'
                    }`}
                  >
                    <span
                      className={`text-sm font-medium ${isPlaceholder ? 'text-slate-400' : 'text-blue-400'}`}
                    >
                      {item.label}
                      {isPlaceholder && (
                        <span className="ml-2 text-xs text-slate-600">(coming soon)</span>
                      )}
                    </span>
                    <span className="block text-xs text-slate-500 mt-0.5">{item.description}</span>
                  </Tag>
                )
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Footer */}
      <div className="mt-8 pt-6 border-t border-slate-800">
        <p className="text-xs text-slate-600">Post-Training Explorer &middot; 2026</p>
        <p className="text-xs text-slate-700 mt-1">Built with React, D3.js, and SmolLM2-360M</p>
      </div>
    </div>
  )
}
