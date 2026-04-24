import { useTranslation } from 'react-i18next'
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
  const { t } = useTranslation()
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
      <h2 className="text-2xl font-bold text-white mb-2">{t('stop.epilogue.headline')}</h2>
      <p className="text-slate-400 mb-8">{t('stop.epilogue.subheadline')}</p>

      {/* Action buttons */}
      <div className="flex gap-4 justify-center mb-8">
        <button
          onClick={startExplore}
          className="px-8 py-3 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-semibold text-lg transition-colors shadow-lg shadow-blue-600/25"
        >
          {t('landing.exploreFree')}
        </button>
        <button
          onClick={startTrain}
          className="px-8 py-3 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg font-semibold text-lg transition-colors shadow-lg shadow-emerald-600/25"
        >
          {t('stop.epilogue.trainYourModel')}
        </button>
      </div>

      {/* Deep Dives */}
      <div className="text-left mb-8">
        <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-3">
          {t('stop.epilogue.bonusDeepDives')}
        </h3>
        <p className="text-xs text-slate-500 mb-3">{t('stop.epilogue.bonusDeepDivesP')}</p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {[
            {
              key: 'context',
              label: t('stop.epilogue.contextWindows'),
              desc: t('stop.epilogue.contextWindowsDesc'),
            },
            {
              key: 'transformers',
              label: t('stop.epilogue.transformersAttention'),
              desc: t('stop.epilogue.transformersAttentionDesc'),
            },
            {
              key: 'lora',
              label: t('stop.epilogue.lora'),
              desc: t('stop.epilogue.loraDesc'),
            },
            {
              key: 'rlalgorithms',
              label: 'RL Algorithm Landscape',
              desc: 'PPO, DPO, GRPO, RLOO, ReMax, RLVR, KDRL — the evolution of reinforcement learning for LLMs',
            },
            {
              key: 'quantization',
              label: 'Quantization & Deployment',
              desc: 'FP32 to INT4, three-layer alignment, MX formats, and why smaller does not always mean faster',
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

      {/* What's Coming in 2026 and Beyond */}
      <div className="text-left mb-8">
        <h3 className="text-sm font-semibold text-cyan-400 uppercase tracking-wide mb-3">
          What's Coming in Post-Training
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="p-4 rounded-lg bg-cyan-950/20 border border-cyan-800/30">
            <span className="text-sm font-semibold text-cyan-300">Process Reward Models</span>
            <p className="text-xs text-slate-400 mt-1">
              Instead of rewarding only the final answer, reward each step of reasoning. This
              teaches models HOW to think, not just WHAT to answer. Enables much better debugging of
              model reasoning failures.
            </p>
          </div>
          <div className="p-4 rounded-lg bg-cyan-950/20 border border-cyan-800/30">
            <span className="text-sm font-semibold text-cyan-300">Multi-Agent RL (MARL)</span>
            <p className="text-xs text-slate-400 mt-1">
              Multiple models training together — negotiating, competing, and collaborating. Think
              AlphaStar or OpenAI Five, but for enterprise workflows where multiple AI agents need
              to coordinate.
            </p>
          </div>
          <div className="p-4 rounded-lg bg-cyan-950/20 border border-cyan-800/30">
            <span className="text-sm font-semibold text-cyan-300">RLVR Beyond Math & Code</span>
            <p className="text-xs text-slate-400 mt-1">
              Verifiable rewards are expanding from math and code (where answers are checkable) to
              science, engineering, and legal reasoning — domains where correctness can be partially
              verified.
            </p>
          </div>
          <div className="p-4 rounded-lg bg-cyan-950/20 border border-cyan-800/30">
            <span className="text-sm font-semibold text-cyan-300">On-Device Post-Training</span>
            <p className="text-xs text-slate-400 mt-1">
              Fine-tuning models directly on edge devices — phones, IoT, laptops — enabling
              personalization without sending data to the cloud. Privacy-preserving AI adaptation.
            </p>
          </div>
        </div>
      </div>

      {/* AINOS Research Advisor Teaser */}
      <div className="text-left mb-8">
        <h3 className="text-sm font-semibold text-indigo-400 uppercase tracking-wide mb-3">
          {t('stop.epilogue.whatComesNext')}
        </h3>
        <div className="p-4 rounded-lg bg-indigo-950/20 border border-indigo-800/30">
          <p className="text-sm text-slate-300 mb-3">{t('stop.epilogue.ainosP1')}</p>
          <p className="text-sm text-slate-400">
            {t('stop.epilogue.ainosP2', {
              defaultValue:
                "That's the core idea behind AINOS — an AI-powered research advisor that examines training artifacts, identifies opportunities for improvement, and suggests specific experiments with cost estimates. It amplifies engineer judgment rather than replacing it: the AI proposes, you approve, the system executes. The infrastructure implications for storage and compute teams are significant — automated experiment loops mean predictable, bursty GPU demand and structured artifact management at scale.",
            })}
          </p>
          <p className="text-xs text-indigo-500/70 mt-3 italic">
            {t('stop.epilogue.ainosComingSoon')}
          </p>
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
        <p className="text-xs text-slate-600">{t('stop.epilogue.footer')}</p>
        <p className="text-xs text-slate-700 mt-1">{t('stop.epilogue.footerBuiltWith')}</p>
      </div>
    </div>
  )
}
