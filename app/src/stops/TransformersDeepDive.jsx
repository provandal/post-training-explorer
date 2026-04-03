// Bonus stop: Deep dive into Transformer architecture and attention layers.
// Accessible from Explore mode via the LoRA Weights aside in SFTComparison.

import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import SectionTabs from '../components/SectionTabs'
import AttentionHeatmap from '../components/AttentionHeatmap'
import ForwardPassSteps from '../components/ForwardPassSteps'

export default function TransformersDeepDive() {
  const { t } = useTranslation()
  const [section, setSection] = useState('overview')

  const TABS = [
    { id: 'overview', label: t('tabs.howItWorks') },
    { id: 'attention', label: t('tabs.attentionExplorer') },
    { id: 'steps', label: t('tabs.stepByStep') },
  ]

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <span className="text-xs font-semibold text-violet-400 uppercase tracking-wide">
          {t('deepdive.transformers.bonusLabel')}
        </span>
        <h2 className="text-xl font-bold text-white mt-1">{t('deepdive.transformers.title')}</h2>
        <p className="text-sm text-slate-400 mt-2">{t('deepdive.transformers.subtitle')}</p>
      </div>

      <SectionTabs tabs={TABS} active={section} onSelect={setSection} color="violet" />

      {/* Tab 1: How It Works — condensed reference */}
      {section === 'overview' && (
        <div className="space-y-6">
          {/* Architecture diagram */}
          <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h3 className="text-base font-semibold text-violet-400 mb-3">
              {t('deepdive.transformers.simplifiedArch')}
            </h3>
            <div className="space-y-2 font-mono text-xs">
              <div className="flex items-center gap-3">
                <span className="w-24 text-right text-slate-500">Input</span>
                <div className="flex-1 p-2 rounded bg-slate-800 border border-slate-700 text-slate-300">
                  "Classify this storage I/O workload: IOPS: 45000..."
                </div>
              </div>
              <div className="flex items-center gap-3">
                <span className="w-24 text-right text-slate-500">&darr;</span>
                <span className="text-slate-600">Tokenize &rarr; convert to token IDs</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="w-24 text-right text-blue-400">Embedding</span>
                <div className="flex-1 p-2 rounded bg-blue-950/30 border border-blue-800/30 text-blue-300">
                  Each token &rarr; 960-dimensional vector (SmolLM2's hidden size)
                </div>
              </div>
              <div className="flex items-center gap-3">
                <span className="w-24 text-right text-slate-500">&darr;</span>
                <span className="text-slate-600">
                  + positional encoding (where each token is in the sequence)
                </span>
              </div>

              {/* Transformer layers block */}
              <div className="flex items-stretch gap-3">
                <span className="w-24 text-right text-violet-400 pt-2">
                  Layers
                  <br />
                  (&times;32)
                </span>
                <div className="flex-1 p-3 rounded bg-violet-950/20 border-2 border-violet-800/40 space-y-2">
                  <div className="p-2 rounded bg-violet-900/30 border border-violet-700/30 text-violet-300">
                    Self-Attention &mdash; "which other tokens should I pay attention to?"
                  </div>
                  <div className="text-center text-slate-600">&darr; add &amp; normalize</div>
                  <div className="p-2 rounded bg-slate-800 border border-slate-700 text-slate-300">
                    Feed-Forward Network &mdash; process each token independently
                  </div>
                  <div className="text-center text-slate-600">&darr; add &amp; normalize</div>
                  <p className="text-xs text-violet-400/60 italic">
                    This block repeats 32 times. Each is called a "layer."
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <span className="w-24 text-right text-slate-500">&darr;</span>
                <span className="text-slate-600">Final layer norm</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="w-24 text-right text-green-400">Output</span>
                <div className="flex-1 p-2 rounded bg-green-950/30 border border-green-800/30 text-green-300">
                  Probability distribution over ~49,152 possible next tokens
                </div>
              </div>
            </div>
          </div>

          {/* Layer abstraction */}
          <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h3 className="text-base font-semibold text-violet-400 mb-3">
              {t('deepdive.transformers.layersBuildHeading')}
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed mb-3">
              {t('deepdive.transformers.layersBuildP')}
            </p>
            <div className="space-y-2">
              <div className="flex gap-3 items-center">
                <div className="w-20 text-right">
                  <span className="text-xs font-mono text-blue-400">Layers 1-8</span>
                </div>
                <div className="flex-1 h-8 rounded bg-blue-950/30 border border-blue-800/30 flex items-center px-3">
                  <span className="text-xs text-blue-300">
                    {t('deepdive.transformers.earlyLayers')}
                  </span>
                </div>
              </div>
              <div className="flex gap-3 items-center">
                <div className="w-20 text-right">
                  <span className="text-xs font-mono text-violet-400">Layers 9-20</span>
                </div>
                <div className="flex-1 h-8 rounded bg-violet-950/30 border border-violet-800/30 flex items-center px-3">
                  <span className="text-xs text-violet-300">
                    {t('deepdive.transformers.middleLayers')}
                  </span>
                </div>
              </div>
              <div className="flex gap-3 items-center">
                <div className="w-20 text-right">
                  <span className="text-xs font-mono text-green-400">Layers 21-32</span>
                </div>
                <div className="flex-1 h-8 rounded bg-green-950/30 border border-green-800/30 flex items-center px-3">
                  <span className="text-xs text-green-300">
                    {t('deepdive.transformers.lateLayers')}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Multi-head attention summary */}
          <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h3 className="text-base font-semibold text-violet-400 mb-3">
              {t('deepdive.transformers.multiHeadHeading')}
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="p-3 rounded bg-slate-800/50 border border-slate-700/30">
                <h4 className="text-xs font-semibold text-violet-300 mb-1">
                  {t('deepdive.transformers.syntaxHeads')}
                </h4>
                <p className="text-xs text-slate-400">{t('deepdive.transformers.syntaxHeadsP')}</p>
              </div>
              <div className="p-3 rounded bg-slate-800/50 border border-slate-700/30">
                <h4 className="text-xs font-semibold text-violet-300 mb-1">
                  {t('deepdive.transformers.valueLinkingHeads')}
                </h4>
                <p className="text-xs text-slate-400">
                  {t('deepdive.transformers.valueLinkingHeadsP')}
                </p>
              </div>
              <div className="p-3 rounded bg-slate-800/50 border border-slate-700/30">
                <h4 className="text-xs font-semibold text-violet-300 mb-1">
                  {t('deepdive.transformers.crossMetricHeads')}
                </h4>
                <p className="text-xs text-slate-400">
                  {t('deepdive.transformers.crossMetricHeadsP')}
                </p>
              </div>
            </div>
            <p className="text-xs text-slate-500 mt-3">
              Explore these patterns interactively in the{' '}
              <button
                onClick={() => setSection('attention')}
                className="text-violet-400 underline hover:text-violet-300"
              >
                Attention Explorer
              </button>{' '}
              tab.
            </p>
          </div>

          {/* SmolLM2 spec card */}
          <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/50">
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">
              {t('deepdive.transformers.archSummary')}
            </h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-center text-xs">
              <div className="p-2 rounded bg-slate-900/50">
                <div className="text-lg font-bold text-slate-200">32</div>
                <div className="text-slate-500">{t('deepdive.transformers.layers32')}</div>
              </div>
              <div className="p-2 rounded bg-slate-900/50">
                <div className="text-lg font-bold text-slate-200">15</div>
                <div className="text-slate-500">{t('deepdive.transformers.headsPerLayer')}</div>
              </div>
              <div className="p-2 rounded bg-slate-900/50">
                <div className="text-lg font-bold text-slate-200">960</div>
                <div className="text-slate-500">{t('deepdive.transformers.hiddenDim')}</div>
              </div>
              <div className="p-2 rounded bg-slate-900/50">
                <div className="text-lg font-bold text-slate-200">49,152</div>
                <div className="text-slate-500">{t('deepdive.transformers.vocabSize')}</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Attention Explorer */}
      {section === 'attention' && <AttentionHeatmap />}

      {/* Tab 3: Step-by-Step Forward Pass */}
      {section === 'steps' && <ForwardPassSteps />}

      <SectionTabs tabs={TABS} active={section} onSelect={setSection} color="violet" />
    </div>
  )
}
