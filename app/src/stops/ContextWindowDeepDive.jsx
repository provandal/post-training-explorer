// Bonus stop: Deep dive into context windows, context rot, and management strategies.
// Accessible from Explore mode via the "Context window is finite" aside in RAGSimple.

import { useTranslation } from 'react-i18next'

export default function ContextWindowDeepDive() {
  const { t } = useTranslation()

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <span className="text-xs font-semibold text-purple-400 uppercase tracking-wide">
          {t('deepdive.context.bonusLabel')}
        </span>
        <h2 className="text-xl font-bold text-white mt-1">{t('deepdive.context.title')}</h2>
        <p className="text-sm text-slate-400 mt-2">{t('deepdive.context.subtitle')}</p>
      </div>

      {/* What is a context window */}
      <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-purple-400 mb-3">
          {t('deepdive.context.whatIs')}
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          {t('deepdive.context.whatIsP')}
        </p>
        <div className="p-4 rounded-lg bg-slate-900/50 border border-slate-700/30 font-mono text-xs text-slate-400 space-y-1">
          <div className="flex items-center gap-2">
            <span className="w-20 text-right text-purple-400">{t('deepdive.context.system')}</span>
            <div className="flex-1 h-5 bg-purple-900/40 rounded flex items-center px-2 text-purple-300">
              "You are a storage I/O expert..."
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-20 text-right text-yellow-400">{t('deepdive.context.ragDocs')}</span>
            <div className="flex-1 h-5 bg-yellow-900/30 rounded flex items-center px-2 text-yellow-300">
              {t('deepdive.context.ragDocsDesc')}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-20 text-right text-blue-400">{t('deepdive.context.history')}</span>
            <div className="flex-1 h-5 bg-blue-900/30 rounded flex items-center px-2 text-blue-300">
              {t('deepdive.context.historyDesc')}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-20 text-right text-green-400">{t('deepdive.context.query')}</span>
            <div className="flex-1 h-5 bg-green-900/30 rounded flex items-center px-2 text-green-300">
              "Classify this I/O pattern: IOPS: 45000..."
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-20 text-right text-slate-500">{t('deepdive.context.response')}</span>
            <div className="flex-1 h-5 bg-slate-800 rounded border border-dashed border-slate-600 flex items-center px-2 text-slate-500">
              {t('deepdive.context.responseDesc')}
            </div>
          </div>
          <div className="mt-2 pt-2 border-t border-slate-700/50 text-slate-500 text-center">
            {t('deepdive.context.totalFit')}
          </div>
        </div>
        <div className="mt-3 grid grid-cols-3 gap-2 text-xs text-slate-400">
          <div className="p-2 rounded bg-slate-900/30">
            <span className="font-semibold text-slate-300">SmolLM2-360M</span>
            <br />
            2,048 tokens
          </div>
          <div className="p-2 rounded bg-slate-900/30">
            <span className="font-semibold text-slate-300">GPT-4o / Claude</span>
            <br />
            128K&ndash;200K tokens
          </div>
          <div className="p-2 rounded bg-slate-900/30">
            <span className="font-semibold text-slate-300">Gemini 1.5 Pro</span>
            <br />
            1M+ tokens
          </div>
        </div>
      </div>

      {/* How context is built */}
      <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-purple-400 mb-3">
          {t('deepdive.context.howBuilt')}
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-4">
          {t('deepdive.context.howBuiltP')}
        </p>
        <div className="space-y-3">
          <div className="flex gap-3 items-start">
            <span className="text-xs font-mono text-purple-400 w-28 shrink-0 pt-0.5">
              {t('deepdive.context.systemPrompt')}
            </span>
            <p className="text-xs text-slate-400">{t('deepdive.context.systemPromptP')}</p>
          </div>
          <div className="flex gap-3 items-start">
            <span className="text-xs font-mono text-yellow-400 w-28 shrink-0 pt-0.5">
              {t('deepdive.context.ragInjection')}
            </span>
            <p className="text-xs text-slate-400">{t('deepdive.context.ragInjectionP')}</p>
          </div>
          <div className="flex gap-3 items-start">
            <span className="text-xs font-mono text-blue-400 w-28 shrink-0 pt-0.5">
              {t('deepdive.context.conversation')}
            </span>
            <p className="text-xs text-slate-400">{t('deepdive.context.conversationP')}</p>
          </div>
          <div className="flex gap-3 items-start">
            <span className="text-xs font-mono text-orange-400 w-28 shrink-0 pt-0.5">
              {t('deepdive.context.fewShot')}
            </span>
            <p className="text-xs text-slate-400">{t('deepdive.context.fewShotP')}</p>
          </div>
        </div>
      </div>

      {/* Context rot */}
      <div className="p-5 rounded-lg bg-red-950/20 border border-red-800/30">
        <h3 className="text-base font-semibold text-red-400 mb-3">
          {t('deepdive.context.contextRot')}
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          {t('deepdive.context.contextRotP')}
        </p>
        <div className="space-y-3">
          <div className="p-3 rounded bg-slate-800/50">
            <h4 className="text-sm font-semibold text-red-300 mb-1">
              {t('deepdive.context.lostInMiddle')}
            </h4>
            <p className="text-xs text-slate-400">
              {t('deepdive.context.lostInMiddleP', {
                defaultValue:
                  'Research shows that models pay the most attention to the beginning and end of the context window. Information placed in the middle is disproportionately ignored. If your most relevant RAG document lands in the middle of 10 retrieved passages, the model may miss it entirely.',
              })}
            </p>
          </div>
          <div className="p-3 rounded bg-slate-800/50">
            <h4 className="text-sm font-semibold text-red-300 mb-1">
              {t('deepdive.context.attentionDilution')}
            </h4>
            <p className="text-xs text-slate-400">{t('deepdive.context.attentionDilutionP')}</p>
          </div>
          <div className="p-3 rounded bg-slate-800/50">
            <h4 className="text-sm font-semibold text-red-300 mb-1">
              {t('deepdive.context.contradictory')}
            </h4>
            <p className="text-xs text-slate-400">{t('deepdive.context.contradictoryP')}</p>
          </div>
        </div>
      </div>

      {/* Management strategies */}
      <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-purple-400 mb-3">
          {t('deepdive.context.strategies')}
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="p-3 rounded bg-slate-800/50 border border-slate-700/30">
            <h4 className="text-xs font-semibold text-purple-300 uppercase tracking-wide mb-2">
              {t('deepdive.context.slidingWindow')}
            </h4>
            <p className="text-xs text-slate-400">{t('deepdive.context.slidingWindowP')}</p>
          </div>
          <div className="p-3 rounded bg-slate-800/50 border border-slate-700/30">
            <h4 className="text-xs font-semibold text-purple-300 uppercase tracking-wide mb-2">
              {t('deepdive.context.summarization')}
            </h4>
            <p className="text-xs text-slate-400">{t('deepdive.context.summarizationP')}</p>
          </div>
          <div className="p-3 rounded bg-slate-800/50 border border-slate-700/30">
            <h4 className="text-xs font-semibold text-purple-300 uppercase tracking-wide mb-2">
              {t('deepdive.context.structuredMemory')}
            </h4>
            <p className="text-xs text-slate-400">{t('deepdive.context.structuredMemoryP')}</p>
          </div>
          <div className="p-3 rounded bg-slate-800/50 border border-slate-700/30">
            <h4 className="text-xs font-semibold text-purple-300 uppercase tracking-wide mb-2">
              {t('deepdive.context.ragAsRelief')}
            </h4>
            <p className="text-xs text-slate-400">{t('deepdive.context.ragAsReliefP')}</p>
          </div>
        </div>
      </div>

      {/* Connection to post-training */}
      <div className="p-4 rounded-lg bg-gradient-to-r from-purple-950/30 to-slate-800 border border-purple-800/30">
        <h4 className="text-sm font-semibold text-purple-400 mb-2">
          {t('deepdive.context.postTrainingConnection')}
        </h4>
        <p className="text-sm text-slate-300 leading-relaxed mb-2">
          {t('deepdive.context.postTrainingP1', {
            defaultValue:
              "Context management is an inference-time problem — you're engineering around the model's limitations. Post-training attacks the problem from the other direction: if the model already knows your domain, you need less context to get a good answer.",
          })}
        </p>
        <p className="text-sm text-slate-400 leading-relaxed">
          {t('deepdive.context.postTrainingP2')}
        </p>
      </div>
    </div>
  )
}
