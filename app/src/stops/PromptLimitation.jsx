import { useTranslation, Trans } from 'react-i18next'
import ModelOutput from '../components/ModelOutput'

const HARD_INPUT =
  'IOPS: 38000 | Latency: 0.5ms | Block Size: 8K | Read/Write: 55/45 | Sequential: 22% | Queue Depth: 64'

const FEW_SHOT_RESPONSE_HARD = `Classification: OLTP Database

This pattern shows high IOPS with low latency and small block sizes. However, the read/write ratio of 55/45 is more balanced than typical OLTP workloads, and the queue depth of 64 is quite high. This could potentially be a VDI workload as well, but the small block size of 8K and random nature (78% random) lean more toward OLTP Database. Though I should note there is some uncertainty in this classification.`

export default function PromptLimitation() {
  const { t } = useTranslation()

  return (
    <div className="max-w-4xl mx-auto">
      {/* Intro — bridge from few-shot */}
      <div className="mb-5 p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-orange-400 mb-3">
          {t('stop.promptLimitation.heading')}
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          {t('stop.promptLimitation.introP1')}
        </p>
        <p className="text-sm text-slate-400 leading-relaxed">
          {t('stop.promptLimitation.introP2')}
        </p>
      </div>

      {/* The hard example */}
      <div className="mb-4">
        <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
          {t('stop.promptLimitation.ambiguousLabel')}
        </label>
        <div className="bg-slate-800 border border-yellow-700/50 rounded-lg p-3 font-mono text-sm text-yellow-200">
          {HARD_INPUT}
        </div>
        <p className="text-xs text-yellow-500/70 mt-1">
          {t('stop.promptLimitation.ambiguousHint')}
        </p>
      </div>

      <ModelOutput
        label={t('stop.promptLimitation.modelLabel')}
        text={FEW_SHOT_RESPONSE_HARD}
        variant="base"
        isCorrect={false}
      />

      {/* Three problems */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className="p-4 rounded-lg bg-red-950/20 border border-red-800/30">
          <h4 className="text-sm font-semibold text-red-400 mb-2">
            {t('stop.promptLimitation.problem1Title')}
          </h4>
          <p className="text-xs text-slate-400">{t('stop.promptLimitation.problem1P')}</p>
        </div>
        <div className="p-4 rounded-lg bg-red-950/20 border border-red-800/30">
          <h4 className="text-sm font-semibold text-red-400 mb-2">
            {t('stop.promptLimitation.problem2Title')}
          </h4>
          <p className="text-xs text-slate-400">{t('stop.promptLimitation.problem2P')}</p>
        </div>
        <div className="p-4 rounded-lg bg-red-950/20 border border-red-800/30">
          <h4 className="text-sm font-semibold text-red-400 mb-2">
            {t('stop.promptLimitation.problem3Title')}
          </h4>
          <p className="text-xs text-slate-400">
            <Trans i18nKey="stop.promptLimitation.problem3P" components={{ 1: <em /> }} />
          </p>
        </div>
      </div>

      {/* Transition callout */}
      <div className="mt-6 p-4 rounded-lg bg-gradient-to-r from-orange-950/30 to-yellow-950/30 border border-orange-800/30">
        <p className="text-sm text-slate-300">
          <Trans
            i18nKey="stop.promptLimitation.transition"
            components={{
              1: <span className="font-semibold text-orange-400" />,
              2: <span className="text-yellow-400 font-semibold" />,
              3: <span className="text-slate-200 font-semibold" />,
            }}
          />
        </p>
      </div>
    </div>
  )
}
