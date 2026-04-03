import { useTranslation, Trans } from 'react-i18next'
import ModelOutput from '../components/ModelOutput'

const EXAMPLE_INPUT =
  'IOPS: 38000 | Latency: 0.5ms | Block Size: 8K | Read/Write: 55/45 | Sequential: 22% | Queue Depth: 64'

const RAG_VERBOSE_RESPONSE = `Based on the retrieved reference patterns, this I/O profile presents an interesting case. The retrieved documents show two potential matches:

1. OLTP Database (71% similarity) - The high IOPS and low latency are consistent with OLTP workloads. The small block size of 8K also supports this classification. However, I should note that the read/write ratio of 55/45 is more balanced than the typical OLTP range of 60-80/20-40.

2. VDI Virtual Desktop (68% similarity) - The balanced read/write ratio and high queue depth of 64 are more characteristic of VDI workloads. The small block size could fit either pattern.

Given the ambiguity, this pattern could potentially be classified as either OLTP Database or VDI Virtual Desktop. The balanced read/write ratio and high queue depth suggest VDI might be slightly more appropriate, but without additional context about the specific deployment environment, infrastructure configuration, and application characteristics, I cannot make a definitive determination. It would be advisable to collect additional metrics such as I/O size distribution histograms, temporal patterns, and application-level metadata to make a more informed classification.

My best assessment is that this is likely a VDI Virtual Desktop workload, though OLTP Database cannot be ruled out entirely.`

const DESIRED_RESPONSE = `Classification: VDI Virtual Desktop
Confidence: Medium
Key indicators: Balanced R/W ratio (55/45) and high queue depth (64) distinguish this from OLTP. Small blocks (8K) with random access are consistent with VDI.`

export default function RAGLimitation() {
  const { t } = useTranslation()

  return (
    <div className="max-w-4xl mx-auto">
      {/* Intro — bridge from RAG success */}
      <div className="mb-5 p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-yellow-400 mb-3">
          {t('stop.ragLimitation.heading')}
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          {t('stop.ragLimitation.introP1')}
        </p>
        <p className="text-sm text-slate-400 leading-relaxed">
          <Trans i18nKey="stop.ragLimitation.introP2" components={{ 1: <em /> }} />
        </p>
      </div>

      {/* Input */}
      <div className="mb-4">
        <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
          {t('stop.ragLimitation.inputLabel')}
        </label>
        <div className="bg-slate-800 border border-yellow-700/50 rounded-lg p-3 font-mono text-sm text-yellow-200">
          {EXAMPLE_INPUT}
        </div>
      </div>

      {/* RAG response - correct but problematic */}
      <ModelOutput
        label={t('stop.ragLimitation.ragModelLabel')}
        text={RAG_VERBOSE_RESPONSE}
        variant="rag"
        isCorrect={true}
      />

      {/* What we wanted */}
      <div className="mt-4">
        <ModelOutput
          label={t('stop.ragLimitation.desiredLabel')}
          text={DESIRED_RESPONSE}
          variant="default"
        />
      </div>

      {/* The problems */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-3">
        <div className="p-4 rounded-lg bg-yellow-950/20 border border-yellow-800/30">
          <h4 className="text-sm font-semibold text-yellow-400 mb-2">
            {t('stop.ragLimitation.rightAnswerWrongDelivery')}
          </h4>
          <p className="text-xs text-slate-400">{t('stop.ragLimitation.rightAnswerP')}</p>
        </div>
        <div className="p-4 rounded-lg bg-yellow-950/20 border border-yellow-800/30">
          <h4 className="text-sm font-semibold text-yellow-400 mb-2">
            {t('stop.ragLimitation.behaviorVsKnowledge')}
          </h4>
          <p className="text-xs text-slate-400">
            <Trans
              i18nKey="stop.ragLimitation.behaviorVsKnowledgeP"
              components={{ 1: <em />, 2: <em /> }}
            />
          </p>
        </div>
      </div>

      {/* Transition to post-training */}
      <div className="mt-6 p-4 rounded-lg bg-gradient-to-r from-yellow-950/30 to-slate-800 border border-slate-600/30">
        <p className="text-sm text-slate-300">
          <Trans
            i18nKey="stop.ragLimitation.transition"
            components={{
              1: <span className="font-semibold text-yellow-400" />,
              2: <em />,
              3: <em />,
              4: <span className="text-slate-100 font-semibold" />,
            }}
          />
        </p>
      </div>
    </div>
  )
}
