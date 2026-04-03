import { useState } from 'react'
import { useTranslation, Trans } from 'react-i18next'
import ModelOutput from '../components/ModelOutput'
import AnimatedPipeline from '../components/AnimatedPipeline'
import VectorSearchViz from '../components/VectorSearchViz'
import SectionTabs from '../components/SectionTabs'
import useStore from '../store'

const EXAMPLE_INPUT =
  'IOPS: 45000 | Latency: 0.3ms | Block Size: 8K | Read/Write: 70/30 | Sequential: 15% | Queue Depth: 32'

const FAILED_FEWSHOT = `Classification: OLTP Database

This pattern shows high IOPS with low latency and small block sizes. However, I should note there is some uncertainty...`

const KNOWLEDGE_BASE = [
  {
    id: 1,
    pattern: 'IOPS: 30K-80K, Block: 4-8K, Latency: <1ms, Sequential: 10-25%, R/W: 60-80/20-40',
    label: 'OLTP Database',
    description:
      'Transaction processing workloads (MySQL, PostgreSQL, Oracle). Characterized by high IOPS, small random I/O, very low latency requirements.',
    similarity: 0.94,
  },
  {
    id: 2,
    pattern: 'IOPS: 20K-60K, Block: 4-16K, Latency: <1ms, Sequential: 15-30%, R/W: 50-70/30-50',
    label: 'VDI Virtual Desktop',
    description:
      'Virtual desktop infrastructure. High IOPS, small blocks, but more balanced read/write ratio and higher queue depth than OLTP.',
    similarity: 0.71,
  },
  {
    id: 3,
    pattern: 'IOPS: 10K-30K, Block: 128K-1M, Latency: 0.5-3ms, Sequential: 85-95%, R/W: 90-99/1-10',
    label: 'AI ML Training',
    description:
      'Machine learning training workloads. Large sequential reads dominate. High throughput with moderate IOPS.',
    similarity: 0.23,
  },
]

const RAG_RESPONSE = `Classification: OLTP Database

Based on the retrieved reference patterns, this I/O profile closely matches OLTP Database workloads. The key indicators are:
- High IOPS (45,000) within the 30K-80K range typical of transaction processing
- Very low latency (0.3ms) consistent with OLTP requirements (<1ms)
- Small block size (8K) in the 4-8K range characteristic of database operations
- Predominantly random access (only 15% sequential) matching OLTP patterns
- Read-heavy ratio (70/30) within the expected 60-80/20-40 range

The retrieved reference pattern shows 94% similarity to known OLTP Database profiles.`

const RAG_PIPELINE_STEPS = [
  { icon: '\u{1F4DD}', label: 'Your Query', desc: 'I/O metric string' },
  { icon: '\u{1F9EE}', label: 'Embed', desc: 'Convert to vector' },
  { icon: '\u{1F50D}', label: 'Search', desc: 'Find nearest vectors' },
  { icon: '\u{1F4CB}', label: 'Retrieve', desc: 'Top-K documents' },
  { icon: '\u{1F4E6}', label: 'Augment', desc: 'Inject into prompt' },
  { icon: '\u{1F916}', label: 'Generate', desc: 'LLM responds' },
]

export default function RAGSimple() {
  const { t } = useTranslation()
  const [section, setSection] = useState('problem')
  const [showContextAside, setShowContextAside] = useState(false)
  const setActiveQuadrant = useStore((s) => s.setActiveQuadrant)
  const setMode = useStore((s) => s.setMode)
  const tabs = [
    { id: 'problem', label: t('tabs.theProblem') },
    { id: 'concept', label: t('tabs.howRagWorks') },
    { id: 'demo', label: t('tabs.seeItWork') },
    { id: 'deepdive', label: t('tabs.deepDiveVectorSearch') },
  ]

  return (
    <div className="max-w-5xl mx-auto">
      {/* Section tabs (top) */}
      <div className="mb-6">
        <SectionTabs tabs={tabs} active={section} onSelect={setSection} color="yellow" />
      </div>

      {/* ==================== PROBLEM ==================== */}
      {section === 'problem' && (
        <div className="space-y-4">
          <div className="p-4 rounded-lg bg-red-950/20 border border-red-800/30">
            <h3 className="text-base font-semibold text-red-400 mb-2">
              {t('stop.ragSimple.problem.heading')}
            </h3>
            <p className="text-sm text-slate-300 mb-3">{t('stop.ragSimple.problem.intro')}</p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="p-3 rounded bg-slate-800/50">
                <h4 className="text-sm font-semibold text-red-300 mb-1">
                  {t('stop.ragSimple.problem.contextFinite')}
                </h4>
                <p className="text-xs text-slate-400">
                  {t('stop.ragSimple.problem.contextFiniteP')}
                </p>
                <button
                  onClick={() => setShowContextAside(!showContextAside)}
                  className="mt-2 text-xs text-purple-400 hover:text-purple-300 underline underline-offset-4 cursor-pointer"
                >
                  {showContextAside ? 'Hide' : t('stop.ragSimple.problem.whyMatter')}
                </button>
                {showContextAside && (
                  <div className="mt-2 p-3 rounded bg-purple-950/20 border border-purple-800/30">
                    <p className="text-xs text-slate-300 leading-relaxed mb-2">
                      <Trans
                        i18nKey="stop.ragSimple.problem.contextRotP"
                        components={{
                          1: <strong className="text-purple-300" />,
                          2: <strong className="text-purple-300" />,
                        }}
                      />
                    </p>
                    <button
                      onClick={() => {
                        setActiveQuadrant('context')
                        setMode('explore')
                      }}
                      className="text-xs font-semibold text-purple-400 hover:text-purple-300 underline underline-offset-4 cursor-pointer"
                    >
                      {t('stop.ragSimple.problem.contextDeepDive')}
                    </button>
                  </div>
                )}
              </div>
              <div className="p-3 rounded bg-slate-800/50">
                <h4 className="text-sm font-semibold text-red-300 mb-1">
                  {t('stop.ragSimple.problem.static')}
                </h4>
                <p className="text-xs text-slate-400">{t('stop.ragSimple.problem.staticP')}</p>
              </div>
              <div className="p-3 rounded bg-slate-800/50">
                <h4 className="text-sm font-semibold text-red-300 mb-1">
                  {t('stop.ragSimple.problem.cantScale')}
                </h4>
                <p className="text-xs text-slate-400">{t('stop.ragSimple.problem.cantScaleP')}</p>
              </div>
            </div>
          </div>

          <div className="p-4 rounded-lg bg-yellow-950/20 border border-yellow-800/30">
            <p className="text-sm text-slate-300">
              <Trans
                i18nKey="stop.ragSimple.problem.whatIfLookUp"
                components={{ 1: <span className="font-semibold text-yellow-400" /> }}
              />
            </p>
            <p className="text-sm text-slate-400 mt-2">{t('stop.ragSimple.problem.thatsRag')}</p>
          </div>
        </div>
      )}

      {/* ==================== CONCEPT ==================== */}
      {section === 'concept' && (
        <div className="space-y-4">
          <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h3 className="text-base font-semibold text-yellow-400 mb-2">
              {t('stop.ragSimple.concept.ragEquals')}
            </h3>
            <p className="text-sm text-slate-300 mb-4">
              <Trans
                i18nKey="stop.ragSimple.concept.ragP"
                components={{ 1: <strong className="text-yellow-300" /> }}
              />
            </p>

            {/* Animated pipeline */}
            <AnimatedPipeline steps={RAG_PIPELINE_STEPS} autoPlay speed={900} />
          </div>

          {/* Key concept cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/50">
              <h4 className="text-sm font-semibold text-yellow-400 mb-2">
                {t('stop.ragSimple.concept.embeddings')}
              </h4>
              <p className="text-xs text-slate-400 mb-2">
                {t('stop.ragSimple.concept.embeddingsP')}
              </p>
              <div className="p-2 rounded bg-slate-900 font-mono text-xs text-slate-500">
                "High IOPS, small blocks" → [0.82, -0.14, 0.67, 0.31, ...]
                <br />
                "Many random reads, 4K" → [0.79, -0.11, 0.71, 0.28, ...]
                <br />
                <span className="text-green-400">
                  {t('stop.ragSimple.concept.similarMeaningSimilarNumbers')}
                </span>
              </div>
            </div>
            <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/50">
              <h4 className="text-sm font-semibold text-yellow-400 mb-2">
                {t('stop.ragSimple.concept.vectorSearch')}
              </h4>
              <p className="text-xs text-slate-400 mb-2">
                {t('stop.ragSimple.concept.vectorSearchP')}
              </p>
              <div className="p-2 rounded bg-slate-900 font-mono text-xs text-slate-500">
                cosine_similarity(query, doc) = (q · d) / (|q| × |d|)
                <br />
                <span className="text-green-400">Range: 0 (unrelated) to 1 (identical)</span>
              </div>
            </div>
          </div>

          {/* Critical point */}
          <div className="p-3 rounded bg-blue-950/20 border border-blue-800/30">
            <p className="text-xs text-blue-300">
              <Trans
                i18nKey="stop.ragSimple.concept.keyInsight"
                components={{
                  1: <strong />,
                  2: <em />,
                  3: <em />,
                }}
              />
            </p>
          </div>
        </div>
      )}

      {/* ==================== DEMO ==================== */}
      {section === 'demo' && (
        <div className="space-y-4">
          {/* Input */}
          <div>
            <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
              {t('stop.ragSimple.demo.queryLabel')}
            </label>
            <div className="bg-slate-800 border border-slate-600 rounded-lg p-3 font-mono text-sm text-slate-200">
              {EXAMPLE_INPUT}
            </div>
          </div>

          {/* Retrieval results */}
          <div>
            <h4 className="text-xs font-semibold text-yellow-400 uppercase tracking-wide mb-2">
              {t('stop.ragSimple.demo.step1')}
            </h4>
            <div className="space-y-2">
              {KNOWLEDGE_BASE.map((doc) => (
                <div
                  key={doc.id}
                  className={`p-3 rounded-lg border ${
                    doc.similarity > 0.8
                      ? 'border-green-700/50 bg-green-950/20'
                      : doc.similarity > 0.5
                        ? 'border-yellow-700/50 bg-yellow-950/10'
                        : 'border-slate-700/50 bg-slate-800/30'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-semibold text-slate-300">{doc.label}</span>
                    <span
                      className={`text-xs px-2 py-0.5 rounded-full font-semibold ${
                        doc.similarity > 0.8
                          ? 'bg-green-900/50 text-green-400'
                          : 'bg-slate-700 text-slate-400'
                      }`}
                    >
                      {t('stop.ragSimple.demo.match', {
                        percent: (doc.similarity * 100).toFixed(0),
                      })}
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 font-mono">{doc.pattern}</p>
                  <p className="text-xs text-slate-400 mt-1">{doc.description}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Augmented response */}
          <div>
            <h4 className="text-xs font-semibold text-yellow-400 uppercase tracking-wide mb-2">
              {t('stop.ragSimple.demo.step2')}
            </h4>
            <ModelOutput
              label={t('stop.ragSimple.demo.modelLabel')}
              text={RAG_RESPONSE}
              variant="rag"
              isCorrect={true}
            />
          </div>

          <div className="p-3 rounded bg-green-950/20 border border-green-800/30">
            <p className="text-sm text-green-300">{t('stop.ragSimple.demo.resultP')}</p>
          </div>
        </div>
      )}

      {/* ==================== DEEP DIVE ==================== */}
      {section === 'deepdive' && (
        <div className="space-y-4">
          <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h3 className="text-base font-semibold text-yellow-400 mb-2">
              {t('stop.ragSimple.deepdive.heading')}
            </h3>
            <p className="text-sm text-slate-300 mb-4">{t('stop.ragSimple.deepdive.p')}</p>
            <VectorSearchViz autoPlay />
          </div>

          {/* What's happening at each step */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/50">
              <h4 className="text-xs font-semibold text-yellow-300 uppercase tracking-wide mb-2">
                {t('stop.ragSimple.deepdive.whyCluster')}
              </h4>
              <p className="text-xs text-slate-400">{t('stop.ragSimple.deepdive.whyClusterP')}</p>
            </div>
            <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/50">
              <h4 className="text-xs font-semibold text-yellow-300 uppercase tracking-wide mb-2">
                {t('stop.ragSimple.deepdive.infraForVector')}
              </h4>
              <p className="text-xs text-slate-400">
                <Trans
                  i18nKey="stop.ragSimple.deepdive.infraForVectorP"
                  components={{ 1: <strong /> }}
                />
              </p>
            </div>
          </div>

          {/* Where storage fits */}
          <div className="p-3 rounded bg-yellow-950/20 border border-yellow-800/30">
            <h4 className="text-xs font-semibold text-yellow-400 mb-1">
              {t('stop.ragSimple.deepdive.whereStorageFits')}
            </h4>
            <div className="grid grid-cols-3 gap-3 mt-2 text-xs text-slate-400">
              <div>
                <span className="font-semibold text-slate-300">
                  {t('stop.ragSimple.deepdive.embeddingIndexes')}
                </span>
                <br />
                {t('stop.ragSimple.deepdive.embeddingIndexesP')}
              </div>
              <div>
                <span className="font-semibold text-slate-300">
                  {t('stop.ragSimple.deepdive.sourceDocuments')}
                </span>
                <br />
                {t('stop.ragSimple.deepdive.sourceDocumentsP')}
              </div>
              <div>
                <span className="font-semibold text-slate-300">
                  {t('stop.ragSimple.deepdive.queryLatency')}
                </span>
                <br />
                {t('stop.ragSimple.deepdive.queryLatencyP')}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Section tabs (bottom) */}
      <div className="mt-6">
        <SectionTabs tabs={tabs} active={section} onSelect={setSection} color="yellow" />
      </div>
    </div>
  )
}
