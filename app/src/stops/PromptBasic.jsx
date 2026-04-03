import { useState } from 'react'
import { useTranslation, Trans } from 'react-i18next'
import ModelOutput from '../components/ModelOutput'
import TokenProbChart from '../components/TokenProbChart'
import SectionTabs from '../components/SectionTabs'
import { isLoaded, getModelOutput, getTokenProbsForChart } from '../data/loadArtifacts'
import useStore from '../store'

// ---------------------------------------------------------------------------
// Precomputed data
// ---------------------------------------------------------------------------

const EXAMPLE_INPUT =
  'IOPS: 45000 | Latency: 0.3ms | Block Size: 8K | Read/Write: 70/30 | Sequential: 15% | Queue Depth: 32'

const BASE_RESPONSE = `This appears to be some kind of storage performance metrics. The IOPS value is 45000 which is relatively high. The latency is 0.3ms which is quite low. The block size is 8K. There is a 70/30 read to write ratio with 15% sequential access and a queue depth of 32. These metrics could be from various types of storage workloads depending on the specific use case and configuration being measured.`

const FALLBACK_BASE_TOKEN_PROBS = [
  { token: 'This', probability: 0.18 },
  { token: 'The', probability: 0.14 },
  { token: 'Based', probability: 0.09 },
  { token: 'OLTP', probability: 0.04 },
  { token: 'These', probability: 0.07 },
  { token: 'It', probability: 0.06 },
  { token: 'Storage', probability: 0.05 },
  { token: 'High', probability: 0.04 },
  { token: 'Looking', probability: 0.04 },
  { token: 'I', probability: 0.03 },
  { token: 'Database', probability: 0.03 },
  { token: 'A', probability: 0.03 },
  { token: 'With', probability: 0.02 },
  { token: 'VDI', probability: 0.02 },
  { token: 'Given', probability: 0.02 },
]

// ---------------------------------------------------------------------------
// Prompt variants for "What is Prompting?" tab
// ---------------------------------------------------------------------------

const PROMPT_VARIANTS = [
  {
    id: 'vague',
    style: 'Vague',
    tagColor: 'bg-red-900/50 text-red-400',
    prompt: `What is this?\n\n${EXAMPLE_INPUT}`,
    response: `This is a set of numerical parameters or specifications, possibly related to computer storage or disk performance benchmarking. IOPS stands for Input/Output Operations Per Second, which measures how fast a storage device can read and write. The other values describe characteristics of the workload being measured.`,
    verdict:
      'The model has no idea what we actually want. It explains the acronyms like a dictionary.',
    verdictColor: 'text-red-300',
  },
  {
    id: 'structured',
    style: 'Structured',
    tagColor: 'bg-yellow-900/50 text-yellow-400',
    prompt: `You are a storage I/O expert. Given the following metrics, classify the workload into one of these categories: OLTP Database, OLAP Analytics, AI/ML Training, Video Streaming, VDI Virtual Desktop, Backup/Archive.\n\nMetrics:\n${EXAMPLE_INPUT}`,
    response: `Based on the provided metrics, this workload appears to be consistent with an OLTP Database pattern. The high IOPS (45,000) combined with very low latency (0.3ms) and small block size (8K) are characteristic of transactional database workloads. The predominantly random access pattern (only 15% sequential) further supports this classification.`,
    verdict:
      'Much better! Giving it categories and a role narrows its focus. But it hedges with "appears to be."',
    verdictColor: 'text-yellow-300',
  },
  {
    id: 'instruction',
    style: 'Instruction-tuned',
    tagColor: 'bg-green-900/50 text-green-400',
    prompt: `Classify the following storage I/O pattern into exactly one category.\n\nCategories: OLTP Database, OLAP Analytics, AI/ML Training, Video Streaming, VDI Virtual Desktop, Backup/Archive\n\nRespond with ONLY the category name, then a confidence level (High/Medium/Low), then a one-sentence justification.\n\nPattern:\n${EXAMPLE_INPUT}`,
    response: `OLTP Database\nConfidence: High\nHigh IOPS (45K) with sub-millisecond latency and small random I/O blocks are the signature of transaction processing.`,
    verdict:
      'Tight instructions produce tight output. The model follows the exact format we asked for.',
    verdictColor: 'text-green-300',
  },
]

// ---------------------------------------------------------------------------
// Full prompt text for the "See It Work" tab
// ---------------------------------------------------------------------------

const SYSTEM_MESSAGE = `You are a storage workload classification assistant. When given storage I/O metrics, classify the workload into one of these categories: OLTP Database, OLAP Analytics, AI/ML Training, Video Streaming, VDI Virtual Desktop, Backup/Archive.`

const USER_MESSAGE = `Classify this storage I/O pattern into a workload type:\n\n${EXAMPLE_INPUT}`

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function PromptBasic() {
  const { t } = useTranslation()
  const [section, setSection] = useState('problem')
  const [selectedVariant, setSelectedVariant] = useState(0)
  const selectedPromptId = useStore((s) => s.selectedPromptId)

  // Use real token prob data when available, fall back to hardcoded
  const baseTokenProbs = (() => {
    if (!isLoaded()) return FALLBACK_BASE_TOKEN_PROBS
    const real = getTokenProbsForChart('base', selectedPromptId)
    return real.length > 0 ? real : FALLBACK_BASE_TOKEN_PROBS
  })()

  // Use real base model output for the demo tab when available
  const demoBaseResponse = (() => {
    if (!isLoaded()) return BASE_RESPONSE
    const output = getModelOutput('base', 0)
    return output?.generated_text ?? BASE_RESPONSE
  })()

  const tabs = [
    { id: 'problem', label: t('tabs.theProblem') },
    { id: 'concept', label: t('tabs.whatIsPrompting') },
    { id: 'demo', label: t('tabs.seeItWork') },
    { id: 'deepdive', label: t('tabs.underTheCovers') },
  ]

  return (
    <div className="max-w-5xl mx-auto">
      {/* ---- Tab bar (top) ---- */}
      <div className="mb-6">
        <SectionTabs tabs={tabs} active={section} onSelect={setSection} color="orange" />
      </div>

      {/* ==================== THE PROBLEM ==================== */}
      {section === 'problem' && (
        <div className="space-y-5">
          {/* Framing */}
          <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h3 className="text-base font-semibold text-orange-400 mb-3">
              {t('stop.promptBasic.smartModelNoKnowledge')}
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed mb-3">
              {t('stop.promptBasic.smartModelParagraph1')}
            </p>
            <p className="text-sm text-slate-400 leading-relaxed">
              {t('stop.promptBasic.smartModelParagraph2')}
            </p>
          </div>

          {/* The raw input */}
          <div>
            <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
              {t('stop.promptBasic.storageIOPattern')}
            </label>
            <div className="bg-slate-800 border border-slate-600 rounded-lg p-3 font-mono text-sm text-slate-200">
              {EXAMPLE_INPUT}
            </div>
            <p className="text-xs text-slate-500 mt-1">{t('stop.promptBasic.classicOltpHint')}</p>
          </div>

          {/* Base model output */}
          <ModelOutput
            label={t('stop.promptBasic.baseModelResponse')}
            text={BASE_RESPONSE}
            variant="base"
            isCorrect={false}
          />

          {/* Commentary */}
          <div className="p-4 rounded-lg bg-red-950/20 border border-red-800/30">
            <h4 className="text-sm font-semibold text-red-400 mb-2">
              {t('stop.promptBasic.whatWentWrong')}
            </h4>
            <p className="text-sm text-slate-300 leading-relaxed mb-2">
              <Trans i18nKey="stop.promptBasic.whatWentWrongP1" components={{ 1: <em /> }} />
            </p>
            <p className="text-sm text-slate-400 leading-relaxed">
              {t('stop.promptBasic.whatWentWrongP2')}
            </p>
          </div>

          {/* The six categories */}
          <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/50">
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">
              {t('stop.promptBasic.sixCategories')}
            </h4>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
              {[
                { nameKey: 'categories.oltp', descKey: 'categories.oltpDesc' },
                { nameKey: 'categories.olap', descKey: 'categories.olapDesc' },
                { nameKey: 'categories.aiml', descKey: 'categories.aimlDesc' },
                { nameKey: 'categories.video', descKey: 'categories.videoDesc' },
                { nameKey: 'categories.vdi', descKey: 'categories.vdiDesc' },
                { nameKey: 'categories.backup', descKey: 'categories.backupDesc' },
              ].map((cat) => (
                <div
                  key={cat.nameKey}
                  className="p-2 rounded bg-slate-900/50 border border-slate-700/30"
                >
                  <div className="text-xs font-semibold text-orange-300">{t(cat.nameKey)}</div>
                  <div className="text-xs text-slate-500 mt-0.5">{t(cat.descKey)}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Transition */}
          <div className="p-3 rounded bg-orange-950/20 border border-orange-800/30">
            <p className="text-sm text-orange-300">
              <Trans
                i18nKey="stop.promptBasic.transitionToPrompting"
                components={{ 1: <strong />, 2: <em /> }}
              />
            </p>
          </div>
        </div>
      )}

      {/* ==================== WHAT IS PROMPTING? ==================== */}
      {section === 'concept' && (
        <div className="space-y-5">
          {/* Intro */}
          <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h3 className="text-base font-semibold text-orange-400 mb-3">
              {t('stop.promptBasic.concept.heading')}
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed mb-3">
              {t('stop.promptBasic.concept.p1')}
            </p>
            <p className="text-sm text-slate-400 leading-relaxed">
              <Trans
                i18nKey="stop.promptBasic.concept.p2"
                components={{ 1: <strong className="text-orange-300" /> }}
              />
            </p>
          </div>

          {/* Analogy box */}
          <div className="p-4 rounded-lg bg-blue-950/20 border border-blue-800/30">
            <p className="text-xs text-blue-300 leading-relaxed">
              <Trans i18nKey="stop.promptBasic.concept.keyInsight" components={{ 1: <strong /> }} />
            </p>
          </div>

          {/* Three prompt variants */}
          <div>
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">
              {t('stop.promptBasic.concept.threePromptsHeading')}
            </h4>
            <div className="flex gap-2 mb-4">
              {PROMPT_VARIANTS.map((v, i) => (
                <button
                  key={v.id}
                  onClick={() => setSelectedVariant(i)}
                  className={`px-3 py-1.5 text-xs rounded-md transition-colors ${
                    i === selectedVariant
                      ? 'bg-orange-600 text-white font-semibold'
                      : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                  }`}
                >
                  {v.style}
                </button>
              ))}
            </div>

            {/* Selected variant */}
            {(() => {
              const v = PROMPT_VARIANTS[selectedVariant]
              return (
                <div className="space-y-3">
                  {/* Prompt text */}
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
                        {t('stop.promptBasic.concept.promptLabel')}
                      </span>
                      <span
                        className={`text-xs px-2 py-0.5 rounded-full font-semibold ${v.tagColor}`}
                      >
                        {v.style}
                      </span>
                    </div>
                    <pre className="bg-slate-800 border border-slate-600 rounded-lg p-3 font-mono text-sm text-slate-200 whitespace-pre-wrap leading-relaxed">
                      {v.prompt}
                    </pre>
                  </div>

                  {/* Model response */}
                  <div>
                    <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide block mb-1">
                      {t('stop.promptBasic.concept.modelResponse')}
                    </span>
                    <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-3">
                      <pre className="text-sm text-slate-200 whitespace-pre-wrap font-mono leading-relaxed">
                        {v.response}
                      </pre>
                    </div>
                  </div>

                  {/* Verdict */}
                  <div className="p-3 rounded bg-slate-800/30 border border-slate-700/30">
                    <p className={`text-sm ${v.verdictColor}`}>{v.verdict}</p>
                  </div>
                </div>
              )
            })()}
          </div>

          {/* Progression summary */}
          <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/50">
            <h4 className="text-sm font-semibold text-orange-400 mb-3">
              {t('stop.promptBasic.concept.whatChanged')}
            </h4>
            <div className="space-y-2">
              <div className="flex gap-3 items-start">
                <span className="text-xs font-mono text-red-400 w-24 shrink-0 pt-0.5">
                  {t('stop.promptBasic.variant.vague')}
                </span>
                <p className="text-xs text-slate-400">
                  {t('stop.promptBasic.concept.vagueExplain')}
                </p>
              </div>
              <div className="flex gap-3 items-start">
                <span className="text-xs font-mono text-yellow-400 w-24 shrink-0 pt-0.5">
                  {t('stop.promptBasic.variant.structured')}
                </span>
                <p className="text-xs text-slate-400">
                  {t('stop.promptBasic.concept.structuredExplain')}
                </p>
              </div>
              <div className="flex gap-3 items-start">
                <span className="text-xs font-mono text-green-400 w-24 shrink-0 pt-0.5">
                  {t('stop.promptBasic.variant.instruction')}
                </span>
                <p className="text-xs text-slate-400">
                  {t('stop.promptBasic.concept.instructionExplain')}
                </p>
              </div>
            </div>
          </div>

          {/* Limitations teaser */}
          <div className="p-3 rounded bg-orange-950/20 border border-orange-800/30">
            <p className="text-sm text-orange-300">
              <Trans
                i18nKey="stop.promptBasic.concept.limitationsTeaser"
                components={{ 1: <em /> }}
              />
            </p>
          </div>
        </div>
      )}

      {/* ==================== SEE IT WORK ==================== */}
      {section === 'demo' && (
        <div className="space-y-5">
          {/* Input */}
          <div>
            <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
              {t('stop.promptBasic.storageIOPattern')}
            </label>
            <div className="bg-slate-800 border border-slate-600 rounded-lg p-3 font-mono text-sm text-slate-200">
              {EXAMPLE_INPUT}
            </div>
          </div>

          {/* Show the actual prompt */}
          <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h4 className="text-xs font-semibold text-orange-400 uppercase tracking-wide mb-3">
              {t('stop.promptBasic.demo.whatModelSees')}
            </h4>
            <p className="text-xs text-slate-400 mb-3">
              {t('stop.promptBasic.demo.whatModelSeesP')}
            </p>

            {/* System message */}
            <div className="mb-3">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
                  {t('stop.promptBasic.demo.systemMessage')}
                </span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-slate-700 text-slate-400">
                  {t('stop.promptBasic.demo.setsTheRole')}
                </span>
              </div>
              <pre className="bg-slate-900 border border-slate-700 rounded-lg p-3 font-mono text-xs text-slate-300 whitespace-pre-wrap leading-relaxed">
                {SYSTEM_MESSAGE}
              </pre>
            </div>

            {/* User message */}
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
                  {t('stop.promptBasic.demo.userMessage')}
                </span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-slate-700 text-slate-400">
                  {t('stop.promptBasic.demo.yourQuery')}
                </span>
              </div>
              <pre className="bg-slate-900 border border-slate-700 rounded-lg p-3 font-mono text-xs text-slate-300 whitespace-pre-wrap leading-relaxed">
                {USER_MESSAGE}
              </pre>
            </div>
          </div>

          {/* Model response */}
          <ModelOutput
            label={t('stop.promptBasic.demo.baseModelResponseNoFt')}
            text={demoBaseResponse}
            variant="base"
            isCorrect={false}
          />

          {/* Explanation */}
          <div className="p-4 rounded-lg bg-red-950/20 border border-red-800/30">
            <h4 className="text-sm font-semibold text-red-400 mb-2">
              {t('stop.promptBasic.demo.resultIncorrect')}
            </h4>
            <p className="text-sm text-slate-300 leading-relaxed mb-2">
              {t('stop.promptBasic.demo.resultP1')}
            </p>
            <p className="text-sm text-slate-400 leading-relaxed">
              <Trans
                i18nKey="stop.promptBasic.demo.resultP2"
                components={{ 1: <strong className="text-green-400" /> }}
              />
            </p>
          </div>

          {/* What prompting can vs. cannot do */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="p-3 rounded-lg bg-green-950/20 border border-green-800/30">
              <h4 className="text-xs font-semibold text-green-400 uppercase tracking-wide mb-2">
                {t('stop.promptBasic.demo.whatPromptingCanDo')}
              </h4>
              <ul className="text-xs text-slate-400 space-y-1.5">
                {t('stop.promptBasic.demo.canDoList', { returnObjects: true }).map((item, i) => (
                  <li key={i}>- {item}</li>
                ))}
              </ul>
            </div>
            <div className="p-3 rounded-lg bg-red-950/20 border border-red-800/30">
              <h4 className="text-xs font-semibold text-red-400 uppercase tracking-wide mb-2">
                {t('stop.promptBasic.demo.whatPromptingCannotDo')}
              </h4>
              <ul className="text-xs text-slate-400 space-y-1.5">
                {t('stop.promptBasic.demo.cannotDoList', { returnObjects: true }).map((item, i) => (
                  <li key={i}>- {item}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* ==================== UNDER THE COVERS ==================== */}
      {section === 'deepdive' && (
        <div className="space-y-5">
          {/* Intro to token probs */}
          <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h3 className="text-base font-semibold text-orange-400 mb-3">
              {t('stop.promptBasic.deepdive.tokenProbsHeading')}
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed mb-3">
              <Trans
                i18nKey="stop.promptBasic.deepdive.tokenProbsP1"
                components={{ 1: <strong className="text-orange-300" /> }}
              />
            </p>
            <p className="text-sm text-slate-400 leading-relaxed">
              <Trans
                i18nKey="stop.promptBasic.deepdive.tokenProbsP2"
                components={{ 1: <strong className="text-slate-200" /> }}
              />
            </p>
          </div>

          {/* Token prob chart */}
          <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/50">
            <TokenProbChart
              data={baseTokenProbs}
              label={t('stop.promptBasic.deepdive.baseModelLabel')}
              highlightToken="OLTP"
            />
          </div>

          {/* Reading the chart */}
          <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h4 className="text-sm font-semibold text-orange-400 mb-3">
              {t('stop.promptBasic.deepdive.readingTheChart')}
            </h4>
            <div className="space-y-3">
              <div className="flex gap-3 items-start">
                <span className="text-lg leading-none mt-0.5">1.</span>
                <p className="text-sm text-slate-300 leading-relaxed">
                  <Trans
                    i18nKey="stop.promptBasic.deepdive.chart1"
                    components={{ 1: <strong className="text-slate-200" /> }}
                  />
                </p>
              </div>
              <div className="flex gap-3 items-start">
                <span className="text-lg leading-none mt-0.5">2.</span>
                <p className="text-sm text-slate-300 leading-relaxed">
                  <Trans
                    i18nKey="stop.promptBasic.deepdive.chart2"
                    components={{ 1: <strong className="text-green-400" /> }}
                  />
                </p>
              </div>
              <div className="flex gap-3 items-start">
                <span className="text-lg leading-none mt-0.5">3.</span>
                <p className="text-sm text-slate-300 leading-relaxed">
                  <Trans
                    i18nKey="stop.promptBasic.deepdive.chart3"
                    components={{ 1: <strong className="text-slate-200" /> }}
                  />
                </p>
              </div>
            </div>
          </div>

          {/* Why this matters */}
          <div className="p-4 rounded-lg bg-orange-950/20 border border-orange-800/30">
            <h4 className="text-sm font-semibold text-orange-400 mb-2">
              {t('stop.promptBasic.deepdive.whyThisMatters')}
            </h4>
            <p className="text-sm text-slate-300 leading-relaxed mb-2">
              {t('stop.promptBasic.deepdive.whyThisMattersP1')}
            </p>
            <p className="text-sm text-slate-400 leading-relaxed">
              <Trans
                i18nKey="stop.promptBasic.deepdive.whyThisMattersP2"
                components={{ 1: <strong className="text-orange-300" /> }}
              />
            </p>
          </div>

          {/* Technical detail */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/50">
              <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wide mb-2">
                {t('stop.promptBasic.deepdive.whatIsAToken')}
              </h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                {t('stop.promptBasic.deepdive.whatIsATokenP')}
              </p>
            </div>
            <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/50">
              <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wide mb-2">
                {t('stop.promptBasic.deepdive.tempAndSampling')}
              </h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                {t('stop.promptBasic.deepdive.tempAndSamplingP')}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ---- Tab bar (bottom) ---- */}
      <div className="mt-6">
        <SectionTabs tabs={tabs} active={section} onSelect={setSection} color="orange" />
      </div>
    </div>
  )
}
