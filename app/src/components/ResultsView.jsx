import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import useStore from '../store'
import { getAccuracySummary } from '../data/loadArtifacts'

const INFRA_TABLE = [
  {
    technique: 'Traditional ML',
    timeT4: '0.4 sec',
    timeA100: 'N/A',
    gpuMem: 'None',
    modelSize: '~50 KB',
    hardware: 'CPU',
  },
  {
    technique: 'SFT',
    timeT4: '~12 min',
    timeA100: '~4 min',
    gpuMem: '~15 GB',
    modelSize: '~700 MB',
    hardware: 'GPU',
  },
  {
    technique: 'DPO',
    timeT4: '~8 min',
    timeA100: '~3 min',
    gpuMem: '~15 GB',
    modelSize: '~700 MB',
    hardware: 'GPU',
  },
  {
    technique: 'GRPO',
    timeT4: '~35 min',
    timeA100: '~10 min',
    gpuMem: '~15 GB',
    modelSize: '~700 MB',
    hardware: 'GPU',
  },
  {
    technique: 'ONNX Export',
    timeT4: '~5 min',
    timeA100: '~3 min',
    gpuMem: '~8 GB',
    modelSize: '~180 MB',
    hardware: 'GPU',
  },
  {
    technique: 'Full Pipeline',
    timeT4: '~60 min',
    timeA100: '~20 min',
    gpuMem: '~15 GB',
    modelSize: '~180 MB',
    hardware: 'GPU',
  },
]

function AccuracyBar({ label, accuracy, color }) {
  const pct = Math.round(accuracy * 100)
  const barWidth = Math.max(pct, 2) // minimum visible width
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-slate-400 w-14 text-right font-mono">{label}</span>
      <div className="flex-1 h-7 bg-slate-800/60 rounded overflow-hidden relative">
        <div
          className={`h-full rounded ${color} transition-all duration-700`}
          style={{ width: `${barWidth}%` }}
        />
        <span className="absolute inset-0 flex items-center px-3 text-xs font-bold text-white">
          {pct}%
        </span>
      </div>
    </div>
  )
}

function AccordionCard({ title, summary, detail }) {
  const [open, setOpen] = useState(false)
  return (
    <div
      className="cursor-pointer rounded-lg bg-slate-800/40 border border-slate-700/40 hover:border-slate-600/60 transition-colors"
      onClick={() => setOpen(!open)}
    >
      <div className="p-4 flex items-start gap-3">
        <span
          className={`text-slate-500 text-xs mt-0.5 transition-transform flex-shrink-0 ${open ? 'rotate-90' : ''}`}
        >
          \u25B6
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-white">{title}</p>
          <p className="text-xs text-slate-500 mt-0.5">{summary}</p>
          {open && (
            <p className="text-xs text-slate-400 mt-3 leading-relaxed border-t border-slate-700/30 pt-3">
              {detail}
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

export default function ResultsView() {
  const { t } = useTranslation()
  const setMode = useStore((s) => s.setMode)
  const startTour = useStore((s) => s.startTour)
  const startExplore = useStore((s) => s.startExplore)
  const startTrain = useStore((s) => s.startTrain)
  const artifactsLoaded = useStore((s) => s.artifactsLoaded)

  // Get accuracy from precomputed data if loaded
  const accuracySummary = artifactsLoaded ? getAccuracySummary() : null
  const baseAcc = accuracySummary?.base?.accuracy ?? 0
  const sftAcc = accuracySummary?.sft?.accuracy ?? 0.35
  const dpoAcc = accuracySummary?.dpo?.accuracy ?? 0.25
  const grpoAcc = accuracySummary?.grpo?.accuracy ?? 0.35

  const DEBUGGING_STORIES = [
    {
      title: t('results.debug1Title'),
      summary: t('results.debug1Summary'),
      detail: t('results.debug1Detail'),
    },
    {
      title: t('results.debug2Title'),
      summary: t('results.debug2Summary'),
      detail: t('results.debug2Detail'),
    },
    {
      title: t('results.debug3Title'),
      summary: t('results.debug3Summary'),
      detail: t('results.debug3Detail'),
    },
    {
      title: t('results.debug4Title'),
      summary: t('results.debug4Summary'),
      detail: t('results.debug4Detail'),
    },
    {
      title: t('results.debug5Title'),
      summary: t('results.debug5Summary'),
      detail: t('results.debug5Detail'),
    },
  ]

  const TAKEAWAYS = [
    { num: 1, title: t('results.takeaway1Title'), text: t('results.takeaway1P') },
    { num: 2, title: t('results.takeaway2Title'), text: t('results.takeaway2P') },
    { num: 3, title: t('results.takeaway3Title'), text: t('results.takeaway3P') },
    { num: 4, title: t('results.takeaway4Title'), text: t('results.takeaway4P') },
    { num: 5, title: t('results.takeaway5Title'), text: t('results.takeaway5P') },
  ]

  return (
    <div className="min-h-screen">
      {/* Header bar */}
      <div className="sticky top-0 z-10 bg-slate-900/95 backdrop-blur border-b border-slate-800 px-6 py-3 flex items-center justify-between">
        <h1 className="text-lg font-bold text-white">{t('results.headerTitle')}</h1>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setMode('landing')}
            className="text-xs text-slate-500 hover:text-slate-300 px-2 py-1 rounded border border-slate-700 hover:border-slate-500 transition-colors"
          >
            {t('nav.home')}
          </button>
          <button
            onClick={startTour}
            className="text-xs text-blue-400 hover:text-blue-300 px-2 py-1 rounded border border-blue-700/50 hover:border-blue-500 transition-colors"
          >
            {t('nav.guidedTour')}
          </button>
          <button
            onClick={startExplore}
            className="text-xs text-slate-400 hover:text-slate-300 px-2 py-1 rounded border border-slate-700 hover:border-slate-500 transition-colors"
          >
            {t('landing.explore')}
          </button>
          <button
            onClick={startTrain}
            className="text-xs text-emerald-400 hover:text-emerald-300 px-2 py-1 rounded border border-emerald-700/50 hover:border-emerald-500 transition-colors"
          >
            {t('nav.train')}
          </button>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-6 py-10 space-y-14">
        {/* Section A: Hero */}
        <section className="text-center">
          <h2 className="text-3xl font-extrabold text-white mb-3">{t('results.heroTitle')}</h2>
          <p className="text-slate-400 max-w-2xl mx-auto leading-relaxed">{t('results.heroP')}</p>
        </section>

        {/* Section B: Right Tool for the Right Job */}
        <section>
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-4">
            {t('results.rightTool')}
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Structured Data card */}
            <div className="p-5 rounded-lg bg-emerald-950/20 border border-emerald-800/40">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-xs font-bold text-emerald-400 bg-emerald-950/50 px-2 py-0.5 rounded">
                  {t('results.winner')}
                </span>
                <span className="text-sm font-semibold text-white">
                  {t('results.traditionalMl')}
                </span>
              </div>
              <h4 className="text-base font-bold text-emerald-400 mb-3">
                {t('results.structuredData')}
              </h4>
              <div className="space-y-1.5 text-xs text-slate-400">
                <p>
                  <span className="text-slate-500">{t('results.accuracy')}</span>{' '}
                  <span className="text-white font-semibold">
                    {t('results.structuredAccuracy')}
                  </span>
                </p>
                <p>
                  <span className="text-slate-500">{t('results.trainingTimeLabel')}</span>{' '}
                  {t('results.structuredTrainingTime')}
                </p>
                <p>
                  <span className="text-slate-500">{t('results.hardwareLabel')}</span>{' '}
                  {t('results.structuredHardware')}
                </p>
                <p>
                  <span className="text-slate-500">{t('results.modelSizeLabel')}</span>{' '}
                  {t('results.structuredModelSize')}
                </p>
              </div>
              <p className="text-xs text-emerald-400/80 mt-3 pt-3 border-t border-emerald-800/30 leading-relaxed">
                {t('results.structuredAdvice')}
              </p>
            </div>

            {/* Unstructured Data card */}
            <div className="p-5 rounded-lg bg-blue-950/20 border border-blue-800/40">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-xs font-bold text-blue-400 bg-blue-950/50 px-2 py-0.5 rounded">
                  {t('results.winner')}
                </span>
                <span className="text-sm font-semibold text-white">
                  {t('results.llmFineTuning')}
                </span>
              </div>
              <h4 className="text-base font-bold text-blue-400 mb-3">
                {t('results.unstructuredData')}
              </h4>
              <div className="space-y-1.5 text-xs text-slate-400">
                <p>
                  <span className="text-slate-500">{t('results.accuracy')}</span>{' '}
                  <span className="text-white font-semibold">
                    {t('results.unstructuredAccuracy')}
                  </span>{' '}
                  <span className="text-slate-600">{t('results.unstructuredVsXgboost')}</span>
                </p>
                <p>
                  <span className="text-slate-500">{t('results.trainingTimeLabel')}</span>{' '}
                  {t('results.unstructuredTrainingTime')}
                </p>
                <p>
                  <span className="text-slate-500">{t('results.hardwareLabel')}</span>{' '}
                  {t('results.unstructuredHardware')}
                </p>
                <p>
                  <span className="text-slate-500">{t('results.modelSizeLabel')}</span>{' '}
                  {t('results.unstructuredModelSize')}
                </p>
              </div>
              <p className="text-xs text-blue-400/80 mt-3 pt-3 border-t border-blue-800/30 leading-relaxed">
                {t('results.unstructuredAdvice')}
              </p>
            </div>
          </div>
        </section>

        {/* Section C: LLM Accuracy Progression */}
        <section>
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-4">
            {t('results.accuracyProgression')}
          </h3>
          <div className="bg-slate-800/30 border border-slate-700/40 rounded-lg p-5 space-y-3">
            <AccuracyBar label="Base" accuracy={baseAcc} color="bg-slate-600" />
            <AccuracyBar label="SFT" accuracy={sftAcc} color="bg-cyan-600" />
            <AccuracyBar label="DPO" accuracy={dpoAcc} color="bg-yellow-600" />
            <AccuracyBar label="GRPO" accuracy={grpoAcc} color="bg-emerald-600" />
            <p className="text-[10px] text-slate-600 text-right">{t('results.testPromptsNote')}</p>
          </div>

          <p className="text-sm text-slate-400 mt-4 leading-relaxed">
            {t('results.accuracyCommentary', {
              1: (chunks) => chunks,
              defaultValue: t('results.accuracyCommentary'),
            })}
          </p>

          {/* What would improve these numbers */}
          <div className="mt-4 p-4 rounded-lg bg-slate-800/20 border border-slate-700/30">
            <p className="text-xs font-semibold text-slate-300 mb-3">
              {t('results.whatWouldImprove')}
            </p>
            <div className="space-y-2 text-xs text-slate-400 leading-relaxed">
              <p>{t('results.moreData', { 1: (c) => c })}</p>
              <p>{t('results.largerModel', { 1: (c) => c })}</p>
              <p>{t('results.unstructuredInput', { 1: (c) => c })}</p>
              <p>{t('results.moreGrpoSteps', { 1: (c) => c })}</p>
            </div>
          </div>
        </section>

        {/* Section D: Debugging Journey */}
        <section>
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-4">
            {t('results.debuggingJourney')}
          </h3>
          <div className="space-y-3">
            {DEBUGGING_STORIES.map((story) => (
              <AccordionCard key={story.title} {...story} />
            ))}
          </div>
        </section>

        {/* Section E: Infrastructure Profile */}
        <section>
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-4">
            {t('results.infraProfile')}
          </h3>
          <div className="overflow-x-auto rounded-lg border border-slate-700/40">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-slate-800/60 text-slate-400">
                  <th className="text-left px-3 py-2 font-semibold">
                    {t('results.infraTable.technique')}
                  </th>
                  <th className="text-left px-3 py-2 font-semibold">
                    {t('results.infraTable.timeT4')}
                  </th>
                  <th className="text-left px-3 py-2 font-semibold">
                    {t('results.infraTable.timeA100')}
                  </th>
                  <th className="text-left px-3 py-2 font-semibold">
                    {t('results.infraTable.gpuMemory')}
                  </th>
                  <th className="text-left px-3 py-2 font-semibold">
                    {t('results.infraTable.modelSize')}
                  </th>
                  <th className="text-left px-3 py-2 font-semibold">
                    {t('results.infraTable.hardware')}
                  </th>
                </tr>
              </thead>
              <tbody>
                {INFRA_TABLE.map((row, i) => {
                  const isLast = row.technique === 'Full Pipeline'
                  return (
                    <tr
                      key={row.technique}
                      className={`border-t border-slate-700/30 ${
                        isLast
                          ? 'bg-slate-800/40 font-semibold text-white'
                          : i % 2 === 0
                            ? 'text-slate-300'
                            : 'bg-slate-800/20 text-slate-300'
                      }`}
                    >
                      <td className="px-3 py-2">{row.technique}</td>
                      <td className="px-3 py-2">{row.timeT4}</td>
                      <td className="px-3 py-2">{row.timeA100}</td>
                      <td className="px-3 py-2">{row.gpuMem}</td>
                      <td className="px-3 py-2">{row.modelSize}</td>
                      <td className="px-3 py-2">{row.hardware}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-slate-500 mt-3 leading-relaxed">{t('results.infraNote')}</p>
        </section>

        {/* Section F: Key Takeaways */}
        <section>
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-4">
            {t('results.keyTakeaways')}
          </h3>
          <div className="space-y-3">
            {TAKEAWAYS.map((tk) => (
              <div
                key={tk.num}
                className="flex items-start gap-4 p-4 rounded-lg bg-slate-800/40 border border-slate-700/40"
              >
                <span className="flex-shrink-0 w-7 h-7 rounded-full bg-purple-950/50 border border-purple-800/40 text-purple-400 text-xs font-bold flex items-center justify-center">
                  {tk.num}
                </span>
                <div>
                  <p className="text-sm font-semibold text-white">{tk.title}</p>
                  <p className="text-xs text-slate-400 mt-1 leading-relaxed">{tk.text}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Section G: Footer links */}
        <section className="flex gap-4 justify-center pt-4 border-t border-slate-800">
          <button
            onClick={startTour}
            className="text-sm text-blue-400 hover:text-blue-300 transition-colors"
          >
            &larr; {t('nav.guidedTour')}
          </button>
          <button
            onClick={startExplore}
            className="text-sm text-slate-400 hover:text-slate-300 transition-colors"
          >
            {t('nav.exploreFreelyNav')}
          </button>
          <button
            onClick={startTrain}
            className="text-sm text-emerald-400 hover:text-emerald-300 transition-colors"
          >
            {t('results.trainYourModel')}
          </button>
        </section>
      </div>
    </div>
  )
}
