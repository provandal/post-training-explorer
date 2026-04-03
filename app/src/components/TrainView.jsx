import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import useStore from '../store'
import LiveInferencePanel from './LiveInferencePanel'

const GITHUB_REPO = 'provandal/post-training-explorer'
const COLAB_BASE = `https://colab.research.google.com/github/${GITHUB_REPO}/blob/main/notebooks`

const STEP_STYLES = [
  { step: 1, color: 'text-blue-400', bg: 'bg-blue-950/30', border: 'border-blue-800/40' },
  { step: 2, color: 'text-cyan-400', bg: 'bg-cyan-950/30', border: 'border-cyan-800/40' },
  { step: 3, color: 'text-yellow-400', bg: 'bg-yellow-950/30', border: 'border-yellow-800/40' },
  { step: 4, color: 'text-emerald-400', bg: 'bg-emerald-950/30', border: 'border-emerald-800/40' },
  { step: 5, color: 'text-purple-400', bg: 'bg-purple-950/30', border: 'border-purple-800/40' },
]

function getPipelineSteps(t) {
  return STEP_STYLES.map((s) => ({
    ...s,
    name: t(`train.step${s.step}Name`),
    desc: t(`train.step${s.step}Desc`),
    input: t(`train.step${s.step}Input`),
    output: t(`train.step${s.step}Output`),
    detail: t(`train.step${s.step}Detail`),
  }))
}

const NOTEBOOK_STYLES = [
  {
    num: 1,
    file: 'Post_Training_Pipeline.ipynb',
    color: 'border-blue-700/50 hover:border-blue-500',
    accent: 'text-blue-400',
  },
  {
    num: 2,
    file: 'Traditional_ML_Comparison.ipynb',
    color: 'border-orange-700/50 hover:border-orange-500',
    accent: 'text-orange-400',
  },
  {
    num: 3,
    file: 'Realistic_LLM_Use_Case.ipynb',
    color: 'border-emerald-700/50 hover:border-emerald-500',
    accent: 'text-emerald-400',
  },
]

function getNotebooks(t) {
  return NOTEBOOK_STYLES.map((nb) => ({
    ...nb,
    title: t(`train.notebook${nb.num}Title`),
    desc: t(`train.notebook${nb.num}Desc`),
    runtime: t(`train.notebook${nb.num}Runtime`),
    time: t(`train.notebook${nb.num}Time`),
  }))
}

const DATA_FORMAT_STAGES = ['sft', 'dpo', 'grpo']

function getDataFormats(t) {
  return DATA_FORMAT_STAGES.map((stage) => ({
    stage: stage.toUpperCase(),
    format: t(`train.${stage}Format`),
    example:
      stage === 'sft'
        ? '{"prompt": "Classify... IOPS: 45,000...", "completion": " OLTP Database"}'
        : stage === 'dpo'
          ? '{"prompt": "Classify...", "chosen": " OLTP Database", "rejected": " AI ML Training"}'
          : 'reward(output) = 1.0 if output.strip() == label else 0.0',
    note: t(`train.${stage}Note`),
  }))
}

export default function TrainView() {
  const { t } = useTranslation()
  const setMode = useStore((s) => s.setMode)
  const startTour = useStore((s) => s.startTour)
  const startExplore = useStore((s) => s.startExplore)
  const [expandedStep, setExpandedStep] = useState(null)

  const PIPELINE_STEPS = getPipelineSteps(t)
  const NOTEBOOKS = getNotebooks(t)
  const DATA_FORMATS = getDataFormats(t)

  return (
    <div className="min-h-screen">
      {/* Header bar */}
      <div className="sticky top-0 z-10 bg-slate-900/95 backdrop-blur border-b border-slate-800 px-6 py-3 flex items-center justify-between">
        <h1 className="text-lg font-bold text-white">{t('train.headerTitle')}</h1>
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
            {t('nav.exploreFreelyNav')}
          </button>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-6 py-10 space-y-12">
        {/* Hero */}
        <section className="text-center">
          <h2 className="text-3xl font-extrabold text-white mb-3">{t('train.heroTitle')}</h2>
          <p className="text-slate-400 max-w-2xl mx-auto leading-relaxed">{t('train.heroP')}</p>
        </section>

        {/* Prerequisites */}
        <section>
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-4">
            {t('train.prerequisites')}
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <a
              href="https://colab.research.google.com"
              target="_blank"
              rel="noopener noreferrer"
              className="p-5 rounded-lg bg-slate-800/40 border border-blue-800/40 hover:border-blue-500 transition-colors group"
            >
              <p className="text-sm font-semibold text-blue-400 mb-2">{t('train.googleColab')}</p>
              <p className="text-xs text-slate-400 leading-relaxed mb-3">
                {t('train.googleColabP')}
              </p>
              <span className="inline-block text-xs font-semibold text-blue-400 bg-blue-950/40 px-3 py-1.5 rounded group-hover:bg-blue-900/50 transition-colors">
                {t('train.openColab')}
              </span>
            </a>
            <a
              href="https://huggingface.co/join"
              target="_blank"
              rel="noopener noreferrer"
              className="p-5 rounded-lg bg-slate-800/40 border border-yellow-800/40 hover:border-yellow-500 transition-colors group"
            >
              <p className="text-sm font-semibold text-yellow-400 mb-2">{t('train.huggingFace')}</p>
              <p className="text-xs text-slate-400 leading-relaxed mb-3">
                {t('train.huggingFaceP')}
              </p>
              <span className="inline-block text-xs font-semibold text-yellow-400 bg-yellow-950/40 px-3 py-1.5 rounded group-hover:bg-yellow-900/50 transition-colors">
                {t('train.createFreeAccount')}
              </span>
            </a>
          </div>
        </section>

        {/* Pipeline Overview */}
        <section>
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-4">
            {t('train.pipelineOverview')}
          </h3>
          <p className="text-xs text-slate-500 mb-3">{t('train.clickToLearnMore')}</p>
          <div className="flex flex-col gap-3">
            {PIPELINE_STEPS.map((s, i) => {
              const isExpanded = expandedStep === s.step
              return (
                <div
                  key={s.step}
                  className="cursor-pointer"
                  onClick={() => setExpandedStep(isExpanded ? null : s.step)}
                >
                  <div className="flex items-start gap-4">
                    {/* Step number + connector */}
                    <div className="flex flex-col items-center flex-shrink-0">
                      <div
                        className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${s.bg} ${s.color} border ${s.border} relative`}
                      >
                        {s.step}
                        <span
                          className={`absolute -right-1 -top-1 text-[10px] ${s.color} transition-transform ${isExpanded ? 'rotate-90' : ''}`}
                        >
                          ▶
                        </span>
                      </div>
                      {i < PIPELINE_STEPS.length - 1 && (
                        <div className="w-px h-6 bg-slate-700/50" />
                      )}
                    </div>
                    {/* Content */}
                    <div
                      className={`flex-1 p-3 rounded-lg ${s.bg} border ${s.border} hover:brightness-125 transition-all`}
                    >
                      <p className={`text-sm font-semibold ${s.color}`}>
                        {s.name} <span className="text-slate-500 font-normal">— {s.desc}</span>
                      </p>
                      <div className="flex gap-6 mt-1 text-xs text-slate-500">
                        <span>
                          {t('train.in')} <span className="text-slate-400">{s.input}</span>
                        </span>
                        <span>
                          {t('train.out')} <span className="text-slate-400">{s.output}</span>
                        </span>
                      </div>
                      {/* Expandable detail */}
                      {isExpanded && (
                        <p className="text-xs text-slate-400 mt-2 leading-relaxed pl-0 border-t border-slate-700/30 pt-2">
                          {s.detail}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </section>

        {/* Available Notebooks */}
        <section>
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-4">
            {t('train.availableNotebooks')}
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {NOTEBOOKS.map((nb) => (
              <div
                key={nb.title}
                className={`p-5 rounded-lg bg-slate-800/50 border ${nb.color} flex flex-col transition-colors`}
              >
                <h4 className={`text-sm font-bold ${nb.accent} mb-2`}>{nb.title}</h4>
                <p className="text-xs text-slate-400 leading-relaxed flex-1 mb-3">{nb.desc}</p>
                <div className="text-xs text-slate-600 mb-3 space-y-0.5">
                  <p>{nb.runtime}</p>
                  <p>{nb.time}</p>
                </div>
                <a
                  href={`${COLAB_BASE}/${nb.file}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block text-center py-2 px-4 rounded-md bg-slate-700 hover:bg-slate-600 text-white text-xs font-semibold transition-colors"
                >
                  {t('train.openInColab')}
                </a>
              </div>
            ))}
          </div>
        </section>

        {/* Data Flow Diagram */}
        <section>
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-4">
            {t('train.dataFormatByStage')}
          </h3>
          <div className="space-y-3">
            {DATA_FORMATS.map((d) => (
              <div
                key={d.stage}
                className="p-4 rounded-lg bg-slate-800/40 border border-slate-700/40"
              >
                <div className="flex items-center gap-3 mb-2">
                  <span className="text-xs font-bold text-cyan-400 bg-cyan-950/40 px-2 py-0.5 rounded">
                    {d.stage}
                  </span>
                  <span className="text-sm text-slate-300 font-medium">{d.format}</span>
                </div>
                <pre className="text-xs text-slate-500 bg-slate-900/50 rounded px-3 py-2 overflow-x-auto font-mono">
                  {d.example}
                </pre>
                <p className="text-xs text-slate-600 mt-2">{d.note}</p>
              </div>
            ))}
          </div>
        </section>

        {/* What to Expect */}
        <section>
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-4">
            {t('train.whatToExpect')}
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[
              {
                label: t('train.runtimeT4'),
                value: t('train.runtimeT4Value'),
                detail: t('train.runtimeT4Detail'),
              },
              {
                label: t('train.runtimeA100'),
                value: t('train.runtimeA100Value'),
                detail: t('train.runtimeA100Detail'),
              },
              {
                label: t('train.gpuMemory'),
                value: t('train.gpuMemoryValue'),
                detail: t('train.gpuMemoryDetail'),
              },
              {
                label: t('train.outputSize'),
                value: t('train.outputSizeValue'),
                detail: t('train.outputSizeDetail'),
              },
            ].map((item) => (
              <div
                key={item.label}
                className="p-4 rounded-lg bg-slate-800/40 border border-slate-700/40"
              >
                <p className="text-xs text-slate-500">{item.label}</p>
                <p className="text-lg font-bold text-white">{item.value}</p>
                <p className="text-xs text-slate-600 mt-1">{item.detail}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Live Inference — the capstone */}
        <section>
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-4">
            {t('train.tryYourModel')}
          </h3>
          <p className="text-sm text-slate-400 mb-4">{t('train.tryYourModelP')}</p>
          <LiveInferencePanel />
        </section>

        {/* Links back */}
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
            {t('nav.exploreFreelyNav')} &rarr;
          </button>
        </section>
      </div>
    </div>
  )
}
