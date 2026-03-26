import useStore from '../store'
import LiveInferencePanel from './LiveInferencePanel'

const GITHUB_REPO = 'provandal/post-training-explorer'
const COLAB_BASE = `https://colab.research.google.com/github/${GITHUB_REPO}/blob/main/notebooks`

const PIPELINE_STEPS = [
  {
    step: 1,
    name: 'Generate Data',
    desc: 'Synthetic I/O metrics',
    input: 'Workload profiles',
    output: 'Training dataset',
    color: 'text-blue-400',
    bg: 'bg-blue-950/30',
    border: 'border-blue-800/40',
  },
  {
    step: 2,
    name: 'SFT',
    desc: 'Supervised Fine-Tuning',
    input: 'prompt + completion pairs',
    output: 'LoRA adapter (task knowledge)',
    color: 'text-cyan-400',
    bg: 'bg-cyan-950/30',
    border: 'border-cyan-800/40',
  },
  {
    step: 3,
    name: 'DPO',
    desc: 'Direct Preference Optimization',
    input: 'chosen / rejected pairs',
    output: 'Refined adapter (style)',
    color: 'text-yellow-400',
    bg: 'bg-yellow-950/30',
    border: 'border-yellow-800/40',
  },
  {
    step: 4,
    name: 'GRPO',
    desc: 'Group Relative Policy Optimization',
    input: 'prompts + reward function',
    output: 'Optimized adapter (accuracy)',
    color: 'text-emerald-400',
    bg: 'bg-emerald-950/30',
    border: 'border-emerald-800/40',
  },
  {
    step: 5,
    name: 'Export to ONNX',
    desc: 'Convert for browser inference',
    input: 'Merged model weights',
    output: 'ONNX model on HuggingFace',
    color: 'text-purple-400',
    bg: 'bg-purple-950/30',
    border: 'border-purple-800/40',
  },
]

const NOTEBOOKS = [
  {
    title: 'Post-Training Pipeline',
    file: 'Post_Training_Pipeline.ipynb',
    desc: 'The full SFT \u2192 DPO \u2192 GRPO pipeline. Train a SmolLM2-360M model to classify storage I/O workloads, then export to ONNX for browser inference.',
    runtime: 'GPU required (T4 or A100)',
    time: '~60 min on T4, ~20 min on A100',
    color: 'border-blue-700/50 hover:border-blue-500',
    accent: 'text-blue-400',
  },
  {
    title: 'Traditional ML Comparison',
    file: 'Traditional_ML_Comparison.ipynb',
    desc: 'Random Forest & XGBoost on the same classification task. Shows when traditional ML is the better choice — faster, smaller, and often more accurate on structured numeric data.',
    runtime: 'CPU only',
    time: '~2 min',
    color: 'border-orange-700/50 hover:border-orange-500',
    accent: 'text-orange-400',
  },
  {
    title: 'Realistic LLM Use Case',
    file: 'Realistic_LLM_Use_Case.ipynb',
    desc: 'Unstructured storage error log classification — a task where LLMs genuinely outperform traditional ML. Compares TF-IDF + XGBoost against SFT fine-tuning.',
    runtime: 'GPU required (T4 or A100)',
    time: '~20 min on T4',
    color: 'border-emerald-700/50 hover:border-emerald-500',
    accent: 'text-emerald-400',
  },
]

const DATA_FORMATS = [
  {
    stage: 'SFT',
    format: 'prompt / completion',
    example: '{"prompt": "Classify... IOPS: 45,000...", "completion": " OLTP Database"}',
    note: 'Label-only completions. Prompt tokens are masked in the loss.',
  },
  {
    stage: 'DPO',
    format: 'prompt / chosen / rejected',
    example: '{"prompt": "Classify...", "chosen": " OLTP Database", "rejected": " AI ML Training"}',
    note: 'Preference pairs teach style, not new knowledge.',
  },
  {
    stage: 'GRPO',
    format: 'prompt + reward function',
    example: 'reward(output) = 1.0 if output.strip() == label else 0.0',
    note: 'K=8 candidates per prompt, scored by binary reward.',
  },
]

export default function TrainView() {
  const setMode = useStore((s) => s.setMode)
  const startTour = useStore((s) => s.startTour)
  const startExplore = useStore((s) => s.startExplore)

  return (
    <div className="min-h-screen">
      {/* Header bar */}
      <div className="sticky top-0 z-10 bg-slate-900/95 backdrop-blur border-b border-slate-800 px-6 py-3 flex items-center justify-between">
        <h1 className="text-lg font-bold text-white">Train Your Model</h1>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setMode('landing')}
            className="text-xs text-slate-500 hover:text-slate-300 px-2 py-1 rounded border border-slate-700 hover:border-slate-500 transition-colors"
          >
            Home
          </button>
          <button
            onClick={startTour}
            className="text-xs text-blue-400 hover:text-blue-300 px-2 py-1 rounded border border-blue-700/50 hover:border-blue-500 transition-colors"
          >
            Guided Tour
          </button>
          <button
            onClick={startExplore}
            className="text-xs text-slate-400 hover:text-slate-300 px-2 py-1 rounded border border-slate-700 hover:border-slate-500 transition-colors"
          >
            Explore Freely
          </button>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-6 py-10 space-y-12">
        {/* Hero */}
        <section className="text-center">
          <h2 className="text-3xl font-extrabold text-white mb-3">
            Train Your Own Model
          </h2>
          <p className="text-slate-400 max-w-2xl mx-auto leading-relaxed">
            Run the same training pipeline used to build this demo. You'll fine-tune
            a SmolLM2-360M language model to classify storage I/O workloads using SFT,
            DPO, and GRPO — then test it live in your browser.
          </p>
        </section>

        {/* Prerequisites */}
        <section>
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-4">
            Prerequisites
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[
              {
                label: 'Google Account',
                detail: 'For Google Colab (free tier works)',
              },
              {
                label: 'HuggingFace Account',
                detail: 'Free — for model hosting and ONNX export',
              },
              {
                label: 'GPU Runtime',
                detail: 'Colab provides free T4 / A100 GPUs',
              },
            ].map((prereq) => (
              <div
                key={prereq.label}
                className="p-4 rounded-lg bg-slate-800/40 border border-slate-700/40"
              >
                <p className="text-sm font-semibold text-white">{prereq.label}</p>
                <p className="text-xs text-slate-500 mt-1">{prereq.detail}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Pipeline Overview */}
        <section>
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-4">
            Pipeline Overview
          </h3>
          <div className="flex flex-col gap-3">
            {PIPELINE_STEPS.map((s, i) => (
              <div key={s.step} className="flex items-start gap-4">
                {/* Step number + connector */}
                <div className="flex flex-col items-center flex-shrink-0">
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${s.bg} ${s.color} border ${s.border}`}
                  >
                    {s.step}
                  </div>
                  {i < PIPELINE_STEPS.length - 1 && (
                    <div className="w-px h-6 bg-slate-700/50" />
                  )}
                </div>
                {/* Content */}
                <div className={`flex-1 p-3 rounded-lg ${s.bg} border ${s.border}`}>
                  <p className={`text-sm font-semibold ${s.color}`}>
                    {s.name}{' '}
                    <span className="text-slate-500 font-normal">— {s.desc}</span>
                  </p>
                  <div className="flex gap-6 mt-1 text-xs text-slate-500">
                    <span>
                      In: <span className="text-slate-400">{s.input}</span>
                    </span>
                    <span>
                      Out: <span className="text-slate-400">{s.output}</span>
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Available Notebooks */}
        <section>
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-4">
            Available Notebooks
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {NOTEBOOKS.map((nb) => (
              <div
                key={nb.title}
                className={`p-5 rounded-lg bg-slate-800/50 border ${nb.color} flex flex-col transition-colors`}
              >
                <h4 className={`text-sm font-bold ${nb.accent} mb-2`}>
                  {nb.title}
                </h4>
                <p className="text-xs text-slate-400 leading-relaxed flex-1 mb-3">
                  {nb.desc}
                </p>
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
                  Open in Colab
                </a>
              </div>
            ))}
          </div>
        </section>

        {/* Data Flow Diagram */}
        <section>
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-4">
            Data Format by Stage
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
                  <span className="text-sm text-slate-300 font-medium">
                    {d.format}
                  </span>
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
            What to Expect
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[
              {
                label: 'Runtime (360M on T4)',
                value: '~60 minutes total',
                detail: 'SFT ~12 min, DPO ~8 min, GRPO ~35 min, Export ~5 min',
              },
              {
                label: 'Runtime (360M on A100)',
                value: '~20 minutes total',
                detail: 'SFT ~4 min, DPO ~3 min, GRPO ~10 min, Export ~3 min',
              },
              {
                label: 'GPU Memory',
                value: '~15 GB VRAM (T4)',
                detail: 'LoRA adapters keep memory footprint small',
              },
              {
                label: 'Output Size',
                value: '~180 MB per ONNX model',
                detail: 'Two models exported: base (untrained) and GRPO (trained)',
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
            Try Your Model
          </h3>
          <p className="text-sm text-slate-400 mb-4">
            After training and exporting to ONNX, download the model to your browser
            and run inference locally. Compare the untrained base model against your
            GRPO-trained model.
          </p>
          <LiveInferencePanel />
        </section>

        {/* Links back */}
        <section className="flex gap-4 justify-center pt-4 border-t border-slate-800">
          <button
            onClick={startTour}
            className="text-sm text-blue-400 hover:text-blue-300 transition-colors"
          >
            &larr; Guided Tour
          </button>
          <button
            onClick={startExplore}
            className="text-sm text-slate-400 hover:text-slate-300 transition-colors"
          >
            Explore Freely &rarr;
          </button>
        </section>
      </div>
    </div>
  )
}
