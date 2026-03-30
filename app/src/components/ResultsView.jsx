import { useState } from 'react'
import useStore from '../store'
import { getAccuracySummary } from '../data/loadArtifacts'

const DEBUGGING_STORIES = [
  {
    title: 'The Model Invented Its Own Labels',
    summary: "Small models can't memorize valid outputs from training data alone",
    detail:
      'During GRPO training, the 360M model generated classifications like "Vogeek Podcast", "Games Volumes", "RAID OPS", and "Fiction Writing" — labels that don\'t exist. The model had learned the concept of classification but not the vocabulary. Fix: include the 6 valid labels directly in the prompt. Key insight: small models need constraints in-context. This is a great example of why prompt engineering matters even after fine-tuning.',
  },
  {
    title: 'GRPO Training Stuck at 0%',
    summary: 'The reward function was too rigid for freeform model output',
    detail:
      "GRPO's accuracy curve was flat at 0% for all 180 training steps. The reward function only accepted exact matches like \"Classification: OLTP Database\", but the model generated freeform text with extra words, reasoning, and varied formatting. The model was actually learning — it was getting answers right — but the scoring function couldn't see it. Fix: robust extraction with keyword matching and fallback scanning. Lesson: in RL, the reward function IS the specification. If it's too narrow, the model can't learn.",
  },
  {
    title: 'The Invisible Space That Broke Everything',
    summary: 'A single trailing space caused a BPE tokenization mismatch',
    detail:
      'SFT accuracy was stuck at ~20% despite the loss curve looking normal. The bug: our training prompts ended with "Classification: " (trailing space) but the completion started with " OLTP Database" (leading space). The tokenizer merged them into a single token, but at inference time the prompt\'s trailing space tokenized differently. The completion token the model had learned to generate didn\'t match what it saw. Fix: prompt ends with "Classification:" (no space), completion starts with " OLTP Database" (space). One character, 15% accuracy swing.',
  },
  {
    title: 'Temperature: The Knob Nobody Tells You About',
    summary: 'High sampling temperature caused hallucinations even with correct prompts',
    detail:
      'Even after adding valid labels to the prompt, GRPO sampling at temperature=0.8 still produced garbage outputs. The model "knew" the right answer but the high randomness in sampling caused it to wander into nonsense tokens. Lowering to temperature=0.5 with top_p=0.85 eliminated extraction failures entirely. For RL-based training methods, sampling temperature is a critical hyperparameter — too high and the model can\'t learn from its own outputs, too low and it can\'t explore.',
  },
  {
    title: 'Post-Training Is Debugging, Not Magic',
    summary: 'The full pipeline required multiple re-runs and alignment across all stages',
    detail:
      "We ran the pipeline at least 5 times end-to-end before getting reasonable results. Each fix (prompt format, temperature, extraction logic, BPE alignment) required re-training the ENTIRE pipeline — SFT, then DPO, then GRPO — because each stage builds on the previous one. Running GRPO with a new prompt format but an OLD SFT adapter produced worse results than starting over. This is the reality of post-training work: it's iterative, it requires alignment across every stage, and the debugging skills matter as much as the ML knowledge.",
  },
]

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

const TAKEAWAYS = [
  {
    num: 1,
    title: 'Use the right tool',
    text: 'Structured numeric data \u2192 Random Forest. Unstructured text \u2192 LLM fine-tuning. Neither is universally better.',
  },
  {
    num: 2,
    title: 'Small models have real limits',
    text: "360M parameters can learn tasks but can't memorize valid outputs. Constraints must be in-context.",
  },
  {
    num: 3,
    title: 'Post-training is iterative',
    text: 'Expect 3-5 full pipeline runs. Every change to prompts, hyperparameters, or evaluation requires retraining all downstream stages.',
  },
  {
    num: 4,
    title: 'The reward function IS the specification',
    text: "In GRPO/RLVR, if your reward function can't score correctly, the model can't learn. Invest in robust evaluation.",
  },
  {
    num: 5,
    title: 'Infrastructure matters',
    text: 'The same techniques that built ChatGPT run on a free Colab GPU in an hour. The barrier is knowledge, not hardware.',
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

  return (
    <div className="min-h-screen">
      {/* Header bar */}
      <div className="sticky top-0 z-10 bg-slate-900/95 backdrop-blur border-b border-slate-800 px-6 py-3 flex items-center justify-between">
        <h1 className="text-lg font-bold text-white">Results & Learnings</h1>
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
            Explore
          </button>
          <button
            onClick={startTrain}
            className="text-xs text-emerald-400 hover:text-emerald-300 px-2 py-1 rounded border border-emerald-700/50 hover:border-emerald-500 transition-colors"
          >
            Train
          </button>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-6 py-10 space-y-14">
        {/* Section A: Hero */}
        <section className="text-center">
          <h2 className="text-3xl font-extrabold text-white mb-3">What We Learned</h2>
          <p className="text-slate-400 max-w-2xl mx-auto leading-relaxed">
            Building a post-training pipeline from scratch — the honest results, the debugging
            journey, and when to use (and not use) an LLM.
          </p>
        </section>

        {/* Section B: Right Tool for the Right Job */}
        <section>
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-4">
            The Right Tool for the Right Job
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Structured Data card */}
            <div className="p-5 rounded-lg bg-emerald-950/20 border border-emerald-800/40">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-xs font-bold text-emerald-400 bg-emerald-950/50 px-2 py-0.5 rounded">
                  WINNER
                </span>
                <span className="text-sm font-semibold text-white">Traditional ML</span>
              </div>
              <h4 className="text-base font-bold text-emerald-400 mb-3">
                Structured Data (I/O Metrics)
              </h4>
              <div className="space-y-1.5 text-xs text-slate-400">
                <p>
                  <span className="text-slate-500">Accuracy:</span>{' '}
                  <span className="text-white font-semibold">~97%</span>
                </p>
                <p>
                  <span className="text-slate-500">Training time:</span> 0.4 seconds
                </p>
                <p>
                  <span className="text-slate-500">Hardware:</span> CPU only
                </p>
                <p>
                  <span className="text-slate-500">Model size:</span> ~50 KB
                </p>
              </div>
              <p className="text-xs text-emerald-400/80 mt-3 pt-3 border-t border-emerald-800/30 leading-relaxed">
                If your features are numbers in a table, use a tree-based model.
              </p>
            </div>

            {/* Unstructured Data card */}
            <div className="p-5 rounded-lg bg-blue-950/20 border border-blue-800/40">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-xs font-bold text-blue-400 bg-blue-950/50 px-2 py-0.5 rounded">
                  WINNER
                </span>
                <span className="text-sm font-semibold text-white">LLM Fine-Tuning (SFT)</span>
              </div>
              <h4 className="text-base font-bold text-blue-400 mb-3">
                Unstructured Data (Error Logs)
              </h4>
              <div className="space-y-1.5 text-xs text-slate-400">
                <p>
                  <span className="text-slate-500">Accuracy:</span>{' '}
                  <span className="text-white font-semibold">~80-90%</span>{' '}
                  <span className="text-slate-600">(vs ~70-80% XGBoost)</span>
                </p>
                <p>
                  <span className="text-slate-500">Training time:</span> ~20 min (T4 GPU)
                </p>
                <p>
                  <span className="text-slate-500">Hardware:</span> GPU required
                </p>
                <p>
                  <span className="text-slate-500">Model size:</span> ~180 MB (ONNX)
                </p>
              </div>
              <p className="text-xs text-blue-400/80 mt-3 pt-3 border-t border-blue-800/30 leading-relaxed">
                If your input is natural language with variable phrasing, LLMs understand context
                that bag-of-words models miss.
              </p>
            </div>
          </div>
        </section>

        {/* Section C: LLM Accuracy Progression */}
        <section>
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-4">
            Our LLM Accuracy Progression
          </h3>
          <div className="bg-slate-800/30 border border-slate-700/40 rounded-lg p-5 space-y-3">
            <AccuracyBar label="Base" accuracy={baseAcc} color="bg-slate-600" />
            <AccuracyBar label="SFT" accuracy={sftAcc} color="bg-cyan-600" />
            <AccuracyBar label="DPO" accuracy={dpoAcc} color="bg-yellow-600" />
            <AccuracyBar label="GRPO" accuracy={grpoAcc} color="bg-emerald-600" />
            <p className="text-[10px] text-slate-600 text-right">20 test prompts per variant</p>
          </div>

          <p className="text-sm text-slate-400 mt-4 leading-relaxed">
            These numbers look low — and they are. That's the point. A 360M-parameter model
            classifying structured numeric data is the wrong tool for this job. Random Forest gets
            97% in under a second. But the <em className="text-slate-300">techniques</em> we used
            (SFT, DPO, GRPO) are the same ones used to build ChatGPT and DeepSeek R1. The value is
            in learning the process, not in the accuracy on this particular task.
          </p>

          {/* What would improve these numbers */}
          <div className="mt-4 p-4 rounded-lg bg-slate-800/20 border border-slate-700/30">
            <p className="text-xs font-semibold text-slate-300 mb-3">
              What would improve these numbers?
            </p>
            <div className="space-y-2 text-xs text-slate-400 leading-relaxed">
              <p>
                <span className="text-slate-300 font-medium">More training data:</span> We used ~720
                samples. Production fine-tuning uses 10K-100K+ examples.
              </p>
              <p>
                <span className="text-slate-300 font-medium">Larger model:</span> SmolLM2-360M was
                chosen for browser inference. A 1.7B or 7B model would score significantly higher.
              </p>
              <p>
                <span className="text-slate-300 font-medium">Unstructured input:</span> On error log
                classification (text, not numbers), the same SFT technique achieves ~80-90%. The
                task matters more than the technique.
              </p>
              <p>
                <span className="text-slate-300 font-medium">More GRPO steps:</span> We ran 180
                steps. DeepSeek R1 ran thousands. RL improves with more compute.
              </p>
            </div>
          </div>
        </section>

        {/* Section D: Debugging Journey */}
        <section>
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-4">
            The Debugging Journey
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
            Infrastructure Profile
          </h3>
          <div className="overflow-x-auto rounded-lg border border-slate-700/40">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-slate-800/60 text-slate-400">
                  <th className="text-left px-3 py-2 font-semibold">Technique</th>
                  <th className="text-left px-3 py-2 font-semibold">Time (T4)</th>
                  <th className="text-left px-3 py-2 font-semibold">Time (A100)</th>
                  <th className="text-left px-3 py-2 font-semibold">GPU Memory</th>
                  <th className="text-left px-3 py-2 font-semibold">Model Size</th>
                  <th className="text-left px-3 py-2 font-semibold">Hardware</th>
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
          <p className="text-xs text-slate-500 mt-3 leading-relaxed">
            GRPO takes 3x longer than SFT — reinforcement learning generates K=8 candidates per
            prompt and scores each one. This is where GPU burst capacity matters. For infrastructure
            teams: expect sustained GPU utilization during GRPO, not the bursty pattern of SFT.
          </p>
        </section>

        {/* Section F: Key Takeaways */}
        <section>
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-4">
            Key Takeaways
          </h3>
          <div className="space-y-3">
            {TAKEAWAYS.map((t) => (
              <div
                key={t.num}
                className="flex items-start gap-4 p-4 rounded-lg bg-slate-800/40 border border-slate-700/40"
              >
                <span className="flex-shrink-0 w-7 h-7 rounded-full bg-purple-950/50 border border-purple-800/40 text-purple-400 text-xs font-bold flex items-center justify-center">
                  {t.num}
                </span>
                <div>
                  <p className="text-sm font-semibold text-white">{t.title}</p>
                  <p className="text-xs text-slate-400 mt-1 leading-relaxed">{t.text}</p>
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
            &larr; Guided Tour
          </button>
          <button
            onClick={startExplore}
            className="text-sm text-slate-400 hover:text-slate-300 transition-colors"
          >
            Explore Freely
          </button>
          <button
            onClick={startTrain}
            className="text-sm text-emerald-400 hover:text-emerald-300 transition-colors"
          >
            Train Your Model &rarr;
          </button>
        </section>
      </div>
    </div>
  )
}
