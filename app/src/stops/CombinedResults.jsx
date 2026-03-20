import { useState } from 'react'
import ModelOutput from '../components/ModelOutput'

const EXAMPLE_INPUT = "IOPS: 38000 | Latency: 0.5ms | Block Size: 8K | Read/Write: 55/45 | Sequential: 22% | Queue Depth: 64"
const CORRECT_LABEL = "VDI Virtual Desktop"

// Progressive improvement through each technique
const STAGES = [
  {
    id: 'base',
    label: 'Base Model',
    variant: 'base',
    correct: false,
    output: `These storage metrics indicate a workload with 38000 IOPS at 0.5ms latency. Block size is 8K, read/write ratio is nearly balanced at 55/45, with 22% sequential access and a queue depth of 64. The high IOPS and low latency suggest this is a performance-sensitive database application, possibly OLTP or a similar transactional workload.`,
    note: 'Describes but doesn\'t classify. Guesses OLTP.',
  },
  {
    id: 'fewshot',
    label: '+ Few-Shot',
    variant: 'default',
    correct: false,
    output: `Classification: OLTP Database\n\nThis pattern shows high IOPS with low latency and small block sizes. However, the read/write ratio of 55/45 is more balanced than typical OLTP workloads. Though I should note there is some uncertainty in this classification.`,
    note: 'Follows format but wrong classification. Sees the ambiguity but calls it wrong.',
  },
  {
    id: 'rag',
    label: '+ RAG',
    variant: 'rag',
    correct: true,
    output: `Based on the retrieved reference patterns, this I/O profile presents an interesting case... [verbose analysis]... My best assessment is that this is likely a VDI Virtual Desktop workload, though OLTP Database cannot be ruled out entirely.`,
    note: 'Gets the right answer but buried in verbose hedging.',
  },
  {
    id: 'sft',
    label: '+ SFT',
    variant: 'sft',
    correct: true,
    output: `Classification: VDI Virtual Desktop\nConfidence: Medium\nKey indicators: Nearly balanced R/W ratio (55/45) distinguishes this from OLTP despite similar IOPS and block size. High queue depth (64) and moderate sequential ratio (22%) are characteristic of multi-user desktop virtualization.`,
    note: 'Correct, formatted, explains reasoning. But confidence could be more calibrated.',
  },
  {
    id: 'dpo',
    label: '+ DPO',
    variant: 'dpo',
    correct: true,
    output: `Classification: VDI Virtual Desktop\nConfidence: Medium\nKey indicators: Balanced R/W ratio (55/45) and high queue depth (64) distinguish this from OLTP. Small blocks (8K) with random access (78%) are consistent with VDI mixed application workloads.`,
    note: 'Tighter phrasing, more confident, better structured.',
  },
  {
    id: 'grpo',
    label: '+ GRPO',
    variant: 'grpo',
    correct: true,
    output: `Classification: VDI Virtual Desktop\nConfidence: Medium-High\nKey indicators: Balanced R/W (55/45) + high queue depth (64) = multi-user pattern. Block size (8K) and random access differentiate from streaming/analytics. Queue depth alone rules out OLTP (typical: 16-48).`,
    note: 'Most precise. Uses definitive reasoning ("rules out") and specific thresholds.',
  },
  {
    id: 'combined',
    label: 'GRPO + RAG',
    variant: 'grpo',
    correct: true,
    output: `Classification: VDI Virtual Desktop\nConfidence: High\nKey indicators: Balanced R/W (55/45) + high queue depth (64) match VDI reference patterns (retrieved: 94% similarity). Block size (8K) with random access (78%) consistent with multi-user desktop virtualization. Queue depth alone differentiates from OLTP (reference: OLTP typical 16-48 vs VDI 32-128).`,
    note: 'Best of both: RAG provides reference data, GRPO-trained model reasons precisely about it.',
  },
]

export default function CombinedResults({ explore = false }) {
  const [visibleStages, setVisibleStages] = useState(1)

  const revealNext = () => {
    if (visibleStages < STAGES.length) {
      setVisibleStages(visibleStages + 1)
    }
  }

  return (
    <div className="max-w-5xl mx-auto">
      {/* Input */}
      <div className="mb-4">
        <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
          Hard Example: Ambiguous I/O Pattern
        </label>
        <div className="bg-slate-800 border border-yellow-700/50 rounded-lg p-3 font-mono text-sm text-yellow-200">
          {EXAMPLE_INPUT}
        </div>
        <p className="text-xs text-slate-500 mt-1">
          Correct: <span className="text-emerald-400 font-semibold">{CORRECT_LABEL}</span> — Watch the progressive improvement.
        </p>
      </div>

      {/* Reveal control */}
      <div className="flex items-center gap-3 mb-4">
        <button
          onClick={revealNext}
          disabled={visibleStages >= STAGES.length}
          className="px-4 py-2 text-sm bg-cyan-700 hover:bg-cyan-600 disabled:bg-slate-700 disabled:text-slate-500 rounded-md transition-colors"
        >
          Add next technique ({visibleStages}/{STAGES.length})
        </button>
        <button
          onClick={() => setVisibleStages(STAGES.length)}
          className="px-4 py-2 text-sm bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-md transition-colors"
        >
          Show all
        </button>
      </div>

      {/* Progressive stages */}
      <div className="space-y-3">
        {STAGES.slice(0, visibleStages).map((stage, i) => (
          <div key={stage.id} className="flex gap-3 items-start">
            {/* Step indicator */}
            <div className="flex flex-col items-center flex-shrink-0 pt-3">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${
                stage.correct
                  ? 'bg-emerald-900/50 text-emerald-400 border border-emerald-700'
                  : 'bg-red-900/50 text-red-400 border border-red-700'
              }`}>
                {i + 1}
              </div>
              {i < visibleStages - 1 && (
                <div className="w-px h-full bg-slate-700 mt-1" />
              )}
            </div>

            {/* Content */}
            <div className="flex-1 min-w-0">
              <ModelOutput
                label={stage.label}
                text={stage.output}
                variant={stage.variant}
                isCorrect={stage.correct}
              />
              <p className="text-xs text-slate-500 mt-1 italic">{stage.note}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Summary when all revealed */}
      {visibleStages === STAGES.length && (
        <div className="mt-6 p-4 rounded-lg bg-gradient-to-r from-cyan-950/30 to-emerald-950/30 border border-cyan-800/30">
          <h4 className="text-sm font-semibold text-cyan-400 mb-2">The Full Journey</h4>
          <div className="grid grid-cols-7 gap-1 text-center text-xs mb-3">
            {STAGES.map((s) => (
              <div key={s.id} className={`py-1 px-0.5 rounded ${s.correct ? 'bg-emerald-900/30 text-emerald-400' : 'bg-red-900/30 text-red-400'}`}>
                {s.label.replace('+ ', '')}
              </div>
            ))}
          </div>
          <p className="text-sm text-slate-300">
            Each technique built on the last. Prompting gave format. RAG gave knowledge.
            SFT gave classification skill. DPO gave style. GRPO gave reasoning precision.
            The combination in the upper-right quadrant is greater than the sum of its parts.
          </p>
        </div>
      )}
    </div>
  )
}
