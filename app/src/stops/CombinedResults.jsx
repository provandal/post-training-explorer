import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import ModelOutput from '../components/ModelOutput'
import PatternPicker from '../components/PatternPicker'
import useStore from '../store'
import {
  isLoaded,
  getTestPrompts,
  getModelOutput,
  formatPromptMetrics,
  getAccuracySummary,
  getResourceUtilization,
} from '../data/loadArtifacts'

const FALLBACK_INPUT =
  'IOPS: 38000 | Latency: 0.5ms | Block Size: 8K | Read/Write: 55/45 | Sequential: 22% | Queue Depth: 64'
const FALLBACK_LABEL = 'VDI Virtual Desktop'

// Fallback hardcoded stages (used when precomputed_results.json is not available)
const FALLBACK_STAGES = [
  {
    id: 'base',
    label: 'Base Model',
    variant: 'base',
    correct: false,
    output: `These storage metrics indicate a workload with 38000 IOPS at 0.5ms latency. Block size is 8K, read/write ratio is nearly balanced at 55/45, with 22% sequential access and a queue depth of 64. The high IOPS and low latency suggest this is a performance-sensitive database application, possibly OLTP or a similar transactional workload.`,
    note: "Describes but doesn't classify. Guesses OLTP.",
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

/**
 * Check if a generated_text contains the correct label.
 */
function checkCorrect(generatedText, trueLabel) {
  if (!generatedText || !trueLabel) return false
  return generatedText.toLowerCase().includes(trueLabel.toLowerCase())
}

/**
 * Build stages from real precomputed data for a given test prompt.
 */
function buildRealStages(promptId, trueLabel) {
  const baseOut = getModelOutput('base', promptId)
  const sftOut = getModelOutput('sft', promptId)
  const dpoOut = getModelOutput('dpo', promptId)
  const grpoOut = getModelOutput('grpo', promptId)

  // If we don't have at least base and one trained model, fall back
  if (!baseOut) return null

  const stages = [
    {
      id: 'base',
      label: 'Base Model',
      variant: 'base',
      correct: checkCorrect(baseOut.generated_text, trueLabel),
      output: baseOut.generated_text,
      note: checkCorrect(baseOut.generated_text, trueLabel)
        ? 'Base model gets it right, but output is unstructured.'
        : "Describes but doesn't classify correctly.",
    },
  ]

  // Few-shot is a prompting technique — no precomputed data for it.
  // We keep a placeholder note.
  stages.push({
    id: 'fewshot',
    label: '+ Few-Shot',
    variant: 'default',
    correct: false,
    output:
      '(Few-shot is a prompting technique — results vary by prompt design, not by model training.)',
    note: "Few-shot prompting helps format but doesn't guarantee correctness.",
  })

  // RAG is also a prompting strategy — keep curated
  stages.push({
    id: 'rag',
    label: '+ RAG',
    variant: 'rag',
    correct: true,
    output: '(RAG retrieves reference patterns. Correct answer, but verbose.)',
    note: "RAG provides knowledge but doesn't control output format.",
  })

  if (sftOut) {
    stages.push({
      id: 'sft',
      label: '+ SFT',
      variant: 'sft',
      correct: checkCorrect(sftOut.generated_text, trueLabel),
      output: sftOut.generated_text,
      note: checkCorrect(sftOut.generated_text, trueLabel)
        ? 'Correct, formatted, explains reasoning.'
        : 'Structured format but misclassified.',
    })
  }

  if (dpoOut) {
    stages.push({
      id: 'dpo',
      label: '+ DPO',
      variant: 'dpo',
      correct: checkCorrect(dpoOut.generated_text, trueLabel),
      output: dpoOut.generated_text,
      note: checkCorrect(dpoOut.generated_text, trueLabel)
        ? 'Tighter phrasing, more confident.'
        : 'Better style but misclassified.',
    })
  }

  if (grpoOut) {
    stages.push({
      id: 'grpo',
      label: '+ GRPO',
      variant: 'grpo',
      correct: checkCorrect(grpoOut.generated_text, trueLabel),
      output: grpoOut.generated_text,
      note: checkCorrect(grpoOut.generated_text, trueLabel)
        ? 'Most precise. Uses definitive reasoning and specific thresholds.'
        : 'Best reasoning style despite misclassification.',
    })
  }

  return stages
}

export default function CombinedResults() {
  const { t } = useTranslation()
  const [visibleStages, setVisibleStages] = useState(1)
  const selectedPromptId = useStore((s) => s.selectedPromptId)

  const hasRealData = isLoaded()

  // Build stages from real data or use fallback
  let stages = FALLBACK_STAGES
  let inputDisplay = FALLBACK_INPUT
  let correctLabel = FALLBACK_LABEL

  if (hasRealData) {
    const testPrompts = getTestPrompts()
    const selected = testPrompts.find((p) => p.id === selectedPromptId) ?? testPrompts[0]
    if (selected) {
      const realStages = buildRealStages(selected.id, selected.true_label)
      if (realStages) {
        stages = realStages
        inputDisplay = formatPromptMetrics(selected)
        correctLabel = selected.true_label
      }
    }
  }

  // Get accuracy summary for the footer
  const summary = hasRealData ? getAccuracySummary() : null
  const resourceUtil = hasRealData ? getResourceUtilization() : null

  const revealNext = () => {
    if (visibleStages < stages.length) {
      setVisibleStages(visibleStages + 1)
    }
  }

  return (
    <div className="max-w-5xl mx-auto">
      {/* HW environment badge */}
      {resourceUtil && (
        <div className="mb-4 flex items-center gap-2 text-xs text-slate-500">
          <span className="px-2 py-1 rounded bg-slate-800 border border-slate-700/50 text-slate-400">
            Trained on: {resourceUtil.gpu_name || 'GPU'} ·{' '}
            {resourceUtil.gpu_memory_total_gb ? `${resourceUtil.gpu_memory_total_gb} GB` : ''} ·
            SmolLM2-360M
          </span>
        </div>
      )}

      {/* Pattern picker (only when real data is available) */}
      {hasRealData && <PatternPicker onChange={() => setVisibleStages(1)} />}

      {/* Input */}
      <div className="mb-4">
        <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
          {hasRealData ? t('stop.combined.selectedPattern') : t('stop.combined.hardExample')}
        </label>
        <div className="bg-slate-800 border border-yellow-700/50 rounded-lg p-3 font-mono text-sm text-yellow-200">
          {inputDisplay}
        </div>
        <p className="text-xs text-slate-500 mt-1">
          {t('stop.combined.correctLabel')}{' '}
          <span className="text-emerald-400 font-semibold">{correctLabel}</span> —{' '}
          {t('stop.combined.watchProgression')}
        </p>
      </div>

      {/* Reveal control */}
      <div className="flex items-center gap-3 mb-4">
        <button
          onClick={revealNext}
          disabled={visibleStages >= stages.length}
          className="px-4 py-2 text-sm bg-cyan-700 hover:bg-cyan-600 disabled:bg-slate-700 disabled:text-slate-500 rounded-md transition-colors"
        >
          {t('stop.combined.addNext', { visible: visibleStages, total: stages.length })}
        </button>
        <button
          onClick={() => setVisibleStages(stages.length)}
          className="px-4 py-2 text-sm bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-md transition-colors"
        >
          {t('stop.combined.showAll')}
        </button>
      </div>

      {/* Progressive stages */}
      <div className="space-y-3">
        {stages.slice(0, visibleStages).map((stage, i) => (
          <div key={stage.id} className="flex gap-3 items-start">
            {/* Step indicator */}
            <div className="flex flex-col items-center flex-shrink-0 pt-3">
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${
                  stage.correct
                    ? 'bg-emerald-900/50 text-emerald-400 border border-emerald-700'
                    : 'bg-red-900/50 text-red-400 border border-red-700'
                }`}
              >
                {i + 1}
              </div>
              {i < visibleStages - 1 && <div className="w-px h-full bg-slate-700 mt-1" />}
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
      {visibleStages === stages.length && (
        <div className="mt-6 p-4 rounded-lg bg-gradient-to-r from-cyan-950/30 to-emerald-950/30 border border-cyan-800/30">
          <h4 className="text-sm font-semibold text-cyan-400 mb-2">
            {t('stop.combined.fullJourney')}
          </h4>
          <div className="grid grid-cols-7 gap-1 text-center text-xs mb-3">
            {stages.map((s) => (
              <div
                key={s.id}
                className={`py-1 px-0.5 rounded ${s.correct ? 'bg-emerald-900/30 text-emerald-400' : 'bg-red-900/30 text-red-400'}`}
              >
                {s.label.replace('+ ', '')}
              </div>
            ))}
          </div>

          {/* Accuracy summary from real data */}
          {summary && (
            <div className="grid grid-cols-4 gap-2 text-center text-xs mb-3">
              {Object.entries(summary).map(
                ([variant, s]) =>
                  s && (
                    <div key={variant} className="py-1 px-1 rounded bg-slate-800/50">
                      <div className="font-bold text-slate-200">
                        {(s.accuracy * 100).toFixed(0)}%
                      </div>
                      <div className="text-slate-500">{variant.toUpperCase()}</div>
                    </div>
                  ),
              )}
            </div>
          )}

          <p className="text-sm text-slate-300">{t('stop.combined.fullJourneyP')}</p>
        </div>
      )}
    </div>
  )
}
