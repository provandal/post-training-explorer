import { useState } from 'react'
import ModelOutput from '../components/ModelOutput'
import TokenProbChart from '../components/TokenProbChart'
import { isLoaded, getTokenProbsForChart } from '../data/loadArtifacts'
import useStore from '../store'

const EXAMPLE_INPUT =
  'IOPS: 45000 | Latency: 0.3ms | Block Size: 8K | Read/Write: 70/30 | Sequential: 15% | Queue Depth: 32'

const FEW_SHOT_EXAMPLES = [
  {
    input:
      'IOPS: 3000 | Latency: 8ms | Block Size: 512K | Read/Write: 99/1 | Sequential: 95% | Queue Depth: 4',
    output: 'Classification: Video Streaming',
  },
  {
    input:
      'IOPS: 5000 | Latency: 25ms | Block Size: 1024K | Read/Write: 8/92 | Sequential: 94% | Queue Depth: 2',
    output: 'Classification: Backup Archive',
  },
  {
    input:
      'IOPS: 55000 | Latency: 0.4ms | Block Size: 4K | Read/Write: 75/25 | Sequential: 12% | Queue Depth: 48',
    output: 'Classification: OLTP Database',
  },
]

const FEW_SHOT_RESPONSE = `Classification: OLTP Database

The high IOPS of 45000 with very low latency of 0.3ms and small block size of 8K is characteristic of an OLTP database workload. The predominantly random access pattern (only 15% sequential) with a moderate read/write ratio further supports this classification.`

const FALLBACK_BASE_PROBS = [
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

const FALLBACK_FEW_SHOT_PROBS = [
  { token: 'Classification', probability: 0.42 },
  { token: 'OLTP', probability: 0.18 },
  { token: 'The', probability: 0.06 },
  { token: 'This', probability: 0.05 },
  { token: 'Based', probability: 0.04 },
  { token: 'Database', probability: 0.04 },
  { token: 'VDI', probability: 0.03 },
  { token: 'High', probability: 0.02 },
  { token: 'It', probability: 0.02 },
  { token: 'Storage', probability: 0.02 },
  { token: 'These', probability: 0.01 },
  { token: 'OL', probability: 0.01 },
  { token: 'AI', probability: 0.01 },
  { token: 'Backup', probability: 0.01 },
  { token: 'Video', probability: 0.01 },
]

export default function PromptFewShot() {
  const [showExamples, setShowExamples] = useState(true)
  const [showProbs, setShowProbs] = useState(false)
  const selectedPromptId = useStore((s) => s.selectedPromptId)

  // Use real base token prob data when available, fall back to hardcoded
  const baseTokenProbs = (() => {
    if (!isLoaded()) return FALLBACK_BASE_PROBS
    const real = getTokenProbsForChart('base', selectedPromptId)
    return real.length > 0 ? real : FALLBACK_BASE_PROBS
  })()

  return (
    <div className="max-w-4xl mx-auto">
      {/* Intro — bridge from basic prompting */}
      <div className="mb-5 p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-orange-400 mb-3">
          What if we showed the model some examples first?
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          What you just saw was <strong className="text-orange-300">zero-shot</strong> prompting
          &mdash; instructions only, no examples. The model knew what we wanted but had never seen a
          correct answer. Now we try <strong className="text-orange-300">few-shot</strong>{' '}
          prompting: we prepend a handful of labeled examples directly into the prompt so the model
          can mimic the pattern. The naming is literal &mdash; zero examples, one example, or a few.
        </p>
        <p className="text-sm text-slate-400 leading-relaxed">
          Below are three input &rarr; output pairs we inject before the real question. The model
          sees them and learns the expected format on the fly &mdash; no training required, just a
          longer prompt.
        </p>
      </div>

      {/* Few-shot examples */}
      <div className="mb-4">
        <button
          onClick={() => setShowExamples(!showExamples)}
          className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2 hover:text-slate-300 cursor-pointer"
        >
          {showExamples ? 'Hide' : 'Show'} Few-Shot Examples in Prompt
        </button>
        {showExamples && (
          <div className="bg-slate-800 border border-orange-800/30 rounded-lg p-3 space-y-2">
            <p className="text-xs text-orange-400 italic">
              These examples are prepended to the prompt:
            </p>
            {FEW_SHOT_EXAMPLES.map((ex, i) => (
              <div key={i} className="text-xs font-mono border-l-2 border-orange-700/50 pl-2">
                <div className="text-slate-400">{ex.input}</div>
                <div className="text-orange-300">{ex.output}</div>
              </div>
            ))}
            <div className="border-t border-slate-700 pt-2 mt-2">
              <p className="text-xs text-slate-500">Now classify:</p>
              <div className="text-xs font-mono text-slate-300">{EXAMPLE_INPUT}</div>
            </div>
          </div>
        )}
      </div>

      {/* Model output */}
      <ModelOutput
        label="Base Model + Few-Shot Prompt"
        text={FEW_SHOT_RESPONSE}
        variant="default"
        isCorrect={true}
      />

      <div className="mt-3 p-3 rounded bg-green-950/20 border border-green-800/30">
        <p className="text-sm text-green-300">
          With few-shot examples, the model follows the format and gets the right answer. But it
          needed 3 examples eating up your context window to get here.
        </p>
      </div>

      {/* Token probability comparison */}
      <button
        onClick={() => setShowProbs(!showProbs)}
        className="mt-4 text-sm text-blue-400 hover:text-blue-300 underline underline-offset-4"
      >
        {showProbs ? 'Hide' : 'Show'} how few-shot shifted probabilities
      </button>

      {showProbs && (
        <div className="mt-3 p-4 rounded-lg bg-slate-800/50 border border-slate-700/50">
          <p className="text-xs text-slate-400 mb-3">
            Compare the first-token probabilities. Without examples, the model spreads probability
            across generic words. With few-shot examples, "Classification" jumps to 42% — the model
            learned the format. But "OLTP" is still only at 18%. It's better, but not confident.
          </p>
          <TokenProbChart
            data={baseTokenProbs}
            comparisonData={FALLBACK_FEW_SHOT_PROBS}
            label="First Token: Basic Prompt vs Few-Shot"
            comparisonLabel="Few-Shot"
            highlightToken="Classification"
          />
        </div>
      )}
    </div>
  )
}
