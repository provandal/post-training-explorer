import { useState } from 'react'
import ModelOutput from '../components/ModelOutput'
import TokenProbChart from '../components/TokenProbChart'

// Precomputed data for the basic prompt stop
// (Will be replaced by import from precomputed.js once agent finishes)
const EXAMPLE_INPUT = "IOPS: 45000 | Latency: 0.3ms | Block Size: 8K | Read/Write: 70/30 | Sequential: 15% | Queue Depth: 32"

const BASE_RESPONSE = `This appears to be some kind of storage performance metrics. The IOPS value is 45000 which is relatively high. The latency is 0.3ms which is quite low. The block size is 8K. There is a 70/30 read to write ratio with 15% sequential access and a queue depth of 32. These metrics could be from various types of storage workloads depending on the specific use case and configuration being measured.`

const BASE_TOKEN_PROBS = [
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

export default function PromptBasic({ explore = false }) {
  const [userInput, setUserInput] = useState(EXAMPLE_INPUT)
  const [showProbs, setShowProbs] = useState(false)

  return (
    <div className="max-w-4xl mx-auto">
      {/* Input area */}
      <div className="mb-4">
        <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
          Storage I/O Pattern
        </label>
        <div className="bg-slate-800 border border-slate-600 rounded-lg p-3 font-mono text-sm text-slate-200">
          {userInput}
        </div>
        <p className="text-xs text-slate-500 mt-1">
          Prompt: "Classify this storage I/O pattern into a workload type."
        </p>
      </div>

      {/* Model output */}
      <ModelOutput
        label="Base Model (SmolLM2-360M, no fine-tuning)"
        text={BASE_RESPONSE}
        variant="base"
        isCorrect={false}
      />

      <div className="mt-3 p-3 rounded bg-red-950/20 border border-red-800/30">
        <p className="text-sm text-red-300">
          The base model describes the numbers but doesn't actually classify the workload.
          It has no concept of storage workload types — it's just a general-purpose language model.
        </p>
      </div>

      {/* Token probabilities toggle */}
      <button
        onClick={() => setShowProbs(!showProbs)}
        className="mt-4 text-sm text-blue-400 hover:text-blue-300 underline underline-offset-4"
      >
        {showProbs ? 'Hide' : 'Show'} token probabilities (under the covers)
      </button>

      {showProbs && (
        <div className="mt-3 p-4 rounded-lg bg-slate-800/50 border border-slate-700/50">
          <p className="text-xs text-slate-400 mb-3">
            These are the model's top predicted first tokens. Notice how the probability is spread across
            generic words like "This", "The", "Based". The correct answer "OLTP" has only 4% probability.
            The model isn't even trying to classify — it's just starting a generic response.
          </p>
          <TokenProbChart
            data={BASE_TOKEN_PROBS}
            label="Base Model: First Token Probabilities"
            highlightToken="OLTP"
          />
        </div>
      )}
    </div>
  )
}
