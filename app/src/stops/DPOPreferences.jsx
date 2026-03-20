import { useState } from 'react'
import useStore from '../store'
import ModelOutput from '../components/ModelOutput'

const PREFERENCE_PAIRS = [
  {
    input: "IOPS: 45000 | Latency: 0.3ms | Block Size: 8K | Read/Write: 70/30 | Sequential: 15% | Queue Depth: 32",
    responseA: `Classification: OLTP Database
Confidence: High
Key indicators: High IOPS (45K) with very low latency (0.3ms) and small block size (8K) are hallmarks of transaction processing. Random-dominant access (85%) with moderate read bias confirms OLTP pattern.`,
    responseB: `Classification: OLTP Database
This could potentially be an OLTP Database workload. The IOPS are quite high at 45000, and the block size of 8K is relatively small. The latency is low at 0.3ms. However, it could also possibly be a VDI workload given the mixed read/write ratio. On balance, OLTP Database seems most likely but I'm not entirely certain. The queue depth of 32 and 15% sequential access are also consistent with this classification, though other workload types occasionally show similar patterns.`,
    betterIs: 'A',
    whyABetter: 'Concise, confident, structured format with key indicators highlighted',
    whyBWorse: 'Correct answer but verbose, hedging, mentions alternatives unnecessarily',
  },
  {
    input: "IOPS: 2500 | Latency: 18ms | Block Size: 512K | Read/Write: 5/95 | Sequential: 93% | Queue Depth: 2",
    responseA: `Classification: Backup Archive
Confidence: High
Key indicators: Very large sequential writes (512K blocks, 93% sequential) with heavy write bias (5/95) and low queue depth (2) indicate batch backup or archive operations. High latency tolerance (18ms) is consistent with background archival processes.`,
    responseB: `Classification: Backup Archive
The storage metrics indicate this is a backup or archive workload.`,
    betterIs: 'A',
    whyABetter: 'Includes explanation and reasoning, not just the label',
    whyBWorse: 'Too terse — correct but provides no actionable information',
  },
  {
    input: "IOPS: 22000 | Latency: 1.2ms | Block Size: 256K | Read/Write: 96/4 | Sequential: 88% | Queue Depth: 64",
    responseA: `Classification: AI ML Training
Confidence: High
Key indicators: Large sequential reads (256K, 88% sequential) with extreme read bias (96/4) and high queue depth (64) indicate data pipeline feeding GPU training. Moderate IOPS with high throughput profile is characteristic of deep learning data loaders.`,
    responseB: `Classification: AI ML Training
Confidence: Medium-High
Key indicators: Large block sequential reads (256K, 88% sequential) with heavy read bias (96/4). High queue depth (64) suggests parallel data loading.
Note: Could be confused with Video Streaming, but IOPS (22K) and queue depth are too high for streaming.`,
    betterIs: 'B',
    whyABetter: null,
    whyBWorse: null,
  },
]

export default function DPOPreferences() {
  const [currentPair, setCurrentPair] = useState(0)
  const [userChoice, setUserChoice] = useState(null)
  const [showReveal, setShowReveal] = useState(false)
  const addPreference = useStore((s) => s.addPreference)

  const pair = PREFERENCE_PAIRS[currentPair]

  const handleChoice = (choice) => {
    setUserChoice(choice)
    setShowReveal(true)
    addPreference({ pairIndex: currentPair, choice })
  }

  const nextPair = () => {
    if (currentPair < PREFERENCE_PAIRS.length - 1) {
      setCurrentPair(currentPair + 1)
      setUserChoice(null)
      setShowReveal(false)
    }
  }

  return (
    <div className="max-w-5xl mx-auto">
      {/* Progress */}
      <div className="flex items-center gap-2 mb-4">
        <span className="text-xs text-slate-500">Preference pair</span>
        {PREFERENCE_PAIRS.map((_, i) => (
          <button
            key={i}
            onClick={() => { setCurrentPair(i); setUserChoice(null); setShowReveal(false) }}
            className={`w-8 h-8 rounded-full text-xs font-semibold transition-colors ${
              i === currentPair
                ? 'bg-pink-600 text-white'
                : i < currentPair
                ? 'bg-pink-900/50 text-pink-400'
                : 'bg-slate-700 text-slate-500'
            }`}
          >
            {i + 1}
          </button>
        ))}
      </div>

      {/* Input */}
      <div className="mb-4 bg-slate-800 border border-slate-600 rounded-lg p-3 font-mono text-sm text-slate-200">
        {pair.input}
      </div>

      {/* Instruction */}
      {!userChoice && (
        <p className="text-sm text-pink-300 mb-4 font-semibold">
          Both responses are correct. Which one do you prefer? Click to choose.
        </p>
      )}

      {/* Side by side responses */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div
          onClick={() => !userChoice && handleChoice('A')}
          className={`cursor-pointer transition-all ${!userChoice ? 'hover:scale-[1.01]' : ''} ${
            userChoice === 'A' ? 'ring-2 ring-green-500' : userChoice === 'B' ? 'opacity-50' : ''
          }`}
        >
          <ModelOutput
            label="Response A"
            text={pair.responseA}
            variant={userChoice === 'A' ? 'chosen' : userChoice === 'B' ? 'rejected' : 'default'}
          />
        </div>
        <div
          onClick={() => !userChoice && handleChoice('B')}
          className={`cursor-pointer transition-all ${!userChoice ? 'hover:scale-[1.01]' : ''} ${
            userChoice === 'B' ? 'ring-2 ring-green-500' : userChoice === 'A' ? 'opacity-50' : ''
          }`}
        >
          <ModelOutput
            label="Response B"
            text={pair.responseB}
            variant={userChoice === 'B' ? 'chosen' : userChoice === 'A' ? 'rejected' : 'default'}
          />
        </div>
      </div>

      {/* Reveal after choice */}
      {showReveal && (
        <div className="mt-4 p-4 rounded-lg bg-pink-950/20 border border-pink-800/30">
          <p className="text-sm text-slate-300 mb-2">
            <span className="font-semibold text-pink-400">You picked Response {userChoice}.</span>{' '}
            {userChoice === pair.betterIs
              ? "That matches the training data's preference."
              : pair.betterIs
              ? `Interesting! The training data preferred Response ${pair.betterIs}, but preferences are subjective — that's the whole point.`
              : "This one is genuinely ambiguous — reasonable people disagree, and that's informative too."
            }
          </p>
          {pair.whyABetter && (
            <p className="text-xs text-slate-400">
              <strong>Why A is typically preferred:</strong> {pair.whyABetter}
            </p>
          )}
          {pair.whyBWorse && (
            <p className="text-xs text-slate-400">
              <strong>Why B is typically less preferred:</strong> {pair.whyBWorse}
            </p>
          )}

          <p className="text-xs text-slate-500 mt-3 italic">
            You just did exactly what DPO training data looks like: comparing two outputs and saying
            which is better. 400 pairs like this is all it takes to reshape the model's style.
          </p>

          {currentPair < PREFERENCE_PAIRS.length - 1 && (
            <button onClick={nextPair} className="mt-3 px-4 py-1.5 text-sm bg-pink-700 hover:bg-pink-600 rounded-md transition-colors">
              Next pair &rarr;
            </button>
          )}
        </div>
      )}
    </div>
  )
}
