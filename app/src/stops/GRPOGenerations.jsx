import { useState } from 'react'

const GRPO_EXAMPLE = {
  input: "IOPS: 12000 | Latency: 4.5ms | Block Size: 128K | Read/Write: 92/8 | Sequential: 78% | Queue Depth: 8",
  correctLabel: "OLAP Analytics",
  generations: [
    {
      id: 1,
      text: "Classification: OLAP Analytics\nConfidence: High\nKey indicators: Large block sequential reads (128K, 78% sequential) with heavy read bias (92/8) indicate analytical query processing.",
      reward: 1.0,
      correct: true,
    },
    {
      id: 2,
      text: "Classification: AI ML Training\nConfidence: Medium\nKey indicators: Large block sequential reads often indicate training data loading. Read-heavy pattern consistent with data pipeline.",
      reward: 0.0,
      correct: false,
    },
    {
      id: 3,
      text: "Classification: OLAP Analytics\nConfidence: Medium-High\nKey indicators: Moderate IOPS with large blocks and high sequential ratio point to scan-heavy warehouse queries.",
      reward: 1.0,
      correct: true,
    },
    {
      id: 4,
      text: "Classification: Video Streaming\nConfidence: Low\nKey indicators: Sequential reads with large blocks. However, IOPS is too high for typical streaming.",
      reward: 0.0,
      correct: false,
    },
    {
      id: 5,
      text: "Classification: AI ML Training\nConfidence: Medium\nKey indicators: Sequential read pattern with large blocks suggests data loading for model training.",
      reward: 0.0,
      correct: false,
    },
    {
      id: 6,
      text: "Classification: OLAP Analytics\nConfidence: High\nKey indicators: Read-heavy (92/8) with large sequential I/O (128K, 78%) and moderate queue depth is classic analytics/BI workload pattern.",
      reward: 1.0,
      correct: true,
    },
    {
      id: 7,
      text: "Classification: OLAP Analytics\nConfidence: Medium\nKey indicators: Large blocks with sequential reads suggest analytical processing or data warehouse queries.",
      reward: 1.0,
      correct: true,
    },
    {
      id: 8,
      text: "Classification: AI ML Training\nConfidence: Medium-High\nKey indicators: High sequential percentage with large blocks and extreme read bias matches GPU training data loading patterns.",
      reward: 0.0,
      correct: false,
    },
  ],
}

// Compute group statistics
const rewards = GRPO_EXAMPLE.generations.map(g => g.reward)
const meanReward = rewards.reduce((a, b) => a + b, 0) / rewards.length
const stdReward = Math.sqrt(rewards.reduce((sum, r) => sum + (r - meanReward) ** 2, 0) / rewards.length)

export default function GRPOGenerations() {
  const [revealedCount, setRevealedCount] = useState(0)
  const [showStats, setShowStats] = useState(false)

  const revealNext = () => {
    if (revealedCount < GRPO_EXAMPLE.generations.length) {
      setRevealedCount(revealedCount + 1)
    }
    if (revealedCount + 1 === GRPO_EXAMPLE.generations.length) {
      setShowStats(true)
    }
  }

  const revealAll = () => {
    setRevealedCount(GRPO_EXAMPLE.generations.length)
    setShowStats(true)
  }

  const correctCount = GRPO_EXAMPLE.generations.filter(g => g.correct).length

  return (
    <div className="max-w-5xl mx-auto">
      {/* Input */}
      <div className="mb-4">
        <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
          Challenge Input (Ambiguous: OLAP vs AI Training)
        </label>
        <div className="bg-slate-800 border border-slate-600 rounded-lg p-3 font-mono text-sm text-slate-200">
          {GRPO_EXAMPLE.input}
        </div>
        <p className="text-xs text-slate-500 mt-1">
          Correct answer: <span className="text-emerald-400 font-semibold">{GRPO_EXAMPLE.correctLabel}</span> |
          The model generates <strong>8 attempts</strong>. Each is scored: correct = 1.0, incorrect = 0.0.
        </p>
      </div>

      {/* Reveal controls */}
      <div className="flex gap-2 mb-4">
        <button
          onClick={revealNext}
          disabled={revealedCount >= GRPO_EXAMPLE.generations.length}
          className="px-4 py-2 text-sm bg-emerald-700 hover:bg-emerald-600 disabled:bg-slate-700 disabled:text-slate-500 rounded-md transition-colors"
        >
          Reveal next generation ({revealedCount}/{GRPO_EXAMPLE.generations.length})
        </button>
        <button
          onClick={revealAll}
          className="px-4 py-2 text-sm bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-md transition-colors"
        >
          Reveal all
        </button>
      </div>

      {/* Generation cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
        {GRPO_EXAMPLE.generations.map((gen, i) => {
          const isRevealed = i < revealedCount
          const advantage = gen.reward - meanReward

          return (
            <div
              key={gen.id}
              className={`p-3 rounded-lg border transition-all duration-500 ${
                !isRevealed
                  ? 'border-slate-700/30 bg-slate-800/20 opacity-30'
                  : gen.correct
                  ? 'border-emerald-700/50 bg-emerald-950/20'
                  : 'border-red-800/50 bg-red-950/20'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-slate-500">
                  Generation #{gen.id}
                </span>
                {isRevealed && (
                  <div className="flex items-center gap-2">
                    <span className={`text-xs px-2 py-0.5 rounded-full font-semibold ${
                      gen.correct
                        ? 'bg-emerald-900/50 text-emerald-400'
                        : 'bg-red-900/50 text-red-400'
                    }`}>
                      Reward: {gen.reward.toFixed(1)}
                    </span>
                    {showStats && (
                      <span className={`text-xs px-2 py-0.5 rounded-full ${
                        advantage > 0
                          ? 'bg-emerald-900/30 text-emerald-300'
                          : 'bg-red-900/30 text-red-300'
                      }`}>
                        Adv: {advantage > 0 ? '+' : ''}{advantage.toFixed(2)}
                      </span>
                    )}
                  </div>
                )}
              </div>
              <pre className={`text-xs font-mono whitespace-pre-wrap leading-relaxed ${
                isRevealed ? 'text-slate-200' : 'text-slate-600'
              }`}>
                {isRevealed ? gen.text : 'Click "Reveal next" to see this generation...'}
              </pre>
            </div>
          )
        })}
      </div>

      {/* Group statistics */}
      {showStats && (
        <div className="p-4 rounded-lg bg-emerald-950/20 border border-emerald-800/30">
          <h4 className="text-sm font-semibold text-emerald-400 mb-3">Group Statistics (GRPO)</h4>
          <div className="grid grid-cols-4 gap-4 text-center mb-3">
            <div>
              <div className="text-2xl font-bold text-emerald-400">{correctCount}/8</div>
              <div className="text-xs text-slate-500">Correct</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-slate-200">{meanReward.toFixed(3)}</div>
              <div className="text-xs text-slate-500">Mean Reward</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-slate-200">{stdReward.toFixed(3)}</div>
              <div className="text-xs text-slate-500">Std Dev</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-slate-200">0</div>
              <div className="text-xs text-slate-500">Human Labels Needed</div>
            </div>
          </div>

          <p className="text-xs text-slate-400">
            <strong className="text-emerald-300">How GRPO learns:</strong> The mean reward ({meanReward.toFixed(3)}) is the baseline.
            The {correctCount} correct generations get <span className="text-emerald-300">positive advantage</span> (+{(1.0 - meanReward).toFixed(2)}) — they're reinforced.
            The {8 - correctCount} incorrect generations get <span className="text-red-300">negative advantage</span> ({(0 - meanReward).toFixed(2)}) — they're suppressed.
            No critic network, no reward model — just group statistics.
          </p>

          <p className="text-xs text-slate-500 mt-2 italic">
            This is the technique behind DeepSeek R1. The answer itself is the teacher.
            For our task, "correct classification" is verifiable. For math, "correct answer" is verifiable.
            That's why it's called Reinforcement Learning with Verifiable Rewards (RLVR).
          </p>
        </div>
      )}
    </div>
  )
}
