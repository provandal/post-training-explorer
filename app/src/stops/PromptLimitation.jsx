import ModelOutput from '../components/ModelOutput'

const HARD_INPUT = "IOPS: 38000 | Latency: 0.5ms | Block Size: 8K | Read/Write: 55/45 | Sequential: 22% | Queue Depth: 64"

const FEW_SHOT_RESPONSE_HARD = `Classification: OLTP Database

This pattern shows high IOPS with low latency and small block sizes. However, the read/write ratio of 55/45 is more balanced than typical OLTP workloads, and the queue depth of 64 is quite high. This could potentially be a VDI workload as well, but the small block size of 8K and random nature (78% random) lean more toward OLTP Database. Though I should note there is some uncertainty in this classification.`

export default function PromptLimitation() {
  return (
    <div className="max-w-4xl mx-auto">
      {/* The hard example */}
      <div className="mb-4">
        <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
          Ambiguous I/O Pattern
        </label>
        <div className="bg-slate-800 border border-yellow-700/50 rounded-lg p-3 font-mono text-sm text-yellow-200">
          {HARD_INPUT}
        </div>
        <p className="text-xs text-yellow-500/70 mt-1">
          This pattern blurs the line between OLTP Database and VDI. Correct answer: VDI Virtual Desktop.
        </p>
      </div>

      <ModelOutput
        label="Few-Shot Response (with 3 examples in prompt)"
        text={FEW_SHOT_RESPONSE_HARD}
        variant="base"
        isCorrect={false}
      />

      {/* Three problems */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className="p-4 rounded-lg bg-red-950/20 border border-red-800/30">
          <h4 className="text-sm font-semibold text-red-400 mb-2">Problem 1: Wrong Answer</h4>
          <p className="text-xs text-slate-400">
            The model got it wrong — this is VDI, not OLTP. The balanced read/write ratio (55/45)
            and high queue depth (64) are VDI indicators, but the model fixated on block size.
          </p>
        </div>
        <div className="p-4 rounded-lg bg-red-950/20 border border-red-800/30">
          <h4 className="text-sm font-semibold text-red-400 mb-2">Problem 2: Hedging</h4>
          <p className="text-xs text-slate-400">
            Even when it gets answers right, the model hedges: "could potentially be", "I should note".
            Your ops team needs a clear answer, not uncertainty.
          </p>
        </div>
        <div className="p-4 rounded-lg bg-red-950/20 border border-red-800/30">
          <h4 className="text-sm font-semibold text-red-400 mb-2">Problem 3: Ephemeral</h4>
          <p className="text-xs text-slate-400">
            Those few-shot examples disappear after every conversation. You're spending context
            window tokens teaching the model something it should just <em>know</em>.
          </p>
        </div>
      </div>

      {/* Transition callout */}
      <div className="mt-6 p-4 rounded-lg bg-gradient-to-r from-orange-950/30 to-yellow-950/30 border border-orange-800/30">
        <p className="text-sm text-slate-300">
          <span className="font-semibold text-orange-400">The ceiling of prompting:</span>{' '}
          You can make the model better by giving it more context, but you can't make it
          fundamentally smarter or change its behavior. For that, we need to move up the map
          to <span className="text-yellow-400 font-semibold">RAG</span> — or across to{' '}
          <span className="text-slate-200 font-semibold">Post Training</span>.
        </p>
      </div>
    </div>
  )
}
