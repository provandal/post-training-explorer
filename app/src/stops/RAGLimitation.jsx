import ModelOutput from '../components/ModelOutput'

const EXAMPLE_INPUT = "IOPS: 38000 | Latency: 0.5ms | Block Size: 8K | Read/Write: 55/45 | Sequential: 22% | Queue Depth: 64"

const RAG_VERBOSE_RESPONSE = `Based on the retrieved reference patterns, this I/O profile presents an interesting case. The retrieved documents show two potential matches:

1. OLTP Database (71% similarity) - The high IOPS and low latency are consistent with OLTP workloads. The small block size of 8K also supports this classification. However, I should note that the read/write ratio of 55/45 is more balanced than the typical OLTP range of 60-80/20-40.

2. VDI Virtual Desktop (68% similarity) - The balanced read/write ratio and high queue depth of 64 are more characteristic of VDI workloads. The small block size could fit either pattern.

Given the ambiguity, this pattern could potentially be classified as either OLTP Database or VDI Virtual Desktop. The balanced read/write ratio and high queue depth suggest VDI might be slightly more appropriate, but without additional context about the specific deployment environment, infrastructure configuration, and application characteristics, I cannot make a definitive determination. It would be advisable to collect additional metrics such as I/O size distribution histograms, temporal patterns, and application-level metadata to make a more informed classification.

My best assessment is that this is likely a VDI Virtual Desktop workload, though OLTP Database cannot be ruled out entirely.`

const DESIRED_RESPONSE = `Classification: VDI Virtual Desktop
Confidence: Medium
Key indicators: Balanced R/W ratio (55/45) and high queue depth (64) distinguish this from OLTP. Small blocks (8K) with random access are consistent with VDI.`

export default function RAGLimitation() {
  return (
    <div className="max-w-4xl mx-auto">
      {/* Intro — bridge from RAG success */}
      <div className="mb-5 p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-yellow-400 mb-3">
          RAG gave the model knowledge. It didn't give it discipline.
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          In the previous stop, RAG worked beautifully &mdash; the model pulled the right
          reference pattern and classified correctly. But that was a clear-cut case. Now
          we give it the same ambiguous I/O pattern that tripped up few-shot prompting
          and see whether having a knowledge base is enough.
        </p>
        <p className="text-sm text-slate-400 leading-relaxed">
          Spoiler: the model gets the right answer this time. The problem is <em>how</em> it
          delivers it.
        </p>
      </div>

      {/* Input */}
      <div className="mb-4">
        <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
          Ambiguous I/O Pattern (same one that tripped up few-shot)
        </label>
        <div className="bg-slate-800 border border-yellow-700/50 rounded-lg p-3 font-mono text-sm text-yellow-200">
          {EXAMPLE_INPUT}
        </div>
      </div>

      {/* RAG response - correct but problematic */}
      <ModelOutput
        label="Base Model + RAG Context"
        text={RAG_VERBOSE_RESPONSE}
        variant="rag"
        isCorrect={true}
      />

      {/* What we wanted */}
      <div className="mt-4">
        <ModelOutput
          label="What your team actually needs"
          text={DESIRED_RESPONSE}
          variant="default"
        />
      </div>

      {/* The problems */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-3">
        <div className="p-4 rounded-lg bg-yellow-950/20 border border-yellow-800/30">
          <h4 className="text-sm font-semibold text-yellow-400 mb-2">Right Answer, Wrong Delivery</h4>
          <p className="text-xs text-slate-400">
            The model eventually gets to VDI, but buried it in 6 paragraphs of hedging.
            Your ops team doesn't have time to read a dissertation on every classification.
          </p>
        </div>
        <div className="p-4 rounded-lg bg-yellow-950/20 border border-yellow-800/30">
          <h4 className="text-sm font-semibold text-yellow-400 mb-2">Behavior vs. Knowledge</h4>
          <p className="text-xs text-slate-400">
            RAG solved the <em>knowledge</em> problem — the model has the right reference patterns.
            But it didn't solve the <em>behavior</em> problem — the model doesn't know your team's
            preferred format, confidence level, or conciseness.
          </p>
        </div>
      </div>

      {/* Transition to post-training */}
      <div className="mt-6 p-4 rounded-lg bg-gradient-to-r from-yellow-950/30 to-slate-800 border border-slate-600/30">
        <p className="text-sm text-slate-300">
          <span className="font-semibold text-yellow-400">The ceiling of RAG:</span>{' '}
          RAG changes what the model <em>sees</em>. But to change how the model <em>behaves</em>
          — its format, confidence, style — you need to change the model itself.
          That's what <span className="text-slate-100 font-semibold">Post Training</span> does.
          We're crossing from the left side of the map to the right.
        </p>
      </div>
    </div>
  )
}
