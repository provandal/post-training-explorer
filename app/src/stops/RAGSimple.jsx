import { useState } from 'react'
import ModelOutput from '../components/ModelOutput'

const EXAMPLE_INPUT = "IOPS: 45000 | Latency: 0.3ms | Block Size: 8K | Read/Write: 70/30 | Sequential: 15% | Queue Depth: 32"

const KNOWLEDGE_BASE = [
  {
    id: 1,
    pattern: "IOPS: 30K-80K, Block: 4-8K, Latency: <1ms, Sequential: 10-25%, R/W: 60-80/20-40",
    label: "OLTP Database",
    description: "Transaction processing workloads (MySQL, PostgreSQL, Oracle). Characterized by high IOPS, small random I/O, very low latency requirements.",
    similarity: 0.94,
  },
  {
    id: 2,
    pattern: "IOPS: 20K-60K, Block: 4-16K, Latency: <1ms, Sequential: 15-30%, R/W: 50-70/30-50",
    label: "VDI Virtual Desktop",
    description: "Virtual desktop infrastructure. High IOPS, small blocks, but more balanced read/write ratio and higher queue depth than OLTP.",
    similarity: 0.71,
  },
  {
    id: 3,
    pattern: "IOPS: 10K-30K, Block: 128K-1M, Latency: 0.5-3ms, Sequential: 85-95%, R/W: 90-99/1-10",
    label: "AI ML Training",
    description: "Machine learning training workloads. Large sequential reads dominate. High throughput with moderate IOPS.",
    similarity: 0.23,
  },
]

const RAG_RESPONSE = `Classification: OLTP Database

Based on the retrieved reference patterns, this I/O profile closely matches OLTP Database workloads. The key indicators are:
- High IOPS (45,000) within the 30K-80K range typical of transaction processing
- Very low latency (0.3ms) consistent with OLTP requirements (<1ms)
- Small block size (8K) in the 4-8K range characteristic of database operations
- Predominantly random access (only 15% sequential) matching OLTP patterns
- Read-heavy ratio (70/30) within the expected 60-80/20-40 range

The retrieved reference pattern shows 94% similarity to known OLTP Database profiles.`

export default function RAGSimple({ explore = false }) {
  const [showRetrieval, setShowRetrieval] = useState(true)

  return (
    <div className="max-w-4xl mx-auto">
      {/* Input */}
      <div className="mb-4">
        <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
          Query I/O Pattern
        </label>
        <div className="bg-slate-800 border border-slate-600 rounded-lg p-3 font-mono text-sm text-slate-200">
          {EXAMPLE_INPUT}
        </div>
      </div>

      {/* Retrieval results */}
      <button
        onClick={() => setShowRetrieval(!showRetrieval)}
        className="text-xs font-semibold text-yellow-400 uppercase tracking-wide mb-2 hover:text-yellow-300 cursor-pointer"
      >
        {showRetrieval ? 'Hide' : 'Show'} Retrieved Knowledge
      </button>

      {showRetrieval && (
        <div className="mb-4 space-y-2">
          {KNOWLEDGE_BASE.map((doc) => (
            <div
              key={doc.id}
              className={`p-3 rounded-lg border ${
                doc.similarity > 0.8
                  ? 'border-green-700/50 bg-green-950/20'
                  : doc.similarity > 0.5
                  ? 'border-yellow-700/50 bg-yellow-950/10'
                  : 'border-slate-700/50 bg-slate-800/30'
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-semibold text-slate-300">{doc.label}</span>
                <span className={`text-xs px-2 py-0.5 rounded-full ${
                  doc.similarity > 0.8 ? 'bg-green-900/50 text-green-400' : 'bg-slate-700 text-slate-400'
                }`}>
                  {(doc.similarity * 100).toFixed(0)}% match
                </span>
              </div>
              <p className="text-xs text-slate-500 font-mono">{doc.pattern}</p>
              <p className="text-xs text-slate-400 mt-1">{doc.description}</p>
            </div>
          ))}
          <p className="text-xs text-slate-500 italic">
            The retrieval system found 3 similar patterns. The top match (94% similarity) is OLTP Database.
            These retrieved documents are injected into the prompt as context.
          </p>
        </div>
      )}

      {/* RAG-augmented response */}
      <ModelOutput
        label="Base Model + RAG Context"
        text={RAG_RESPONSE}
        variant="rag"
        isCorrect={true}
      />

      <div className="mt-3 p-3 rounded bg-green-950/20 border border-green-800/30">
        <p className="text-sm text-green-300">
          With retrieved context, the model nails it. It cites the reference patterns and
          explains its reasoning. The knowledge base acts as an expert reference library.
        </p>
      </div>

      {/* How it works diagram */}
      <div className="mt-6 p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">How RAG Works</h4>
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <div className="px-3 py-2 rounded bg-slate-700 text-slate-300 font-mono">Your query</div>
          <span className="text-slate-600">&rarr;</span>
          <div className="px-3 py-2 rounded bg-yellow-900/30 border border-yellow-800/30 text-yellow-300">
            Vector search
          </div>
          <span className="text-slate-600">&rarr;</span>
          <div className="px-3 py-2 rounded bg-yellow-900/30 border border-yellow-800/30 text-yellow-300">
            Top-K matches
          </div>
          <span className="text-slate-600">&rarr;</span>
          <div className="px-3 py-2 rounded bg-slate-700 text-slate-300">
            Augmented prompt
          </div>
          <span className="text-slate-600">&rarr;</span>
          <div className="px-3 py-2 rounded bg-blue-900/30 border border-blue-800/30 text-blue-300">
            LLM
          </div>
        </div>
        <p className="text-xs text-slate-500 mt-3">
          Key insight: RAG doesn't change the model at all. It changes what the model <em>sees</em>.
          The model weights are identical — only the input is different.
        </p>
      </div>
    </div>
  )
}
