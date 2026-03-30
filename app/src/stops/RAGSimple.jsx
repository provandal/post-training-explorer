import { useState } from 'react'
import ModelOutput from '../components/ModelOutput'
import AnimatedPipeline from '../components/AnimatedPipeline'
import VectorSearchViz from '../components/VectorSearchViz'
import SectionTabs from '../components/SectionTabs'
import useStore from '../store'

const EXAMPLE_INPUT =
  'IOPS: 45000 | Latency: 0.3ms | Block Size: 8K | Read/Write: 70/30 | Sequential: 15% | Queue Depth: 32'

const FAILED_FEWSHOT = `Classification: OLTP Database

This pattern shows high IOPS with low latency and small block sizes. However, I should note there is some uncertainty...`

const KNOWLEDGE_BASE = [
  {
    id: 1,
    pattern: 'IOPS: 30K-80K, Block: 4-8K, Latency: <1ms, Sequential: 10-25%, R/W: 60-80/20-40',
    label: 'OLTP Database',
    description:
      'Transaction processing workloads (MySQL, PostgreSQL, Oracle). Characterized by high IOPS, small random I/O, very low latency requirements.',
    similarity: 0.94,
  },
  {
    id: 2,
    pattern: 'IOPS: 20K-60K, Block: 4-16K, Latency: <1ms, Sequential: 15-30%, R/W: 50-70/30-50',
    label: 'VDI Virtual Desktop',
    description:
      'Virtual desktop infrastructure. High IOPS, small blocks, but more balanced read/write ratio and higher queue depth than OLTP.',
    similarity: 0.71,
  },
  {
    id: 3,
    pattern: 'IOPS: 10K-30K, Block: 128K-1M, Latency: 0.5-3ms, Sequential: 85-95%, R/W: 90-99/1-10',
    label: 'AI ML Training',
    description:
      'Machine learning training workloads. Large sequential reads dominate. High throughput with moderate IOPS.',
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

const RAG_PIPELINE_STEPS = [
  { icon: '\u{1F4DD}', label: 'Your Query', desc: 'I/O metric string' },
  { icon: '\u{1F9EE}', label: 'Embed', desc: 'Convert to vector' },
  { icon: '\u{1F50D}', label: 'Search', desc: 'Find nearest vectors' },
  { icon: '\u{1F4CB}', label: 'Retrieve', desc: 'Top-K documents' },
  { icon: '\u{1F4E6}', label: 'Augment', desc: 'Inject into prompt' },
  { icon: '\u{1F916}', label: 'Generate', desc: 'LLM responds' },
]

export default function RAGSimple() {
  const [section, setSection] = useState('problem')
  const [showContextAside, setShowContextAside] = useState(false)
  const setActiveQuadrant = useStore((s) => s.setActiveQuadrant)
  const setMode = useStore((s) => s.setMode)
  const tabs = [
    { id: 'problem', label: 'The Problem' },
    { id: 'concept', label: 'How RAG Works' },
    { id: 'demo', label: 'See It Work' },
    { id: 'deepdive', label: 'Deep Dive: Vector Search' },
  ]

  return (
    <div className="max-w-5xl mx-auto">
      {/* Section tabs (top) */}
      <div className="mb-6">
        <SectionTabs tabs={tabs} active={section} onSelect={setSection} color="yellow" />
      </div>

      {/* ==================== PROBLEM ==================== */}
      {section === 'problem' && (
        <div className="space-y-4">
          <div className="p-4 rounded-lg bg-red-950/20 border border-red-800/30">
            <h3 className="text-base font-semibold text-red-400 mb-2">
              Few-shot prompting has limits
            </h3>
            <p className="text-sm text-slate-300 mb-3">
              In the previous stop, we added examples to the prompt and the model improved. But
              there are real problems with this approach:
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="p-3 rounded bg-slate-800/50">
                <h4 className="text-sm font-semibold text-red-300 mb-1">
                  Context window is finite
                </h4>
                <p className="text-xs text-slate-400">
                  Every example you add consumes tokens. With 6 workload categories and multiple
                  examples each, you'd use thousands of tokens just for examples &mdash; leaving
                  less room for the actual query and response.
                </p>
                <button
                  onClick={() => setShowContextAside(!showContextAside)}
                  className="mt-2 text-xs text-purple-400 hover:text-purple-300 underline underline-offset-4 cursor-pointer"
                >
                  {showContextAside ? 'Hide' : 'Why does this matter?'}
                </button>
                {showContextAside && (
                  <div className="mt-2 p-3 rounded bg-purple-950/20 border border-purple-800/30">
                    <p className="text-xs text-slate-300 leading-relaxed mb-2">
                      Language models process everything &mdash; system prompt, conversation
                      history, RAG documents, few-shot examples, and their own response &mdash;
                      inside a single fixed-size{' '}
                      <strong className="text-purple-300">context window</strong>. When that window
                      fills up, older content gets dropped or summarized, leading to{' '}
                      <strong className="text-purple-300">context rot</strong>: the model quietly
                      forgets instructions, contradicts earlier answers, or misses key details
                      buried in the middle of long inputs.
                    </p>
                    <button
                      onClick={() => {
                        setActiveQuadrant('context')
                        setMode('explore')
                      }}
                      className="text-xs font-semibold text-purple-400 hover:text-purple-300 underline underline-offset-4 cursor-pointer"
                    >
                      Deep dive: Context windows, rot, and management strategies &rarr;
                    </button>
                  </div>
                )}
              </div>
              <div className="p-3 rounded bg-slate-800/50">
                <h4 className="text-sm font-semibold text-red-300 mb-1">Static and inflexible</h4>
                <p className="text-xs text-slate-400">
                  The same examples appear every time, regardless of what the user is asking about.
                  If they ask about a backup pattern, the OLTP examples are wasted context.
                </p>
              </div>
              <div className="p-3 rounded bg-slate-800/50">
                <h4 className="text-sm font-semibold text-red-300 mb-1">
                  Can't scale to real knowledge
                </h4>
                <p className="text-xs text-slate-400">
                  Your organization has hundreds of documented patterns, runbooks, and vendor specs.
                  You can't fit all that into a prompt.
                </p>
              </div>
            </div>
          </div>

          <div className="p-4 rounded-lg bg-yellow-950/20 border border-yellow-800/30">
            <p className="text-sm text-slate-300">
              <span className="font-semibold text-yellow-400">
                What if the model could look things up?
              </span>{' '}
              Instead of cramming everything into the prompt, what if the model had access to a
              reference library — and could search it for just the relevant information before
              answering?
            </p>
            <p className="text-sm text-slate-400 mt-2">That's exactly what RAG does.</p>
          </div>
        </div>
      )}

      {/* ==================== CONCEPT ==================== */}
      {section === 'concept' && (
        <div className="space-y-4">
          <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h3 className="text-base font-semibold text-yellow-400 mb-2">
              RAG = Retrieval-Augmented Generation
            </h3>
            <p className="text-sm text-slate-300 mb-4">
              RAG adds a "lookup" step before the model generates a response. Instead of relying
              only on what's in the prompt or what the model memorized during training, RAG
              <strong className="text-yellow-300"> searches a knowledge base</strong> for
              information relevant to the current query and includes it in the prompt automatically.
            </p>

            {/* Animated pipeline */}
            <AnimatedPipeline steps={RAG_PIPELINE_STEPS} autoPlay speed={900} />
          </div>

          {/* Key concept cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/50">
              <h4 className="text-sm font-semibold text-yellow-400 mb-2">Embeddings</h4>
              <p className="text-xs text-slate-400 mb-2">
                Text is converted into a list of numbers (a "vector") that captures its meaning.
                Similar texts produce similar vectors. This is done by an embedding model — a
                separate, small neural network specialized for this task.
              </p>
              <div className="p-2 rounded bg-slate-900 font-mono text-xs text-slate-500">
                "High IOPS, small blocks" → [0.82, -0.14, 0.67, 0.31, ...]
                <br />
                "Many random reads, 4K" → [0.79, -0.11, 0.71, 0.28, ...]
                <br />
                <span className="text-green-400">↑ Similar meaning = similar numbers</span>
              </div>
            </div>
            <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/50">
              <h4 className="text-sm font-semibold text-yellow-400 mb-2">Vector Search</h4>
              <p className="text-xs text-slate-400 mb-2">
                Your query is embedded into the same vector space as the knowledge base. Then we
                find the nearest neighbors — the entries whose vectors are most similar to your
                query's vector. This is "semantic search": it finds relevant content even if the
                exact words don't match.
              </p>
              <div className="p-2 rounded bg-slate-900 font-mono text-xs text-slate-500">
                cosine_similarity(query, doc) = (q · d) / (|q| × |d|)
                <br />
                <span className="text-green-400">Range: 0 (unrelated) to 1 (identical)</span>
              </div>
            </div>
          </div>

          {/* Critical point */}
          <div className="p-3 rounded bg-blue-950/20 border border-blue-800/30">
            <p className="text-xs text-blue-300">
              <strong>Key insight:</strong> RAG does not change the model's weights at all. The
              model is identical — only the input it receives is different. RAG changes what the
              model <em>sees</em>, not how it <em>thinks</em>.
            </p>
          </div>
        </div>
      )}

      {/* ==================== DEMO ==================== */}
      {section === 'demo' && (
        <div className="space-y-4">
          {/* Input */}
          <div>
            <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
              Query I/O Pattern
            </label>
            <div className="bg-slate-800 border border-slate-600 rounded-lg p-3 font-mono text-sm text-slate-200">
              {EXAMPLE_INPUT}
            </div>
          </div>

          {/* Retrieval results */}
          <div>
            <h4 className="text-xs font-semibold text-yellow-400 uppercase tracking-wide mb-2">
              Step 1: Retrieved from Knowledge Base
            </h4>
            <div className="space-y-2">
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
                    <span
                      className={`text-xs px-2 py-0.5 rounded-full font-semibold ${
                        doc.similarity > 0.8
                          ? 'bg-green-900/50 text-green-400'
                          : 'bg-slate-700 text-slate-400'
                      }`}
                    >
                      {(doc.similarity * 100).toFixed(0)}% match
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 font-mono">{doc.pattern}</p>
                  <p className="text-xs text-slate-400 mt-1">{doc.description}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Augmented response */}
          <div>
            <h4 className="text-xs font-semibold text-yellow-400 uppercase tracking-wide mb-2">
              Step 2: Model responds with retrieved context
            </h4>
            <ModelOutput
              label="Base Model + RAG Context"
              text={RAG_RESPONSE}
              variant="rag"
              isCorrect={true}
            />
          </div>

          <div className="p-3 rounded bg-green-950/20 border border-green-800/30">
            <p className="text-sm text-green-300">
              With retrieved context, the model nails it — correct classification, cites the
              reference patterns, and explains its reasoning using data from the knowledge base.
            </p>
          </div>
        </div>
      )}

      {/* ==================== DEEP DIVE ==================== */}
      {section === 'deepdive' && (
        <div className="space-y-4">
          <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h3 className="text-base font-semibold text-yellow-400 mb-2">
              How Vector Search Actually Works
            </h3>
            <p className="text-sm text-slate-300 mb-4">
              Each entry in the knowledge base has been pre-converted into an embedding vector
              (hundreds of numbers). When your query arrives, it's also embedded. Then we find which
              stored vectors are closest to the query vector. Here's what that looks like in 2D
              (real embeddings have hundreds of dimensions, but the clustering principle is the
              same):
            </p>
            <VectorSearchViz autoPlay />
          </div>

          {/* What's happening at each step */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/50">
              <h4 className="text-xs font-semibold text-yellow-300 uppercase tracking-wide mb-2">
                Why similar patterns cluster
              </h4>
              <p className="text-xs text-slate-400">
                The embedding model was trained on millions of text pairs, learning that "high IOPS
                with small blocks" and "transaction processing database" are semantically related —
                so they get similar vectors. OLTP entries cluster together, backup entries cluster
                together, etc. This happens automatically from the meaning of the text.
              </p>
            </div>
            <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/50">
              <h4 className="text-xs font-semibold text-yellow-300 uppercase tracking-wide mb-2">
                Infrastructure for vector search
              </h4>
              <p className="text-xs text-slate-400">
                The embedding vectors are stored in a <strong>vector database</strong> (e.g.,
                Pinecone, Milvus, pgvector). For our 13-entry demo, an in-memory array works fine.
                At enterprise scale with millions of entries, you need optimized approximate nearest
                neighbor (ANN) indexes — and that means dedicated storage and compute for the vector
                DB.
              </p>
            </div>
          </div>

          {/* Where storage fits */}
          <div className="p-3 rounded bg-yellow-950/20 border border-yellow-800/30">
            <h4 className="text-xs font-semibold text-yellow-400 mb-1">
              Where storage fits in RAG
            </h4>
            <div className="grid grid-cols-3 gap-3 mt-2 text-xs text-slate-400">
              <div>
                <span className="font-semibold text-slate-300">Embedding indexes</span>
                <br />
                ~1.5 KB per entry (768-dim float32). 1M entries ≈ 1.5 GB.
              </div>
              <div>
                <span className="font-semibold text-slate-300">Source documents</span>
                <br />
                The original text. Often larger than the embeddings themselves.
              </div>
              <div>
                <span className="font-semibold text-slate-300">Query latency</span>
                <br />
                Vector search adds 10-50ms per query. ANN indexes trade accuracy for speed.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Section tabs (bottom) */}
      <div className="mt-6">
        <SectionTabs tabs={tabs} active={section} onSelect={setSection} color="yellow" />
      </div>
    </div>
  )
}
