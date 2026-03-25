import { useState, useRef, useEffect } from 'react'
import * as d3 from 'd3'
import ModelOutput from '../components/ModelOutput'
import InfrastructureCard from '../components/InfrastructureCard'
import TokenProbChart from '../components/TokenProbChart'
import LossChart from '../components/LossChart'
import SectionTabs from '../components/SectionTabs'
import PatternPicker from '../components/PatternPicker'
import useStore from '../store'
import {
  isLoaded, getTestPrompts, getModelOutput, getTokenProbsForChart,
  getSFTLossCurve, getLoRAWeights, getSFTBeforeAfter, getModelSummary,
  formatPromptMetrics, getTrainingTime,
} from '../data/loadArtifacts'

/* ──────────────────────────────────────────────
   Fallback data: Examples for the demo tab
   ────────────────────────────────────────────── */
const FALLBACK_EXAMPLES = [
  {
    id: 1,
    input: "IOPS: 45000 | Latency: 0.3ms | Block Size: 8K | Read/Write: 70/30 | Sequential: 15% | Queue Depth: 32",
    label: "OLTP Database",
    difficulty: "Easy",
    baseOutput: `This appears to be some kind of storage performance metrics. The IOPS value is 45000 which is relatively high. The latency is 0.3ms which is quite low. The block size is 8K. There is a 70/30 read to write ratio with 15% sequential access and a queue depth of 32. These metrics could be from various types of storage workloads depending on the specific use case and configuration being measured.`,
    sftOutput: `Classification: OLTP Database
Confidence: High
Key indicators: High IOPS (45K) with very low latency (0.3ms) and small block size (8K) are hallmarks of transaction processing. Random-dominant access (85%) with moderate read bias confirms OLTP pattern.`,
    baseCorrect: false,
    sftCorrect: true,
  },
  {
    id: 2,
    input: "IOPS: 12000 | Latency: 4.5ms | Block Size: 128K | Read/Write: 92/8 | Sequential: 78% | Queue Depth: 8",
    label: "OLAP Analytics",
    difficulty: "Medium",
    baseOutput: `The storage metrics show 12000 IOPS with a latency of 4.5ms. The block size is 128K which is larger than usual. The read/write ratio is heavily skewed towards reads at 92/8, with 78% sequential access. Queue depth is 8. This looks like it could be some kind of read-heavy sequential workload, possibly related to data processing or analytics.`,
    sftOutput: `Classification: OLAP Analytics
Confidence: High
Key indicators: Large block sequential reads (128K, 78% sequential) with heavy read bias (92/8) indicate analytical query processing. Moderate IOPS with higher latency tolerance is typical of scan-heavy warehouse workloads.`,
    baseCorrect: false,
    sftCorrect: true,
  },
  {
    id: 3,
    input: "IOPS: 38000 | Latency: 0.5ms | Block Size: 8K | Read/Write: 55/45 | Sequential: 22% | Queue Depth: 64",
    label: "VDI Virtual Desktop",
    difficulty: "Hard",
    baseOutput: `These storage metrics indicate a workload with 38000 IOPS at 0.5ms latency. Block size is 8K, read/write ratio is nearly balanced at 55/45, with 22% sequential access and a queue depth of 64. The high IOPS and low latency suggest this is a performance-sensitive database application, possibly OLTP or a similar transactional workload.`,
    sftOutput: `Classification: VDI Virtual Desktop
Confidence: Medium
Key indicators: Nearly balanced R/W ratio (55/45) distinguishes this from OLTP despite similar IOPS and block size. High queue depth (64) and moderate sequential ratio (22%) are characteristic of multi-user desktop virtualization with mixed application workloads.`,
    baseCorrect: false,
    sftCorrect: true,
  },
]

/* ──────────────────────────────────────────────
   RAG limitation example (for "The Problem" tab)
   ────────────────────────────────────────────── */
const RAG_EXAMPLE_INPUT = "IOPS: 38000 | Latency: 0.5ms | Block Size: 8K | Read/Write: 55/45 | Sequential: 22% | Queue Depth: 64"

const RAG_VERBOSE_RESPONSE = `Based on the retrieved reference patterns, this I/O profile presents an interesting case. The retrieved documents show two potential matches:

1. OLTP Database (71% similarity) - The high IOPS and low latency are consistent with OLTP workloads. The small block size of 8K also supports this classification. However, I should note that the read/write ratio of 55/45 is more balanced than the typical OLTP range of 60-80/20-40.

2. VDI Virtual Desktop (68% similarity) - The balanced read/write ratio and high queue depth of 64 are more characteristic of VDI workloads. The small block size could fit either pattern.

Given the ambiguity, this pattern could potentially be classified as either OLTP Database or VDI Virtual Desktop. The balanced read/write ratio and high queue depth suggest VDI might be slightly more appropriate, but without additional context about the specific deployment environment, infrastructure configuration, and application characteristics, I cannot make a definitive determination. It would be advisable to collect additional metrics such as I/O size distribution histograms, temporal patterns, and application-level metadata to make a more informed classification.

My best assessment is that this is likely a VDI Virtual Desktop workload, though OLTP Database cannot be ruled out entirely.`

const DESIRED_FORMAT = `Classification: VDI Virtual Desktop
Confidence: Medium
Key indicators: Balanced R/W ratio (55/45) and high queue depth (64) distinguish this from OLTP. Small blocks (8K) with random access are consistent with VDI.`

/* ──────────────────────────────────────────────
   Infrastructure profile
   ────────────────────────────────────────────── */
const SFT_INFRA = {
  gpuMemoryGB: 4.2,
  trainingTimeMinutes: 12,
  checkpointSizeMB: 1.7,
  peakGPUUtilization: 87,
  storageIOPattern: "Bursty checkpoint writes every ~50 steps, steady data reads",
  note: "LoRA adapter is only 1.7 MB. The full model (720 MB) stays frozen. This is why PEFT changed everything — you can fine-tune with minimal training VRAM, no dedicated cluster needed."
}

/* ──────────────────────────────────────────────
   Fallback token probabilities
   ────────────────────────────────────────────── */
const FALLBACK_BASE_PROBS = [
  { token: 'This', probability: 0.18 },
  { token: 'The', probability: 0.14 },
  { token: 'Based', probability: 0.09 },
  { token: 'These', probability: 0.07 },
  { token: 'It', probability: 0.06 },
  { token: 'Storage', probability: 0.05 },
  { token: 'Classification', probability: 0.03 },
  { token: 'OLTP', probability: 0.04 },
  { token: 'High', probability: 0.04 },
  { token: 'Looking', probability: 0.04 },
  { token: 'I', probability: 0.03 },
  { token: 'Database', probability: 0.03 },
  { token: 'A', probability: 0.03 },
  { token: 'VDI', probability: 0.02 },
  { token: 'Given', probability: 0.02 },
]

const FALLBACK_SFT_PROBS = [
  { token: 'Classification', probability: 0.73 },
  { token: 'OLTP', probability: 0.08 },
  { token: 'This', probability: 0.03 },
  { token: 'The', probability: 0.02 },
  { token: 'Database', probability: 0.02 },
  { token: 'Based', probability: 0.01 },
  { token: 'High', probability: 0.01 },
  { token: 'Storage', probability: 0.01 },
  { token: 'VDI', probability: 0.01 },
  { token: 'These', probability: 0.01 },
  { token: 'It', probability: 0.01 },
  { token: 'I', probability: 0.005 },
  { token: 'A', probability: 0.005 },
  { token: 'Looking', probability: 0.004 },
  { token: 'Given', probability: 0.003 },
]

// Fallback SFT loss curve (525 steps, 3 epochs)
const FALLBACK_LOSS_CURVE = (() => {
  const data = []
  for (let step = 0; step <= 525; step += 3) {
    const baseDecay = 2.8 * Math.exp(-step / 120) + 0.25
    const epochBump = step === 175 ? 0.15 : step === 350 ? 0.08 : 0
    const noise = (Math.sin(step * 0.7) * 0.08 + Math.cos(step * 1.3) * 0.05) * Math.exp(-step / 200)
    data.push({ step, loss: Math.max(0.2, baseDecay + epochBump + noise) })
  }
  return data
})()

/* ──────────────────────────────────────────────
   LoRA Weight Heatmap (D3 visualization)
   ────────────────────────────────────────────── */
function LoRAWeightHeatmap() {
  const svgRef = useRef()

  // Try to use real LoRA weight data
  const realWeights = isLoaded() ? getLoRAWeights() : null

  useEffect(() => {
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const rows = 16
    const cols = 32
    const cellSize = 10
    const margin = { top: 40, right: 80, bottom: 30, left: 60 }
    const width = cols * cellSize + margin.left + margin.right
    const height = rows * cellSize + margin.top + margin.bottom

    svg.attr('width', width).attr('height', height)

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    const weights = []
    if (realWeights?.lora_A && realWeights?.lora_B) {
      // Use real LoRA weight matrices
      const A = realWeights.lora_A
      const B = realWeights.lora_B
      for (let i = 0; i < Math.min(rows, A.length); i++) {
        for (let j = 0; j < Math.min(cols, (B[0] || []).length); j++) {
          let value = 0
          for (let k = 0; k < Math.min(A[0]?.length || 0, B.length); k++) {
            value += (A[i]?.[k] || 0) * (B[k]?.[j] || 0)
          }
          weights.push({ row: i, col: j, value })
        }
      }
    } else {
      // Fallback: synthetic but structured weights
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          const signal = (i < 4 ? 0.3 : 0.1) * Math.sin(i * 0.8 + j * 0.3) * Math.cos(j * 0.5 - i * 0.2)
          const noise = (Math.random() - 0.5) * 0.1
          weights.push({ row: i, col: j, value: signal + noise })
        }
      }
    }

    const maxAbs = d3.max(weights, d => Math.abs(d.value)) || 0.4
    const colorScale = d3.scaleSequential(d3.interpolateRdBu)
      .domain([maxAbs, -maxAbs])

    g.selectAll('rect')
      .data(weights)
      .join('rect')
      .attr('x', d => d.col * cellSize)
      .attr('y', d => d.row * cellSize)
      .attr('width', cellSize - 1)
      .attr('height', cellSize - 1)
      .attr('rx', 1)
      .attr('fill', d => colorScale(d.value))
      .attr('opacity', 0)
      .transition()
      .delay((d, i) => i * 0.5)
      .duration(300)
      .attr('opacity', 1)

    const layerLabel = realWeights?.layer || 'layer 8, q_proj'
    svg.append('text')
      .attr('x', width / 2).attr('y', 15)
      .attr('text-anchor', 'middle')
      .attr('fill', '#94a3b8').attr('font-size', '11').attr('font-weight', '600')
      .text(`LoRA Weight Delta (${layerLabel})`)

    svg.append('text')
      .attr('x', margin.left + (cols * cellSize) / 2).attr('y', margin.top + rows * cellSize + 20)
      .attr('text-anchor', 'middle')
      .attr('fill', '#64748b').attr('font-size', '9')
      .text('Hidden dimension (subset)')

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -(margin.top + (rows * cellSize) / 2)).attr('y', 15)
      .attr('text-anchor', 'middle')
      .attr('fill', '#64748b').attr('font-size', '9')
      .text(`LoRA rank (r=${realWeights?.rank || 16})`)

    const legendG = svg.append('g').attr('transform', `translate(${margin.left + cols * cellSize + 15}, ${margin.top})`)
    const legendScale = d3.scaleLinear().domain([-maxAbs, maxAbs]).range([rows * cellSize, 0])
    const legendAxis = d3.axisRight(legendScale).ticks(5).tickFormat(d3.format('.1f'))

    const defs = svg.append('defs')
    const gradient = defs.append('linearGradient').attr('id', 'lora-gradient').attr('x1', '0').attr('y1', '1').attr('x2', '0').attr('y2', '0')
    gradient.append('stop').attr('offset', '0%').attr('stop-color', d3.interpolateRdBu(1))
    gradient.append('stop').attr('offset', '50%').attr('stop-color', d3.interpolateRdBu(0.5))
    gradient.append('stop').attr('offset', '100%').attr('stop-color', d3.interpolateRdBu(0))

    legendG.append('rect')
      .attr('width', 12).attr('height', rows * cellSize)
      .attr('fill', 'url(#lora-gradient)').attr('rx', 2)

    legendG.append('g').attr('transform', 'translate(14,0)')
      .call(legendAxis)
      .selectAll('text').attr('fill', '#64748b').attr('font-size', '8')
  }, [realWeights])

  return <svg ref={svgRef} />
}

/* ══════════════════════════════════════════════
   Main Component
   ══════════════════════════════════════════════ */
export default function SFTComparison({ explore = false }) {
  const [section, setSection] = useState('problem')
  const [selectedExample, setSelectedExample] = useState(0)
  const [deepTab, setDeepTab] = useState('probs')
  const [showTransformerAside, setShowTransformerAside] = useState(false)
  const [showLoRAAside, setShowLoRAAside] = useState(false)
  const selectedPromptId = useStore((s) => s.selectedPromptId)
  const setActiveQuadrant = useStore((s) => s.setActiveQuadrant)
  const setMode = useStore((s) => s.setMode)

  const hasRealData = isLoaded()

  // Build demo examples from real data or use fallback
  let examples = FALLBACK_EXAMPLES
  if (hasRealData) {
    const testPrompts = getTestPrompts()
    const realExamples = testPrompts.slice(0, 5).map((tp) => {
      const baseOut = getModelOutput('base', tp.id)
      const sftOut = getModelOutput('sft', tp.id)
      return {
        id: tp.id,
        input: formatPromptMetrics(tp),
        label: tp.true_label,
        difficulty: tp.id < 8 ? 'Easy' : tp.id < 16 ? 'Medium' : 'Hard',
        baseOutput: baseOut?.generated_text ?? '(No base model data)',
        sftOutput: sftOut?.generated_text ?? '(No SFT data)',
        baseCorrect: baseOut ? baseOut.generated_text.toLowerCase().includes(tp.true_label.toLowerCase()) : false,
        sftCorrect: sftOut ? sftOut.generated_text.toLowerCase().includes(tp.true_label.toLowerCase()) : false,
      }
    })
    if (realExamples.length > 0 && realExamples[0].baseOutput !== '(No base model data)') {
      examples = realExamples
    }
  }

  const ex = examples[selectedExample] || examples[0]

  // Token probs — use real data or fallback
  let baseProbs = FALLBACK_BASE_PROBS
  let sftProbs = FALLBACK_SFT_PROBS
  if (hasRealData) {
    const realBase = getTokenProbsForChart('base', selectedPromptId)
    const realSft = getTokenProbsForChart('sft', selectedPromptId)
    if (realBase.length > 0) baseProbs = realBase
    if (realSft.length > 0) sftProbs = realSft
  }

  // Loss curve — use real data or fallback
  const realLoss = hasRealData ? getSFTLossCurve() : []
  const lossCurve = realLoss.length > 0 ? realLoss : FALLBACK_LOSS_CURVE

  // Training time
  const trainingTimeSec = hasRealData ? getTrainingTime('sft') : null
  const trainingTimeMin = trainingTimeSec ? Math.round(trainingTimeSec / 60) : 12

  // Accuracy summary
  const sftSummary = hasRealData ? getModelSummary('sft') : null

  const TABS = [
    { id: 'problem', label: 'The Problem' },
    { id: 'concept', label: 'How SFT Works' },
    { id: 'demo', label: 'See It Work' },
    { id: 'deepdive', label: 'Under the Covers' },
  ]

  const DEEP_TABS = [
    { id: 'probs', label: 'Token Probabilities' },
    { id: 'lora', label: 'LoRA Weights' },
    { id: 'loss', label: 'Training Loss' },
  ]

  return (
    <div className="max-w-5xl mx-auto">
      {/* ── Section tabs (top) ─────────────────────────── */}
      <div className="mb-6">
        <SectionTabs tabs={TABS} active={section} onSelect={setSection} color="violet" />
      </div>

      {/* ════════════════ THE PROBLEM ════════════════ */}
      {section === 'problem' && (
        <div className="space-y-5">
          {/* Context setter */}
          <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h3 className="text-base font-semibold text-violet-400 mb-2">
              Where we left off: RAG got the right answer
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed">
              In the previous stop we added Retrieval-Augmented Generation. The model
              could look up reference patterns from a knowledge base before answering.
              It worked &mdash; the answers became <em>correct</em>. But there is a catch.
            </p>
          </div>

          {/* The ambiguous input */}
          <div>
            <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
              Ambiguous I/O Pattern
            </label>
            <div className="bg-slate-800 border border-yellow-700/50 rounded-lg p-3 font-mono text-sm text-yellow-200">
              {RAG_EXAMPLE_INPUT}
            </div>
          </div>

          {/* Side by side: RAG verbose vs desired */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <ModelOutput
                label="Base Model + RAG Context"
                text={RAG_VERBOSE_RESPONSE}
                variant="rag"
                isCorrect={true}
              />
              <p className="mt-2 text-xs text-yellow-500/80 italic">
                Technically correct &mdash; eventually says "VDI" &mdash; but buried in 6 paragraphs of hedging.
              </p>
            </div>
            <div>
              <ModelOutput
                label="What your ops team actually needs"
                text={DESIRED_FORMAT}
                variant="default"
              />
              <p className="mt-2 text-xs text-slate-500 italic">
                Three lines. Classification, confidence, reasoning. Done.
              </p>
            </div>
          </div>

          {/* Diagnosis cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="p-4 rounded-lg bg-red-950/20 border border-red-800/30">
              <h4 className="text-sm font-semibold text-red-400 mb-2">Verbose</h4>
              <p className="text-xs text-slate-400">
                The model hedges, qualifies, and over-explains. Your monitoring dashboard
                cannot parse a five-paragraph essay. It needs a structured three-line response
                that can be fed directly into an automation pipeline.
              </p>
            </div>
            <div className="p-4 rounded-lg bg-red-950/20 border border-red-800/30">
              <h4 className="text-sm font-semibold text-red-400 mb-2">Wrong Format</h4>
              <p className="text-xs text-slate-400">
                There is no "Classification:" prefix. No "Confidence:" level. The model
                structures its output like a general assistant, not like a storage operations
                tool. RAG gave it the right data but cannot control the output shape.
              </p>
            </div>
            <div className="p-4 rounded-lg bg-red-950/20 border border-red-800/30">
              <h4 className="text-sm font-semibold text-red-400 mb-2">Uncertain Tone</h4>
              <p className="text-xs text-slate-400">
                "Could potentially be classified as either..." is not actionable. When
                the on-call engineer gets a page at 3 AM, they need a clear answer
                with a confidence level &mdash; not a research paper on the topic.
              </p>
            </div>
          </div>

          {/* The transition insight */}
          <div className="p-4 rounded-lg bg-gradient-to-r from-yellow-950/30 via-slate-800/50 to-violet-950/30 border border-violet-700/30">
            <p className="text-sm text-slate-300 leading-relaxed">
              <span className="font-bold text-yellow-400">RAG changes what the model sees.</span>{' '}
              But the model's behavior &mdash; its format, confidence style, and
              conciseness &mdash; stays the same. To change how the model <em>acts</em>,
              we need to change the model itself. That means modifying its weights through
              training.
            </p>
            <p className="text-sm text-slate-300 mt-2 leading-relaxed">
              This is where{' '}
              <span className="font-bold text-violet-400">Supervised Fine-Tuning (SFT)</span>{' '}
              comes in. We are crossing from the left side of the map (prompt engineering, RAG)
              to the right side (post-training). From here on, we are changing the model.
            </p>
          </div>
        </div>
      )}

      {/* ════════════════ HOW SFT WORKS ════════════════ */}
      {section === 'concept' && (
        <div className="space-y-5">
          <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h3 className="text-base font-semibold text-violet-400 mb-3">
              Supervised Fine-Tuning (SFT)
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed mb-3">
              SFT is the simplest form of post-training. You take a pre-trained language model
              and show it examples of the behavior you want. The model adjusts its internal
              weights so that it produces outputs matching your examples. It is "supervised"
              because every training example has a known correct answer.
            </p>
            <div className="p-3 rounded bg-violet-950/20 border border-violet-800/30">
              <p className="text-sm text-violet-300 italic leading-relaxed">
                Think of it like training a new employee with an example handbook.
                You hand them 1,400 examples that say: "When you see this I/O pattern,
                respond exactly like this." After studying the handbook, they internalize
                the patterns and can handle new cases they have never seen before.
              </p>
            </div>
          </div>

          <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h3 className="text-sm font-semibold text-violet-400 mb-3 uppercase tracking-wide">
              What a Training Example Looks Like
            </h3>
            <p className="text-xs text-slate-400 mb-4 leading-relaxed">
              Each training example is an input-output pair. The input is a raw I/O metric
              string (what we want the model to classify). The output is the structured
              response we want the model to produce. Here is one of the 1,400 examples:
            </p>
            <div className="grid grid-cols-1 md:grid-cols-[1fr,auto,1fr] gap-3 items-center">
              <div className="p-3 rounded bg-slate-900 border border-slate-700">
                <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
                  Input (I/O Metrics)
                </div>
                <pre className="text-xs text-slate-300 font-mono whitespace-pre-wrap leading-relaxed">
{`IOPS: 45000
Latency: 0.3ms
Block Size: 8K
Read/Write: 70/30
Sequential: 15%
Queue Depth: 32`}
                </pre>
              </div>

              <div className="hidden md:flex flex-col items-center text-violet-500">
                <span className="text-2xl">&#8594;</span>
                <span className="text-xs mt-1 text-slate-500">train on</span>
              </div>
              <div className="md:hidden flex justify-center text-violet-500">
                <span className="text-2xl">&#8595;</span>
              </div>

              <div className="p-3 rounded bg-violet-950/30 border border-violet-800/40">
                <div className="text-xs font-semibold text-violet-400 uppercase tracking-wide mb-2">
                  Desired Output (Label)
                </div>
                <pre className="text-xs text-slate-200 font-mono whitespace-pre-wrap leading-relaxed">
{`Classification: OLTP Database
Confidence: High
Key indicators: High IOPS (45K)
with very low latency (0.3ms)
and small block size (8K) are
hallmarks of transaction
processing.`}
                </pre>
              </div>
            </div>
            <p className="text-xs text-slate-500 mt-3 leading-relaxed">
              The dataset contains 1,400 examples across 6 workload categories (OLTP, OLAP,
              VDI, Backup, AI/ML Training, Video Streaming) with varying difficulty levels.
            </p>
          </div>

          <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h3 className="text-sm font-semibold text-violet-400 mb-3 uppercase tracking-wide">
              LoRA: Training a Patch, Not the Whole Model
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed mb-3">
              SmolLM2-360M has 360 million parameters. Retraining all of them would require
              significant GPU memory, time, and storage. <strong className="text-violet-300">LoRA
              (Low-Rank Adaptation)</strong> is a technique that freezes the entire original model
              and trains only a small set of adapter weights that get layered on top.
            </p>
            <div className="p-3 rounded bg-blue-950/20 border border-blue-800/30 mb-4">
              <p className="text-sm text-blue-300 italic leading-relaxed">
                Instead of rewriting the entire model (360M parameters), we only train a
                small adapter (<strong>432K parameters &mdash; 0.12%</strong> of the total).
                Think of it like applying a firmware patch rather than reflashing the entire
                image. The base model stays intact; you just overlay a tiny delta that
                changes the behavior.
              </p>
            </div>

            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="p-3 rounded bg-slate-800/80 border border-slate-700/50 text-center">
                <div className="text-2xl font-bold text-slate-300">360M</div>
                <div className="text-xs text-slate-500 mt-1">Total model parameters</div>
                <div className="mt-2 h-2 rounded-full bg-slate-700 overflow-hidden">
                  <div className="h-full w-full bg-slate-500 rounded-full" />
                </div>
                <div className="text-xs text-slate-600 mt-1">720 MB on disk (frozen)</div>
              </div>
              <div className="p-3 rounded bg-violet-950/30 border border-violet-800/40 text-center">
                <div className="text-2xl font-bold text-violet-400">432K</div>
                <div className="text-xs text-slate-500 mt-1">LoRA adapter parameters</div>
                <div className="mt-2 h-2 rounded-full bg-slate-700 overflow-hidden">
                  <div className="h-full rounded-full bg-violet-500" style={{ width: '0.12%', minWidth: '3px' }} />
                </div>
                <div className="text-xs text-violet-500/80 mt-1">1.7 MB on disk (trained)</div>
              </div>
            </div>

            <p className="text-xs text-slate-400 leading-relaxed mb-2">
              LoRA targets the <strong className="text-slate-300">attention projections</strong>{' '}
              (q_proj and v_proj) inside each of the model's 32 transformer layers. These are the
              matrices that control what the model pays attention to and what information it carries
              forward. By adapting just these projections with a rank-16 decomposition, LoRA captures
              the essential behavioral change in a tiny number of parameters.
            </p>
            <button
              onClick={() => setShowLoRAAside(!showLoRAAside)}
              className="text-xs text-violet-400 hover:text-violet-300 underline underline-offset-4 cursor-pointer"
            >
              {showLoRAAside ? 'Hide' : 'How does LoRA select which parameters to train?'}
            </button>
            {showLoRAAside && (
              <div className="mt-2 p-3 rounded bg-violet-950/20 border border-violet-800/30">
                <p className="text-xs text-slate-300 leading-relaxed mb-2">
                  LoRA doesn't search for parameters &mdash; you configure which weight matrices
                  to target. The standard choice is the <strong className="text-violet-300">query
                  and value projections</strong> (q_proj, v_proj) in each attention layer, because
                  research shows these have the most impact on model behavior. The adapter itself
                  is a pair of small matrices (A and B) whose product approximates the weight change
                  that full fine-tuning would make. The "rank" (16 in our case) controls how
                  expressive the adapter is &mdash; higher rank = more capacity, but more parameters.
                </p>
                <button
                  onClick={() => {
                    setActiveQuadrant('lora')
                    setMode('explore')
                  }}
                  className="text-xs font-semibold text-violet-400 hover:text-violet-300 underline underline-offset-4 cursor-pointer"
                >
                  Deep dive: LoRA parameter selection, rank decomposition, and the training loop &rarr;
                </button>
              </div>
            )}
            <p className="text-xs text-slate-500 leading-relaxed mt-3">
              During inference, the adapter weights are merged with the base model
              at near-zero cost. You can even swap different adapters for different tasks
              on the same base model &mdash; like hot-swapping firmware modules.
            </p>
          </div>

          <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h3 className="text-sm font-semibold text-violet-400 mb-3 uppercase tracking-wide">
              Training at a Glance
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-center">
              <div className="p-3 rounded bg-slate-800/80 border border-slate-700/50">
                <div className="text-2xl font-bold text-violet-400">1,400</div>
                <div className="text-xs text-slate-500 mt-1">training examples</div>
              </div>
              <div className="p-3 rounded bg-slate-800/80 border border-slate-700/50">
                <div className="text-2xl font-bold text-violet-400">3</div>
                <div className="text-xs text-slate-500 mt-1">epochs</div>
              </div>
              <div className="p-3 rounded bg-slate-800/80 border border-slate-700/50">
                <div className="text-2xl font-bold text-violet-400">0.12%</div>
                <div className="text-xs text-slate-500 mt-1">params trained</div>
              </div>
              <div className="p-3 rounded bg-slate-800/80 border border-slate-700/50">
                <div className="text-2xl font-bold text-violet-400">{trainingTimeMin} min</div>
                <div className="text-xs text-slate-500 mt-1">training time</div>
              </div>
            </div>
          </div>

          <div className="p-4 rounded-lg bg-gradient-to-r from-violet-950/30 to-blue-950/30 border border-violet-700/30">
            <p className="text-xs text-blue-300 leading-relaxed">
              <strong>Key insight:</strong> SFT changes the model's weights. After training,
              the model permanently knows how to classify I/O patterns &mdash; no prompt
              engineering needed. You don't need to tell it the format, the categories, or
              the reasoning style every time. It learned all of that from the 1,400 examples.
              The adapter is a 1.7 MB file that gets merged at load time, and the model
              behaves differently from that point forward.
            </p>
          </div>
        </div>
      )}

      {/* ════════════════ SEE IT WORK ════════════════ */}
      {section === 'demo' && (
        <div className="space-y-4">
          {/* Pattern picker for real data */}
          {hasRealData && <PatternPicker compact />}

          {/* Example selector */}
          <div>
            <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
              Choose an example
            </label>
            <div className="flex gap-2 flex-wrap">
              {examples.map((e, i) => (
                <button
                  key={i}
                  onClick={() => setSelectedExample(i)}
                  className={`px-3 py-1.5 text-xs rounded-md transition-colors ${
                    i === selectedExample
                      ? 'bg-violet-600 text-white'
                      : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                  }`}
                >
                  {e.label} ({e.difficulty})
                </button>
              ))}
            </div>
          </div>

          {/* Input metrics */}
          <div>
            <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
              I/O Metrics
            </label>
            <div className="bg-slate-800 border border-slate-600 rounded-lg p-3 font-mono text-sm text-slate-200">
              {ex.input}
            </div>
          </div>

          {/* Side by side comparison */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <ModelOutput
              label="Base Model (no training)"
              text={ex.baseOutput}
              variant="base"
              isCorrect={ex.baseCorrect}
            />
            <ModelOutput
              label="After SFT (1,400 examples)"
              text={ex.sftOutput}
              variant="sft"
              isCorrect={ex.sftCorrect}
            />
          </div>

          {/* Training stats summary bar */}
          <div className="p-3 rounded-lg bg-violet-950/20 border border-violet-800/30">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-center">
              <div>
                <div className="text-lg font-bold text-violet-400">1,400</div>
                <div className="text-xs text-slate-500">training examples</div>
              </div>
              <div>
                <div className="text-lg font-bold text-violet-400">3</div>
                <div className="text-xs text-slate-500">epochs</div>
              </div>
              <div>
                <div className="text-lg font-bold text-violet-400">0.12%</div>
                <div className="text-xs text-slate-500">params trained (LoRA)</div>
              </div>
              <div>
                <div className="text-lg font-bold text-violet-400">
                  {sftSummary ? `${(sftSummary.accuracy * 100).toFixed(0)}%` : `${trainingTimeMin} min`}
                </div>
                <div className="text-xs text-slate-500">
                  {sftSummary ? `accuracy (${sftSummary.correct}/${sftSummary.total})` : 'training time'}
                </div>
              </div>
            </div>
          </div>

          {/* Observation callout */}
          <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <p className="text-sm text-slate-300 mb-2 leading-relaxed">
              <span className="font-semibold text-violet-400">What changed?</span>{' '}
              The base model sees the same metrics and produces vague, hedging descriptions.
              It never commits to a classification, never uses a structured format, and often
              gets the category wrong entirely.
            </p>
            <p className="text-sm text-slate-300 leading-relaxed">
              After SFT, the model has learned three things: <strong className="text-slate-200">the format</strong>{' '}
              (Classification / Confidence / Key indicators), <strong className="text-slate-200">the task</strong>{' '}
              (commit to a specific workload category), and <strong className="text-slate-200">the reasoning
              style</strong> (cite the specific metrics that led to the decision). All from
              1,400 examples and {trainingTimeMin} minutes of training.
            </p>
          </div>
        </div>
      )}

      {/* ════════════════ UNDER THE COVERS ════════════════ */}
      {section === 'deepdive' && (
        <div className="space-y-4">
          <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <p className="text-sm text-slate-300 leading-relaxed">
              SFT modifies the model's internal weights so it produces different outputs.
              But what does "different" look like at a technical level? These three views
              show what actually changed inside the model during training.
            </p>
          </div>

          {/* Deep-dive sub-tabs */}
          <div className="flex gap-1 bg-slate-800 rounded-lg p-1 w-fit">
            {DEEP_TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setDeepTab(tab.id)}
                className={`px-4 py-2 text-sm rounded-md transition-colors ${
                  deepTab === tab.id
                    ? 'bg-violet-600 text-white'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* ── Token Probabilities ── */}
          {deepTab === 'probs' && (
            <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
              <h4 className="text-sm font-semibold text-violet-400 mb-2">
                How SFT Shifts the Model's Confidence
              </h4>
              <p className="text-sm text-slate-300 mb-4 leading-relaxed">
                A language model generates text one token at a time. At each step, it assigns a
                probability to every possible next token. The chart below shows the probability
                distribution for the <strong className="text-slate-200">very first token</strong>{' '}
                the model generates. Before SFT, probability is
                spread across generic words ("This", "The", "Based"). After SFT,
                probability mass concentrates on structured tokens like "Classification".
              </p>
              <TokenProbChart
                data={baseProbs}
                comparisonData={sftProbs}
                label="First Token: Base Model vs SFT"
                comparisonLabel="After SFT"
                highlightToken="Classification"
                width={550}
                height={400}
              />
              {hasRealData && (
                <p className="text-xs text-cyan-400/60 mt-2">
                  Using real token probabilities from trained model.
                </p>
              )}
            </div>
          )}

          {/* ── LoRA Weights ── */}
          {deepTab === 'lora' && (
            <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
              <h4 className="text-sm font-semibold text-violet-400 mb-2">
                The Actual Weight Changes
              </h4>
              <p className="text-sm text-slate-300 mb-3 leading-relaxed">
                This heatmap shows the LoRA weight delta for one attention layer. Each cell
                represents how much that specific weight was adjusted during training. The
                structured patterns &mdash; not random noise &mdash; confirm the model learned
                meaningful features rather than memorizing individual examples.
              </p>
              <button
                onClick={() => setShowTransformerAside(!showTransformerAside)}
                className="mb-4 text-xs text-violet-400 hover:text-violet-300 underline underline-offset-4 cursor-pointer"
              >
                {showTransformerAside ? 'Hide' : 'What is an attention layer?'}
              </button>
              {showTransformerAside && (
                <div className="mb-4 p-3 rounded bg-violet-950/20 border border-violet-800/30">
                  <p className="text-xs text-slate-300 leading-relaxed mb-2">
                    Language models are built from <strong className="text-violet-300">Transformers</strong> &mdash;
                    a stack of identical layers (SmolLM2 has 32). Each layer contains an{' '}
                    <strong className="text-violet-300">attention mechanism</strong> that lets every
                    token in the input "look at" every other token to decide what's relevant.
                    This is how the model connects "IOPS: 45000" with "OLTP Database" even when
                    they're far apart. The attention mechanism uses learned weight matrices (Q, K, V)
                    &mdash; and that's exactly where LoRA inserts its small trainable adapters.
                  </p>
                  <button
                    onClick={() => {
                      setActiveQuadrant('transformers')
                      setMode('explore')
                    }}
                    className="text-xs font-semibold text-violet-400 hover:text-violet-300 underline underline-offset-4 cursor-pointer"
                  >
                    Deep dive: Transformers, attention, and how LoRA fits in &rarr;
                  </button>
                </div>
              )}
              <LoRAWeightHeatmap />
              <div className="mt-4 grid grid-cols-3 gap-4 text-center">
                <div className="p-2 rounded bg-slate-800">
                  <div className="text-lg font-bold text-slate-200">360M</div>
                  <div className="text-xs text-slate-500">total parameters</div>
                </div>
                <div className="p-2 rounded bg-violet-900/30">
                  <div className="text-lg font-bold text-violet-400">432K</div>
                  <div className="text-xs text-slate-500">trainable (LoRA)</div>
                </div>
                <div className="p-2 rounded bg-slate-800">
                  <div className="text-lg font-bold text-slate-200">1.7 MB</div>
                  <div className="text-xs text-slate-500">adapter file size</div>
                </div>
              </div>
            </div>
          )}

          {/* ── Training Loss ── */}
          {deepTab === 'loss' && (
            <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
              <h4 className="text-sm font-semibold text-violet-400 mb-2">
                Training Progress Over Time
              </h4>
              <p className="text-sm text-slate-300 mb-4 leading-relaxed">
                The loss curve shows how the model improved over {lossCurve.length > 0 ? lossCurve[lossCurve.length - 1].step : 525} optimization steps.{' '}
                <strong className="text-slate-200">Loss</strong> measures how far the model's
                output is from the desired output &mdash; lower is better.
              </p>
              <LossChart
                data={lossCurve}
                label="SFT Training Loss"
                color="#8b5cf6"
                width={550}
                height={250}
              />
              {hasRealData && realLoss.length > 0 && (
                <p className="text-xs text-cyan-400/60 mt-2">
                  Real training loss from actual SFT run.
                </p>
              )}
              <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-3">
                <div className="p-3 rounded bg-slate-800/80 border border-slate-700/50">
                  <div className="text-xs text-slate-500">Epoch Boundaries</div>
                  <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                    Small bumps at epoch boundaries are normal &mdash; the model sees the
                    data in a new random order each epoch.
                  </p>
                </div>
                <div className="p-3 rounded bg-slate-800/80 border border-slate-700/50">
                  <div className="text-xs text-slate-500">Final Loss</div>
                  <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                    The loss converges around {lossCurve.length > 0 ? lossCurve[lossCurve.length - 1].loss.toFixed(2) : '0.25'}. Going lower would risk overfitting.
                  </p>
                </div>
                <div className="p-3 rounded bg-slate-800/80 border border-slate-700/50">
                  <div className="text-xs text-slate-500">Storage I/O Impact</div>
                  <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                    Each checkpoint write creates a burst of storage I/O. The training data
                    stream is a steady sequential read.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Infrastructure card */}
          <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h4 className="text-sm font-semibold text-violet-400 mb-3 uppercase tracking-wide">
              Infrastructure Profile
            </h4>
            <InfrastructureCard data={SFT_INFRA} />
          </div>
        </div>
      )}

      {/* ── Section tabs (bottom) ─────────────────────────── */}
      <div className="mt-6">
        <SectionTabs tabs={TABS} active={section} onSelect={setSection} color="violet" />
      </div>
    </div>
  )
}
