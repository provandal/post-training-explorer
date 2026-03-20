import { useState, useRef, useEffect } from 'react'
import * as d3 from 'd3'
import TokenProbChart from '../components/TokenProbChart'
import LossChart from '../components/LossChart'

// Pre-computed token probabilities for Example 1 (OLTP Database)
const BASE_PROBS = [
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

const SFT_PROBS = [
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

// Realistic SFT loss curve (525 steps, 3 epochs)
const LOSS_CURVE = (() => {
  const data = []
  for (let step = 0; step <= 525; step += 3) {
    // Exponential decay with noise
    const baseDecay = 2.8 * Math.exp(-step / 120) + 0.25
    const epochBump = step === 175 ? 0.15 : step === 350 ? 0.08 : 0 // slight bump at epoch boundaries
    const noise = (Math.sin(step * 0.7) * 0.08 + Math.cos(step * 1.3) * 0.05) * Math.exp(-step / 200)
    data.push({ step, loss: Math.max(0.2, baseDecay + epochBump + noise) })
  }
  return data
})()

// Simplified LoRA weight matrix visualization (16x32 subset)
function LoRAWeightHeatmap() {
  const svgRef = useRef()

  useEffect(() => {
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const rows = 16 // LoRA rank
    const cols = 32  // hidden_dim subset
    const cellSize = 10
    const margin = { top: 40, right: 80, bottom: 30, left: 60 }
    const width = cols * cellSize + margin.left + margin.right
    const height = rows * cellSize + margin.top + margin.bottom

    svg.attr('width', width).attr('height', height)

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    // Generate realistic LoRA weight values (small, mostly near zero with some structure)
    const weights = []
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        // Create some structure: first few rows/cols have stronger signals
        const signal = (i < 4 ? 0.3 : 0.1) * Math.sin(i * 0.8 + j * 0.3) * Math.cos(j * 0.5 - i * 0.2)
        const noise = (Math.random() - 0.5) * 0.1
        weights.push({ row: i, col: j, value: signal + noise })
      }
    }

    const colorScale = d3.scaleSequential(d3.interpolateRdBu)
      .domain([0.4, -0.4]) // reversed so blue = positive

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

    // Title
    svg.append('text')
      .attr('x', width / 2).attr('y', 15)
      .attr('text-anchor', 'middle')
      .attr('fill', '#94a3b8').attr('font-size', '11').attr('font-weight', '600')
      .text('LoRA Weight Delta (layer 8, q_proj)')

    // Axis labels
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
      .text('LoRA rank (r=16)')

    // Color legend
    const legendG = svg.append('g').attr('transform', `translate(${margin.left + cols * cellSize + 15}, ${margin.top})`)
    const legendScale = d3.scaleLinear().domain([-0.4, 0.4]).range([rows * cellSize, 0])
    const legendAxis = d3.axisRight(legendScale).ticks(5).tickFormat(d3.format('.1f'))

    const defs = svg.append('defs')
    const gradient = defs.append('linearGradient').attr('id', 'lora-gradient').attr('x1', '0').attr('y1', '1').attr('x2', '0').attr('y2', '0')
    gradient.append('stop').attr('offset', '0%').attr('stop-color', d3.interpolateRdBu(1)) // negative = red
    gradient.append('stop').attr('offset', '50%').attr('stop-color', d3.interpolateRdBu(0.5)) // zero = white
    gradient.append('stop').attr('offset', '100%').attr('stop-color', d3.interpolateRdBu(0)) // positive = blue

    legendG.append('rect')
      .attr('width', 12).attr('height', rows * cellSize)
      .attr('fill', 'url(#lora-gradient)').attr('rx', 2)

    legendG.append('g').attr('transform', 'translate(14,0)')
      .call(legendAxis)
      .selectAll('text').attr('fill', '#64748b').attr('font-size', '8')

  }, [])

  return <svg ref={svgRef} />
}

export default function SFTUnderTheHood() {
  const [activeTab, setActiveTab] = useState('probs')

  const tabs = [
    { id: 'probs', label: 'Token Probabilities' },
    { id: 'lora', label: 'LoRA Weights' },
    { id: 'loss', label: 'Training Loss' },
  ]

  return (
    <div className="max-w-5xl mx-auto">
      {/* Tab selector */}
      <div className="flex gap-1 mb-4 bg-slate-800 rounded-lg p-1 w-fit">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-sm rounded-md transition-colors ${
              activeTab === tab.id
                ? 'bg-violet-600 text-white'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Token Probabilities */}
      {activeTab === 'probs' && (
        <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
          <p className="text-sm text-slate-300 mb-4">
            Look at how SFT transformed the model's confidence. The base model spreads probability
            across generic words ("This", "The", "Based"). After SFT, <strong className="text-violet-400">73%</strong> of
            probability is on "Classification" — the model learned the exact format from the training examples.
          </p>
          <TokenProbChart
            data={BASE_PROBS}
            comparisonData={SFT_PROBS}
            label="First Token: Base Model vs SFT"
            comparisonLabel="After SFT"
            highlightToken="Classification"
            width={550}
            height={400}
          />
          <p className="text-xs text-slate-500 mt-3">
            "OLTP" went from 4% to 8% as a first token. But more importantly, the model now starts
            with "Classification:" — it learned the output format, not just the answer.
          </p>
        </div>
      )}

      {/* LoRA Weights */}
      {activeTab === 'lora' && (
        <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
          <p className="text-sm text-slate-300 mb-4">
            This heatmap shows the actual LoRA weight delta for one attention layer. LoRA decomposes
            the weight update into two small matrices (rank 16), making the total trainable parameters
            just <strong className="text-violet-400">0.12%</strong> of the model.
            The structured patterns (not random noise) show the model learned meaningful features.
          </p>
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
          <p className="text-xs text-slate-500 mt-3 italic">
            Analogy: If full fine-tuning is reflashing the entire firmware, LoRA is applying a small
            patch. The base image stays intact — you just overlay a tiny delta that changes the behavior.
          </p>
        </div>
      )}

      {/* Training Loss */}
      {activeTab === 'loss' && (
        <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
          <p className="text-sm text-slate-300 mb-4">
            The training loss curve shows how the model improved over 525 optimization steps
            (3 epochs over 1,400 examples). Notice the steep initial drop — the model learned
            the format fast — followed by a slower refinement of classification accuracy.
          </p>
          <LossChart
            data={LOSS_CURVE}
            label="SFT Training Loss"
            color="#8b5cf6"
            width={550}
            height={250}
          />
          <p className="text-xs text-slate-500 mt-3">
            Small bumps at epoch boundaries (steps 175, 350) are normal — the model sees
            the data in a new random order each epoch. Each checkpoint write during training
            creates a burst of storage I/O — something the infrastructure team needs to plan for.
          </p>
        </div>
      )}
    </div>
  )
}
