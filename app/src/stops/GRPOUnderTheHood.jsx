import { useState, useRef, useEffect } from 'react'
import * as d3 from 'd3'
import LossChart from '../components/LossChart'
import InfrastructureCard from '../components/InfrastructureCard'

// GRPO accuracy improvement over training
const ACCURACY_CURVE = (() => {
  const data = []
  for (let step = 0; step <= 200; step += 2) {
    // Sigmoid-like improvement with noise
    const base = 0.82 / (1 + Math.exp(-0.04 * (step - 60))) + 0.42
    const noise = (Math.sin(step * 0.9) * 0.02 + Math.cos(step * 1.7) * 0.015)
    data.push({ step, loss: Math.min(0.88, Math.max(0.35, base + noise)) })
  }
  return data
})()

// GRPO reward curve
const REWARD_CURVE = (() => {
  const data = []
  for (let step = 0; step <= 200; step += 2) {
    const base = 0.7 / (1 + Math.exp(-0.035 * (step - 70))) + 0.3
    const noise = (Math.sin(step * 0.6) * 0.03 + Math.cos(step * 1.1) * 0.02)
    data.push({ step, loss: Math.min(0.92, Math.max(0.25, base + noise)) })
  }
  return data
})()

// Advantage waterfall visualization
function AdvantageWaterfall() {
  const svgRef = useRef()

  useEffect(() => {
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const width = 500
    const height = 220
    const margin = { top: 30, right: 20, bottom: 40, left: 50 }
    const w = width - margin.left - margin.right
    const h = height - margin.top - margin.bottom

    svg.attr('width', width).attr('height', height)
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    // Data: 8 generations with rewards and advantages
    const generations = [
      { id: 'G1', reward: 1.0, label: 'OLAP', correct: true },
      { id: 'G2', reward: 0.0, label: 'AI/ML', correct: false },
      { id: 'G3', reward: 1.0, label: 'OLAP', correct: true },
      { id: 'G4', reward: 0.0, label: 'Video', correct: false },
      { id: 'G5', reward: 0.0, label: 'AI/ML', correct: false },
      { id: 'G6', reward: 1.0, label: 'OLAP', correct: true },
      { id: 'G7', reward: 1.0, label: 'OLAP', correct: true },
      { id: 'G8', reward: 0.0, label: 'AI/ML', correct: false },
    ]
    const mean = generations.reduce((s, g) => s + g.reward, 0) / generations.length
    generations.forEach(g => { g.advantage = g.reward - mean })

    const x = d3.scaleBand().domain(generations.map(g => g.id)).range([0, w]).padding(0.2)
    const y = d3.scaleLinear().domain([-0.7, 0.7]).range([h, 0])

    // Title
    svg.append('text').attr('x', width / 2).attr('y', 14)
      .attr('text-anchor', 'middle').attr('fill', '#94a3b8')
      .attr('font-size', '11').attr('font-weight', '600')
      .text('GRPO Advantage Scores (8 Generations)')

    // Zero line (mean reward baseline)
    g.append('line')
      .attr('x1', 0).attr('x2', w)
      .attr('y1', y(0)).attr('y2', y(0))
      .attr('stroke', '#64748b').attr('stroke-width', 1.5).attr('stroke-dasharray', '4,3')

    g.append('text')
      .attr('x', w + 5).attr('y', y(0) + 4)
      .attr('fill', '#64748b').attr('font-size', '8')
      .text(`mean=${mean.toFixed(2)}`)

    // Bars
    g.selectAll('.bar')
      .data(generations)
      .join('rect')
      .attr('x', d => x(d.id))
      .attr('y', d => d.advantage > 0 ? y(d.advantage) : y(0))
      .attr('width', x.bandwidth())
      .attr('height', 0)
      .attr('fill', d => d.advantage > 0 ? '#22c55e' : '#ef4444')
      .attr('opacity', 0.8)
      .attr('rx', 3)
      .transition()
      .duration(500)
      .delay((d, i) => i * 100)
      .attr('height', d => Math.abs(y(0) - y(d.advantage)))

    // Labels above/below bars
    g.selectAll('.adv-label')
      .data(generations)
      .join('text')
      .attr('x', d => x(d.id) + x.bandwidth() / 2)
      .attr('y', d => d.advantage > 0 ? y(d.advantage) - 5 : y(d.advantage) + 12)
      .attr('text-anchor', 'middle')
      .attr('fill', d => d.advantage > 0 ? '#22c55e' : '#ef4444')
      .attr('font-size', '9').attr('font-weight', '600')
      .text(d => (d.advantage > 0 ? '+' : '') + d.advantage.toFixed(2))
      .attr('opacity', 0)
      .transition().delay((d, i) => i * 100 + 500).attr('opacity', 1)

    // Classification labels at bottom
    g.selectAll('.class-label')
      .data(generations)
      .join('text')
      .attr('x', d => x(d.id) + x.bandwidth() / 2)
      .attr('y', h + 14)
      .attr('text-anchor', 'middle')
      .attr('fill', d => d.correct ? '#86efac' : '#fca5a5')
      .attr('font-size', '8')
      .text(d => d.label)

    // Generation IDs
    g.selectAll('.gen-label')
      .data(generations)
      .join('text')
      .attr('x', d => x(d.id) + x.bandwidth() / 2)
      .attr('y', h + 28)
      .attr('text-anchor', 'middle')
      .attr('fill', '#64748b').attr('font-size', '8')
      .text(d => d.id)

  }, [])

  return <svg ref={svgRef} />
}

const GRPO_INFRA = {
  gpuMemoryGB: 6.8,
  trainingTimeMinutes: 35,
  checkpointSizeMB: 1.7,
  peakGPUUtilization: 95,
  storageIOPattern: "Heavy compute bursts (generating 8 completions per prompt), periodic checkpoint writes",
  note: "8x generation per prompt means 8x the compute per training step vs SFT. Peak GPU utilization hits 95% during generation bursts."
}

export default function GRPOUnderTheHood() {
  const [activeTab, setActiveTab] = useState('advantage')
  const [showInfra, setShowInfra] = useState(false)

  const tabs = [
    { id: 'advantage', label: 'Advantage Scores' },
    { id: 'accuracy', label: 'Accuracy Over Training' },
    { id: 'reward', label: 'Mean Reward Curve' },
  ]

  return (
    <div className="max-w-5xl mx-auto">
      {/* Tabs */}
      <div className="flex gap-1 mb-4 bg-slate-800 rounded-lg p-1 w-fit">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-sm rounded-md transition-colors ${
              activeTab === tab.id
                ? 'bg-emerald-600 text-white'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === 'advantage' && (
        <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
          <p className="text-sm text-slate-300 mb-4">
            This is the core GRPO mechanism. The group mean reward (0.50) is the baseline.
            Correct classifications (green) get <strong className="text-emerald-400">positive advantage</strong> — they're reinforced.
            Incorrect ones (red) get <strong className="text-red-400">negative advantage</strong> — they're suppressed.
            The model learns to generate more responses like the winners.
          </p>
          <AdvantageWaterfall />
          <p className="text-xs text-slate-500 mt-3">
            Notice: no external reward model, no human labels. The advantage is computed entirely
            from the group's own statistics. This is what makes GRPO 50% cheaper than PPO.
          </p>
        </div>
      )}

      {activeTab === 'accuracy' && (
        <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
          <p className="text-sm text-slate-300 mb-4">
            Classification accuracy improves over training as the model learns from its own
            successes and failures. Starting at ~45% (random-ish), reaching ~85% after 200 steps.
          </p>
          <LossChart
            data={ACCURACY_CURVE}
            label="Classification Accuracy Over Training"
            color="#10b981"
            width={550}
            height={250}
          />
        </div>
      )}

      {activeTab === 'reward' && (
        <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
          <p className="text-sm text-slate-300 mb-4">
            Mean group reward increases as more generations become correct. This curve
            directly reflects the model's improving ability to classify I/O patterns.
          </p>
          <LossChart
            data={REWARD_CURVE}
            label="Mean Group Reward Over Training"
            color="#10b981"
            width={550}
            height={250}
          />
        </div>
      )}

      {/* GRPO formula */}
      <div className="mt-4 p-3 rounded bg-slate-800 border border-slate-700/50">
        <p className="text-xs text-slate-500 mb-1 font-semibold">GRPO update rule (simplified):</p>
        <p className="text-sm text-slate-300 font-mono">
          advantage_i = (reward_i - mean(rewards)) / std(rewards)
        </p>
        <p className="text-sm text-slate-300 font-mono mt-1">
          loss = -sum(advantage_i * log_prob(generation_i))
        </p>
        <p className="text-xs text-slate-500 mt-2">
          Positive advantage = reinforce this generation. Negative = suppress it.
          The std normalization prevents the model from only making tiny updates.
        </p>
      </div>

      {/* Infrastructure */}
      <button
        onClick={() => setShowInfra(!showInfra)}
        className="mt-4 text-sm text-emerald-400 hover:text-emerald-300 underline underline-offset-4"
      >
        {showInfra ? 'Hide' : 'Show'} infrastructure profile
      </button>
      {showInfra && (
        <div className="mt-3">
          <InfrastructureCard data={GRPO_INFRA} />
        </div>
      )}
    </div>
  )
}
