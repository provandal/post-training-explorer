import { useState, useRef, useEffect } from 'react'
import * as d3 from 'd3'
import InfrastructureCard from '../components/InfrastructureCard'

// Probability shift visualization data
const PROB_SHIFT = {
  example: "IOPS: 45000 | Latency: 0.3ms | Block Size: 8K | Read/Write: 70/30 | Sequential: 15% | Queue Depth: 32",
  chosen: "Concise, confident, structured",
  rejected: "Verbose, hedging, uncertain",
  before: { chosenLogProb: -2.1, rejectedLogProb: -1.8 },
  after: { chosenLogProb: -0.9, rejectedLogProb: -3.2 },
}

function ProbabilityShiftChart() {
  const svgRef = useRef()

  useEffect(() => {
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const width = 500
    const height = 250
    const margin = { top: 40, right: 30, bottom: 50, left: 70 }
    const w = width - margin.left - margin.right
    const h = height - margin.top - margin.bottom

    svg.attr('width', width).attr('height', height)
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    // Data
    const categories = ['Chosen\n(concise)', 'Rejected\n(verbose)']
    const beforeData = [PROB_SHIFT.before.chosenLogProb, PROB_SHIFT.before.rejectedLogProb]
    const afterData = [PROB_SHIFT.after.chosenLogProb, PROB_SHIFT.after.rejectedLogProb]

    const x = d3.scaleBand().domain(categories).range([0, w]).padding(0.4)
    const y = d3.scaleLinear().domain([-4, 0]).range([h, 0])

    // Title
    svg.append('text').attr('x', width / 2).attr('y', 16)
      .attr('text-anchor', 'middle').attr('fill', '#94a3b8')
      .attr('font-size', '11').attr('font-weight', '600')
      .text('Log Probability Shift: Before vs After DPO')

    // Grid
    g.selectAll('.grid').data(y.ticks(4)).join('line')
      .attr('x1', 0).attr('x2', w)
      .attr('y1', d => y(d)).attr('y2', d => y(d))
      .attr('stroke', '#1e293b').attr('stroke-dasharray', '2,2')

    // Y axis
    g.append('g').call(d3.axisLeft(y).ticks(4))
      .selectAll('text').attr('fill', '#64748b').attr('font-size', '9')
    g.append('text').attr('transform', 'rotate(-90)')
      .attr('x', -h / 2).attr('y', -50)
      .attr('text-anchor', 'middle').attr('fill', '#64748b').attr('font-size', '9')
      .text('Log Probability (higher = more likely)')

    g.selectAll('.domain').attr('stroke', '#334155')

    const barWidth = x.bandwidth() / 2 - 3

    // Before bars
    g.selectAll('.before')
      .data(categories)
      .join('rect')
      .attr('x', d => x(d))
      .attr('y', h)
      .attr('width', barWidth)
      .attr('height', 0)
      .attr('fill', '#ef4444')
      .attr('opacity', 0.6)
      .attr('rx', 3)
      .transition().duration(700)
      .attr('y', (d, i) => y(beforeData[i]))
      .attr('height', (d, i) => h - y(beforeData[i]))

    // After bars
    g.selectAll('.after')
      .data(categories)
      .join('rect')
      .attr('x', d => x(d) + barWidth + 6)
      .attr('y', h)
      .attr('width', barWidth)
      .attr('height', 0)
      .attr('fill', (d, i) => i === 0 ? '#22c55e' : '#f97316')
      .attr('opacity', 0.8)
      .attr('rx', 3)
      .transition().duration(700).delay(400)
      .attr('y', (d, i) => y(afterData[i]))
      .attr('height', (d, i) => h - y(afterData[i]))

    // Value labels - before
    g.selectAll('.label-before')
      .data(categories)
      .join('text')
      .attr('x', d => x(d) + barWidth / 2)
      .attr('y', (d, i) => y(beforeData[i]) - 5)
      .attr('text-anchor', 'middle')
      .attr('fill', '#ef4444').attr('font-size', '10').attr('font-weight', '600')
      .text((d, i) => beforeData[i].toFixed(1))
      .attr('opacity', 0)
      .transition().delay(700).attr('opacity', 1)

    // Value labels - after
    g.selectAll('.label-after')
      .data(categories)
      .join('text')
      .attr('x', d => x(d) + barWidth + 6 + barWidth / 2)
      .attr('y', (d, i) => y(afterData[i]) - 5)
      .attr('text-anchor', 'middle')
      .attr('fill', (d, i) => i === 0 ? '#22c55e' : '#f97316')
      .attr('font-size', '10').attr('font-weight', '600')
      .text((d, i) => afterData[i].toFixed(1))
      .attr('opacity', 0)
      .transition().delay(1100).attr('opacity', 1)

    // X axis labels
    categories.forEach((cat, i) => {
      cat.split('\n').forEach((line, j) => {
        g.append('text')
          .attr('x', x(cat) + x.bandwidth() / 2)
          .attr('y', h + 15 + j * 12)
          .attr('text-anchor', 'middle')
          .attr('fill', '#94a3b8').attr('font-size', '9')
          .text(line)
      })
    })

    // Legend
    const legend = svg.append('g').attr('transform', `translate(${margin.left + 10}, ${margin.top - 18})`)
    legend.append('rect').attr('width', 10).attr('height', 10).attr('fill', '#ef4444').attr('opacity', 0.6).attr('rx', 2)
    legend.append('text').attr('x', 14).attr('y', 9).attr('fill', '#94a3b8').attr('font-size', '9').text('Before DPO')
    legend.append('rect').attr('x', 100).attr('width', 10).attr('height', 10).attr('fill', '#22c55e').attr('opacity', 0.8).attr('rx', 2)
    legend.append('text').attr('x', 114).attr('y', 9).attr('fill', '#94a3b8').attr('font-size', '9').text('After DPO')

  }, [])

  return <svg ref={svgRef} />
}

const DPO_INFRA = {
  gpuMemoryGB: 5.1,
  trainingTimeMinutes: 8,
  checkpointSizeMB: 1.7,
  peakGPUUtilization: 82,
  modelsInMemory: 1,
  storageIOPattern: "Similar to SFT — reads preference pairs, writes checkpoints periodically",
  note: "DPO needs reference model logits, but these can be precomputed. Effective memory is ~1.2x SFT.",
  vsRLHF: {
    rlhfGPUMemoryGB: 12.8,
    rlhfTrainingTimeMinutes: 45,
    rlhfModelsInMemory: 3,
    dpoModelsInMemory: 1,
    note: "DPO achieves similar alignment with ~60% less compute by skipping the reward model entirely."
  }
}

export default function DPOUnderTheHood() {
  const [showInfra, setShowInfra] = useState(false)

  return (
    <div className="max-w-5xl mx-auto">
      {/* Probability shift */}
      <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50 mb-4">
        <p className="text-sm text-slate-300 mb-4">
          Look at what happened. <strong className="text-red-400">Before DPO</strong>, the model actually
          slightly preferred the verbose response (log prob -1.8 vs -2.1). It was more "cautious" by default.
          <strong className="text-green-400"> After DPO</strong>, the concise response is now
          strongly preferred (-0.9 vs -3.2). Your preference pairs literally reshaped the model's style.
        </p>
        <ProbabilityShiftChart />
      </div>

      {/* DPO vs RLHF explanation */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div className="p-4 rounded-lg bg-red-950/15 border border-red-800/30">
          <h4 className="text-sm font-semibold text-red-400 mb-2">RLHF (the old way)</h4>
          <ol className="text-xs text-slate-400 space-y-1 list-decimal list-inside">
            <li>Collect human preferences</li>
            <li>Train a separate <strong>reward model</strong> on those preferences</li>
            <li>Use PPO to optimize the policy against the reward model</li>
            <li>Need a <strong>value network</strong> for PPO baseline</li>
          </ol>
          <p className="text-xs text-red-300 mt-2 font-semibold">
            3 models in memory simultaneously. 45 minutes training.
          </p>
        </div>
        <div className="p-4 rounded-lg bg-green-950/15 border border-green-800/30">
          <h4 className="text-sm font-semibold text-green-400 mb-2">DPO (the direct way)</h4>
          <ol className="text-xs text-slate-400 space-y-1 list-decimal list-inside">
            <li>Collect the same human preferences</li>
            <li>Optimize directly from preference pairs</li>
            <li>No reward model, no PPO, no value network</li>
            <li>Math proves this is equivalent to implicit reward modeling</li>
          </ol>
          <p className="text-xs text-green-300 mt-2 font-semibold">
            1 model in memory. 8 minutes training. Same result.
          </p>
        </div>
      </div>

      {/* DPO formula (simplified) */}
      <div className="p-3 rounded bg-slate-800 border border-slate-700/50 mb-4">
        <p className="text-xs text-slate-500 mb-1 font-semibold">The DPO insight (simplified):</p>
        <p className="text-sm text-slate-300 font-mono">
          Loss = -log(sigmoid(beta * (log_prob_chosen - log_prob_rejected)))
        </p>
        <p className="text-xs text-slate-500 mt-1">
          Increase log probability of the chosen response, decrease the rejected.
          No intermediate reward model needed. Beta controls how aggressively preferences are enforced.
        </p>
      </div>

      {/* Infrastructure */}
      <button
        onClick={() => setShowInfra(!showInfra)}
        className="text-sm text-pink-400 hover:text-pink-300 underline underline-offset-4"
      >
        {showInfra ? 'Hide' : 'Show'} infrastructure comparison (DPO vs RLHF)
      </button>
      {showInfra && (
        <div className="mt-3">
          <InfrastructureCard data={DPO_INFRA} />
        </div>
      )}
    </div>
  )
}
