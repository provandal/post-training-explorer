import { useState, useRef, useEffect } from 'react'
import * as d3 from 'd3'

const TECHNIQUES = [
  { name: 'Prompting', gpu: 0, time: 0, cost: 1, models: 1, color: '#f97316', storage: 'None (inference only)' },
  { name: 'RAG', gpu: 0, time: 5, cost: 1.2, models: 1, color: '#eab308', storage: 'Vector DB: ~2-10 GB for embeddings' },
  { name: 'SFT\n(LoRA)', gpu: 4.2, time: 12, cost: 3, models: 1, color: '#8b5cf6', storage: 'Checkpoints: 1.7 MB adapter + base model reads' },
  { name: 'DPO\n(LoRA)', gpu: 5.1, time: 8, cost: 4, models: 1, color: '#ec4899', storage: 'Similar to SFT + preference pair dataset' },
  { name: 'RLHF\n(PPO)', gpu: 12.8, time: 45, cost: 10, models: 3, color: '#ef4444', storage: '3x model checkpoints + reward model + value model' },
  { name: 'GRPO', gpu: 6.8, time: 35, cost: 7, models: 1, color: '#10b981', storage: 'Heavy burst I/O during generation phases' },
]

function ComparisonChart({ metric }) {
  const svgRef = useRef()

  useEffect(() => {
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const configs = {
      gpu: { key: 'gpu', label: 'GPU Memory (GB)', format: v => `${v} GB` },
      time: { key: 'time', label: 'Training Time (min)', format: v => `${v} min` },
      cost: { key: 'cost', label: 'Relative Cost', format: v => `${v}x` },
      models: { key: 'models', label: 'Models in Memory', format: v => `${v}` },
    }
    const config = configs[metric]

    const width = 550
    const height = 220
    const margin = { top: 30, right: 40, bottom: 60, left: 50 }
    const w = width - margin.left - margin.right
    const h = height - margin.top - margin.bottom

    svg.attr('width', width).attr('height', height)
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    const x = d3.scaleBand().domain(TECHNIQUES.map(t => t.name)).range([0, w]).padding(0.3)
    const maxVal = d3.max(TECHNIQUES, t => t[config.key])
    const y = d3.scaleLinear().domain([0, maxVal * 1.15]).range([h, 0])

    // Title
    svg.append('text').attr('x', width / 2).attr('y', 16)
      .attr('text-anchor', 'middle').attr('fill', '#94a3b8')
      .attr('font-size', '11').attr('font-weight', '600')
      .text(config.label)

    // Grid
    g.selectAll('.grid').data(y.ticks(4)).join('line')
      .attr('x1', 0).attr('x2', w)
      .attr('y1', d => y(d)).attr('y2', d => y(d))
      .attr('stroke', '#1e293b').attr('stroke-dasharray', '2,2')

    // Y axis
    g.append('g').call(d3.axisLeft(y).ticks(4))
      .selectAll('text').attr('fill', '#64748b').attr('font-size', '9')
    g.selectAll('.domain').attr('stroke', '#334155')

    // Bars
    g.selectAll('.bar')
      .data(TECHNIQUES)
      .join('rect')
      .attr('x', d => x(d.name))
      .attr('y', h)
      .attr('width', x.bandwidth())
      .attr('height', 0)
      .attr('fill', d => d.color)
      .attr('opacity', 0.8)
      .attr('rx', 3)
      .transition().duration(600).delay((d, i) => i * 80)
      .attr('y', d => y(d[config.key]))
      .attr('height', d => h - y(d[config.key]))

    // Value labels
    g.selectAll('.value')
      .data(TECHNIQUES)
      .join('text')
      .attr('x', d => x(d.name) + x.bandwidth() / 2)
      .attr('y', d => y(d[config.key]) - 5)
      .attr('text-anchor', 'middle')
      .attr('fill', '#e2e8f0').attr('font-size', '10').attr('font-weight', '600')
      .text(d => d[config.key] > 0 ? config.format(d[config.key]) : 'N/A')
      .attr('opacity', 0)
      .transition().delay((d, i) => i * 80 + 600).attr('opacity', 1)

    // X axis labels
    TECHNIQUES.forEach(t => {
      const lines = t.name.split('\n')
      lines.forEach((line, j) => {
        g.append('text')
          .attr('x', x(t.name) + x.bandwidth() / 2)
          .attr('y', h + 15 + j * 12)
          .attr('text-anchor', 'middle')
          .attr('fill', t.color).attr('font-size', '9').attr('font-weight', '600')
          .text(line)
      })
    })

  }, [metric])

  return <svg ref={svgRef} />
}

export default function InfrastructureSummary() {
  const [metric, setMetric] = useState('gpu')

  const metrics = [
    { id: 'gpu', label: 'GPU Memory' },
    { id: 'time', label: 'Training Time' },
    { id: 'cost', label: 'Relative Cost' },
    { id: 'models', label: 'Models in Memory' },
  ]

  return (
    <div className="max-w-5xl mx-auto">
      {/* Metric selector */}
      <div className="flex gap-1 mb-4 bg-slate-800 rounded-lg p-1 w-fit">
        {metrics.map((m) => (
          <button
            key={m.id}
            onClick={() => setMetric(m.id)}
            className={`px-4 py-2 text-sm rounded-md transition-colors ${
              metric === m.id
                ? 'bg-cyan-600 text-white'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>

      {/* Chart */}
      <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50 mb-6">
        <ComparisonChart metric={metric} />
      </div>

      {/* Storage I/O patterns table */}
      <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50 mb-4">
        <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">
          Storage I/O Patterns by Technique
        </h4>
        <div className="space-y-2">
          {TECHNIQUES.map((t) => (
            <div key={t.name} className="flex items-start gap-3 text-sm">
              <span className="w-20 flex-shrink-0 font-semibold" style={{ color: t.color }}>
                {t.name.replace('\n', ' ')}
              </span>
              <span className="text-slate-400">{t.storage}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Key takeaways */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className="p-4 rounded-lg bg-violet-950/20 border border-violet-800/30">
          <h4 className="text-sm font-semibold text-violet-400 mb-2">The LoRA Revolution</h4>
          <p className="text-xs text-slate-400">
            SFT and DPO with LoRA need just 4-5 GB GPU memory. That's a single consumer GPU card.
            Full fine-tuning of the same model would need 40+ GB. PEFT made post-training accessible.
          </p>
        </div>
        <div className="p-4 rounded-lg bg-red-950/20 border border-red-800/30">
          <h4 className="text-sm font-semibold text-red-400 mb-2">The RLHF Tax</h4>
          <p className="text-xs text-slate-400">
            RLHF (PPO) needs 3 models in memory simultaneously: policy, reward, and value networks.
            That's 12.8 GB for a 360M model. For 7B models, you're looking at 250+ GB. DPO eliminates this.
          </p>
        </div>
        <div className="p-4 rounded-lg bg-emerald-950/20 border border-emerald-800/30">
          <h4 className="text-sm font-semibold text-emerald-400 mb-2">The GRPO Burst Pattern</h4>
          <p className="text-xs text-slate-400">
            GRPO generates 8 completions per prompt — that's 8x compute per step. But it eliminates
            the reward model entirely. Net result: more GPU time, less GPU memory, no human labelers.
          </p>
        </div>
      </div>

      {/* Scaling note */}
      <div className="mt-4 p-3 rounded bg-slate-800 border border-slate-700/50">
        <p className="text-xs text-slate-400">
          <strong className="text-slate-300">Scaling reality:</strong> All numbers above are for SmolLM2-360M.
          For a production 7B model, multiply GPU memory by ~20x and training time by ~15x.
          For 70B models, multiply by ~200x and ~150x. This is why enterprise post-training is an
          infrastructure problem, not just a data science problem.
        </p>
      </div>
    </div>
  )
}
