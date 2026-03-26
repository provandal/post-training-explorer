import { useState, useRef, useEffect } from 'react'
import * as d3 from 'd3'
import { isLoaded, getResourceUtilization, getTrainingTime, getStorageIOProfile } from '../data/loadArtifacts'

// Build techniques data, optionally using real resource utilization
function buildTechniques() {
  const res = isLoaded() ? getResourceUtilization() : null

  const sftTime = (isLoaded() ? getTrainingTime('sft') : null)
  const dpoTime = (isLoaded() ? getTrainingTime('dpo') : null)
  const grpoTime = (isLoaded() ? getTrainingTime('grpo') : null)

  return [
    { name: 'Prompting', gpu: 0, time: 0, cost: 1, models: 1, color: '#f97316', storage: 'None (inference only)' },
    { name: 'RAG', gpu: 0, time: 5, cost: 1.2, models: 1, color: '#eab308', storage: 'Vector DB: ~2-10 GB for embeddings' },
    { name: 'SFT\n(LoRA)', gpu: 4.2, time: sftTime ? Math.round(sftTime / 60) : 12, cost: 3, models: 1, color: '#8b5cf6',
      storage: `Checkpoints: ${res?.sft_checkpoint_size_mb ? res.sft_checkpoint_size_mb + ' MB' : '1.7 MB'} adapter + base model reads` },
    { name: 'DPO\n(LoRA)', gpu: 5.1, time: dpoTime ? Math.round(dpoTime / 60) : 8, cost: 4, models: 1, color: '#ec4899',
      storage: 'Similar to SFT + preference pair dataset' },
    { name: 'RLHF\n(PPO)', gpu: 12.8, time: 45, cost: 10, models: 3, color: '#ef4444',
      storage: '3x model checkpoints + reward model + value model' },
    { name: 'GRPO', gpu: 6.8, time: grpoTime ? Math.round(grpoTime / 60) : 35, cost: 7, models: 1, color: '#10b981',
      storage: 'Heavy burst I/O during generation phases' },
  ]
}

function ComparisonChart({ metric, techniques }) {
  const svgRef = useRef()

  useEffect(() => {
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const TECHNIQUES = techniques

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

  }, [metric, techniques])

  return <svg ref={svgRef} />
}

// ─── Storage Footprint Chart ──────────────────────────────────────────────────
// Horizontal bars comparing LoRA adapter vs full model checkpoint size
function StorageFootprintChart({ profile }) {
  const svgRef = useRef()

  useEffect(() => {
    if (!profile) return
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const scaling = profile.scaling_projections
    const sizes = ['360M', '7B', '70B']
    const width = 550
    const height = 180
    const margin = { top: 25, right: 110, bottom: 30, left: 70 }
    const w = width - margin.left - margin.right
    const h = height - margin.top - margin.bottom

    svg.attr('width', width).attr('height', height)
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    // Title
    svg.append('text').attr('x', width / 2).attr('y', 14)
      .attr('text-anchor', 'middle').attr('fill', '#94a3b8')
      .attr('font-size', '11').attr('font-weight', '600')
      .text('Checkpoint Size: LoRA Adapter vs Full Model')

    const y = d3.scaleBand().domain(sizes).range([0, h]).padding(0.35)
    const maxVal = d3.max(sizes, s => scaling[s]?.full_finetune_checkpoint_mb || 0)
    const x = d3.scaleLog().domain([1, maxVal * 1.5]).range([0, w]).clamp(true)

    // Grid lines
    const ticks = [1, 10, 100, 1000, 10000, 100000]
    g.selectAll('.grid').data(ticks.filter(t => t <= maxVal * 1.5)).join('line')
      .attr('x1', d => x(d)).attr('x2', d => x(d))
      .attr('y1', 0).attr('y2', h)
      .attr('stroke', '#1e293b').attr('stroke-dasharray', '2,2')

    // Full model bars (background)
    g.selectAll('.full-bar').data(sizes).join('rect')
      .attr('x', 0).attr('y', d => y(d))
      .attr('width', 0).attr('height', y.bandwidth())
      .attr('fill', '#334155').attr('rx', 2).attr('opacity', 0.6)
      .transition().duration(600)
      .attr('width', d => x(scaling[d]?.full_finetune_checkpoint_mb || 1))

    // LoRA adapter bars (overlay)
    g.selectAll('.lora-bar').data(sizes).join('rect')
      .attr('x', 0).attr('y', d => y(d))
      .attr('width', 0).attr('height', y.bandwidth())
      .attr('fill', '#8b5cf6').attr('rx', 2).attr('opacity', 0.9)
      .transition().duration(800).delay(300)
      .attr('width', d => Math.max(3, x(scaling[d]?.lora_adapter_mb || 1)))

    // Labels on Y axis
    g.selectAll('.y-label').data(sizes).join('text')
      .attr('x', -8).attr('y', d => y(d) + y.bandwidth() / 2)
      .attr('text-anchor', 'end').attr('dominant-baseline', 'middle')
      .attr('fill', '#e2e8f0').attr('font-size', '10').attr('font-weight', '600')
      .text(d => d)

    // Size labels at end of bars
    g.selectAll('.full-label').data(sizes).join('text')
      .attr('x', d => x(scaling[d]?.full_finetune_checkpoint_mb || 1) + 4)
      .attr('y', d => y(d) + y.bandwidth() * 0.35)
      .attr('fill', '#64748b').attr('font-size', '9')
      .text(d => {
        const mb = scaling[d]?.full_finetune_checkpoint_mb || 0
        return mb >= 1000 ? `${(mb / 1000).toFixed(0)} GB full` : `${mb} MB full`
      })
      .attr('opacity', 0).transition().delay(600).attr('opacity', 1)

    g.selectAll('.lora-label').data(sizes).join('text')
      .attr('x', d => Math.max(3, x(scaling[d]?.lora_adapter_mb || 1)) + 4)
      .attr('y', d => y(d) + y.bandwidth() * 0.75)
      .attr('fill', '#a78bfa').attr('font-size', '9').attr('font-weight', '600')
      .text(d => {
        const mb = scaling[d]?.lora_adapter_mb || 0
        const pct = scaling[d]?.storage_reduction_pct || 0
        return `${mb} MB adapter (${pct}% smaller)`
      })
      .attr('opacity', 0).transition().delay(800).attr('opacity', 1)

  }, [profile])

  return <svg ref={svgRef} />
}

// ─── I/O Timeline Chart ──────────────────────────────────────────────────────
// Horizontal stacked bars showing read/write/compute phases per technique
function IOTimelineChart({ profile }) {
  const svgRef = useRef()

  useEffect(() => {
    if (!profile?.techniques) return
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const techniqueNames = ['sft', 'dpo', 'grpo']
    const techniqueLabels = { sft: 'SFT', dpo: 'DPO', grpo: 'GRPO' }
    const techniqueColors = { sft: '#8b5cf6', dpo: '#ec4899', grpo: '#10b981' }
    const directionColors = {
      read: '#3b82f6',
      write: '#f97316',
      compute: '#1e293b',
      read_write: '#eab308',
    }

    const width = 550
    const height = 190
    const margin = { top: 25, right: 20, bottom: 40, left: 50 }
    const w = width - margin.left - margin.right
    const h = height - margin.top - margin.bottom

    svg.attr('width', width).attr('height', height)
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    // Title
    svg.append('text').attr('x', width / 2).attr('y', 14)
      .attr('text-anchor', 'middle').attr('fill', '#94a3b8')
      .attr('font-size', '11').attr('font-weight', '600')
      .text('Storage I/O Timeline by Technique')

    const y = d3.scaleBand().domain(techniqueNames).range([0, h]).padding(0.3)

    // Technique labels
    g.selectAll('.tech-label').data(techniqueNames).join('text')
      .attr('x', -8).attr('y', d => y(d) + y.bandwidth() / 2)
      .attr('text-anchor', 'end').attr('dominant-baseline', 'middle')
      .attr('fill', d => techniqueColors[d]).attr('font-size', '11').attr('font-weight', '700')
      .text(d => techniqueLabels[d])

    // Draw stacked horizontal bars for each technique
    techniqueNames.forEach(tech => {
      const events = profile.techniques[tech]?.io_events || []
      let xOffset = 0
      const barY = y(tech)
      const barH = y.bandwidth()

      events.forEach((evt, i) => {
        const evtW = (evt.duration_pct / 100) * w
        const color = directionColors[evt.direction] || '#334155'

        g.append('rect')
          .attr('x', xOffset).attr('y', barY)
          .attr('width', 0).attr('height', barH)
          .attr('fill', color).attr('opacity', evt.direction === 'compute' ? 0.3 : 0.7)
          .attr('rx', i === 0 ? 3 : 0)
          .transition().duration(500).delay(i * 80)
          .attr('width', evtW)

        // Phase labels for larger segments
        if (evt.duration_pct >= 15) {
          g.append('text')
            .attr('x', xOffset + evtW / 2).attr('y', barY + barH / 2)
            .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
            .attr('fill', '#e2e8f0').attr('font-size', '8')
            .text(evt.phase.length > 20 ? evt.phase.slice(0, 18) + '...' : evt.phase)
            .attr('opacity', 0)
            .transition().delay(500 + i * 80).attr('opacity', 1)
        }

        // Size annotation for read/write events with actual I/O
        if (evt.size_mb > 0 && evt.duration_pct >= 10) {
          g.append('text')
            .attr('x', xOffset + evtW / 2).attr('y', barY + barH - 3)
            .attr('text-anchor', 'middle')
            .attr('fill', '#94a3b8').attr('font-size', '7')
            .text(evt.size_mb >= 100 ? `${(evt.size_mb / 1000).toFixed(1)} GB` : `${evt.size_mb} MB`)
            .attr('opacity', 0)
            .transition().delay(600 + i * 80).attr('opacity', 1)
        }

        xOffset += evtW
      })
    })

    // Legend
    const legendData = [
      { label: 'Read', color: directionColors.read },
      { label: 'Write', color: directionColors.write },
      { label: 'Compute', color: '#334155' },
    ]
    const legend = g.append('g').attr('transform', `translate(0, ${h + 12})`)
    legendData.forEach((d, i) => {
      legend.append('rect')
        .attr('x', i * 80).attr('y', 0).attr('width', 10).attr('height', 10)
        .attr('fill', d.color).attr('opacity', d.label === 'Compute' ? 0.3 : 0.7).attr('rx', 2)
      legend.append('text')
        .attr('x', i * 80 + 14).attr('y', 9)
        .attr('fill', '#94a3b8').attr('font-size', '9')
        .text(d.label)
    })

  }, [profile])

  return <svg ref={svgRef} />
}

// ─── Storage Architecture Comparison ──────────────────────────────────────────
// Local SSD vs Network Storage latency bars
function StorageArchitectureChart({ profile }) {
  const svgRef = useRef()

  useEffect(() => {
    if (!profile?.storage_architecture) return
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const arch = profile.storage_architecture
    const items = [
      {
        label: 'LoRA Adapter Save',
        local: arch.adapter_checkpoint.local_ssd_ms,
        network: arch.adapter_checkpoint.network_nfs_ms,
        unit: 'ms',
      },
      {
        label: 'Full Checkpoint Save',
        local: arch.full_checkpoint.local_ssd_ms,
        network: arch.full_checkpoint.network_nfs_ms,
        unit: 'ms',
      },
      {
        label: 'Base Model Load',
        local: arch.model_load.local_ssd_seconds * 1000,
        network: arch.model_load.network_nfs_seconds * 1000,
        unit: 'ms',
      },
    ]

    const width = 550
    const height = 170
    const margin = { top: 25, right: 90, bottom: 25, left: 130 }
    const w = width - margin.left - margin.right
    const h = height - margin.top - margin.bottom

    svg.attr('width', width).attr('height', height)
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    // Title
    svg.append('text').attr('x', width / 2).attr('y', 14)
      .attr('text-anchor', 'middle').attr('fill', '#94a3b8')
      .attr('font-size', '11').attr('font-weight', '600')
      .text('Local SSD vs Network Storage (NFS/GPFS)')

    const maxVal = d3.max(items, d => d.network)
    const y = d3.scaleBand().domain(items.map(d => d.label)).range([0, h]).padding(0.3)
    const x = d3.scaleLog().domain([0.5, maxVal * 1.5]).range([0, w]).clamp(true)

    // Row labels
    g.selectAll('.row-label').data(items).join('text')
      .attr('x', -8).attr('y', d => y(d.label) + y.bandwidth() / 2)
      .attr('text-anchor', 'end').attr('dominant-baseline', 'middle')
      .attr('fill', '#e2e8f0').attr('font-size', '10')
      .text(d => d.label)

    // Network bars (background — wider)
    g.selectAll('.net-bar').data(items).join('rect')
      .attr('x', 0).attr('y', d => y(d.label))
      .attr('width', 0).attr('height', y.bandwidth())
      .attr('fill', '#ef4444').attr('opacity', 0.4).attr('rx', 2)
      .transition().duration(600)
      .attr('width', d => x(d.network))

    // Local bars (overlay — narrower)
    g.selectAll('.local-bar').data(items).join('rect')
      .attr('x', 0).attr('y', d => y(d.label))
      .attr('width', 0).attr('height', y.bandwidth())
      .attr('fill', '#22c55e').attr('opacity', 0.7).attr('rx', 2)
      .transition().duration(600).delay(200)
      .attr('width', d => Math.max(3, x(d.local)))

    // Value labels
    g.selectAll('.net-label').data(items).join('text')
      .attr('x', d => x(d.network) + 4)
      .attr('y', d => y(d.label) + y.bandwidth() * 0.35)
      .attr('fill', '#f87171').attr('font-size', '9')
      .text(d => {
        const v = d.network
        return v >= 1000 ? `${(v / 1000).toFixed(1)}s NFS` : `${v.toFixed(0)}ms NFS`
      })
      .attr('opacity', 0).transition().delay(600).attr('opacity', 1)

    g.selectAll('.local-label').data(items).join('text')
      .attr('x', d => Math.max(3, x(d.local)) + 4)
      .attr('y', d => y(d.label) + y.bandwidth() * 0.75)
      .attr('fill', '#4ade80').attr('font-size', '9').attr('font-weight', '600')
      .text(d => {
        const v = d.local
        return v >= 1000 ? `${(v / 1000).toFixed(1)}s local` : `${v < 1 ? '<1' : v.toFixed(0)}ms local`
      })
      .attr('opacity', 0).transition().delay(800).attr('opacity', 1)

  }, [profile])

  return <svg ref={svgRef} />
}

// ─── Scaling Projections Table ────────────────────────────────────────────────
function ScalingTable({ profile }) {
  if (!profile?.scaling_projections) return null
  const scaling = profile.scaling_projections
  const sizes = ['360M', '7B', '70B']

  const fmt = (mb) => {
    if (mb >= 1000) return `${(mb / 1000).toFixed(0)} GB`
    return `${mb} MB`
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-slate-700">
            <th className="text-left py-2 pr-4 text-slate-400 font-semibold">Model Size</th>
            <th className="text-right py-2 px-3 text-slate-400 font-semibold">Base Model</th>
            <th className="text-right py-2 px-3 text-slate-400 font-semibold">Full Checkpoint</th>
            <th className="text-right py-2 px-3 text-violet-400 font-semibold">LoRA Adapter</th>
            <th className="text-right py-2 pl-3 text-emerald-400 font-semibold">Storage Saved</th>
          </tr>
        </thead>
        <tbody>
          {sizes.map((size) => (
            <tr key={size} className="border-b border-slate-800/50">
              <td className="py-2 pr-4 text-slate-200 font-semibold">{size}</td>
              <td className="py-2 px-3 text-right text-slate-400">{fmt(scaling[size].base_model_mb)}</td>
              <td className="py-2 px-3 text-right text-slate-400">{fmt(scaling[size].full_finetune_checkpoint_mb)}</td>
              <td className="py-2 px-3 text-right text-violet-300 font-semibold">{fmt(scaling[size].lora_adapter_mb)}</td>
              <td className="py-2 pl-3 text-right text-emerald-400 font-semibold">{scaling[size].storage_reduction_pct}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────
export default function InfrastructureSummary() {
  const [metric, setMetric] = useState('gpu')
  const [storageView, setStorageView] = useState('footprint')
  const TECHNIQUES = buildTechniques()
  const profile = isLoaded() ? getStorageIOProfile() : null

  const metrics = [
    { id: 'gpu', label: 'GPU Memory' },
    { id: 'time', label: 'Training Time' },
    { id: 'cost', label: 'Relative Cost' },
    { id: 'models', label: 'Models in Memory' },
  ]

  const storageViews = [
    { id: 'footprint', label: 'Storage Footprint' },
    { id: 'timeline', label: 'I/O Timeline' },
    { id: 'architecture', label: 'Local vs Network' },
  ]

  // Fallback profile for when real data isn't loaded
  const fallbackProfile = {
    base_model_size_mb: 720,
    train_dataset_mb: 0.74,
    techniques: {
      sft: {
        adapter_size_mb: 1.7, adapter_pct_of_base: 0.236, full_checkpoint_size_mb: 720,
        dataset_read_mb: 0.74, epochs: 5, checkpoint_saves: 5,
        total_write_mb: 8.5, total_read_mb: 723.7, training_time_seconds: 394,
        io_events: [
          { phase: 'Model Load', direction: 'read', size_mb: 720, pattern: 'sequential', duration_pct: 5 },
          { phase: 'Dataset Scan (5 epochs)', direction: 'read', size_mb: 3.7, pattern: 'sequential_repeated', duration_pct: 85 },
          { phase: 'Checkpoint Saves (5x)', direction: 'write', size_mb: 8.5, pattern: 'periodic_burst', duration_pct: 5 },
          { phase: 'Final Adapter', direction: 'write', size_mb: 1.7, pattern: 'burst', duration_pct: 5 },
        ],
      },
      dpo: {
        adapter_size_mb: 1.7, adapter_pct_of_base: 0.236, full_checkpoint_size_mb: 720,
        dataset_read_mb: 1.48, epochs: 2, checkpoint_saves: 2,
        total_write_mb: 3.4, total_read_mb: 727.6, training_time_seconds: 317,
        io_events: [
          { phase: 'Base + SFT Load', direction: 'read', size_mb: 721.7, pattern: 'sequential', duration_pct: 8 },
          { phase: 'SFT Merge (in-memory)', direction: 'compute', size_mb: 0, pattern: 'in_memory', duration_pct: 5 },
          { phase: 'Preference Pairs (2 epochs)', direction: 'read', size_mb: 2.96, pattern: 'sequential_repeated', duration_pct: 75 },
          { phase: 'Checkpoint Saves (2x)', direction: 'write', size_mb: 3.4, pattern: 'periodic_burst', duration_pct: 7 },
          { phase: 'Final Adapter', direction: 'write', size_mb: 1.7, pattern: 'burst', duration_pct: 5 },
        ],
      },
      grpo: {
        adapter_size_mb: 1.7, adapter_pct_of_base: 0.236, full_checkpoint_size_mb: 720,
        dataset_read_mb: 0.22, epochs: 2, checkpoint_saves: 2, generations_per_prompt: 8,
        total_write_mb: 3.4, total_read_mb: 721.9, training_time_seconds: 770,
        io_events: [
          { phase: 'Base + SFT Load', direction: 'read', size_mb: 721.7, pattern: 'sequential', duration_pct: 4 },
          { phase: 'SFT Merge (in-memory)', direction: 'compute', size_mb: 0, pattern: 'in_memory', duration_pct: 2 },
          { phase: 'Prompt Dataset', direction: 'read', size_mb: 0.22, pattern: 'sequential', duration_pct: 1 },
          { phase: '8x Generation Bursts', direction: 'compute', size_mb: 0, pattern: 'burst_compute', duration_pct: 60 },
          { phase: 'Reward + Policy Update', direction: 'compute', size_mb: 0, pattern: 'in_memory', duration_pct: 28 },
          { phase: 'Checkpoint Saves (2x)', direction: 'write', size_mb: 3.4, pattern: 'periodic_burst', duration_pct: 5 },
        ],
      },
    },
    scaling_projections: {
      '360M': { base_model_mb: 720, lora_adapter_mb: 1.7, full_finetune_checkpoint_mb: 720, storage_reduction_pct: 99.8 },
      '7B': { base_model_mb: 14000, lora_adapter_mb: 34, full_finetune_checkpoint_mb: 14000, storage_reduction_pct: 99.8 },
      '70B': { base_model_mb: 140000, lora_adapter_mb: 340, full_finetune_checkpoint_mb: 140000, storage_reduction_pct: 99.8 },
    },
    storage_architecture: {
      model_load: { local_ssd_seconds: 0.2, network_nfs_seconds: 3.6, size_mb: 720, label: 'Base model load (720 MB)' },
      adapter_checkpoint: { local_ssd_ms: 1, network_nfs_ms: 8.5, size_mb: 1.7, label: 'LoRA adapter save (1.7 MB)' },
      full_checkpoint: { local_ssd_ms: 240, network_nfs_ms: 3600, size_mb: 720, label: 'Full model checkpoint (720 MB)' },
      key_insight: 'LoRA reduces checkpoint I/O by 424x. On network storage, adapter saves complete in <1ms vs 3600ms for full checkpoints.',
    },
  }

  const displayProfile = profile || fallbackProfile

  return (
    <div className="max-w-5xl mx-auto">
      {/* Compute Metric selector */}
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

      {/* Compute Comparison Chart */}
      <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50 mb-6">
        <ComparisonChart metric={metric} techniques={TECHNIQUES} />
      </div>

      {/* ─── Data curation callout ───────────────────────────── */}
      <div className="p-4 rounded-lg bg-amber-950/20 border border-amber-800/30 mb-6">
        <h4 className="text-sm font-semibold text-amber-400 mb-2">The Real Bottleneck: Data Curation</h4>
        <p className="text-xs text-slate-300 leading-relaxed">
          At 360M parameters, all three techniques train in under 35 minutes on a single GPU.
          The real bottleneck at this scale isn't compute — it's <span className="text-amber-300 font-semibold">data curation</span>:
          labeling 1,400 examples for SFT, creating preference pairs for DPO, designing reward
          functions for GRPO. At 7B–70B scale, training becomes multi-hour to multi-day GPU jobs
          where storage throughput and checkpoint I/O dominate.
        </p>
      </div>

      {/* ─── Storage Infrastructure Section ───────────────────────────── */}
      <h3 className="text-sm font-semibold text-cyan-400 uppercase tracking-wide mb-3">
        Storage Infrastructure Impact
      </h3>

      {/* Storage view selector */}
      <div className="flex gap-1 mb-4 bg-slate-800 rounded-lg p-1 w-fit">
        {storageViews.map((v) => (
          <button
            key={v.id}
            onClick={() => setStorageView(v.id)}
            className={`px-4 py-2 text-sm rounded-md transition-colors ${
              storageView === v.id
                ? 'bg-violet-600 text-white'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {v.label}
          </button>
        ))}
      </div>

      {/* Storage visualization */}
      <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50 mb-4">
        {storageView === 'footprint' && (
          <StorageFootprintChart profile={displayProfile} />
        )}
        {storageView === 'timeline' && (
          <IOTimelineChart profile={displayProfile} />
        )}
        {storageView === 'architecture' && (
          <StorageArchitectureChart profile={displayProfile} />
        )}
      </div>

      {/* Per-technique I/O summary */}
      {displayProfile?.techniques && (
        <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50 mb-4">
          <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">
            Per-Technique Storage I/O
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {[
              { key: 'sft', label: 'SFT', color: '#8b5cf6' },
              { key: 'dpo', label: 'DPO', color: '#ec4899' },
              { key: 'grpo', label: 'GRPO', color: '#10b981' },
            ].map(({ key, label, color }) => {
              const t = displayProfile.techniques[key]
              if (!t) return null
              return (
                <div key={key} className="p-3 rounded bg-slate-800/50 border border-slate-700/30">
                  <h5 className="text-sm font-semibold mb-2" style={{ color }}>{label}</h5>
                  <div className="space-y-1 text-xs text-slate-400">
                    <div className="flex justify-between">
                      <span>Adapter size:</span>
                      <span className="text-violet-300 font-semibold">{t.adapter_size_mb} MB</span>
                    </div>
                    <div className="flex justify-between">
                      <span>% of base model:</span>
                      <span className="text-emerald-400">{t.adapter_pct_of_base}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Total reads:</span>
                      <span className="text-blue-300">{t.total_read_mb >= 1000 ? `${(t.total_read_mb / 1000).toFixed(1)} GB` : `${t.total_read_mb} MB`}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Total writes:</span>
                      <span className="text-orange-300">{t.total_write_mb} MB</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Checkpoint saves:</span>
                      <span className="text-slate-300">{t.checkpoint_saves}x</span>
                    </div>
                    {t.generations_per_prompt && (
                      <div className="flex justify-between">
                        <span>Generations/prompt:</span>
                        <span className="text-emerald-300">{t.generations_per_prompt}x</span>
                      </div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Scaling projections table */}
      <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50 mb-4">
        <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">
          Scaling: Checkpoint Storage by Model Size
        </h4>
        <ScalingTable profile={displayProfile} />
      </div>

      {/* Key takeaways */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className="p-4 rounded-lg bg-violet-950/20 border border-violet-800/30">
          <h4 className="text-sm font-semibold text-violet-400 mb-2">Checkpoint I/O</h4>
          <p className="text-xs text-slate-400">
            LoRA adapters are {displayProfile?.scaling_projections?.['360M']?.storage_reduction_pct || 99.8}% smaller
            than full model checkpoints. At 70B scale, that's 340 MB vs 140 GB per save — the difference
            between milliseconds and minutes on network storage.
          </p>
        </div>
        <div className="p-4 rounded-lg bg-blue-950/20 border border-blue-800/30">
          <h4 className="text-sm font-semibold text-blue-400 mb-2">Read Patterns</h4>
          <p className="text-xs text-slate-400">
            Base model load is the dominant read: {displayProfile?.base_model_size_mb || 720} MB sequential.
            Training datasets are comparatively tiny. DPO and GRPO read the base model + SFT adapter,
            creating a dependency chain on storage throughput.
          </p>
        </div>
        <div className="p-4 rounded-lg bg-emerald-950/20 border border-emerald-800/30">
          <h4 className="text-sm font-semibold text-emerald-400 mb-2">The GRPO Burst Pattern</h4>
          <p className="text-xs text-slate-400">
            GRPO generates 8 completions per prompt — 90% of wall-clock time is GPU compute, not storage I/O.
            But checkpoint saves create write bursts. On shared storage, these bursts can cause
            contention with other training jobs.
          </p>
        </div>
      </div>

      {/* Architecture insight */}
      {displayProfile?.storage_architecture?.key_insight && (
        <div className="mt-4 p-3 rounded bg-slate-800 border border-slate-700/50">
          <p className="text-xs text-slate-400">
            <strong className="text-slate-300">Storage architecture insight:</strong>{' '}
            {displayProfile.storage_architecture.key_insight}
          </p>
        </div>
      )}

      {/* Scaling reality */}
      <div className="mt-4 p-3 rounded bg-slate-800/50 border border-slate-700/30">
        <p className="text-xs text-slate-400">
          <strong className="text-slate-300">Scaling reality:</strong>{' '}
          At demo scale (360M params), GPU time is trivial — the human effort of data curation
          dwarfs compute cost. At production scale (7B–70B), the equation flips: multi-day training
          runs make storage throughput, checkpoint frequency, and network bandwidth the dominant
          infrastructure concerns. Plan for both bottlenecks.
        </p>
      </div>
    </div>
  )
}
