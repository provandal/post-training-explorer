import { useRef, useEffect, useState } from 'react'
import * as d3 from 'd3'

// Animated 2D visualization of vector search / embedding space
// Shows how text gets converted to embeddings and how similarity search works

// Pre-computed "embedding" positions for our knowledge base entries
// Clustered by workload type to show how similar patterns end up near each other
const KB_POINTS = [
  // OLTP cluster (lower-right area)
  { id: 1, label: 'OLTP', x: 0.72, y: 0.28, desc: 'High IOPS, 4-8K blocks, <1ms' },
  { id: 2, label: 'OLTP', x: 0.78, y: 0.32, desc: '50K IOPS, random, low latency' },
  { id: 3, label: 'OLTP', x: 0.69, y: 0.24, desc: 'Transaction processing pattern' },
  // VDI cluster (near OLTP but shifted)
  { id: 4, label: 'VDI', x: 0.62, y: 0.38, desc: 'Balanced R/W, high queue depth' },
  { id: 5, label: 'VDI', x: 0.58, y: 0.42, desc: 'Mixed desktop workloads' },
  // OLAP cluster (upper-left)
  { id: 6, label: 'OLAP', x: 0.25, y: 0.72, desc: 'Large sequential reads, analytics' },
  { id: 7, label: 'OLAP', x: 0.30, y: 0.68, desc: 'Scan-heavy warehouse queries' },
  // AI/ML cluster (upper area)
  { id: 8, label: 'AI/ML', x: 0.35, y: 0.82, desc: 'Sequential reads, large blocks' },
  { id: 9, label: 'AI/ML', x: 0.40, y: 0.78, desc: 'Training data pipeline' },
  // Video cluster (far upper-left)
  { id: 10, label: 'Video', x: 0.15, y: 0.85, desc: 'Streaming, very large blocks' },
  { id: 11, label: 'Video', x: 0.12, y: 0.80, desc: '99% reads, sequential' },
  // Backup cluster (lower-left)
  { id: 12, label: 'Backup', x: 0.18, y: 0.22, desc: '95% writes, sequential' },
  { id: 13, label: 'Backup', x: 0.22, y: 0.18, desc: 'Large block archive operations' },
]

// Query point - an OLTP-like pattern
const QUERY_POINT = { x: 0.74, y: 0.30, label: 'Query' }

// Top-3 nearest neighbors (indices into KB_POINTS)
const TOP_K = [0, 1, 2] // The 3 OLTP entries

const CATEGORY_COLORS = {
  OLTP: '#f97316',
  VDI: '#a855f7',
  OLAP: '#3b82f6',
  'AI/ML': '#10b981',
  Video: '#ec4899',
  Backup: '#64748b',
  Query: '#ef4444',
}

export default function VectorSearchViz({ autoPlay = false }) {
  const svgRef = useRef()
  const [phase, setPhase] = useState(0)
  // Phases: 0=empty, 1=show embeddings, 2=show query, 3=show distances, 4=highlight matches

  useEffect(() => {
    if (autoPlay && phase === 0) {
      setPhase(1)
    }
  }, [autoPlay])

  useEffect(() => {
    if (phase > 0 && phase < 4) {
      const timer = setTimeout(() => setPhase(phase + 1), 1200)
      return () => clearTimeout(timer)
    }
  }, [phase])

  useEffect(() => {
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const width = 520
    const height = 380
    const margin = { top: 35, right: 20, bottom: 35, left: 20 }
    const w = width - margin.left - margin.right
    const h = height - margin.top - margin.bottom

    svg.attr('width', width).attr('height', height)

    // Background
    svg.append('rect').attr('width', width).attr('height', height).attr('fill', '#0c1222').attr('rx', 8)

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    // Title
    svg.append('text').attr('x', width / 2).attr('y', 20)
      .attr('text-anchor', 'middle').attr('fill', '#94a3b8')
      .attr('font-size', '12').attr('font-weight', '600')
      .text('Embedding Space — Where Similar Patterns Live Close Together')

    // Subtle grid
    for (let i = 0; i <= 4; i++) {
      g.append('line').attr('x1', 0).attr('x2', w).attr('y1', h * i / 4).attr('y2', h * i / 4).attr('stroke', '#1e293b').attr('stroke-dasharray', '2,4')
      g.append('line').attr('x1', w * i / 4).attr('x2', w * i / 4).attr('y1', 0).attr('y2', h).attr('stroke', '#1e293b').attr('stroke-dasharray', '2,4')
    }

    // Axis labels
    svg.append('text').attr('x', width / 2).attr('y', height - 8).attr('text-anchor', 'middle').attr('fill', '#475569').attr('font-size', '9').attr('font-style', 'italic').text('Embedding dimension 1 (simplified to 2D)')
    svg.append('text').attr('x', 10).attr('y', height / 2).attr('text-anchor', 'middle').attr('fill', '#475569').attr('font-size', '9').attr('font-style', 'italic').attr('transform', `rotate(-90, 10, ${height / 2})`).text('Embedding dimension 2')

    const xScale = d3.scaleLinear().domain([0, 1]).range([0, w])
    const yScale = d3.scaleLinear().domain([0, 1]).range([h, 0])

    if (phase >= 1) {
      // Draw knowledge base points with cluster halos
      const categories = [...new Set(KB_POINTS.map(p => p.label))]
      categories.forEach(cat => {
        const pts = KB_POINTS.filter(p => p.label === cat)
        const cx = d3.mean(pts, p => xScale(p.x))
        const cy = d3.mean(pts, p => yScale(p.y))
        g.append('ellipse')
          .attr('cx', cx).attr('cy', cy)
          .attr('rx', 0).attr('ry', 0)
          .attr('fill', CATEGORY_COLORS[cat])
          .attr('opacity', 0.07)
          .transition().duration(800)
          .attr('rx', 45).attr('ry', 35)
      })

      // Points
      KB_POINTS.forEach((p, i) => {
        const circle = g.append('circle')
          .attr('cx', xScale(p.x)).attr('cy', yScale(p.y))
          .attr('r', 0)
          .attr('fill', CATEGORY_COLORS[p.label])
          .attr('stroke', phase >= 4 && TOP_K.includes(i) ? '#ffffff' : 'none')
          .attr('stroke-width', 2)
          .attr('opacity', phase >= 4 && !TOP_K.includes(i) ? 0.3 : 0.85)

        circle.transition().duration(500).delay(i * 40).attr('r', phase >= 4 && TOP_K.includes(i) ? 8 : 6)

        // Labels (only show on hover-like state or when highlighted)
        if (phase >= 4 && TOP_K.includes(i)) {
          g.append('text')
            .attr('x', xScale(p.x) + 12).attr('y', yScale(p.y) + 4)
            .attr('fill', '#e2e8f0').attr('font-size', '9').attr('font-weight', '600')
            .text(p.desc)
            .attr('opacity', 0)
            .transition().delay(800).duration(400).attr('opacity', 1)
        }
      })

      // Category legend
      const legendG = g.append('g').attr('transform', `translate(${w - 100}, 5)`)
      categories.forEach((cat, i) => {
        legendG.append('circle').attr('cx', 0).attr('cy', i * 16).attr('r', 4).attr('fill', CATEGORY_COLORS[cat])
        legendG.append('text').attr('x', 8).attr('y', i * 16 + 4).attr('fill', '#94a3b8').attr('font-size', '9').text(cat)
      })
    }

    if (phase >= 2) {
      // Draw query point
      const qx = xScale(QUERY_POINT.x)
      const qy = yScale(QUERY_POINT.y)

      // Pulsing ring
      g.append('circle')
        .attr('cx', qx).attr('cy', qy).attr('r', 0)
        .attr('fill', 'none').attr('stroke', '#ef4444').attr('stroke-width', 2).attr('opacity', 0.4)
        .transition().duration(600).attr('r', 20)
        .transition().duration(600).attr('r', 12).attr('opacity', 0.2)

      // Point
      g.append('circle')
        .attr('cx', qx).attr('cy', qy).attr('r', 0)
        .attr('fill', '#ef4444').attr('stroke', '#fca5a5').attr('stroke-width', 2)
        .transition().duration(400).attr('r', 9)

      // Label
      g.append('text')
        .attr('x', qx).attr('y', qy - 15)
        .attr('text-anchor', 'middle').attr('fill', '#fca5a5')
        .attr('font-size', '10').attr('font-weight', '700')
        .text('YOUR QUERY')
        .attr('opacity', 0).transition().delay(300).duration(300).attr('opacity', 1)
    }

    if (phase >= 3) {
      // Draw distance lines to all points, highlight nearest
      const qx = xScale(QUERY_POINT.x)
      const qy = yScale(QUERY_POINT.y)

      KB_POINTS.forEach((p, i) => {
        const px = xScale(p.x)
        const py = yScale(p.y)
        const isMatch = TOP_K.includes(i)
        const dist = Math.sqrt((p.x - QUERY_POINT.x) ** 2 + (p.y - QUERY_POINT.y) ** 2)

        const line = g.append('line')
          .attr('x1', qx).attr('y1', qy)
          .attr('x2', qx).attr('y2', qy)
          .attr('stroke', isMatch ? '#22c55e' : '#475569')
          .attr('stroke-width', isMatch ? 2 : 0.5)
          .attr('stroke-dasharray', isMatch ? 'none' : '3,3')
          .attr('opacity', isMatch ? 0.8 : 0.2)

        line.transition().duration(400).delay(i * 30)
          .attr('x2', px).attr('y2', py)

        // Similarity score for matches
        if (isMatch) {
          const similarity = (1 - dist / 1.0).toFixed(2)
          const mx = (qx + px) / 2
          const my = (qy + py) / 2
          g.append('text')
            .attr('x', mx).attr('y', my - 6)
            .attr('text-anchor', 'middle').attr('fill', '#86efac')
            .attr('font-size', '9').attr('font-weight', '700')
            .text(`${(similarity * 100).toFixed(0)}%`)
            .attr('opacity', 0)
            .transition().delay(600).duration(300).attr('opacity', 1)
        }
      })
    }

    // Phase description
    const descriptions = [
      '',
      'Knowledge base entries plotted in embedding space. Similar patterns cluster together.',
      'Your I/O pattern is converted to an embedding and placed in the same space.',
      'Cosine similarity measured to every entry. Nearest neighbors highlighted.',
      'Top-3 matches retrieved. These get injected into the prompt as context.',
    ]
    if (phase > 0) {
      svg.append('text').attr('x', width / 2).attr('y', height - 22)
        .attr('text-anchor', 'middle').attr('fill', '#64748b')
        .attr('font-size', '10').attr('font-weight', '500')
        .text(descriptions[phase] || '')
    }

  }, [phase])

  return (
    <div>
      <svg ref={svgRef} />
      <div className="flex gap-2 mt-2">
        {phase === 0 && (
          <button onClick={() => setPhase(1)} className="px-4 py-1.5 text-sm bg-blue-700 hover:bg-blue-600 rounded-md transition-colors">
            Show embedding space
          </button>
        )}
        {phase >= 4 && (
          <button onClick={() => setPhase(0)} className="px-4 py-1.5 text-sm bg-slate-700 hover:bg-slate-600 rounded-md transition-colors text-slate-300">
            Replay animation
          </button>
        )}
      </div>
    </div>
  )
}
