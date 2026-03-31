import { useRef, useEffect, useState } from 'react'
import * as d3 from 'd3'
import {
  TOKENS,
  TOKEN_COLORS,
  TOKEN_TYPE_LABELS,
  ATTENTION_HEADS,
  ATTENTION_LAYERS,
  ATTENTION_WEIGHTS,
  ATTENTION_INSIGHTS,
} from '../data/transformerData'

export default function AttentionHeatmap() {
  const [selectedToken, setSelectedToken] = useState(null)
  const [selectedHead, setSelectedHead] = useState('syntax')
  const [selectedLayer, setSelectedLayer] = useState(0)
  const [hoveredCell, setHoveredCell] = useState(null)
  const heatmapRef = useRef()
  const barRef = useRef()

  const matrixKey = `${selectedLayer}_${selectedHead}`
  const matrix = ATTENTION_WEIGHTS[matrixKey]
  const head = ATTENTION_HEADS.find((h) => h.id === selectedHead)
  const layer = ATTENTION_LAYERS[selectedLayer]

  // --- Heatmap rendering ---
  useEffect(() => {
    const svg = d3.select(heatmapRef.current)
    svg.selectAll('*').remove()

    if (!matrix) return

    const cellSize = 24
    const margin = { top: 90, right: 10, bottom: 10, left: 80 }
    const width = margin.left + cellSize * TOKENS.length + margin.right
    const height = margin.top + cellSize * TOKENS.length + margin.bottom

    svg.attr('width', width).attr('height', height)

    // Background
    svg
      .append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', '#0c1222')
      .attr('rx', 8)

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    // Find max weight for color scale (excluding diagonal for better contrast)
    let maxWeight = 0
    matrix.forEach((row, i) =>
      row.forEach((v, j) => {
        if (i !== j && v > maxWeight) maxWeight = v
      }),
    )
    maxWeight = Math.max(maxWeight, 0.1)

    const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([0, maxWeight])

    // X axis labels (target tokens — rotated)
    TOKENS.forEach((tok, j) => {
      const label = tok.text.length > 6 ? tok.text.slice(0, 6) : tok.text
      g.append('text')
        .attr('x', j * cellSize + cellSize / 2)
        .attr('y', -8)
        .attr('text-anchor', 'start')
        .attr('fill', selectedToken === j ? '#e2e8f0' : '#64748b')
        .attr('font-size', '9')
        .attr('font-family', 'monospace')
        .attr('font-weight', selectedToken === j ? '700' : '400')
        .attr('transform', `rotate(-45, ${j * cellSize + cellSize / 2}, -8)`)
        .text(label)
    })

    // Y axis labels (source tokens)
    TOKENS.forEach((tok, i) => {
      const label = tok.text.length > 8 ? tok.text.slice(0, 8) : tok.text
      g.append('text')
        .attr('x', -6)
        .attr('y', i * cellSize + cellSize / 2 + 3)
        .attr('text-anchor', 'end')
        .attr('fill', selectedToken === i ? '#e2e8f0' : '#64748b')
        .attr('font-size', '9')
        .attr('font-family', 'monospace')
        .attr('font-weight', selectedToken === i ? '700' : '400')
        .text(label)
    })

    // Axis titles
    svg
      .append('text')
      .attr('x', margin.left + (cellSize * TOKENS.length) / 2)
      .attr('y', 12)
      .attr('text-anchor', 'middle')
      .attr('fill', '#475569')
      .attr('font-size', '10')
      .text('Attends TO ↓')

    svg
      .append('text')
      .attr('x', 12)
      .attr('y', margin.top + (cellSize * TOKENS.length) / 2)
      .attr('text-anchor', 'middle')
      .attr('fill', '#475569')
      .attr('font-size', '10')
      .attr('transform', `rotate(-90, 12, ${margin.top + (cellSize * TOKENS.length) / 2})`)
      .text('Attends FROM →')

    // Draw cells
    matrix.forEach((row, i) => {
      row.forEach((value, j) => {
        const isMasked = j > i
        const dimmed = selectedToken !== null && selectedToken !== i

        // Cell rect
        g.append('rect')
          .attr('x', j * cellSize)
          .attr('y', i * cellSize)
          .attr('width', cellSize - 1)
          .attr('height', cellSize - 1)
          .attr('fill', isMasked ? '#0f172a' : colorScale(value))
          .attr('opacity', isMasked ? 0.3 : dimmed ? 0.15 : 1)
          .attr('rx', 2)
          .attr('stroke', hoveredCell?.row === i && hoveredCell?.col === j ? '#e2e8f0' : 'none')
          .attr('stroke-width', 1.5)
          .attr('cursor', isMasked ? 'default' : 'pointer')
          .on('mouseenter', function () {
            if (!isMasked) {
              d3.select(this).attr('stroke', '#e2e8f0').attr('stroke-width', 1.5)
              setHoveredCell({ row: i, col: j, value })
            }
          })
          .on('mouseleave', function () {
            d3.select(this).attr('stroke', 'none')
            setHoveredCell(null)
          })
          .on('click', () => {
            if (!isMasked) setSelectedToken(selectedToken === i ? null : i)
          })

        // Causal mask pattern (diagonal line through masked cells)
        if (isMasked && j === i + 1) {
          // Small "x" in first masked cell per row for visual cue
        }
      })
    })

    // Selected row highlight border
    if (selectedToken !== null) {
      g.append('rect')
        .attr('x', -2)
        .attr('y', selectedToken * cellSize - 2)
        .attr('width', (selectedToken + 1) * cellSize + 3)
        .attr('height', cellSize + 3)
        .attr('fill', 'none')
        .attr('stroke', '#a78bfa')
        .attr('stroke-width', 1.5)
        .attr('rx', 3)
        .attr('opacity', 0.6)
    }

    // Causal mask label
    if (TOKENS.length > 4) {
      const maskX = 4 * cellSize
      const maskY = 1 * cellSize
      g.append('text')
        .attr('x', maskX + cellSize * 3)
        .attr('y', maskY + cellSize)
        .attr('text-anchor', 'middle')
        .attr('fill', '#334155')
        .attr('font-size', '8')
        .attr('font-style', 'italic')
        .text('Causal mask')
    }
  }, [matrix, selectedToken, hoveredCell, selectedLayer, selectedHead])

  // --- Attention bar chart for selected token ---
  useEffect(() => {
    const svg = d3.select(barRef.current)
    svg.selectAll('*').remove()

    if (selectedToken === null || !matrix) return

    const row = matrix[selectedToken]

    // Build labels — only disambiguate tokens that appear more than once
    const textCounts = {}
    TOKENS.forEach((t) => {
      textCounts[t.text] = (textCounts[t.text] || 0) + 1
    })

    const barData = row
      .map((weight, j) => ({
        token: TOKENS[j].text,
        label: textCounts[TOKENS[j].text] > 1 ? `${TOKENS[j].text} (${j})` : TOKENS[j].text,
        weight,
        index: j,
      }))
      .filter((d) => d.index <= selectedToken && d.weight > 0.005) // non-masked, non-zero
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 8)

    const margin = { top: 20, right: 15, bottom: 10, left: 70 }
    const width = 320
    const barHeight = 18
    const height = margin.top + barData.length * barHeight + margin.bottom

    svg.attr('width', width).attr('height', height)

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    const x = d3
      .scaleLinear()
      .domain([0, d3.max(barData, (d) => d.weight) * 1.15])
      .range([0, width - margin.left - margin.right])

    const y = d3
      .scaleBand()
      .domain(barData.map((d) => d.label))
      .range([0, barData.length * barHeight])
      .padding(0.2)

    // Title
    svg
      .append('text')
      .attr('x', width / 2)
      .attr('y', 12)
      .attr('text-anchor', 'middle')
      .attr('fill', '#94a3b8')
      .attr('font-size', '10')
      .text(`"${TOKENS[selectedToken].text}" attends to:`)

    // Y labels
    g.append('g')
      .call(d3.axisLeft(y).tickSize(0))
      .selectAll('text')
      .attr('fill', '#94a3b8')
      .attr('font-size', '9')
      .attr('font-family', 'monospace')
    g.selectAll('.domain').attr('stroke', '#334155')

    // Bars
    g.selectAll('.att-bar')
      .data(barData)
      .join('rect')
      .attr('class', 'att-bar')
      .attr('x', 0)
      .attr('y', (d) => y(d.label))
      .attr('width', 0)
      .attr('height', y.bandwidth())
      .attr('fill', '#8b5cf6')
      .attr('opacity', 0.8)
      .attr('rx', 2)
      .transition()
      .duration(400)
      .attr('width', (d) => x(d.weight))

    // Weight labels
    g.selectAll('.att-label')
      .data(barData)
      .join('text')
      .attr('class', 'att-label')
      .attr('x', (d) => x(d.weight) + 4)
      .attr('y', (d) => y(d.label) + y.bandwidth() / 2 + 3)
      .attr('fill', '#64748b')
      .attr('font-size', '8')
      .text((d) => `${(d.weight * 100).toFixed(1)}%`)
      .attr('opacity', 0)
      .transition()
      .delay(400)
      .attr('opacity', 1)
  }, [selectedToken, matrix])

  // Build insight text
  const insightKey = `${selectedHead}_${selectedToken}`
  const insight = ATTENTION_INSIGHTS[insightKey]

  return (
    <div className="space-y-4">
      {/* Introduction */}
      <p className="text-sm text-slate-400">
        This heatmap shows attention weights for a storage I/O classification prompt. Each cell
        shows how much one token "pays attention to" another. Click a token to highlight its
        attention row. Switch heads and layers to see different patterns.
      </p>

      {/* Token pills */}
      <div>
        <p className="text-xs text-slate-500 mb-2">
          Click a token to explore its attention pattern:
        </p>
        <div className="flex flex-wrap gap-1">
          {TOKENS.map((tok) => {
            const colors = TOKEN_COLORS[tok.type]
            const isSelected = selectedToken === tok.id
            return (
              <button
                key={tok.id}
                onClick={() => setSelectedToken(isSelected ? null : tok.id)}
                className={`px-2 py-1 text-xs font-mono rounded border transition-all duration-200 ${
                  isSelected
                    ? 'ring-2 ring-violet-400 bg-violet-900/50 border-violet-500 text-white'
                    : `${colors.bg} ${colors.border} ${colors.text} hover:brightness-125`
                }`}
                title={TOKEN_TYPE_LABELS[tok.type]}
              >
                {tok.text}
              </button>
            )
          })}
        </div>
        <div className="flex gap-3 mt-2">
          {Object.entries(TOKEN_TYPE_LABELS).map(([type, label]) => (
            <span key={type} className="flex items-center gap-1 text-xs">
              <span
                className={`w-2 h-2 rounded-sm ${TOKEN_COLORS[type].bg} ${TOKEN_COLORS[type].border} border`}
              />
              <span className="text-slate-500">{label}</span>
            </span>
          ))}
        </div>
      </div>

      {/* Controls: Head + Layer selectors */}
      <div className="flex flex-wrap gap-4">
        <div>
          <p className="text-xs text-slate-500 mb-1.5">Attention Head</p>
          <div className="flex gap-1">
            {ATTENTION_HEADS.map((h) => (
              <button
                key={h.id}
                onClick={() => setSelectedHead(h.id)}
                className={`px-3 py-1.5 text-xs rounded-md transition-colors ${
                  selectedHead === h.id
                    ? 'bg-violet-600 text-white font-semibold'
                    : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                }`}
              >
                {h.label}
              </button>
            ))}
          </div>
        </div>
        <div>
          <p className="text-xs text-slate-500 mb-1.5">Layer</p>
          <div className="flex gap-1">
            {ATTENTION_LAYERS.map((l) => (
              <button
                key={l.id}
                onClick={() => setSelectedLayer(l.id)}
                className={`px-3 py-1.5 text-xs rounded-md transition-colors ${
                  selectedLayer === l.id
                    ? 'bg-violet-600 text-white font-semibold'
                    : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                }`}
              >
                {l.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Heatmap + Bar chart side by side */}
      <div className="flex flex-wrap gap-4 items-start">
        <div className="overflow-auto">
          <svg ref={heatmapRef} />
        </div>
        {selectedToken !== null && (
          <div className="min-w-[320px]">
            <svg ref={barRef} />
          </div>
        )}
      </div>

      {/* Hover tooltip */}
      {hoveredCell && hoveredCell.value !== undefined && (
        <div className="text-xs text-slate-400">
          <span className="font-mono text-slate-300">
            "{TOKENS[hoveredCell.row].text}" → "{TOKENS[hoveredCell.col].text}"
          </span>
          : attention weight ={' '}
          <span className="text-violet-300 font-semibold">
            {(hoveredCell.value * 100).toFixed(1)}%
          </span>
        </div>
      )}

      {/* Insight panel */}
      <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <div className="flex items-start gap-2">
          <span className="text-violet-400 text-sm mt-0.5">💡</span>
          <div>
            <p className="text-xs font-semibold text-violet-400 mb-1">
              {head?.label} — {layer?.label}
            </p>
            <p className="text-sm text-slate-300 leading-relaxed">
              {insight ||
                (selectedToken !== null
                  ? `Token "${TOKENS[selectedToken].text}" — ${head?.description}`
                  : head?.description +
                    ' Try clicking a token above to see specific attention patterns.')}
            </p>
            {selectedToken !== null && (
              <p className="text-xs text-slate-500 mt-2">
                Layer progression: try switching layers to see how this pattern evolves from surface
                patterns (early) to task reasoning (late).
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
