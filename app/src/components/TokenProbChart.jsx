import { useRef, useEffect } from 'react'
import * as d3 from 'd3'

// Horizontal bar chart showing token probability distributions
// This is the key "under the covers" visualization that shows
// how post-training shifts model confidence

export default function TokenProbChart({
  data,           // Array of { token, probability }
  comparisonData, // Optional: second distribution for overlay comparison
  label = 'Token Probabilities',
  comparisonLabel = 'After',
  highlightToken = null, // Token to highlight (e.g., the correct answer)
  width = 500,
  height = 350,
}) {
  const svgRef = useRef()

  useEffect(() => {
    if (!data || data.length === 0) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const margin = { top: comparisonData ? 46 : 30, right: 20, bottom: 20, left: 100 }
    const w = width - margin.left - margin.right
    const h = height - margin.top - margin.bottom

    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Take top 15 tokens, sorted by probability
    const topTokens = [...data]
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 15)

    // Y scale: token labels
    const y = d3.scaleBand()
      .domain(topTokens.map(d => d.token))
      .range([0, h])
      .padding(comparisonData ? 0.25 : 0.3)

    // X scale: probability
    const maxProb = Math.max(
      d3.max(topTokens, d => d.probability),
      comparisonData ? d3.max(comparisonData, d => d.probability) : 0
    )
    const x = d3.scaleLinear()
      .domain([0, Math.min(maxProb * 1.2, 1)])
      .range([0, w])

    // Title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .attr('fill', '#94a3b8')
      .attr('font-size', '12')
      .attr('font-weight', '600')
      .text(label)

    // Y axis (token labels)
    g.append('g')
      .call(d3.axisLeft(y).tickSize(0))
      .selectAll('text')
      .attr('fill', d => d === highlightToken ? '#22c55e' : '#cbd5e1')
      .attr('font-size', '11')
      .attr('font-family', 'monospace')
      .attr('font-weight', d => d === highlightToken ? '700' : '400')

    g.selectAll('.domain').attr('stroke', '#334155')

    // X axis (probability)
    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(5).tickFormat(d3.format('.0%')))
      .selectAll('text')
      .attr('fill', '#64748b')
      .attr('font-size', '9')

    // Grid lines
    g.selectAll('.grid-line')
      .data(x.ticks(5))
      .join('line')
      .attr('class', 'grid-line')
      .attr('x1', d => x(d))
      .attr('x2', d => x(d))
      .attr('y1', 0)
      .attr('y2', h)
      .attr('stroke', '#1e293b')
      .attr('stroke-dasharray', '2,2')

    if (comparisonData) {
      // Comparison mode: two bars per token
      const compMap = new Map(comparisonData.map(d => [d.token, d.probability]))
      const barH = y.bandwidth() / 2

      // "Before" bars
      g.selectAll('.bar-before')
        .data(topTokens)
        .join('rect')
        .attr('class', 'bar-before')
        .attr('x', 0)
        .attr('y', d => y(d.token))
        .attr('width', 0)
        .attr('height', barH - 1)
        .attr('fill', '#ef4444')
        .attr('opacity', 0.6)
        .attr('rx', 2)
        .transition()
        .duration(600)
        .attr('width', d => x(d.probability))

      // "After" bars
      g.selectAll('.bar-after')
        .data(topTokens)
        .join('rect')
        .attr('class', 'bar-after')
        .attr('x', 0)
        .attr('y', d => y(d.token) + barH)
        .attr('width', 0)
        .attr('height', barH - 1)
        .attr('fill', d => d.token === highlightToken ? '#22c55e' : '#3b82f6')
        .attr('opacity', 0.8)
        .attr('rx', 2)
        .transition()
        .duration(600)
        .delay(300)
        .attr('width', d => x(compMap.get(d.token) || 0))

      // Probability labels for "after"
      g.selectAll('.prob-after')
        .data(topTokens)
        .join('text')
        .attr('class', 'prob-after')
        .attr('x', d => x(compMap.get(d.token) || 0) + 4)
        .attr('y', d => y(d.token) + barH + barH / 2 + 3)
        .attr('fill', '#94a3b8')
        .attr('font-size', '9')
        .text(d => {
          const p = compMap.get(d.token)
          return p ? `${(p * 100).toFixed(1)}%` : ''
        })
        .attr('opacity', 0)
        .transition()
        .delay(900)
        .attr('opacity', 1)

      // Legend — positioned below the title with breathing room
      const legend = svg.append('g').attr('transform', `translate(${margin.left + 10}, ${margin.top - 12})`)
      legend.append('rect').attr('width', 10).attr('height', 10).attr('fill', '#ef4444').attr('opacity', 0.6).attr('rx', 2)
      legend.append('text').attr('x', 14).attr('y', 9).attr('fill', '#94a3b8').attr('font-size', '9').text('Before (Zero-Shot)')
      legend.append('rect').attr('x', 120).attr('width', 10).attr('height', 10).attr('fill', '#3b82f6').attr('opacity', 0.8).attr('rx', 2)
      legend.append('text').attr('x', 134).attr('y', 9).attr('fill', '#94a3b8').attr('font-size', '9').text('After (Few-Shot)')
    } else {
      // Single distribution
      g.selectAll('.bar')
        .data(topTokens)
        .join('rect')
        .attr('class', 'bar')
        .attr('x', 0)
        .attr('y', d => y(d.token))
        .attr('width', 0)
        .attr('height', y.bandwidth())
        .attr('fill', d => d.token === highlightToken ? '#22c55e' : '#3b82f6')
        .attr('opacity', 0.8)
        .attr('rx', 2)
        .transition()
        .duration(600)
        .attr('width', d => x(d.probability))

      // Probability labels
      g.selectAll('.prob-label')
        .data(topTokens)
        .join('text')
        .attr('class', 'prob-label')
        .attr('x', d => x(d.probability) + 4)
        .attr('y', d => y(d.token) + y.bandwidth() / 2 + 4)
        .attr('fill', '#94a3b8')
        .attr('font-size', '9')
        .text(d => `${(d.probability * 100).toFixed(1)}%`)
        .attr('opacity', 0)
        .transition()
        .delay(600)
        .attr('opacity', 1)
    }
  }, [data, comparisonData, highlightToken, label, comparisonLabel, width, height])

  return <svg ref={svgRef} />
}
