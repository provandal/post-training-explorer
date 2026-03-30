import { useRef, useEffect } from 'react'
import * as d3 from 'd3'

// Training loss curve visualization
// Shows how loss decreases over training steps

export default function LossChart({
  data,
  label = 'Training Loss',
  width = 450,
  height = 200,
  color = '#3b82f6',
}) {
  const svgRef = useRef()

  useEffect(() => {
    if (!data || data.length === 0) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const margin = { top: 25, right: 20, bottom: 35, left: 50 }
    const w = width - margin.left - margin.right
    const h = height - margin.top - margin.bottom

    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    const x = d3
      .scaleLinear()
      .domain(d3.extent(data, (d) => d.step))
      .range([0, w])

    const y = d3
      .scaleLinear()
      .domain([0, d3.max(data, (d) => d.loss) * 1.1])
      .range([h, 0])

    // Title
    svg
      .append('text')
      .attr('x', width / 2)
      .attr('y', 14)
      .attr('text-anchor', 'middle')
      .attr('fill', '#94a3b8')
      .attr('font-size', '11')
      .attr('font-weight', '600')
      .text(label)

    // Grid lines
    g.selectAll('.grid-y')
      .data(y.ticks(4))
      .join('line')
      .attr('x1', 0)
      .attr('x2', w)
      .attr('y1', (d) => y(d))
      .attr('y2', (d) => y(d))
      .attr('stroke', '#1e293b')
      .attr('stroke-dasharray', '2,2')

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(6))
      .selectAll('text')
      .attr('fill', '#64748b')
      .attr('font-size', '9')

    g.append('g')
      .call(d3.axisLeft(y).ticks(4))
      .selectAll('text')
      .attr('fill', '#64748b')
      .attr('font-size', '9')

    g.selectAll('.domain').attr('stroke', '#334155')

    // Axis labels
    g.append('text')
      .attr('x', w / 2)
      .attr('y', h + 30)
      .attr('text-anchor', 'middle')
      .attr('fill', '#64748b')
      .attr('font-size', '9')
      .text('Training Step')

    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -h / 2)
      .attr('y', -35)
      .attr('text-anchor', 'middle')
      .attr('fill', '#64748b')
      .attr('font-size', '9')
      .text('Loss')

    // Area fill
    const area = d3
      .area()
      .x((d) => x(d.step))
      .y0(h)
      .y1((d) => y(d.loss))
      .curve(d3.curveMonotoneX)

    g.append('path').datum(data).attr('fill', color).attr('opacity', 0.1).attr('d', area)

    // Line
    const line = d3
      .line()
      .x((d) => x(d.step))
      .y((d) => y(d.loss))
      .curve(d3.curveMonotoneX)

    const path = g
      .append('path')
      .datum(data)
      .attr('fill', 'none')
      .attr('stroke', color)
      .attr('stroke-width', 2)
      .attr('d', line)

    // Animate the line drawing
    const totalLength = path.node().getTotalLength()
    path
      .attr('stroke-dasharray', totalLength)
      .attr('stroke-dashoffset', totalLength)
      .transition()
      .duration(1500)
      .ease(d3.easeLinear)
      .attr('stroke-dashoffset', 0)
  }, [data, label, width, height, color])

  return <svg ref={svgRef} />
}
