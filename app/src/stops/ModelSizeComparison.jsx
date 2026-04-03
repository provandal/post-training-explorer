import { useState, useRef, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import * as d3 from 'd3'
import SectionTabs from '../components/SectionTabs'
import { isLoaded, getModelSizeComparison } from '../data/loadArtifacts'

// ---------------------------------------------------------------------------
// Fallback data — works without precomputed_results.json
// ---------------------------------------------------------------------------
const FALLBACK_COMPARISON = {
  models: ['360M', '1.7B'],
  accuracy_by_technique: {
    base: { '360M': 0.0, '1.7B': 0.1 },
    sft: { '360M': 0.65, '1.7B': 0.8 },
    dpo: { '360M': 0.6, '1.7B': 0.75 },
    grpo: { '360M': 0.85, '1.7B': 0.95 },
  },
  training_time_minutes: {
    sft: { '360M': 12, '1.7B': 55 },
    dpo: { '360M': 8, '1.7B': 35 },
    grpo: { '360M': 35, '1.7B': 150 },
  },
  gpu_memory_gb: { '360M': 3.2, '1.7B': 12.5 },
  head_to_head: [
    {
      prompt_id: 0,
      prompt_snippet:
        'IOPS: 45,000 | Throughput: 180 MB/s | Latency: 850 us | R/W: 72/28 | Random: 93% | Block: 8 KB | QD: 64',
      true_label: 'OLTP Database',
      results: {
        '360M': {
          sft: {
            predicted: 'OLTP Database',
            text: 'Classification: OLTP Database\nReason: High random IOPS with small block sizes indicate transactional database operations.',
          },
          grpo: {
            predicted: 'OLTP Database',
            text: 'Classification: OLTP Database\nReason: Small block random reads dominate, consistent with index lookups and row fetches.',
          },
        },
        '1.7B': {
          sft: {
            predicted: 'OLTP Database',
            text: 'Classification: OLTP Database\nReason: High IOPS with predominantly random, small-block I/O and deep queue depth are hallmarks of transactional database workloads.',
          },
          grpo: {
            predicted: 'OLTP Database',
            text: 'Classification: OLTP Database\nReason: Random small-block I/O at 93% with high IOPS and deep queue depth is characteristic of OLTP database operations.',
          },
        },
      },
    },
    {
      prompt_id: 1,
      prompt_snippet:
        'IOPS: 1,200 | Throughput: 2,800 MB/s | Latency: 8,500 us | R/W: 95/5 | Random: 12% | Block: 512 KB | QD: 8',
      true_label: 'OLAP Analytics',
      results: {
        '360M': {
          sft: {
            predicted: 'OLAP Analytics',
            text: 'Classification: OLAP Analytics\nReason: Large sequential reads with high throughput indicate analytical table scans.',
          },
          grpo: {
            predicted: 'OLAP Analytics',
            text: 'Classification: OLAP Analytics\nReason: Predominantly sequential read pattern with large blocks suggests data warehouse queries.',
          },
        },
        '1.7B': {
          sft: {
            predicted: 'OLAP Analytics',
            text: 'Classification: OLAP Analytics\nReason: Very high throughput with large sequential reads and minimal writes indicate heavy analytical or data warehouse query processing.',
          },
          grpo: {
            predicted: 'OLAP Analytics',
            text: 'Classification: OLAP Analytics\nReason: Large block sequential reads at high throughput with read-dominant access pattern indicate analytical query processing.',
          },
        },
      },
    },
    {
      prompt_id: 2,
      prompt_snippet:
        'IOPS: 3,500 | Throughput: 4,200 MB/s | Latency: 3,000 us | R/W: 96/4 | Random: 35% | Block: 256 KB | QD: 32',
      true_label: 'AI ML Training',
      results: {
        '360M': {
          sft: {
            predicted: 'Video Streaming',
            text: 'Classification: Video Streaming\nReason: Sequential reads with large block sizes indicate media streaming.',
          },
          grpo: {
            predicted: 'AI ML Training',
            text: 'Classification: AI ML Training\nReason: High throughput reads with mixed access patterns suggest training data pipeline loading.',
          },
        },
        '1.7B': {
          sft: {
            predicted: 'AI ML Training',
            text: 'Classification: AI ML Training\nReason: Very high throughput with large block reads and mixed access patterns indicate GPU training data ingestion.',
          },
          grpo: {
            predicted: 'AI ML Training',
            text: 'Classification: AI ML Training\nReason: High throughput reads with mixed sequential/random access and large blocks are characteristic of ML training data pipelines.',
          },
        },
      },
    },
    {
      prompt_id: 3,
      prompt_snippet:
        'IOPS: 500 | Throughput: 1,500 MB/s | Latency: 6,000 us | R/W: 97/3 | Random: 18% | Block: 1024 KB | QD: 4',
      true_label: 'Video Streaming',
      results: {
        '360M': {
          sft: {
            predicted: 'OLAP Analytics',
            text: 'Classification: OLAP Analytics\nReason: Large sequential reads with high throughput indicate analytical table scans.',
          },
          grpo: {
            predicted: 'Video Streaming',
            text: 'Classification: Video Streaming\nReason: Steady sequential reads with large block sizes indicate media streaming operations.',
          },
        },
        '1.7B': {
          sft: {
            predicted: 'Video Streaming',
            text: 'Classification: Video Streaming\nReason: Low IOPS with high throughput sequential reads and very large blocks indicate streaming media delivery.',
          },
          grpo: {
            predicted: 'Video Streaming',
            text: 'Classification: Video Streaming\nReason: Sequential reads with very large blocks, low queue depth, and read-dominant pattern indicate video streaming.',
          },
        },
      },
    },
    {
      prompt_id: 4,
      prompt_snippet:
        'IOPS: 15,000 | Throughput: 90 MB/s | Latency: 1,200 us | R/W: 58/42 | Random: 88% | Block: 8 KB | QD: 32',
      true_label: 'VDI Virtual Desktop',
      results: {
        '360M': {
          sft: {
            predicted: 'OLTP Database',
            text: 'Classification: OLTP Database\nReason: High random IOPS with small block sizes indicate transactional database operations.',
          },
          grpo: {
            predicted: 'VDI Virtual Desktop',
            text: 'Classification: VDI Virtual Desktop\nReason: Mixed read/write random I/O with small blocks indicates virtual desktop user activity.',
          },
        },
        '1.7B': {
          sft: {
            predicted: 'VDI Virtual Desktop',
            text: 'Classification: VDI Virtual Desktop\nReason: Balanced read/write ratio with high random IOPS and small blocks indicate multiple concurrent desktop sessions.',
          },
          grpo: {
            predicted: 'VDI Virtual Desktop',
            text: 'Classification: VDI Virtual Desktop\nReason: Mixed read-write random small-block I/O with moderate IOPS suggests concurrent virtual desktop sessions.',
          },
        },
      },
    },
  ],
}

// ---------------------------------------------------------------------------
// Tab definitions
// ---------------------------------------------------------------------------
// Tab definitions are inside the component to access t()

// ---------------------------------------------------------------------------
// Tab 1: Why Size Matters
// ---------------------------------------------------------------------------
function WhySizeMatters({ data, t }) {
  return (
    <div className="space-y-6">
      <div className="p-5 rounded-lg bg-slate-800/40 border border-slate-700/50">
        <h3 className="text-lg font-semibold text-cyan-400 mb-3">
          {t('deepdive.modelSize.whySizeMatters.heading')}
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-4">
          {t('deepdive.modelSize.whySizeMatters.p1')}
        </p>
        <p className="text-sm text-slate-300 leading-relaxed">
          {t('deepdive.modelSize.whySizeMatters.p2')}
        </p>
      </div>

      {/* Stat callout cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 rounded-lg bg-cyan-950/20 border border-cyan-800/30 text-center">
          <div className="text-3xl font-bold text-cyan-400 mb-1">360M</div>
          <div className="text-xs text-slate-400 mb-2">{t('deepdive.modelSize.parameters')}</div>
          <div className="space-y-1 text-sm text-slate-300">
            <div>~{data.gpu_memory_gb?.['360M'] ?? 3.2} GB VRAM</div>
            <div>~{data.training_time_minutes?.sft?.['360M'] ?? 12} min SFT training</div>
          </div>
        </div>
        <div className="p-4 rounded-lg bg-cyan-950/20 border border-cyan-800/30 text-center">
          <div className="text-3xl font-bold text-cyan-400 mb-1">1.7B</div>
          <div className="text-xs text-slate-400 mb-2">{t('deepdive.modelSize.parameters')}</div>
          <div className="space-y-1 text-sm text-slate-300">
            <div>~{data.gpu_memory_gb?.['1.7B'] ?? 12.5} GB VRAM</div>
            <div>~{data.training_time_minutes?.sft?.['1.7B'] ?? 55} min SFT training</div>
          </div>
        </div>
        <div className="p-4 rounded-lg bg-slate-800/40 border border-slate-700/50 text-center">
          <div className="text-3xl font-bold text-emerald-400 mb-1">
            +
            {Math.round(
              ((data.accuracy_by_technique?.grpo?.['1.7B'] ?? 0.95) -
                (data.accuracy_by_technique?.grpo?.['360M'] ?? 0.85)) *
                100,
            )}
            %
          </div>
          <div className="text-xs text-slate-400 mb-2">{t('deepdive.modelSize.accuracyDelta')}</div>
          <div className="text-sm text-slate-300">{t('deepdive.modelSize.biggerBetter')}</div>
        </div>
      </div>

      {/* Infrastructure implications */}
      <div className="p-5 rounded-lg bg-slate-800/40 border border-slate-700/50">
        <h3 className="text-base font-semibold text-cyan-400 mb-3">
          {t('deepdive.modelSize.infraImplications')}
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          {t('deepdive.modelSize.infraP1')}
        </p>
        <p className="text-sm text-slate-300 leading-relaxed">{t('deepdive.modelSize.infraP2')}</p>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tab 2: Head-to-Head
// ---------------------------------------------------------------------------
function HeadToHead({ data, t }) {
  const [promptIdx, setPromptIdx] = useState(0)
  const [technique, setTechnique] = useState('sft')

  const prompts = data.head_to_head ?? []
  const current = prompts[promptIdx] ?? null

  if (!current) {
    return <p className="text-sm text-slate-500">{t('deepdive.modelSize.headToHead.noData')}</p>
  }

  const results360 = current.results?.['360M']?.[technique]
  const results17B = current.results?.['1.7B']?.[technique]
  const correct360 = results360?.predicted === current.true_label
  const correct17B = results17B?.predicted === current.true_label

  return (
    <div className="space-y-5">
      {/* Prompt selector */}
      <div>
        <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-1.5">
          {t('deepdive.modelSize.headToHead.testPattern')}
        </label>
        <div className="flex flex-wrap gap-2">
          {prompts.map((p, i) => (
            <button
              key={p.prompt_id}
              onClick={() => setPromptIdx(i)}
              className={`px-3 py-1.5 text-xs rounded-md transition-colors ${
                promptIdx === i
                  ? 'bg-cyan-600 text-white font-semibold'
                  : 'bg-slate-700 hover:bg-slate-600 text-slate-300'
              }`}
            >
              #{p.prompt_id + 1} — {p.true_label}
            </button>
          ))}
        </div>
      </div>

      {/* Input prompt */}
      <div className="bg-slate-800 border border-slate-600 rounded-lg p-3 font-mono text-sm text-slate-200">
        {current.prompt_snippet}
      </div>
      <p className="text-xs text-slate-500">
        {t('deepdive.modelSize.headToHead.groundTruth')}{' '}
        <span className="text-cyan-400 font-semibold">{current.true_label}</span>
      </p>

      {/* Technique toggle */}
      <div className="flex gap-2">
        {['sft', 'grpo'].map((tech) => (
          <button
            key={tech}
            onClick={() => setTechnique(tech)}
            className={`px-4 py-2 text-sm rounded-md transition-colors ${
              technique === tech
                ? 'bg-cyan-600 text-white font-semibold'
                : 'bg-slate-700 hover:bg-slate-600 text-slate-300'
            }`}
          >
            {tech.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Side-by-side comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* 360M column */}
        <div
          className={`p-4 rounded-lg border ${
            correct360
              ? 'border-emerald-700/50 bg-emerald-950/15'
              : 'border-red-800/50 bg-red-950/15'
          }`}
        >
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-semibold text-blue-400">SmolLM2-360M</span>
            <span
              className={`text-xs px-2 py-0.5 rounded-full font-semibold ${
                correct360 ? 'bg-emerald-900/50 text-emerald-400' : 'bg-red-900/50 text-red-400'
              }`}
            >
              {correct360
                ? t('deepdive.modelSize.headToHead.correct')
                : t('deepdive.modelSize.headToHead.incorrect')}
            </span>
          </div>
          <div className="text-xs text-slate-400 mb-1">
            Predicted:{' '}
            <span className={correct360 ? 'text-emerald-300' : 'text-red-300'}>
              {results360?.predicted ?? 'N/A'}
            </span>
          </div>
          <pre className="text-xs font-mono whitespace-pre-wrap leading-relaxed text-slate-200 mt-2 p-3 rounded bg-slate-800/60">
            {results360?.text ?? t('deepdive.modelSize.headToHead.noOutput')}
          </pre>
        </div>

        {/* 1.7B column */}
        <div
          className={`p-4 rounded-lg border ${
            correct17B
              ? 'border-emerald-700/50 bg-emerald-950/15'
              : 'border-red-800/50 bg-red-950/15'
          }`}
        >
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-semibold text-cyan-400">SmolLM2-1.7B</span>
            <span
              className={`text-xs px-2 py-0.5 rounded-full font-semibold ${
                correct17B ? 'bg-emerald-900/50 text-emerald-400' : 'bg-red-900/50 text-red-400'
              }`}
            >
              {correct17B
                ? t('deepdive.modelSize.headToHead.correct')
                : t('deepdive.modelSize.headToHead.incorrect')}
            </span>
          </div>
          <div className="text-xs text-slate-400 mb-1">
            Predicted:{' '}
            <span className={correct17B ? 'text-emerald-300' : 'text-red-300'}>
              {results17B?.predicted ?? 'N/A'}
            </span>
          </div>
          <pre className="text-xs font-mono whitespace-pre-wrap leading-relaxed text-slate-200 mt-2 p-3 rounded bg-slate-800/60">
            {results17B?.text ?? t('deepdive.modelSize.headToHead.noOutput')}
          </pre>
        </div>
      </div>

      {/* Insight */}
      <div className="p-3 rounded-lg bg-cyan-950/20 border border-cyan-800/30">
        <p className="text-xs text-slate-400 leading-relaxed">
          <strong className="text-cyan-300">Notice:</strong> The 1.7B model tends to produce more
          detailed reasoning and is less likely to confuse similar workload categories. On ambiguous
          patterns (like AI/ML vs Video Streaming), the extra capacity helps the larger model
          distinguish subtle differences in the metrics.
        </p>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tab 3: Scaling Analysis (D3 grouped bar chart)
// ---------------------------------------------------------------------------
function ScalingAnalysis({ data, t }) {
  const svgRef = useRef()

  const techniques = ['base', 'sft', 'dpo', 'grpo']
  const techniqueLabels = { base: 'Base', sft: 'SFT', dpo: 'DPO', grpo: 'GRPO' }
  const models = data.models ?? ['360M', '1.7B']
  const modelColors = { '360M': '#3b82f6', '1.7B': '#06b6d4' }

  useEffect(() => {
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const width = 550
    const height = 300
    const margin = { top: 40, right: 30, bottom: 50, left: 55 }
    const w = width - margin.left - margin.right
    const h = height - margin.top - margin.bottom

    svg.attr('width', width).attr('height', height)
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    // Scales
    const x0 = d3.scaleBand().domain(techniques).range([0, w]).padding(0.25)

    const x1 = d3.scaleBand().domain(models).range([0, x0.bandwidth()]).padding(0.1)

    const y = d3.scaleLinear().domain([0, 100]).range([h, 0])

    // Title
    svg
      .append('text')
      .attr('x', width / 2)
      .attr('y', 18)
      .attr('text-anchor', 'middle')
      .attr('fill', '#94a3b8')
      .attr('font-size', '12')
      .attr('font-weight', '600')
      .text('Accuracy by Technique and Model Size')

    // X axis
    g.append('g')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(x0).tickFormat((d) => techniqueLabels[d]))
      .call((g) => g.select('.domain').attr('stroke', '#475569'))
      .call((g) => g.selectAll('.tick line').attr('stroke', '#475569'))
      .call((g) => g.selectAll('.tick text').attr('fill', '#94a3b8').attr('font-size', '11'))

    // X axis label
    g.append('text')
      .attr('x', w / 2)
      .attr('y', h + 40)
      .attr('text-anchor', 'middle')
      .attr('fill', '#64748b')
      .attr('font-size', '10')
      .text('Post-Training Technique')

    // Y axis
    g.append('g')
      .call(
        d3
          .axisLeft(y)
          .ticks(5)
          .tickFormat((d) => `${d}%`),
      )
      .call((g) => g.select('.domain').attr('stroke', '#475569'))
      .call((g) => g.selectAll('.tick line').attr('stroke', '#475569'))
      .call((g) => g.selectAll('.tick text').attr('fill', '#94a3b8').attr('font-size', '10'))

    // Y axis label
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -h / 2)
      .attr('y', -42)
      .attr('text-anchor', 'middle')
      .attr('fill', '#64748b')
      .attr('font-size', '10')
      .text('Accuracy (%)')

    // Grid lines
    g.append('g')
      .call(d3.axisLeft(y).ticks(5).tickSize(-w).tickFormat(''))
      .call((g) => g.select('.domain').remove())
      .call((g) =>
        g.selectAll('.tick line').attr('stroke', '#334155').attr('stroke-dasharray', '2,2'),
      )

    // Bars
    techniques.forEach((tech) => {
      models.forEach((model) => {
        const accuracy = (data.accuracy_by_technique?.[tech]?.[model] ?? 0) * 100
        g.append('rect')
          .attr('x', x0(tech) + x1(model))
          .attr('y', y(0))
          .attr('width', x1.bandwidth())
          .attr('height', 0)
          .attr('fill', modelColors[model])
          .attr('opacity', 0.85)
          .attr('rx', 3)
          .transition()
          .duration(600)
          .delay(techniques.indexOf(tech) * 120)
          .attr('y', y(accuracy))
          .attr('height', h - y(accuracy))

        // Value label on top of bar
        g.append('text')
          .attr('x', x0(tech) + x1(model) + x1.bandwidth() / 2)
          .attr('y', y(accuracy) - 5)
          .attr('text-anchor', 'middle')
          .attr('fill', modelColors[model])
          .attr('font-size', '9')
          .attr('font-weight', '600')
          .attr('opacity', 0)
          .text(`${Math.round(accuracy)}%`)
          .transition()
          .delay(techniques.indexOf(tech) * 120 + 600)
          .attr('opacity', 1)
      })
    })

    // Legend
    const legend = svg
      .append('g')
      .attr('transform', `translate(${width - 140}, ${margin.top - 20})`)
    models.forEach((model, i) => {
      const lg = legend.append('g').attr('transform', `translate(${i * 70}, 0)`)
      lg.append('rect')
        .attr('width', 12)
        .attr('height', 12)
        .attr('rx', 2)
        .attr('fill', modelColors[model])
        .attr('opacity', 0.85)
      lg.append('text')
        .attr('x', 16)
        .attr('y', 10)
        .attr('fill', '#94a3b8')
        .attr('font-size', '10')
        .text(model)
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data])

  // Training time data for the table below the chart
  const trainingTechniques = ['sft', 'dpo', 'grpo']

  return (
    <div className="space-y-6">
      {/* D3 Chart */}
      <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <svg ref={svgRef} />
        <p className="text-xs text-slate-500 mt-3">
          Across all techniques, the 1.7B model consistently outperforms the 360M model. The gap
          narrows with GRPO, suggesting that reinforcement learning with verifiable rewards helps
          smaller models punch above their weight.
        </p>
      </div>

      {/* Training time comparison */}
      <div className="p-5 rounded-lg bg-slate-800/40 border border-slate-700/50">
        <h4 className="text-sm font-semibold text-cyan-400 mb-3">
          {t('deepdive.modelSize.scaling.trainingTimeComparison')}
        </h4>
        <div className="grid grid-cols-3 gap-4">
          {trainingTechniques.map((tech) => {
            const time360 = data.training_time_minutes?.[tech]?.['360M'] ?? '—'
            const time17B = data.training_time_minutes?.[tech]?.['1.7B'] ?? '—'
            const multiplier =
              typeof time360 === 'number' && typeof time17B === 'number'
                ? `${(time17B / time360).toFixed(1)}x`
                : '—'
            return (
              <div
                key={tech}
                className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30 text-center"
              >
                <div className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
                  {tech.toUpperCase()}
                </div>
                <div className="flex items-center justify-center gap-3">
                  <div>
                    <div className="text-lg font-bold text-blue-400">{time360}</div>
                    <div className="text-xs text-slate-500">360M</div>
                  </div>
                  <div className="text-slate-600">vs</div>
                  <div>
                    <div className="text-lg font-bold text-cyan-400">{time17B}</div>
                    <div className="text-xs text-slate-500">1.7B</div>
                  </div>
                </div>
                <div className="text-xs text-slate-500 mt-1">{multiplier} longer</div>
              </div>
            )
          })}
        </div>
        <p className="text-xs text-slate-500 mt-3">
          GRPO on 1.7B takes ~150 minutes — generating 8 completions per prompt with a larger model
          is compute-intensive. For infrastructure teams, this means planning for sustained GPU
          reservations and higher I/O throughput during training data loading.
        </p>
      </div>
    </div>
  )
}

// ===========================================================================
// Main component
// ===========================================================================
export default function ModelSizeComparison() {
  const { t } = useTranslation()
  const [activeTab, setActiveTab] = useState('why')

  // Use real data if loaded, otherwise fallback
  const data = (isLoaded() && getModelSizeComparison()) || FALLBACK_COMPARISON

  const TABS = [
    { id: 'why', label: t('tabs.whySizeMatters') },
    { id: 'headtohead', label: t('tabs.headToHead') },
    { id: 'scaling', label: t('tabs.scalingAnalysis') },
  ]

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-white mb-1">{t('deepdive.modelSize.title')}</h2>
        <p className="text-sm text-slate-400">{t('deepdive.modelSize.subtitle')}</p>
      </div>

      <SectionTabs tabs={TABS} active={activeTab} onSelect={setActiveTab} color="cyan" />

      {activeTab === 'why' && <WhySizeMatters data={data} t={t} />}
      {activeTab === 'headtohead' && <HeadToHead data={data} t={t} />}
      {activeTab === 'scaling' && <ScalingAnalysis data={data} t={t} />}

      <SectionTabs tabs={TABS} active={activeTab} onSelect={setActiveTab} color="cyan" />
    </div>
  )
}
