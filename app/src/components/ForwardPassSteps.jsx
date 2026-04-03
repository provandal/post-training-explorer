import { useRef, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import * as d3 from 'd3'
import TokenProbChart from './TokenProbChart'
import {
  TOKENS,
  TOKEN_COLORS,
  FORWARD_PASS_STEPS,
  EMBEDDING_DIMS,
  OUTPUT_PROBS,
  getForwardPassSteps,
} from '../data/transformerData'

export default function ForwardPassSteps() {
  const { t } = useTranslation()
  const [currentStep, setCurrentStep] = useState(0)
  const steps = getForwardPassSteps(t)
  const step = steps[currentStep]

  return (
    <div className="space-y-5">
      <p className="text-sm text-slate-400">{t('forwardPass.steps.intro')}</p>

      {/* Step indicator */}
      <div className="flex items-center gap-0 justify-center">
        {steps.map((s, i) => (
          <div key={s.id} className="flex items-center">
            <button
              onClick={() => setCurrentStep(i)}
              className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold transition-all duration-300 border-2 ${
                i === currentStep
                  ? 'border-violet-500 bg-violet-950/60 text-violet-300 shadow-lg shadow-violet-500/20 scale-110'
                  : i < currentStep
                    ? 'border-slate-500 bg-slate-800/50 text-slate-400'
                    : 'border-slate-700/40 bg-slate-800/20 text-slate-600 opacity-50'
              }`}
            >
              {i + 1}
            </button>
            {i < steps.length - 1 && (
              <div
                className={`w-8 h-0.5 transition-colors duration-300 ${
                  i < currentStep ? 'bg-slate-500' : 'bg-slate-700/30'
                }`}
              />
            )}
          </div>
        ))}
      </div>

      {/* Step title */}
      <div className="text-center">
        <h3 className="text-lg font-bold text-white">{step.title}</h3>
        <p className="text-sm text-violet-400">{step.subtitle}</p>
      </div>

      {/* Visualization area */}
      <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50 min-h-[280px]">
        {currentStep === 0 && <TokenizeViz />}
        {currentStep === 1 && <EmbeddingViz />}
        {currentStep === 2 && <PositionViz />}
        {currentStep === 3 && <AttentionViz />}
        {currentStep === 4 && <FFNViz />}
        {currentStep === 5 && <PredictViz />}
      </div>

      {/* Formula */}
      <div className="p-3 rounded-lg bg-slate-900/50 border border-slate-700/30 font-mono text-xs text-violet-300">
        {step.formula}
      </div>

      {/* Explanation */}
      <p className="text-sm text-slate-300 leading-relaxed">{step.explanation}</p>

      {/* Navigation */}
      <div className="flex justify-between items-center">
        <button
          onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
          disabled={currentStep === 0}
          className={`px-4 py-2 text-sm rounded-md transition-colors ${
            currentStep === 0
              ? 'bg-slate-800/30 text-slate-600 cursor-not-allowed'
              : 'bg-slate-700 text-slate-200 hover:bg-slate-600'
          }`}
        >
          {t('forwardPass.steps.previous')}
        </button>
        <span className="text-xs text-slate-500">
          {t('forwardPass.steps.stepOf', { current: currentStep + 1, total: steps.length })}
        </span>
        <button
          onClick={() => setCurrentStep(Math.min(steps.length - 1, currentStep + 1))}
          disabled={currentStep === steps.length - 1}
          className={`px-4 py-2 text-sm rounded-md transition-colors ${
            currentStep === steps.length - 1
              ? 'bg-slate-800/30 text-slate-600 cursor-not-allowed'
              : 'bg-violet-700 text-white hover:bg-violet-600'
          }`}
        >
          {t('forwardPass.steps.nextStep')}
        </button>
      </div>
    </div>
  )
}

// --- Step 0: Tokenization ---
function TokenizeViz() {
  const { t } = useTranslation()
  const [phase, setPhase] = useState(0) // 0=raw text, 1=split tokens, 2=show IDs

  useEffect(() => {
    if (phase < 2) {
      const timer = setTimeout(() => setPhase(phase + 1), 1000)
      return () => clearTimeout(timer)
    }
  }, [phase])

  return (
    <div className="space-y-4">
      {/* Raw text */}
      <div>
        <p className="text-xs text-slate-500 mb-1">{t('forwardPass.steps.inputText')}</p>
        <div className="p-3 rounded bg-slate-900/50 font-mono text-sm text-slate-300 border border-slate-700/30">
          Classify this I/O workload: IOPS: 45000 | Latency: 0.3ms | Block:
        </div>
      </div>

      {/* Split tokens */}
      {phase >= 1 && (
        <div>
          <p className="text-xs text-slate-500 mb-1">
            {t('forwardPass.steps.bpeTokenization', { count: TOKENS.length })}
          </p>
          <div className="flex flex-wrap gap-1">
            {TOKENS.map((tok, i) => {
              const colors = TOKEN_COLORS[tok.type]
              return (
                <div
                  key={tok.id}
                  className={`flex flex-col items-center transition-all duration-500 ${
                    phase >= 1 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'
                  }`}
                  style={{ transitionDelay: `${i * 40}ms` }}
                >
                  <span
                    className={`px-2 py-1 text-xs font-mono rounded border ${colors.bg} ${colors.border} ${colors.text}`}
                  >
                    {tok.text}
                  </span>
                  {phase >= 2 && (
                    <span
                      className="text-[10px] text-slate-500 mt-0.5 font-mono transition-opacity duration-300"
                      style={{ transitionDelay: `${i * 30 + 500}ms` }}
                    >
                      {tok.tokenId}
                    </span>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {phase >= 2 && (
        <p className="text-xs text-slate-500 italic">{t('forwardPass.steps.splitNotice')}</p>
      )}
    </div>
  )
}

// --- Step 1: Embedding ---
function EmbeddingViz() {
  const { t } = useTranslation()
  const svgRef = useRef()

  useEffect(() => {
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const { labels, tokens } = EMBEDDING_DIMS
    const tokenWidth = 85
    const barHeight = 12
    const margin = { top: 30, left: 10, bottom: 20 }
    const width = margin.left + tokens.length * tokenWidth
    const height = margin.top + labels.length * (barHeight + 3) + margin.bottom

    svg.attr('width', width).attr('height', height)

    tokens.forEach((tok, ti) => {
      const gx = margin.left + ti * tokenWidth

      // Token label
      svg
        .append('text')
        .attr('x', gx + tokenWidth / 2)
        .attr('y', 16)
        .attr('text-anchor', 'middle')
        .attr('fill', '#94a3b8')
        .attr('font-size', '10')
        .attr('font-family', 'monospace')
        .attr('font-weight', '600')
        .text(tok.text)

      // Bars for each dimension
      const barScale = d3
        .scaleLinear()
        .domain([-1, 1])
        .range([0, tokenWidth - 12])

      tok.values.forEach((val, di) => {
        const y = margin.top + di * (barHeight + 3)
        const barW = Math.abs(barScale(val) - barScale(0))
        const barX = val >= 0 ? barScale(0) : barScale(val)

        // Dim label (only for first token)
        if (ti === 0) {
          svg
            .append('text')
            .attr('x', gx - 2)
            .attr('y', y + barHeight / 2 + 3)
            .attr('text-anchor', 'end')
            .attr('fill', '#475569')
            .attr('font-size', '7')
            .text(labels[di])
        }

        // Zero line
        svg
          .append('line')
          .attr('x1', gx + barScale(0))
          .attr('x2', gx + barScale(0))
          .attr('y1', y)
          .attr('y2', y + barHeight)
          .attr('stroke', '#334155')
          .attr('stroke-width', 0.5)

        // Bar
        svg
          .append('rect')
          .attr('x', gx + barX)
          .attr('y', y + 1)
          .attr('width', 0)
          .attr('height', barHeight - 2)
          .attr('fill', val >= 0 ? '#8b5cf6' : '#ef4444')
          .attr('opacity', 0.7)
          .attr('rx', 1)
          .transition()
          .duration(500)
          .delay(ti * 100 + di * 30)
          .attr('width', barW)
      })
    })
  }, [])

  return (
    <div className="space-y-3">
      <p className="text-xs text-slate-500">{t('forwardPass.steps.embeddingIntro')}</p>
      <div className="overflow-x-auto">
        <svg ref={svgRef} />
      </div>
    </div>
  )
}

// --- Step 2: Positional Encoding ---
function PositionViz() {
  const { t } = useTranslation()
  return (
    <div className="space-y-4">
      <p className="text-xs text-slate-500">{t('forwardPass.steps.positionIntro')}</p>
      <div className="flex items-center gap-3 justify-center flex-wrap">
        {TOKENS.slice(0, 6).map((tok, i) => {
          const colors = TOKEN_COLORS[tok.type]
          return (
            <div key={tok.id} className="flex flex-col items-center gap-1">
              <span
                className={`px-2 py-1 text-xs font-mono rounded border ${colors.bg} ${colors.border} ${colors.text}`}
              >
                {tok.text}
              </span>
              <span className="text-[10px] text-slate-600">+</span>
              <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-amber-950/30 border border-amber-800/30 text-amber-400">
                pos {i}
              </span>
              <span className="text-[10px] text-slate-600">=</span>
              <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-violet-950/30 border border-violet-800/30 text-violet-300">
                input[{i}]
              </span>
            </div>
          )
        })}
        <span className="text-slate-600 text-lg">...</span>
      </div>
      <div className="p-3 rounded bg-slate-900/50 border border-slate-700/30 text-xs text-slate-400">
        {t('forwardPass.steps.positionWhy', { 1: (c) => c })}
      </div>
    </div>
  )
}

// --- Step 3: Self-Attention ---
function AttentionViz() {
  const { t } = useTranslation()
  return (
    <div className="space-y-4">
      <p className="text-xs text-slate-500">{t('forwardPass.steps.attentionIntro')}</p>

      {/* Q, K, V diagram */}
      <div className="flex items-center justify-center gap-4 flex-wrap">
        <div className="text-center">
          <div className="p-3 rounded-lg bg-blue-950/30 border border-blue-800/40 w-24">
            <p className="text-xs font-bold text-blue-300">{t('forwardPass.steps.queryQ')}</p>
            <p className="text-[10px] text-blue-400/70">{t('forwardPass.steps.queryQSub')}</p>
          </div>
        </div>

        <div className="text-2xl text-slate-600">×</div>

        <div className="text-center">
          <div className="p-3 rounded-lg bg-emerald-950/30 border border-emerald-800/40 w-24">
            <p className="text-xs font-bold text-emerald-300">{t('forwardPass.steps.keyK')}</p>
            <p className="text-[10px] text-emerald-400/70">{t('forwardPass.steps.keyKSub')}</p>
          </div>
        </div>

        <div className="text-2xl text-slate-600">→</div>

        <div className="text-center">
          <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-600/40 w-24">
            <p className="text-xs font-bold text-slate-300">{t('forwardPass.steps.scores')}</p>
            <p className="text-[10px] text-slate-400/70">{t('forwardPass.steps.scoresSub')}</p>
          </div>
        </div>

        <div className="text-2xl text-slate-600">×</div>

        <div className="text-center">
          <div className="p-3 rounded-lg bg-amber-950/30 border border-amber-800/40 w-24">
            <p className="text-xs font-bold text-amber-300">{t('forwardPass.steps.valueV')}</p>
            <p className="text-[10px] text-amber-400/70">{t('forwardPass.steps.valueVSub')}</p>
          </div>
        </div>
      </div>

      {/* LoRA callout */}
      <div className="p-3 rounded-lg border-2 border-dashed border-violet-600/60 bg-violet-950/20">
        <p className="text-xs text-violet-300">
          {t('forwardPass.steps.loraInsertsHere', { 1: (c) => c })}
        </p>
      </div>

      {/* Mini example */}
      <div className="grid grid-cols-3 gap-2 text-center text-xs">
        <div className="p-2 rounded bg-slate-900/50 border border-slate-700/30">
          <p className="font-mono text-green-400">"45000"</p>
          <p className="text-slate-500 mt-1">Q asks: "which metric label describes me?"</p>
        </div>
        <div className="p-2 rounded bg-slate-900/50 border border-slate-700/30">
          <p className="font-mono text-violet-400">"IOPS"</p>
          <p className="text-slate-500 mt-1">K answers: "I'm the IOPS metric label"</p>
        </div>
        <div className="p-2 rounded bg-slate-900/50 border border-slate-700/30">
          <p className="text-slate-300">High score → strong connection</p>
          <p className="text-slate-500 mt-1">45000 attends to IOPS with high weight</p>
        </div>
      </div>
    </div>
  )
}

// --- Step 4: Feed-Forward Network ---
function FFNViz() {
  const { t } = useTranslation()
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    const timer = setTimeout(() => setExpanded(true), 500)
    return () => clearTimeout(timer)
  }, [])

  return (
    <div className="space-y-4">
      <p className="text-xs text-slate-500">{t('forwardPass.steps.ffnIntro')}</p>

      {/* Expand/compress visualization */}
      <div className="flex items-center justify-center gap-3">
        <div className="text-center">
          <div
            className="mx-auto rounded bg-violet-600/60 transition-all duration-700"
            style={{ width: '40px', height: expanded ? '100px' : '80px' }}
          />
          <p className="text-xs text-slate-400 mt-1 font-mono">960</p>
          <p className="text-[10px] text-slate-500">input</p>
        </div>

        <div className="text-slate-600">→</div>

        <div className="text-center">
          <div
            className="mx-auto rounded transition-all duration-700"
            style={{
              width: expanded ? '160px' : '40px',
              height: expanded ? '100px' : '80px',
              backgroundColor: expanded ? 'rgba(245, 158, 11, 0.4)' : 'rgba(139, 92, 246, 0.4)',
            }}
          />
          <p className="text-xs text-amber-400 mt-1 font-mono">3,840</p>
          <p className="text-[10px] text-slate-500">expand (4×)</p>
        </div>

        <div className="text-slate-600">→</div>
        <div className="text-xs text-slate-500">GELU</div>
        <div className="text-slate-600">→</div>

        <div className="text-center">
          <div
            className="mx-auto rounded bg-emerald-600/50 transition-all duration-700"
            style={{ width: '40px', height: expanded ? '100px' : '80px' }}
          />
          <p className="text-xs text-emerald-400 mt-1 font-mono">960</p>
          <p className="text-[10px] text-slate-500">compress</p>
        </div>
      </div>

      <div className="p-3 rounded bg-slate-900/50 border border-slate-700/30 text-xs text-slate-400">
        {t('forwardPass.steps.ffnWhy', { 1: (c) => c })}
      </div>
    </div>
  )
}

// --- Step 5: Next Token Prediction ---
function PredictViz() {
  const { t } = useTranslation()
  return (
    <div className="space-y-4">
      <p className="text-xs text-slate-500">{t('forwardPass.steps.predictIntro')}</p>

      <TokenProbChart
        data={OUTPUT_PROBS.base}
        comparisonData={OUTPUT_PROBS.finetuned}
        label={t('forwardPass.steps.predictLabel')}
        comparisonLabel={t('forwardPass.steps.predictAfterLabel')}
        highlightToken="OLTP"
        width={500}
        height={320}
      />

      <div className="p-3 rounded-lg border border-emerald-800/30 bg-emerald-950/20 text-xs text-emerald-300">
        {t('forwardPass.steps.predictEffect', { 1: (c) => c })}
      </div>
    </div>
  )
}
