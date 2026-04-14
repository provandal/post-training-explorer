// The four-quadrant optimization framework is adapted from
// "A Survey of Techniques for Maximizing LLM Performance" by Colin Jarvis
// and John Allard, presented at OpenAI DevDay (November 2023).
// https://www.youtube.com/watch?v=ahnGLM-RC1Y

import { useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import useStore from '../store'
import tourSteps from '../data/tourSteps'

// Quadrant style definitions (non-translatable)
const QUADRANT_STYLES = {
  prompt: {
    color: '#f97316',
    bgColor: '#ea8c3c',
    textColor: '#000000',
    col: 0,
    row: 1,
    subKeys: [
      { key: 'quadrant.sub.prompt', x: 0.3, y: 0.75 },
      { key: 'quadrant.sub.fewShot', x: 0.65, y: 0.35 },
    ],
  },
  rag: {
    bgColor: '#e8c840',
    color: '#eab308',
    textColor: '#000000',
    col: 0,
    row: 0,
    subKeys: [
      { key: 'quadrant.sub.simpleRetrieval', x: 0.38, y: 0.75 },
      { key: 'quadrant.sub.optimizeRetrieval', x: 0.6, y: 0.3 },
    ],
  },
  posttraining: {
    color: '#475569',
    bgColor: '#1a2540',
    textColor: '#e2e8f0',
    col: 1,
    row: 1,
    subKeys: [
      { key: 'quadrant.sub.sft', x: 0.22, y: 0.72 },
      { key: 'quadrant.sub.dpo', x: 0.48, y: 0.55 },
      { key: 'quadrant.sub.grpo', x: 0.74, y: 0.38 },
    ],
  },
  alloptions: {
    color: '#06b6d4',
    bgColor: '#4ab8cc',
    textColor: '#000000',
    col: 1,
    row: 0,
    subKeys: [
      { key: 'quadrant.sub.postTraining', x: 0.45, y: 0.72 },
      { key: 'quadrant.sub.fineTuneWithRag', x: 0.6, y: 0.28 },
    ],
  },
}

const QUADRANT_LABEL_KEYS = {
  prompt: 'quadrant.prompt',
  rag: 'quadrant.rag',
  posttraining: 'quadrant.posttraining',
  alloptions: 'quadrant.alloptions',
}

// Layout constants (viewBox 600 x 520)
const VB_W = 600
const VB_H = 520
const MARGIN = { top: 50, right: 30, bottom: 55, left: 80 }
const GAP = 16
const QUAD_W = (VB_W - MARGIN.left - MARGIN.right - GAP) / 2
const QUAD_H = (VB_H - MARGIN.top - MARGIN.bottom - GAP) / 2

function quadXY(col, row) {
  return {
    x: MARGIN.left + col * (QUAD_W + GAP),
    y: MARGIN.top + row * (QUAD_H + GAP),
  }
}

// Zig-zag waypoints positioned relative to quadrant interiors
function wp(col, row, fracX, fracY) {
  const { x, y } = quadXY(col, row)
  return { x: x + QUAD_W * fracX, y: y + QUAD_H * fracY }
}

const ZIGZAG_POINTS = [
  { ...wp(0, 1, 0.3, 0.75) },
  { ...wp(0, 1, 0.65, 0.35) },
  { ...wp(0, 0, 0.38, 0.75) },
  { ...wp(0, 0, 0.6, 0.3) },
  { ...wp(1, 1, 0.22, 0.72) },
  { ...wp(1, 1, 0.48, 0.55) },
  { ...wp(1, 1, 0.74, 0.38) },
  { ...wp(1, 0, 0.55, 0.5) },
]

export default function QuadrantMap({ size = 'full', interactive = false }) {
  const { t } = useTranslation()
  const mode = useStore((s) => s.mode)
  const currentStep = useStore((s) => s.currentStep)
  const activeQuadrant = useStore((s) => s.activeQuadrant)

  // Build translated quadrant data
  const QUADRANTS = useMemo(() => {
    const result = {}
    for (const [key, style] of Object.entries(QUADRANT_STYLES)) {
      result[key] = {
        ...style,
        label: t(QUADRANT_LABEL_KEYS[key]),
        subLabels: style.subKeys.map((sk) => ({
          label: t(sk.key),
          x: sk.x,
          y: sk.y,
        })),
      }
    }
    return result
  }, [t])

  const currentZigzag = useMemo(() => {
    if (mode !== 'tour') return -1
    return tourSteps[currentStep]?.zigzagPosition ?? -1
  }, [mode, currentStep])

  const activeQ = useMemo(() => {
    if (mode === 'tour') return tourSteps[currentStep]?.quadrant || null
    return activeQuadrant
  }, [mode, currentStep, activeQuadrant])

  // Build the zig-zag SVG path with smooth curves
  const zigzagPath = useMemo(() => {
    const pts = ZIGZAG_POINTS
    if (pts.length < 2) return ''
    let d = `M ${pts[0].x} ${pts[0].y}`
    for (let i = 1; i < pts.length; i++) {
      d += ` L ${pts[i].x} ${pts[i].y}`
    }
    return d
  }, [])

  // Arrow markers for the zig-zag path
  const arrowSegments = useMemo(() => {
    const segments = []
    for (let i = 0; i < ZIGZAG_POINTS.length - 1; i++) {
      const a = ZIGZAG_POINTS[i]
      const b = ZIGZAG_POINTS[i + 1]
      // midpoint for arrow
      segments.push({
        x1: a.x,
        y1: a.y,
        x2: b.x,
        y2: b.y,
        mx: (a.x + b.x) / 2,
        my: (a.y + b.y) / 2,
        angle: (Math.atan2(b.y - a.y, b.x - a.x) * 180) / Math.PI,
      })
    }
    return segments
  }, [])

  return (
    <svg
      viewBox={`0 0 ${VB_W} ${VB_H}`}
      className={
        size === 'mini'
          ? 'w-[220px] h-[190px]'
          : size === 'medium'
            ? 'w-full max-w-[600px]'
            : 'w-full max-w-[680px]'
      }
      style={size === 'mini' ? undefined : { aspectRatio: `${VB_W}/${VB_H}` }}
    >
      {/* Background */}
      <rect width={VB_W} height={VB_H} fill="#0f172a" rx="12" />

      {/* Title */}
      {size !== 'mini' && (
        <text
          x={VB_W / 2}
          y="30"
          textAnchor="middle"
          fill="#94a3b8"
          fontSize="15"
          fontWeight="600"
          fontFamily="system-ui, sans-serif"
        >
          {t('quadrant.title')}
        </text>
      )}

      {/* Y-axis */}
      <line
        x1={MARGIN.left - 20}
        y1={MARGIN.top + QUAD_H * 2 + GAP + 10}
        x2={MARGIN.left - 20}
        y2={MARGIN.top - 10}
        stroke="#475569"
        strokeWidth="2"
      />
      <polygon
        points={`${MARGIN.left - 20},${MARGIN.top - 15} ${MARGIN.left - 24},${MARGIN.top - 5} ${MARGIN.left - 16},${MARGIN.top - 5}`}
        fill="#475569"
      />

      {size !== 'mini' && (
        <>
          <text
            x={MARGIN.left - 35}
            y={MARGIN.top + QUAD_H}
            textAnchor="middle"
            fill="#94a3b8"
            fontSize="12"
            fontWeight="600"
            transform={`rotate(-90, ${MARGIN.left - 35}, ${MARGIN.top + QUAD_H})`}
          >
            {t('quadrant.yAxisLabel')}
          </text>
          <text
            x={MARGIN.left - 50}
            y={MARGIN.top + QUAD_H}
            textAnchor="middle"
            fill="#64748b"
            fontSize="11"
            fontStyle="italic"
            transform={`rotate(-90, ${MARGIN.left - 50}, ${MARGIN.top + QUAD_H})`}
          >
            {t('quadrant.yAxisSub')}
          </text>
        </>
      )}

      {/* X-axis */}
      <line
        x1={MARGIN.left - 10}
        y1={VB_H - MARGIN.bottom + 20}
        x2={VB_W - MARGIN.right + 10}
        y2={VB_H - MARGIN.bottom + 20}
        stroke="#475569"
        strokeWidth="2"
      />
      <polygon
        points={`${VB_W - MARGIN.right + 15},${VB_H - MARGIN.bottom + 20} ${VB_W - MARGIN.right + 5},${VB_H - MARGIN.bottom + 16} ${VB_W - MARGIN.right + 5},${VB_H - MARGIN.bottom + 24}`}
        fill="#475569"
      />

      {size !== 'mini' && (
        <>
          <text
            x={MARGIN.left + QUAD_W}
            y={VB_H - 18}
            textAnchor="middle"
            fill="#94a3b8"
            fontSize="12"
            fontWeight="600"
          >
            {t('quadrant.xAxisLabel')}
          </text>
          <text
            x={MARGIN.left + QUAD_W}
            y={VB_H - 4}
            textAnchor="middle"
            fill="#64748b"
            fontSize="11"
            fontStyle="italic"
          >
            {t('quadrant.xAxisSub')}
          </text>
        </>
      )}

      {/* LLM icon */}
      {size !== 'mini' && (
        <g transform={`translate(${MARGIN.left - 22}, ${VB_H - MARGIN.bottom + 10})`}>
          <rect
            x="-18"
            y="-18"
            width="36"
            height="30"
            rx="6"
            fill="#1e1b4b"
            stroke="#6366f1"
            strokeWidth="1.5"
          />
          <text
            x="0"
            y="0"
            textAnchor="middle"
            fill="#a5b4fc"
            fontSize="10"
            fontWeight="700"
            fontFamily="system-ui, sans-serif"
          >
            LLM
          </text>
        </g>
      )}

      {/* Quadrants */}
      {Object.entries(QUADRANTS).map(([key, q]) => {
        const { x: qx, y: qy } = quadXY(q.col, q.row)
        const isActive = activeQ === key
        const opacity = activeQ === null ? 1 : isActive ? 1 : 0.3

        return (
          <g
            key={key}
            opacity={opacity}
            className={interactive ? 'cursor-pointer' : ''}
            onClick={interactive ? () => useStore.getState().setActiveQuadrant(key) : undefined}
            style={{ transition: 'opacity 0.3s' }}
          >
            {/* Glow for active */}
            {isActive && (
              <rect
                x={qx - 4}
                y={qy - 4}
                width={QUAD_W + 8}
                height={QUAD_H + 8}
                rx="14"
                fill="none"
                stroke={q.color}
                strokeWidth="2.5"
                opacity="0.7"
              />
            )}

            {/* Background */}
            <rect x={qx} y={qy} width={QUAD_W} height={QUAD_H} rx="10" fill={q.bgColor} />

            {/* Label */}
            <text
              x={qx + QUAD_W / 2}
              y={qy + QUAD_H / 2 + (size === 'mini' ? 5 : 6)}
              textAnchor="middle"
              fill={q.textColor}
              fontSize={size === 'mini' ? '14' : size === 'medium' ? '18' : '22'}
              fontWeight="800"
              fontFamily="system-ui, sans-serif"
            >
              {q.label}
            </text>

            {/* Sub-labels with dots (medium and full) */}
            {size !== 'mini' &&
              q.subLabels.map((sub) => {
                const sx = qx + QUAD_W * sub.x
                const sy = qy + QUAD_H * sub.y
                return (
                  <g key={sub.label}>
                    <rect
                      x={sx - 3}
                      y={sy - 3}
                      width="6"
                      height="6"
                      rx="1"
                      fill={q.textColor}
                      opacity="0.5"
                    />
                    <text
                      x={sx + 8}
                      y={sy + 4}
                      fill={q.textColor}
                      fontSize="10"
                      opacity="0.7"
                      fontFamily="system-ui, sans-serif"
                    >
                      {sub.label}
                    </text>
                  </g>
                )
              })}
          </g>
        )
      })}

      {/* Zig-zag path */}
      <path
        d={zigzagPath}
        fill="none"
        stroke="#94a3b8"
        strokeWidth="2.5"
        strokeDasharray="8 4"
        opacity="0.6"
      />

      {/* Arrow heads at midpoints of each segment */}
      {size !== 'mini' &&
        arrowSegments.map((seg, i) => (
          <polygon
            key={i}
            points="-5,-4 5,0 -5,4"
            fill="#94a3b8"
            opacity="0.5"
            transform={`translate(${seg.mx}, ${seg.my}) rotate(${seg.angle})`}
          />
        ))}

      {/* Zig-zag waypoints */}
      {ZIGZAG_POINTS.map((point, i) => {
        const isCurrentStop = currentZigzag === i
        const isPastStop = currentZigzag > i

        return (
          <g key={i}>
            <circle
              cx={point.x}
              cy={point.y}
              r={isCurrentStop ? 8 : 5}
              fill={isCurrentStop ? '#3b82f6' : isPastStop ? '#22c55e' : '#334155'}
              stroke={isCurrentStop ? '#93c5fd' : isPastStop ? '#86efac' : '#64748b'}
              strokeWidth={isCurrentStop ? 2.5 : 1}
            />
            {isCurrentStop && (
              <circle
                cx={point.x}
                cy={point.y}
                r="8"
                fill="none"
                stroke="#3b82f6"
                strokeWidth="2"
                opacity="0.5"
              >
                <animate attributeName="r" values="8;16;8" dur="2s" repeatCount="indefinite" />
                <animate
                  attributeName="opacity"
                  values="0.5;0;0.5"
                  dur="2s"
                  repeatCount="indefinite"
                />
              </circle>
            )}
          </g>
        )
      })}
    </svg>
  )
}
