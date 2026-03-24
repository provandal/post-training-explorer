import { useState, useRef, useEffect } from 'react'
import * as d3 from 'd3'
import LossChart from '../components/LossChart'
import InfrastructureCard from '../components/InfrastructureCard'
import SectionTabs from '../components/SectionTabs'
import { isLoaded, getGRPOGenerationLogs, getGRPOAccuracyCurve, getGRPORewardCurve, getGRPOGroupStatistics, getTrainingTime } from '../data/loadArtifacts'

// ---------------------------------------------------------------------------
// Precomputed data: GRPO generation example (interactive demo)
// ---------------------------------------------------------------------------
const GRPO_EXAMPLE = {
  input: "IOPS: 12000 | Latency: 4.5ms | Block Size: 128K | Read/Write: 92/8 | Sequential: 78% | Queue Depth: 8",
  correctLabel: "OLAP Analytics",
  generations: [
    {
      id: 1,
      text: "Classification: OLAP Analytics\nConfidence: High\nKey indicators: Large block sequential reads (128K, 78% sequential) with heavy read bias (92/8) indicate analytical query processing.",
      reward: 1.0,
      correct: true,
    },
    {
      id: 2,
      text: "Classification: AI ML Training\nConfidence: Medium\nKey indicators: Large block sequential reads often indicate training data loading. Read-heavy pattern consistent with data pipeline.",
      reward: 0.0,
      correct: false,
    },
    {
      id: 3,
      text: "Classification: OLAP Analytics\nConfidence: Medium-High\nKey indicators: Moderate IOPS with large blocks and high sequential ratio point to scan-heavy warehouse queries.",
      reward: 1.0,
      correct: true,
    },
    {
      id: 4,
      text: "Classification: Video Streaming\nConfidence: Low\nKey indicators: Sequential reads with large blocks. However, IOPS is too high for typical streaming.",
      reward: 0.0,
      correct: false,
    },
    {
      id: 5,
      text: "Classification: AI ML Training\nConfidence: Medium\nKey indicators: Sequential read pattern with large blocks suggests data loading for model training.",
      reward: 0.0,
      correct: false,
    },
    {
      id: 6,
      text: "Classification: OLAP Analytics\nConfidence: High\nKey indicators: Read-heavy (92/8) with large sequential I/O (128K, 78%) and moderate queue depth is classic analytics/BI workload pattern.",
      reward: 1.0,
      correct: true,
    },
    {
      id: 7,
      text: "Classification: OLAP Analytics\nConfidence: Medium\nKey indicators: Large blocks with sequential reads suggest analytical processing or data warehouse queries.",
      reward: 1.0,
      correct: true,
    },
    {
      id: 8,
      text: "Classification: AI ML Training\nConfidence: Medium-High\nKey indicators: High sequential percentage with large blocks and extreme read bias matches GPU training data loading patterns.",
      reward: 0.0,
      correct: false,
    },
  ],
}

// Group statistics (precomputed)
const rewards = GRPO_EXAMPLE.generations.map(g => g.reward)
const meanReward = rewards.reduce((a, b) => a + b, 0) / rewards.length
const stdReward = Math.sqrt(rewards.reduce((sum, r) => sum + (r - meanReward) ** 2, 0) / rewards.length)

// ---------------------------------------------------------------------------
// Precomputed data: Training curves (Under the Covers)
// ---------------------------------------------------------------------------
const FALLBACK_ACCURACY_CURVE = (() => {
  const data = []
  for (let step = 0; step <= 200; step += 2) {
    const base = 0.82 / (1 + Math.exp(-0.04 * (step - 60))) + 0.42
    const noise = (Math.sin(step * 0.9) * 0.02 + Math.cos(step * 1.7) * 0.015)
    data.push({ step, loss: Math.min(0.88, Math.max(0.35, base + noise)) })
  }
  return data
})()

const FALLBACK_REWARD_CURVE = (() => {
  const data = []
  for (let step = 0; step <= 200; step += 2) {
    const base = 0.7 / (1 + Math.exp(-0.035 * (step - 70))) + 0.3
    const noise = (Math.sin(step * 0.6) * 0.03 + Math.cos(step * 1.1) * 0.02)
    data.push({ step, loss: Math.min(0.92, Math.max(0.25, base + noise)) })
  }
  return data
})()

// ---------------------------------------------------------------------------
// Precomputed data: Infrastructure profile
// ---------------------------------------------------------------------------
const GRPO_INFRA = {
  gpuMemoryGB: 6.8,
  trainingTimeMinutes: 35,
  checkpointSizeMB: 1.7,
  peakGPUUtilization: 95,
  storageIOPattern: "Heavy compute bursts (generating 8 completions per prompt), periodic checkpoint writes",
  note: "8x generation per prompt means 8x the compute per training step vs SFT. Peak GPU utilization hits 95% during generation bursts."
}

// ---------------------------------------------------------------------------
// D3 visualization: Advantage waterfall
// ---------------------------------------------------------------------------
function AdvantageWaterfall() {
  const svgRef = useRef()

  useEffect(() => {
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const width = 500
    const height = 220
    const margin = { top: 30, right: 20, bottom: 40, left: 50 }
    const w = width - margin.left - margin.right
    const h = height - margin.top - margin.bottom

    svg.attr('width', width).attr('height', height)
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    const generations = [
      { id: 'G1', reward: 1.0, label: 'OLAP', correct: true },
      { id: 'G2', reward: 0.0, label: 'AI/ML', correct: false },
      { id: 'G3', reward: 1.0, label: 'OLAP', correct: true },
      { id: 'G4', reward: 0.0, label: 'Video', correct: false },
      { id: 'G5', reward: 0.0, label: 'AI/ML', correct: false },
      { id: 'G6', reward: 1.0, label: 'OLAP', correct: true },
      { id: 'G7', reward: 1.0, label: 'OLAP', correct: true },
      { id: 'G8', reward: 0.0, label: 'AI/ML', correct: false },
    ]
    const mean = generations.reduce((s, g) => s + g.reward, 0) / generations.length
    generations.forEach(g => { g.advantage = g.reward - mean })

    const x = d3.scaleBand().domain(generations.map(g => g.id)).range([0, w]).padding(0.2)
    const y = d3.scaleLinear().domain([-0.7, 0.7]).range([h, 0])

    svg.append('text').attr('x', width / 2).attr('y', 14)
      .attr('text-anchor', 'middle').attr('fill', '#94a3b8')
      .attr('font-size', '11').attr('font-weight', '600')
      .text('GRPO Advantage Scores (8 Generations)')

    g.append('line')
      .attr('x1', 0).attr('x2', w)
      .attr('y1', y(0)).attr('y2', y(0))
      .attr('stroke', '#64748b').attr('stroke-width', 1.5).attr('stroke-dasharray', '4,3')

    g.append('text')
      .attr('x', w + 5).attr('y', y(0) + 4)
      .attr('fill', '#64748b').attr('font-size', '8')
      .text(`mean=${mean.toFixed(2)}`)

    g.selectAll('.bar')
      .data(generations)
      .join('rect')
      .attr('x', d => x(d.id))
      .attr('y', d => d.advantage > 0 ? y(d.advantage) : y(0))
      .attr('width', x.bandwidth())
      .attr('height', 0)
      .attr('fill', d => d.advantage > 0 ? '#22c55e' : '#ef4444')
      .attr('opacity', 0.8)
      .attr('rx', 3)
      .transition()
      .duration(500)
      .delay((d, i) => i * 100)
      .attr('height', d => Math.abs(y(0) - y(d.advantage)))

    g.selectAll('.adv-label')
      .data(generations)
      .join('text')
      .attr('x', d => x(d.id) + x.bandwidth() / 2)
      .attr('y', d => d.advantage > 0 ? y(d.advantage) - 5 : y(d.advantage) + 12)
      .attr('text-anchor', 'middle')
      .attr('fill', d => d.advantage > 0 ? '#22c55e' : '#ef4444')
      .attr('font-size', '9').attr('font-weight', '600')
      .text(d => (d.advantage > 0 ? '+' : '') + d.advantage.toFixed(2))
      .attr('opacity', 0)
      .transition().delay((d, i) => i * 100 + 500).attr('opacity', 1)

    g.selectAll('.class-label')
      .data(generations)
      .join('text')
      .attr('x', d => x(d.id) + x.bandwidth() / 2)
      .attr('y', h + 14)
      .attr('text-anchor', 'middle')
      .attr('fill', d => d.correct ? '#86efac' : '#fca5a5')
      .attr('font-size', '8')
      .text(d => d.label)

    g.selectAll('.gen-label')
      .data(generations)
      .join('text')
      .attr('x', d => x(d.id) + x.bandwidth() / 2)
      .attr('y', h + 28)
      .attr('text-anchor', 'middle')
      .attr('fill', '#64748b').attr('font-size', '8')
      .text(d => d.id)
  }, [])

  return <svg ref={svgRef} />
}

// ===========================================================================
// Main component
// ===========================================================================
export default function GRPOGenerations({ explore = false }) {
  const [section, setSection] = useState('problem')

  // --- generation reveal state (demo tab) ---
  const [revealedCount, setRevealedCount] = useState(0)
  const [showStats, setShowStats] = useState(false)

  // --- under the covers state (deepdive tab) ---
  const [vizTab, setVizTab] = useState('advantage')

  // --- Resolve real vs fallback data ---
  const realLogs = isLoaded() ? getGRPOGenerationLogs() : null
  const realFirstExample = realLogs?.examples?.[0] ?? null

  // Use real generation example if available, otherwise fallback
  const activeExample = realFirstExample
    ? {
        input: realFirstExample.input,
        correctLabel: realFirstExample.true_label,
        generations: realFirstExample.generations.map((g, i) => ({
          id: i + 1,
          text: g.text ?? g.generated_text ?? `Classification: ${g.label ?? 'Unknown'}`,
          reward: g.reward ?? (g.correct ? 1.0 : 0.0),
          correct: g.correct ?? false,
        })),
      }
    : GRPO_EXAMPLE
  const usingRealData = !!realFirstExample

  // Accuracy & reward curves: map real data to LossChart's {step, loss} format
  const rawAccuracy = isLoaded() ? getGRPOAccuracyCurve() : []
  const accuracyCurve = rawAccuracy.length > 0
    ? rawAccuracy.map(d => ({ step: d.step, loss: d.accuracy ?? d.loss }))
    : FALLBACK_ACCURACY_CURVE

  const rawReward = isLoaded() ? getGRPORewardCurve() : []
  const rewardCurve = rawReward.length > 0
    ? rawReward.map(d => ({ step: d.step, loss: d.mean_reward ?? d.loss }))
    : FALLBACK_REWARD_CURVE

  // Training time
  const grpoTrainingTime = getTrainingTime('grpo')

  // Group statistics for active example
  const activeRewards = activeExample.generations.map(g => g.reward)
  const activeMeanReward = activeRewards.reduce((a, b) => a + b, 0) / activeRewards.length
  const activeStdReward = Math.sqrt(activeRewards.reduce((sum, r) => sum + (r - activeMeanReward) ** 2, 0) / activeRewards.length)

  const revealNext = () => {
    if (revealedCount < activeExample.generations.length) {
      setRevealedCount(revealedCount + 1)
    }
    if (revealedCount + 1 === activeExample.generations.length) {
      setShowStats(true)
    }
  }

  const revealAll = () => {
    setRevealedCount(activeExample.generations.length)
    setShowStats(true)
  }

  const correctCount = activeExample.generations.filter(g => g.correct).length

  const tabs = [
    { id: 'problem', label: 'The Problem' },
    { id: 'concept', label: 'How GRPO Works' },
    { id: 'demo', label: 'See It Work' },
    { id: 'deepdive', label: 'Under the Covers' },
  ]

  const vizTabs = [
    { id: 'advantage', label: 'Advantage Scores' },
    { id: 'accuracy', label: 'Accuracy Over Training' },
    { id: 'reward', label: 'Mean Reward Curve' },
  ]

  return (
    <div className="max-w-5xl mx-auto">
      {/* ---- Tab bar (top) ---- */}
      <div className="mb-6">
        <SectionTabs tabs={tabs} active={section} onSelect={setSection} color="emerald" />
      </div>

      {/* ================================================================= */}
      {/* TAB 1 -- The Problem                                              */}
      {/* ================================================================= */}
      {section === 'problem' && (
        <div className="space-y-6">
          <div className="p-5 rounded-lg bg-slate-800/40 border border-slate-700/50">
            <h3 className="text-lg font-semibold text-emerald-400 mb-3">
              DPO improved style, but humans are still in the loop
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed mb-4">
              DPO was a big step forward: it removed the reward model and simplified the
              training pipeline. But it still required{' '}
              <strong className="text-pink-400">400 human-labeled preference pairs</strong>.
              Someone had to look at output pairs and decide which was better. For our
              storage I/O classifier, that meant an engineer spending hours reviewing
              outputs, comparing styles, and clicking "this one is better."
            </p>

            <div className="p-4 rounded-lg bg-emerald-950/20 border border-emerald-800/30 mb-4">
              <p className="text-sm text-emerald-300 font-semibold mb-2">
                The scaling bottleneck
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
                <div className="p-3 rounded bg-slate-800/50">
                  <div className="text-2xl font-bold text-pink-400">400</div>
                  <div className="text-xs text-slate-400">Preference pairs we labeled</div>
                </div>
                <div className="p-3 rounded bg-slate-800/50">
                  <div className="text-2xl font-bold text-pink-400">~3 hrs</div>
                  <div className="text-xs text-slate-400">Human annotation time</div>
                </div>
                <div className="p-3 rounded bg-slate-800/50">
                  <div className="text-2xl font-bold text-pink-400">$$$</div>
                  <div className="text-xs text-slate-400">Doesn't scale to new tasks</div>
                </div>
              </div>
            </div>
          </div>

          {/* The insight */}
          <div className="p-5 rounded-lg bg-slate-800/40 border border-slate-700/50">
            <h3 className="text-base font-semibold text-emerald-400 mb-3">
              What if the answer itself could be the teacher?
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed mb-4">
              For many tasks, there is a clear right answer. Our storage I/O classification
              is one of them: "OLAP Analytics" is either the correct label or it isn't. We
              don't need a human to tell us that &mdash; we can check programmatically.
              Math problems, code correctness, factual lookups &mdash; all of these have
              <strong className="text-emerald-300"> verifiable answers</strong>.
            </p>
            <p className="text-sm text-slate-300 leading-relaxed">
              This is the frontier of post-training:{' '}
              <strong className="text-emerald-400">
                Reinforcement Learning with Verifiable Rewards (RLVR)
              </strong>. Instead of collecting human preferences, we let the model generate
              multiple attempts, check which ones are correct, and use that signal to
              improve. The algorithm that makes this work is called{' '}
              <strong className="text-emerald-400">GRPO</strong> &mdash; Group Relative
              Policy Optimization &mdash; and it's the technique behind DeepSeek R1.
            </p>
          </div>

          {/* Transition */}
          <div className="p-4 rounded-lg bg-emerald-950/20 border border-emerald-800/30">
            <p className="text-sm text-emerald-300 leading-relaxed">
              <strong>Can the model learn from its own attempts?</strong> That's GRPO.
              Head to the next tab to see how it works.
            </p>
          </div>
        </div>
      )}

      {/* ================================================================= */}
      {/* TAB 2 -- How GRPO Works                                           */}
      {/* ================================================================= */}
      {section === 'concept' && (
        <div className="space-y-6">
          {/* Headline */}
          <div className="p-5 rounded-lg bg-slate-800/40 border border-slate-700/50">
            <h3 className="text-lg font-semibold text-emerald-400 mb-1">
              Group Relative Policy Optimization
            </h3>
            <p className="text-xs text-slate-500 mb-4">
              The technique behind DeepSeek R1.
            </p>
            <p className="text-sm text-slate-300 leading-relaxed mb-4">
              GRPO takes a radically different approach from both RLHF and DPO. Instead of
              collecting preferences from humans, the model <strong>generates multiple
              attempts</strong> for the same input, <strong>scores them with a verifiable
              reward</strong> (did it get the right answer?), and uses the{' '}
              <strong>group statistics as a baseline</strong> to figure out which attempts
              were above-average and which were below.
            </p>

            {/* The four-step process */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-5">
              <div className="p-4 rounded-lg bg-emerald-950/15 border border-emerald-800/30">
                <div className="text-2xl mb-2">1.</div>
                <h4 className="text-sm font-semibold text-emerald-400 mb-1">Generate a group</h4>
                <p className="text-xs text-slate-400">
                  Given one input prompt, the model generates <strong>8 different
                  completions</strong>. Each might classify the I/O pattern differently, with
                  varying confidence and reasoning.
                </p>
              </div>
              <div className="p-4 rounded-lg bg-emerald-950/15 border border-emerald-800/30">
                <div className="text-2xl mb-2">2.</div>
                <h4 className="text-sm font-semibold text-emerald-400 mb-1">Score with verifiable reward</h4>
                <p className="text-xs text-slate-400">
                  Each completion is scored: <strong>correct classification = 1</strong>,{' '}
                  <strong>wrong = 0</strong>. No reward model, no human labeler. The answer
                  key is the teacher.
                </p>
              </div>
              <div className="p-4 rounded-lg bg-emerald-950/15 border border-emerald-800/30">
                <div className="text-2xl mb-2">3.</div>
                <h4 className="text-sm font-semibold text-emerald-400 mb-1">Compute group mean</h4>
                <p className="text-xs text-slate-400">
                  The average reward across all 8 generations becomes the{' '}
                  <strong>baseline</strong>. No separate critic network or reward model
                  needed &mdash; just simple group statistics.
                </p>
              </div>
              <div className="p-4 rounded-lg bg-emerald-950/15 border border-emerald-800/30">
                <div className="text-2xl mb-2">4.</div>
                <h4 className="text-sm font-semibold text-emerald-400 mb-1">Compute relative advantage</h4>
                <p className="text-xs text-slate-400">
                  Compare each generation's reward to the group average.{' '}
                  <strong className="text-emerald-300">Above average = reinforce.</strong>{' '}
                  <strong className="text-red-300">Below average = suppress.</strong>
                </p>
              </div>
            </div>

            {/* Key difference from PPO */}
            <div className="p-4 rounded bg-slate-800 border border-slate-700/50 mb-5">
              <p className="text-sm text-slate-300 leading-relaxed">
                <strong className="text-emerald-300">Key difference from PPO:</strong>{' '}
                Traditional RLHF with PPO requires a separately-trained critic network and a
                separate reward model &mdash; both expensive to train and maintain. GRPO replaces
                both with <strong>group statistics</strong>: just compute the mean and standard
                deviation of the group's rewards. This means roughly{' '}
                <strong className="text-emerald-300">50% less compute</strong> compared to PPO,
                with no extra models to train.
              </p>
            </div>

            {/* The advantage formula */}
            <div className="p-4 rounded bg-slate-800 border border-slate-700/50 mb-5">
              <p className="text-xs text-slate-500 mb-2 font-semibold uppercase tracking-wide">
                The GRPO advantage formula (simplified)
              </p>
              <p className="text-base text-slate-200 font-mono mb-3">
                advantage_i = (reward_i - mean(rewards)) / std(rewards)
              </p>
              <div className="text-sm text-slate-400 space-y-2 leading-relaxed">
                <p>
                  <strong className="text-slate-300">In plain English:</strong> For each
                  generation in the group, subtract the group's mean reward. If you got a
                  reward of 1.0 and the group mean is 0.5, your advantage is positive &mdash;
                  you did better than average. If you got 0.0, your advantage is negative &mdash;
                  you did worse. Dividing by standard deviation normalizes the scores so the
                  model makes meaningful gradient updates.
                </p>
                <p>
                  <span className="text-emerald-300 font-semibold">Positive advantage</span>{' '}
                  &rarr; increase the probability of generating this response.
                  <br />
                  <span className="text-red-300 font-semibold">Negative advantage</span>{' '}
                  &rarr; decrease the probability of generating this response.
                </p>
              </div>
            </div>

            {/* Study group analogy */}
            <div className="p-4 rounded bg-slate-800 border border-slate-700/50 mb-5">
              <p className="text-sm text-slate-300 leading-relaxed">
                <strong className="text-emerald-300">Analogy:</strong> Imagine a study group
                where everyone answers the same exam question, then they grade each other's
                homework. The answer key is the teacher &mdash; no human grader needed.
                Students who scored above the group average get encouragement ("do more of
                this"), and those who scored below get correction ("do less of this"). Over
                time, the whole group converges on consistently right answers.
              </p>
            </div>
          </div>

          {/* Key insight box */}
          <div className="p-4 rounded-lg bg-blue-950/20 border border-blue-800/30">
            <p className="text-sm text-blue-300 leading-relaxed">
              <strong>Key insight:</strong> GRPO needs no human labels and no reward model.
              The answer itself is the teacher. For any task with a verifiable correct
              answer &mdash; classification, math, code &mdash; the model can improve
              itself purely from its own attempts.
            </p>
          </div>

          {/* Training stats comparison */}
          <div className="p-5 rounded-lg bg-slate-800/40 border border-slate-700/50">
            <h4 className="text-sm font-semibold text-emerald-400 mb-3">
              Training time comparison across techniques
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
              <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/30">
                <div className="text-2xl font-bold text-sky-400">12 min</div>
                <div className="text-xs text-slate-400 mt-1">SFT (Supervised Fine-Tuning)</div>
                <div className="text-xs text-slate-500 mt-0.5">Needs labeled examples</div>
              </div>
              <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/30">
                <div className="text-2xl font-bold text-violet-400">8 min</div>
                <div className="text-xs text-slate-400 mt-1">DPO (Direct Preference)</div>
                <div className="text-xs text-pink-400 mt-0.5">+ human preference labels</div>
              </div>
              <div className="p-4 rounded-lg bg-emerald-950/30 border border-emerald-800/30">
                <div className="text-2xl font-bold text-emerald-400">{grpoTrainingTime ? `${Math.round(grpoTrainingTime / 60)} min` : '35 min'}</div>
                <div className="text-xs text-slate-400 mt-1">GRPO (Group Relative)</div>
                <div className="text-xs text-emerald-400 mt-0.5">No labels needed</div>
              </div>
            </div>
            <p className="text-xs text-slate-500 mt-3">
              GRPO takes longer per run because it generates 8 completions per prompt (8x
              the compute per step). But it eliminates the hours of human labeling that DPO
              requires, making it far more scalable to new tasks.
            </p>
          </div>

          {/* RLVR */}
          <div className="p-4 rounded-lg bg-emerald-950/20 border border-emerald-800/30">
            <h4 className="text-sm font-semibold text-emerald-400 mb-2">
              RLVR: Reinforcement Learning with Verifiable Rewards
            </h4>
            <p className="text-sm text-slate-300 leading-relaxed">
              GRPO is powerful, but it only works when you can <em>automatically check</em>{' '}
              if an answer is correct. This class of problems is called{' '}
              <strong>Reinforcement Learning with Verifiable Rewards (RLVR)</strong>.
              Examples:
            </p>
            <ul className="text-sm text-slate-400 mt-2 space-y-1 list-disc list-inside">
              <li>
                <strong>Our task:</strong> Does "Classification: OLAP Analytics" match the
                ground truth label? Verifiable.
              </li>
              <li>
                <strong>Math:</strong> Does the model's answer equal 42? Verifiable.
              </li>
              <li>
                <strong>Code:</strong> Do the unit tests pass? Verifiable.
              </li>
              <li>
                <strong className="text-slate-500">Creative writing:</strong> Is this poem
                "good"? <em>Not</em> verifiable &mdash; you'd still need DPO or RLHF for
                this.
              </li>
            </ul>
            <p className="text-xs text-slate-500 mt-3 italic">
              RLVR is what makes GRPO practical. For any task with a clear right/wrong
              answer, you can skip human labeling entirely.
            </p>
          </div>
        </div>
      )}

      {/* ================================================================= */}
      {/* TAB 3 -- See It Work (generation reveal)                          */}
      {/* ================================================================= */}
      {section === 'demo' && (
        <div className="space-y-4">
          <div className="p-4 rounded-lg bg-slate-800/40 border border-slate-700/50 mb-2">
            <p className="text-sm text-slate-300 leading-relaxed">
              Below is an ambiguous I/O pattern that could plausibly be OLAP Analytics or
              AI/ML Training. The model generates <strong className="text-emerald-400">8
              attempts</strong>. Each is automatically scored: correct classification = 1.0,
              incorrect = 0.0. Reveal them one at a time or all at once to see how the
              group splits.
            </p>
          </div>

          {/* Real data indicator */}
          {usingRealData && (
            <div className="flex items-center gap-1.5 text-xs text-emerald-400/80 mb-2">
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400" />
              Using real training data
            </div>
          )}

          {/* Input */}
          <div className="mb-4">
            <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
              Challenge Input (Ambiguous: OLAP vs AI Training)
            </label>
            <div className="bg-slate-800 border border-slate-600 rounded-lg p-3 font-mono text-sm text-slate-200">
              {activeExample.input}
            </div>
            <p className="text-xs text-slate-500 mt-1">
              Correct answer: <span className="text-emerald-400 font-semibold">{activeExample.correctLabel}</span> |
              The model generates <strong>{activeExample.generations.length} attempts</strong>. Each is scored: correct = 1.0, incorrect = 0.0.
            </p>
          </div>

          {/* Reveal controls */}
          <div className="flex gap-2 mb-4">
            <button
              onClick={revealNext}
              disabled={revealedCount >= activeExample.generations.length}
              className="px-4 py-2 text-sm bg-emerald-700 hover:bg-emerald-600 disabled:bg-slate-700 disabled:text-slate-500 rounded-md transition-colors"
            >
              Reveal next generation ({revealedCount}/{activeExample.generations.length})
            </button>
            <button
              onClick={revealAll}
              className="px-4 py-2 text-sm bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-md transition-colors"
            >
              Reveal all
            </button>
          </div>

          {/* Generation cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
            {activeExample.generations.map((gen, i) => {
              const isRevealed = i < revealedCount
              const advantage = gen.reward - activeMeanReward

              return (
                <div
                  key={gen.id}
                  className={`p-3 rounded-lg border transition-all duration-500 ${
                    !isRevealed
                      ? 'border-slate-700/30 bg-slate-800/20 opacity-30'
                      : gen.correct
                      ? 'border-emerald-700/50 bg-emerald-950/20'
                      : 'border-red-800/50 bg-red-950/20'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-semibold text-slate-500">
                      Generation #{gen.id}
                    </span>
                    {isRevealed && (
                      <div className="flex items-center gap-2">
                        <span className={`text-xs px-2 py-0.5 rounded-full font-semibold ${
                          gen.correct
                            ? 'bg-emerald-900/50 text-emerald-400'
                            : 'bg-red-900/50 text-red-400'
                        }`}>
                          Reward: {gen.reward.toFixed(1)}
                        </span>
                        {showStats && (
                          <span className={`text-xs px-2 py-0.5 rounded-full ${
                            advantage > 0
                              ? 'bg-emerald-900/30 text-emerald-300'
                              : 'bg-red-900/30 text-red-300'
                          }`}>
                            Adv: {advantage > 0 ? '+' : ''}{advantage.toFixed(2)}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                  <pre className={`text-xs font-mono whitespace-pre-wrap leading-relaxed ${
                    isRevealed ? 'text-slate-200' : 'text-slate-600'
                  }`}>
                    {isRevealed ? gen.text : 'Click "Reveal next" to see this generation...'}
                  </pre>
                </div>
              )
            })}
          </div>

          {/* Group statistics */}
          {showStats && (
            <div className="p-4 rounded-lg bg-emerald-950/20 border border-emerald-800/30">
              <h4 className="text-sm font-semibold text-emerald-400 mb-3">Group Statistics (GRPO)</h4>
              <div className="grid grid-cols-4 gap-4 text-center mb-3">
                <div>
                  <div className="text-2xl font-bold text-emerald-400">{correctCount}/{activeExample.generations.length}</div>
                  <div className="text-xs text-slate-500">Correct</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-slate-200">{activeMeanReward.toFixed(3)}</div>
                  <div className="text-xs text-slate-500">Mean Reward</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-slate-200">{activeStdReward.toFixed(3)}</div>
                  <div className="text-xs text-slate-500">Std Dev</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-slate-200">0</div>
                  <div className="text-xs text-slate-500">Human Labels Needed</div>
                </div>
              </div>

              <p className="text-xs text-slate-400">
                <strong className="text-emerald-300">How GRPO learns:</strong> The mean reward ({activeMeanReward.toFixed(3)}) is the baseline.
                The {correctCount} correct generations get <span className="text-emerald-300">positive advantage</span> (+{(1.0 - activeMeanReward).toFixed(2)}) &mdash; they're reinforced.
                The {activeExample.generations.length - correctCount} incorrect generations get <span className="text-red-300">negative advantage</span> ({(0 - activeMeanReward).toFixed(2)}) &mdash; they're suppressed.
                No critic network, no reward model &mdash; just group statistics.
              </p>

              <p className="text-xs text-slate-500 mt-2 italic">
                This is the technique behind DeepSeek R1. The answer itself is the teacher.
                For our task, "correct classification" is verifiable. For math, "correct answer" is verifiable.
                That's why it's called Reinforcement Learning with Verifiable Rewards (RLVR).
              </p>
            </div>
          )}
        </div>
      )}

      {/* ================================================================= */}
      {/* TAB 4 -- Under the Covers                                         */}
      {/* ================================================================= */}
      {section === 'deepdive' && (
        <div className="space-y-6">
          {isLoaded() && (
            <div className="flex items-center gap-1.5 text-xs text-emerald-400/80">
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400" />
              Using real training data
            </div>
          )}

          {/* Viz sub-tabs */}
          <div className="flex gap-1 bg-slate-800 rounded-lg p-1 w-fit">
            {vizTabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setVizTab(tab.id)}
                className={`px-4 py-2 text-sm rounded-md transition-colors ${
                  vizTab === tab.id
                    ? 'bg-emerald-600 text-white font-semibold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Advantage waterfall */}
          {vizTab === 'advantage' && (
            <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
              <p className="text-sm text-slate-300 mb-4 leading-relaxed">
                This is the core GRPO mechanism. The group mean reward (0.50) is the baseline.
                Correct classifications (green) get <strong className="text-emerald-400">positive advantage</strong> &mdash; they're reinforced.
                Incorrect ones (red) get <strong className="text-red-400">negative advantage</strong> &mdash; they're suppressed.
                The model learns to generate more responses like the winners.
              </p>
              <AdvantageWaterfall />
              <p className="text-xs text-slate-500 mt-3">
                Notice: no external reward model, no human labels. The advantage is computed entirely
                from the group's own statistics. This is what makes GRPO 50% cheaper than PPO.
              </p>
            </div>
          )}

          {/* Accuracy over training */}
          {vizTab === 'accuracy' && (
            <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
              <p className="text-sm text-slate-300 mb-4 leading-relaxed">
                Classification accuracy improves over training as the model learns from its own
                successes and failures. Starting at ~45% (random-ish), reaching ~85% after 200 steps.
                Each step involves generating 8 completions, scoring them, and updating the model
                toward the correct ones.
              </p>
              <LossChart
                data={accuracyCurve}
                label="Classification Accuracy Over Training"
                color="#10b981"
                width={550}
                height={250}
              />
            </div>
          )}

          {/* Mean reward curve */}
          {vizTab === 'reward' && (
            <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
              <p className="text-sm text-slate-300 mb-4 leading-relaxed">
                Mean group reward increases as more generations become correct. This curve
                directly reflects the model's improving ability to classify I/O patterns.
                A mean reward of 1.0 would mean every generation in every group is correct.
              </p>
              <LossChart
                data={rewardCurve}
                label="Mean Group Reward Over Training"
                color="#10b981"
                width={550}
                height={250}
              />
            </div>
          )}

          {/* GRPO formula */}
          <div className="p-4 rounded bg-slate-800 border border-slate-700/50">
            <p className="text-xs text-slate-500 mb-2 font-semibold uppercase tracking-wide">
              GRPO update rule (simplified)
            </p>
            <p className="text-sm text-slate-300 font-mono">
              advantage_i = (reward_i - mean(rewards)) / std(rewards)
            </p>
            <p className="text-sm text-slate-300 font-mono mt-1">
              loss = -sum(advantage_i * log_prob(generation_i))
            </p>
            <p className="text-xs text-slate-500 mt-3 leading-relaxed">
              <strong className="text-slate-400">Reading the formula:</strong> The advantage
              tells the model how much better (or worse) each generation was compared to the
              group. Multiplying by log_prob means "adjust the probability of generating this
              response by this much." Positive advantage = reinforce. Negative = suppress.
              The std normalization prevents the model from only making tiny updates.
            </p>
          </div>

          {/* Infrastructure -- always visible */}
          <InfrastructureCard data={GRPO_INFRA} />
        </div>
      )}

      {/* ---- Tab bar (bottom) ---- */}
      <div className="mt-6">
        <SectionTabs tabs={tabs} active={section} onSelect={setSection} color="emerald" />
      </div>
    </div>
  )
}
