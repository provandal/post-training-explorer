import { useState, useRef, useEffect } from 'react'
import * as d3 from 'd3'
import useStore from '../store'
import ModelOutput from '../components/ModelOutput'
import InfrastructureCard from '../components/InfrastructureCard'

// ---------------------------------------------------------------------------
// Precomputed data: Preference pairs (interactive demo)
// ---------------------------------------------------------------------------
const PREFERENCE_PAIRS = [
  {
    input: "IOPS: 45000 | Latency: 0.3ms | Block Size: 8K | Read/Write: 70/30 | Sequential: 15% | Queue Depth: 32",
    responseA: `Classification: OLTP Database
Confidence: High
Key indicators: High IOPS (45K) with very low latency (0.3ms) and small block size (8K) are hallmarks of transaction processing. Random-dominant access (85%) with moderate read bias confirms OLTP pattern.`,
    responseB: `Classification: OLTP Database
This could potentially be an OLTP Database workload. The IOPS are quite high at 45000, and the block size of 8K is relatively small. The latency is low at 0.3ms. However, it could also possibly be a VDI workload given the mixed read/write ratio. On balance, OLTP Database seems most likely but I'm not entirely certain. The queue depth of 32 and 15% sequential access are also consistent with this classification, though other workload types occasionally show similar patterns.`,
    betterIs: 'A',
    whyABetter: 'Concise, confident, structured format with key indicators highlighted',
    whyBWorse: 'Correct answer but verbose, hedging, mentions alternatives unnecessarily',
  },
  {
    input: "IOPS: 2500 | Latency: 18ms | Block Size: 512K | Read/Write: 5/95 | Sequential: 93% | Queue Depth: 2",
    responseA: `Classification: Backup Archive
Confidence: High
Key indicators: Very large sequential writes (512K blocks, 93% sequential) with heavy write bias (5/95) and low queue depth (2) indicate batch backup or archive operations. High latency tolerance (18ms) is consistent with background archival processes.`,
    responseB: `Classification: Backup Archive
The storage metrics indicate this is a backup or archive workload.`,
    betterIs: 'A',
    whyABetter: 'Includes explanation and reasoning, not just the label',
    whyBWorse: 'Too terse — correct but provides no actionable information',
  },
  {
    input: "IOPS: 22000 | Latency: 1.2ms | Block Size: 256K | Read/Write: 96/4 | Sequential: 88% | Queue Depth: 64",
    responseA: `Classification: AI ML Training
Confidence: High
Key indicators: Large sequential reads (256K, 88% sequential) with extreme read bias (96/4) and high queue depth (64) indicate data pipeline feeding GPU training. Moderate IOPS with high throughput profile is characteristic of deep learning data loaders.`,
    responseB: `Classification: AI ML Training
Confidence: Medium-High
Key indicators: Large block sequential reads (256K, 88% sequential) with heavy read bias (96/4). High queue depth (64) suggests parallel data loading.
Note: Could be confused with Video Streaming, but IOPS (22K) and queue depth are too high for streaming.`,
    betterIs: 'B',
    whyABetter: null,
    whyBWorse: null,
  },
]

// ---------------------------------------------------------------------------
// Precomputed data: Probability shift (Under the Covers)
// ---------------------------------------------------------------------------
const PROB_SHIFT = {
  example: "IOPS: 45000 | Latency: 0.3ms | Block Size: 8K | Read/Write: 70/30 | Sequential: 15% | Queue Depth: 32",
  chosen: "Concise, confident, structured",
  rejected: "Verbose, hedging, uncertain",
  before: { chosenLogProb: -2.1, rejectedLogProb: -1.8 },
  after: { chosenLogProb: -0.9, rejectedLogProb: -3.2 },
}

// ---------------------------------------------------------------------------
// Precomputed data: Infrastructure profile
// ---------------------------------------------------------------------------
const DPO_INFRA = {
  gpuMemoryGB: 5.1,
  trainingTimeMinutes: 8,
  checkpointSizeMB: 1.7,
  peakGPUUtilization: 82,
  modelsInMemory: 1,
  storageIOPattern: "Similar to SFT \u2014 reads preference pairs, writes checkpoints periodically",
  note: "DPO needs reference model logits, but these can be precomputed. Effective memory is ~1.2x SFT.",
  vsRLHF: {
    rlhfGPUMemoryGB: 12.8,
    rlhfTrainingTimeMinutes: 45,
    rlhfModelsInMemory: 3,
    dpoModelsInMemory: 1,
    note: "DPO achieves similar alignment with ~60% less compute by skipping the reward model entirely."
  }
}

// ---------------------------------------------------------------------------
// D3 visualization: Probability shift chart
// ---------------------------------------------------------------------------
function ProbabilityShiftChart() {
  const svgRef = useRef()

  useEffect(() => {
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const width = 500
    const height = 250
    const margin = { top: 40, right: 30, bottom: 50, left: 70 }
    const w = width - margin.left - margin.right
    const h = height - margin.top - margin.bottom

    svg.attr('width', width).attr('height', height)
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    const categories = ['Chosen\n(concise)', 'Rejected\n(verbose)']
    const beforeData = [PROB_SHIFT.before.chosenLogProb, PROB_SHIFT.before.rejectedLogProb]
    const afterData = [PROB_SHIFT.after.chosenLogProb, PROB_SHIFT.after.rejectedLogProb]

    const x = d3.scaleBand().domain(categories).range([0, w]).padding(0.4)
    const y = d3.scaleLinear().domain([-4, 0]).range([h, 0])

    svg.append('text').attr('x', width / 2).attr('y', 16)
      .attr('text-anchor', 'middle').attr('fill', '#94a3b8')
      .attr('font-size', '11').attr('font-weight', '600')
      .text('Log Probability Shift: Before vs After DPO')

    g.selectAll('.grid').data(y.ticks(4)).join('line')
      .attr('x1', 0).attr('x2', w)
      .attr('y1', d => y(d)).attr('y2', d => y(d))
      .attr('stroke', '#1e293b').attr('stroke-dasharray', '2,2')

    g.append('g').call(d3.axisLeft(y).ticks(4))
      .selectAll('text').attr('fill', '#64748b').attr('font-size', '9')
    g.append('text').attr('transform', 'rotate(-90)')
      .attr('x', -h / 2).attr('y', -50)
      .attr('text-anchor', 'middle').attr('fill', '#64748b').attr('font-size', '9')
      .text('Log Probability (higher = more likely)')

    g.selectAll('.domain').attr('stroke', '#334155')

    const barWidth = x.bandwidth() / 2 - 3

    g.selectAll('.before')
      .data(categories)
      .join('rect')
      .attr('x', d => x(d))
      .attr('y', h)
      .attr('width', barWidth)
      .attr('height', 0)
      .attr('fill', '#ef4444')
      .attr('opacity', 0.6)
      .attr('rx', 3)
      .transition().duration(700)
      .attr('y', (d, i) => y(beforeData[i]))
      .attr('height', (d, i) => h - y(beforeData[i]))

    g.selectAll('.after')
      .data(categories)
      .join('rect')
      .attr('x', d => x(d) + barWidth + 6)
      .attr('y', h)
      .attr('width', barWidth)
      .attr('height', 0)
      .attr('fill', (d, i) => i === 0 ? '#22c55e' : '#f97316')
      .attr('opacity', 0.8)
      .attr('rx', 3)
      .transition().duration(700).delay(400)
      .attr('y', (d, i) => y(afterData[i]))
      .attr('height', (d, i) => h - y(afterData[i]))

    g.selectAll('.label-before')
      .data(categories)
      .join('text')
      .attr('x', d => x(d) + barWidth / 2)
      .attr('y', (d, i) => y(beforeData[i]) - 5)
      .attr('text-anchor', 'middle')
      .attr('fill', '#ef4444').attr('font-size', '10').attr('font-weight', '600')
      .text((d, i) => beforeData[i].toFixed(1))
      .attr('opacity', 0)
      .transition().delay(700).attr('opacity', 1)

    g.selectAll('.label-after')
      .data(categories)
      .join('text')
      .attr('x', d => x(d) + barWidth + 6 + barWidth / 2)
      .attr('y', (d, i) => y(afterData[i]) - 5)
      .attr('text-anchor', 'middle')
      .attr('fill', (d, i) => i === 0 ? '#22c55e' : '#f97316')
      .attr('font-size', '10').attr('font-weight', '600')
      .text((d, i) => afterData[i].toFixed(1))
      .attr('opacity', 0)
      .transition().delay(1100).attr('opacity', 1)

    categories.forEach((cat) => {
      cat.split('\n').forEach((line, j) => {
        g.append('text')
          .attr('x', x(cat) + x.bandwidth() / 2)
          .attr('y', h + 15 + j * 12)
          .attr('text-anchor', 'middle')
          .attr('fill', '#94a3b8').attr('font-size', '9')
          .text(line)
      })
    })

    const legend = svg.append('g').attr('transform', `translate(${margin.left + 10}, ${margin.top - 18})`)
    legend.append('rect').attr('width', 10).attr('height', 10).attr('fill', '#ef4444').attr('opacity', 0.6).attr('rx', 2)
    legend.append('text').attr('x', 14).attr('y', 9).attr('fill', '#94a3b8').attr('font-size', '9').text('Before DPO')
    legend.append('rect').attr('x', 100).attr('width', 10).attr('height', 10).attr('fill', '#22c55e').attr('opacity', 0.8).attr('rx', 2)
    legend.append('text').attr('x', 114).attr('y', 9).attr('fill', '#94a3b8').attr('font-size', '9').text('After DPO')
  }, [])

  return <svg ref={svgRef} />
}

// ===========================================================================
// Main component
// ===========================================================================
export default function DPOPreferences({ explore = false }) {
  const [section, setSection] = useState('problem')

  // --- preference picker state ---
  const [currentPair, setCurrentPair] = useState(0)
  const [userChoice, setUserChoice] = useState(null)
  const [showReveal, setShowReveal] = useState(false)
  const addPreference = useStore((s) => s.addPreference)

  const pair = PREFERENCE_PAIRS[currentPair]

  const handleChoice = (choice) => {
    setUserChoice(choice)
    setShowReveal(true)
    addPreference({ pairIndex: currentPair, choice })
  }

  const nextPair = () => {
    if (currentPair < PREFERENCE_PAIRS.length - 1) {
      setCurrentPair(currentPair + 1)
      setUserChoice(null)
      setShowReveal(false)
    }
  }

  const tabs = [
    { id: 'problem', label: 'The Problem' },
    { id: 'concept', label: 'How DPO Works' },
    { id: 'demo', label: 'See It Work' },
    { id: 'deepdive', label: 'Under the Covers' },
  ]

  return (
    <div className="max-w-5xl mx-auto">
      {/* ---- Tab bar ---- */}
      <div className="flex gap-1 mb-6 bg-slate-800 rounded-lg p-1 w-fit">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setSection(tab.id)}
            className={`px-4 py-2 text-sm rounded-md transition-colors ${
              section === tab.id
                ? 'bg-pink-600 text-white font-semibold'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* ================================================================= */}
      {/* TAB 1 -- The Problem                                              */}
      {/* ================================================================= */}
      {section === 'problem' && (
        <div className="space-y-6">
          <div className="p-5 rounded-lg bg-slate-800/40 border border-slate-700/50">
            <h3 className="text-lg font-semibold text-pink-400 mb-3">
              SFT taught the model WHAT to say. But not HOW to say it.
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed mb-4">
              After supervised fine-tuning, our SmolLM2-360M model can correctly classify
              storage I/O patterns. It knows that high IOPS with small blocks and low
              latency means OLTP. That's the "what." But look at the difference in
              <em> how</em> two correct answers can sound:
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <ModelOutput
                label="Response A — Concise & Confident"
                variant="chosen"
                text={`Classification: OLTP Database
Confidence: High
Key indicators: High IOPS (45K) with very low latency (0.3ms) and small block size (8K) are hallmarks of transaction processing. Random-dominant access (85%) with moderate read bias confirms OLTP pattern.`}
              />
              <ModelOutput
                label="Response B — Verbose & Hedging"
                variant="rejected"
                text={`Classification: OLTP Database
This could potentially be an OLTP Database workload. The IOPS are quite high at 45000, and the block size of 8K is relatively small. The latency is low at 0.3ms. However, it could also possibly be a VDI workload given the mixed read/write ratio. On balance, OLTP Database seems most likely but I'm not entirely certain.`}
              />
            </div>

            <div className="p-4 rounded-lg bg-pink-950/20 border border-pink-800/30">
              <p className="text-sm text-pink-300 font-semibold mb-2">
                Both answers are correct. So why does it matter?
              </p>
              <p className="text-sm text-slate-400 leading-relaxed mb-3">
                Imagine it's 3 AM during an incident and your monitoring pipeline surfaces
                one of these responses. Response A gives you a clear classification, a
                confidence level, and the reasoning in a scannable format. Response B
                waffles, hedges, and buries the answer in qualifications. In production,
                that difference matters.
              </p>
              <p className="text-sm text-slate-400 leading-relaxed mb-3">
                The model defaults to the verbose, hedging style because cautious language
                was more common in its pretraining data. Phrases like "could potentially
                be" and "I'm not entirely certain" are safe bets when you've been trained
                on internet text where hedging is everywhere.
              </p>
              <p className="text-sm text-slate-400 leading-relaxed">
                SFT can't fix this because it only trains on "correct vs. incorrect." It
                has no mechanism to express <em>"correct AND well-formatted"</em> vs.{' '}
                <em>"correct but unhelpful."</em>
              </p>
            </div>
          </div>

          {/* Transition */}
          <div className="p-4 rounded bg-pink-950/20 border border-pink-800/30">
            <p className="text-sm text-pink-300">
              We need a way to teach the model our <strong>style preferences</strong> —
              not just what answer to give, but how to give it. That's exactly what{' '}
              <strong className="text-pink-200">Direct Preference Optimization (DPO)</strong>{' '}
              does. Head to the next tab to see how it works.
            </p>
          </div>
        </div>
      )}

      {/* ================================================================= */}
      {/* TAB 2 -- How DPO Works                                            */}
      {/* ================================================================= */}
      {section === 'concept' && (
        <div className="space-y-6">
          {/* Core idea */}
          <div className="p-5 rounded-lg bg-slate-800/40 border border-slate-700/50">
            <h3 className="text-lg font-semibold text-pink-400 mb-3">
              Teaching with comparisons, not examples
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed mb-3">
              SFT teaches by showing the model examples: "given this input, produce this
              output." <strong className="text-pink-300">DPO (Direct Preference
              Optimization)</strong> teaches a different way — through comparisons. Instead
              of saying "here's the right answer," you say "this answer is{' '}
              <em>better than</em> that one."
            </p>
            <p className="text-sm text-slate-400 leading-relaxed">
              Think of it like training a new hire. SFT is handing them a style guide and
              saying "write reports like this." DPO is showing them two reports side by side
              and saying "this one is better — do you see why?" Both work, but comparisons
              teach nuances that are hard to spell out in a single example.
            </p>
          </div>

          {/* RLHF: the old way */}
          <div className="p-5 rounded-lg bg-slate-800/40 border border-slate-700/50">
            <h3 className="text-lg font-semibold text-pink-400 mb-3">
              The backstory: why not RLHF?
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed mb-4">
              Before DPO, the standard approach to preference learning was{' '}
              <strong>Reinforcement Learning from Human Feedback (RLHF)</strong>. It works,
              but it's complicated and expensive:
            </p>
            <ol className="text-sm text-slate-300 space-y-3 list-decimal list-inside mb-4">
              <li>
                <strong className="text-red-400">Collect preferences.</strong> Show humans
                two model outputs for the same input. They pick the better one. (This is
                exactly what you'll do in the "See It Work" tab.)
              </li>
              <li>
                <strong className="text-red-400">Train a reward model.</strong> A separate
                neural network learns to predict which response a human would prefer. That's
                Model #2 in GPU memory.
              </li>
              <li>
                <strong className="text-red-400">Optimize with PPO.</strong> Use Proximal
                Policy Optimization (a reinforcement learning algorithm) to maximize the
                reward model's score. This requires a value network too — Model #3 in memory.
              </li>
            </ol>
            <p className="text-sm text-slate-400 italic">
              Three models running simultaneously. Unstable training dynamics. 45 minutes for
              our small 360M model. It works, but the engineering cost is steep.
            </p>
          </div>

          {/* DPO: the insight */}
          <div className="p-5 rounded-lg bg-pink-950/15 border border-pink-800/30">
            <h3 className="text-lg font-semibold text-pink-400 mb-3">
              DPO: skip the reward model entirely
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed mb-4">
              In 2023, Rafailov et al. published a key insight: there is a{' '}
              <strong>closed-form solution</strong> that maps directly from preference data
              to the optimal policy. You don't need to train a reward model first and then
              do RL against it. You can collapse those two steps into a single training loss.
            </p>

            {/* RLHF vs DPO comparison */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-5">
              <div className="p-4 rounded-lg bg-red-950/20 border border-red-800/30">
                <h4 className="text-sm font-semibold text-red-400 mb-2">RLHF (3 models)</h4>
                <div className="text-xs text-slate-400 space-y-1.5">
                  <p>1. Policy model (the LLM you're training)</p>
                  <p>2. Reward model (predicts human preference)</p>
                  <p>3. Value network (PPO baseline estimator)</p>
                </div>
                <p className="text-xs text-red-300 mt-3 font-semibold">
                  12.8 GB GPU memory. 45 min training. Unstable.
                </p>
              </div>
              <div className="p-4 rounded-lg bg-green-950/20 border border-green-800/30">
                <h4 className="text-sm font-semibold text-green-400 mb-2">DPO (1 model)</h4>
                <div className="text-xs text-slate-400 space-y-1.5">
                  <p>1. Policy model (the LLM you're training)</p>
                  <p className="text-slate-600 line-through">2. No reward model needed</p>
                  <p className="text-slate-600 line-through">3. No value network needed</p>
                </div>
                <p className="text-xs text-green-300 mt-3 font-semibold">
                  5.1 GB GPU memory. 8 min training. Stable.
                </p>
              </div>
            </div>

            {/* Analogy */}
            <div className="p-4 rounded bg-slate-800 border border-slate-700/50 mb-5">
              <p className="text-sm text-slate-300 leading-relaxed">
                <strong className="text-pink-300">Analogy:</strong> RLHF is like hiring a
                quality inspector (the reward model) to check every report your employee
                writes, then giving the employee feedback based on the inspector's scores.
                DPO is like the employee learning directly from your feedback — no
                middleman. Same outcome, less overhead.
              </p>
            </div>

            {/* Formula */}
            <div className="p-4 rounded bg-slate-800 border border-slate-700/50 mb-5">
              <p className="text-xs text-slate-500 mb-2 font-semibold uppercase tracking-wide">
                The DPO loss (simplified)
              </p>
              <p className="text-base text-slate-200 font-mono mb-3">
                Loss = -log(sigma(beta * (log P(chosen) - log P(rejected))))
              </p>
              <div className="text-sm text-slate-400 space-y-2 leading-relaxed">
                <p>
                  <strong className="text-slate-300">In plain English:</strong> For each
                  preference pair, compare how likely the model thinks the chosen response is
                  vs. the rejected one. If the model already agrees with the human preference,
                  the loss is low. If it disagrees, the loss is high and the gradient pushes
                  the model to increase the probability of the chosen response and decrease
                  the rejected one.
                </p>
                <p>
                  <strong className="text-pink-300">beta</strong> controls how aggressively
                  preferences are enforced. A high beta means "stick closely to the reference
                  model, just nudge preferences." A low beta means "reshape outputs
                  dramatically." We used beta = 0.1 for our training run.
                </p>
              </div>
            </div>

            {/* Key insight */}
            <div className="p-4 rounded-lg bg-blue-950/20 border border-blue-800/30">
              <p className="text-sm text-blue-300 leading-relaxed">
                <strong>Key insight:</strong> DPO doesn't need a separate reward model. It
                learns preferences directly from chosen/rejected pairs — same result as RLHF
                with 60% less compute. For our demo: 400 preference pairs, 0.12% of
                parameters trained via LoRA, 8 minutes of training on a single GPU.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ================================================================= */}
      {/* TAB 3 -- See It Work (interactive preference picker)              */}
      {/* ================================================================= */}
      {section === 'demo' && (
        <div className="space-y-4">
          <div className="p-4 rounded-lg bg-slate-800/40 border border-slate-700/50 mb-2">
            <p className="text-sm text-slate-300 leading-relaxed">
              You're about to do exactly what DPO training data looks like: comparing two
              model outputs and choosing the one you prefer. We have{' '}
              <strong className="text-pink-400">3 preference pairs</strong> below. Both
              responses in each pair are <em>correct</em> classifications — the
              difference is in style, confidence, and helpfulness. Click the one you'd
              rather see in a production monitoring system.
            </p>
          </div>

          {/* Progress dots */}
          <div className="flex items-center gap-2 mb-4">
            <span className="text-xs text-slate-500">Preference pair</span>
            {PREFERENCE_PAIRS.map((_, i) => (
              <button
                key={i}
                onClick={() => { setCurrentPair(i); setUserChoice(null); setShowReveal(false) }}
                className={`w-8 h-8 rounded-full text-xs font-semibold transition-colors ${
                  i === currentPair
                    ? 'bg-pink-600 text-white'
                    : i < currentPair
                    ? 'bg-pink-900/50 text-pink-400'
                    : 'bg-slate-700 text-slate-500'
                }`}
              >
                {i + 1}
              </button>
            ))}
          </div>

          {/* Input */}
          <div className="mb-4 bg-slate-800 border border-slate-600 rounded-lg p-3 font-mono text-sm text-slate-200">
            {pair.input}
          </div>

          {/* Instruction */}
          {!userChoice && (
            <p className="text-sm text-pink-300 mb-4 font-semibold">
              Both responses are correct. Which one do you prefer? Click to choose.
            </p>
          )}

          {/* Side-by-side responses */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div
              onClick={() => !userChoice && handleChoice('A')}
              className={`cursor-pointer transition-all ${!userChoice ? 'hover:scale-[1.01]' : ''} ${
                userChoice === 'A' ? 'ring-2 ring-green-500' : userChoice === 'B' ? 'opacity-50' : ''
              }`}
            >
              <ModelOutput
                label="Response A"
                text={pair.responseA}
                variant={userChoice === 'A' ? 'chosen' : userChoice === 'B' ? 'rejected' : 'default'}
              />
            </div>
            <div
              onClick={() => !userChoice && handleChoice('B')}
              className={`cursor-pointer transition-all ${!userChoice ? 'hover:scale-[1.01]' : ''} ${
                userChoice === 'B' ? 'ring-2 ring-green-500' : userChoice === 'A' ? 'opacity-50' : ''
              }`}
            >
              <ModelOutput
                label="Response B"
                text={pair.responseB}
                variant={userChoice === 'B' ? 'chosen' : userChoice === 'A' ? 'rejected' : 'default'}
              />
            </div>
          </div>

          {/* Reveal after choice */}
          {showReveal && (
            <div className="mt-4 p-4 rounded-lg bg-pink-950/20 border border-pink-800/30">
              <p className="text-sm text-slate-300 mb-2">
                <span className="font-semibold text-pink-400">You picked Response {userChoice}.</span>{' '}
                {userChoice === pair.betterIs
                  ? "That matches the training data's preference."
                  : pair.betterIs
                  ? `Interesting! The training data preferred Response ${pair.betterIs}, but preferences are subjective \u2014 that's the whole point.`
                  : "This one is genuinely ambiguous \u2014 reasonable people disagree, and that's informative too."
                }
              </p>
              {pair.whyABetter && (
                <p className="text-xs text-slate-400">
                  <strong>Why A is typically preferred:</strong> {pair.whyABetter}
                </p>
              )}
              {pair.whyBWorse && (
                <p className="text-xs text-slate-400">
                  <strong>Why B is typically less preferred:</strong> {pair.whyBWorse}
                </p>
              )}

              <p className="text-xs text-slate-500 mt-3 italic">
                You just did exactly what DPO training data looks like: comparing two outputs and saying
                which is better. 400 pairs like this is all it takes to reshape the model's style.
              </p>

              {currentPair < PREFERENCE_PAIRS.length - 1 && (
                <button onClick={nextPair} className="mt-3 px-4 py-1.5 text-sm bg-pink-700 hover:bg-pink-600 rounded-md transition-colors">
                  Next pair &rarr;
                </button>
              )}
            </div>
          )}
        </div>
      )}

      {/* ================================================================= */}
      {/* TAB 4 -- Under the Covers                                         */}
      {/* ================================================================= */}
      {section === 'deepdive' && (
        <div className="space-y-6">
          {/* Probability shift visualization */}
          <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h3 className="text-base font-semibold text-pink-400 mb-3">
              How DPO reshapes the model's probabilities
            </h3>
            <p className="text-sm text-slate-300 mb-4 leading-relaxed">
              The chart below shows log probabilities for two response styles — concise
              (chosen) and verbose (rejected) — before and after DPO training. Log
              probabilities are negative numbers where <strong className="text-slate-200">
              higher (closer to 0) means more likely</strong>.
            </p>

            <ProbabilityShiftChart />

            <div className="mt-5 space-y-3">
              <div className="flex gap-3 items-start">
                <span className="text-lg leading-none mt-0.5">1.</span>
                <p className="text-sm text-slate-300 leading-relaxed">
                  <strong className="text-red-400">Before DPO</strong>, the model actually
                  slightly preferred the verbose response (log prob -1.8 vs. -2.1 for
                  concise). Hedging language was more common in its training data, so the
                  model defaulted to "cautious" phrasing.
                </p>
              </div>
              <div className="flex gap-3 items-start">
                <span className="text-lg leading-none mt-0.5">2.</span>
                <p className="text-sm text-slate-300 leading-relaxed">
                  <strong className="text-green-400">After DPO</strong>, the concise response
                  jumped to -0.9 (strongly preferred) while the verbose response dropped to
                  -3.2 (strongly suppressed). That's a complete reversal — the model now
                  generates the concise style by default.
                </p>
              </div>
              <div className="flex gap-3 items-start">
                <span className="text-lg leading-none mt-0.5">3.</span>
                <p className="text-sm text-slate-300 leading-relaxed">
                  <strong className="text-slate-200">The gap matters.</strong> Before DPO the
                  difference was only 0.3 (almost a coin flip). After DPO it's 2.3 — the
                  model is now highly confident about which style to use. Your 400 preference
                  pairs literally reshaped the model's writing style.
                </p>
              </div>
            </div>
          </div>

          {/* DPO loss formula */}
          <div className="p-4 rounded bg-slate-800 border border-slate-700/50">
            <p className="text-xs text-slate-500 mb-1 font-semibold uppercase tracking-wide">
              The DPO insight (simplified)
            </p>
            <p className="text-sm text-slate-300 font-mono mb-2">
              Loss = -log(sigmoid(beta * (log_prob_chosen - log_prob_rejected)))
            </p>
            <p className="text-xs text-slate-500">
              Increase log probability of the chosen response, decrease the rejected.
              No intermediate reward model needed. Beta (0.1 in our run) controls how
              aggressively preferences are enforced.
            </p>
          </div>

          {/* RLHF vs DPO comparison cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 rounded-lg bg-red-950/15 border border-red-800/30">
              <h4 className="text-sm font-semibold text-red-400 mb-2">RLHF (the old way)</h4>
              <ol className="text-xs text-slate-400 space-y-1 list-decimal list-inside">
                <li>Collect human preferences</li>
                <li>Train a separate <strong>reward model</strong> on those preferences</li>
                <li>Use PPO to optimize the policy against the reward model</li>
                <li>Need a <strong>value network</strong> for PPO baseline</li>
              </ol>
              <p className="text-xs text-red-300 mt-2 font-semibold">
                3 models in memory simultaneously. 45 minutes training.
              </p>
            </div>
            <div className="p-4 rounded-lg bg-green-950/15 border border-green-800/30">
              <h4 className="text-sm font-semibold text-green-400 mb-2">DPO (the direct way)</h4>
              <ol className="text-xs text-slate-400 space-y-1 list-decimal list-inside">
                <li>Collect the same human preferences</li>
                <li>Optimize directly from preference pairs</li>
                <li>No reward model, no PPO, no value network</li>
                <li>Math proves this is equivalent to implicit reward modeling</li>
              </ol>
              <p className="text-xs text-green-300 mt-2 font-semibold">
                1 model in memory. 8 minutes training. Same result.
              </p>
            </div>
          </div>

          {/* Infrastructure card - always visible */}
          <div>
            <h3 className="text-base font-semibold text-pink-400 mb-3">
              Infrastructure profile
            </h3>
            <InfrastructureCard data={DPO_INFRA} />
          </div>
        </div>
      )}
    </div>
  )
}
