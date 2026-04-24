// Bonus stop: Deep dive into the evolution of RL algorithms for LLM training.
// Covers slides 30-35 of the SNIA presentation: PPO → DPO/GRPO → emerging methods.

import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import SectionTabs from '../components/SectionTabs'

export default function RLAlgorithmsDeepDive() {
  // eslint-disable-next-line no-unused-vars -- i18n keys will replace hardcoded strings later
  const { t } = useTranslation()
  const [section, setSection] = useState('evolution')

  const TABS = [
    { id: 'evolution', label: 'The Evolution' },
    { id: 'algorithms', label: 'The Algorithms' },
    { id: 'enterprise', label: 'Enterprise Impact' },
  ]

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <span className="text-xs font-semibold text-pink-400 uppercase tracking-wide">
          Bonus Deep Dive
        </span>
        <h2 className="text-xl font-bold text-white mt-1">RL Algorithms for LLM Training</h2>
        <p className="text-sm text-slate-400 mt-2">
          From PPO&rsquo;s 4-model complexity to GRPO&rsquo;s elegant simplification &mdash; how the
          field went from &ldquo;we need four models and a PhD&rdquo; to &ldquo;one reward function
          and group statistics.&rdquo;
        </p>
      </div>

      <SectionTabs tabs={TABS} active={section} onSelect={setSection} color="pink" />

      {/* ── Tab 1: The Evolution ── */}
      {section === 'evolution' && (
        <div className="space-y-6">
          {/* PPO Era */}
          <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h3 className="text-base font-semibold text-pink-400 mb-1">
              PPO Era (2017&ndash;2023)
            </h3>
            <p className="text-xs text-slate-500 mb-3">
              The architecture that made ChatGPT possible
            </p>
            <p className="text-sm text-slate-300 leading-relaxed mb-4">
              Proximal Policy Optimization required{' '}
              <strong className="text-white">four separate models</strong> running simultaneously
              during training. It worked &mdash; spectacularly &mdash; but at enormous cost.
            </p>

            {/* 4-model visual */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
              <div className="p-3 rounded bg-green-950/30 border border-green-800/40 text-center">
                <div className="text-sm font-bold text-green-400">Policy Model</div>
                <div className="text-xs text-slate-500 mt-1">Generates responses</div>
              </div>
              <div className="p-3 rounded bg-orange-950/30 border border-orange-800/40 text-center">
                <div className="text-sm font-bold text-orange-400">Reward Model</div>
                <div className="text-xs text-slate-500 mt-1">Scores quality</div>
              </div>
              <div className="p-3 rounded bg-blue-950/30 border border-blue-800/40 text-center">
                <div className="text-sm font-bold text-blue-400">Critic Model</div>
                <div className="text-xs text-slate-500 mt-1">Estimates expected reward</div>
              </div>
              <div className="p-3 rounded bg-slate-800/50 border border-slate-700/30 text-center">
                <div className="text-sm font-bold text-slate-400">Reference Model</div>
                <div className="text-xs text-slate-500 mt-1">Prevents drift (KL penalty)</div>
              </div>
            </div>

            {/* Data flow */}
            <div className="p-3 rounded bg-slate-900/50 border border-slate-700/30 font-mono text-xs text-slate-400">
              <span className="text-green-400">Policy</span>
              {' → generate → '}
              <span className="text-orange-400">Reward</span>
              {' → score → '}
              <span className="text-blue-400">Critic</span>
              {' → advantage → '}
              <span className="text-green-400">Policy update</span>
              <br />
              <span className="text-slate-500 ml-16">
                ↑ KL penalty from <span className="text-slate-400">Reference</span>
              </span>
            </div>

            <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="p-3 rounded bg-green-950/20 border border-green-800/30">
                <h4 className="text-xs font-semibold text-green-400 uppercase tracking-wide mb-1">
                  Why it was revolutionary
                </h4>
                <p className="text-xs text-slate-400">
                  First reliable method to align LLMs with human preferences. Made ChatGPT, Claude,
                  and every major chat model possible. Stable training via clipped surrogate
                  objective.
                </p>
              </div>
              <div className="p-3 rounded bg-red-950/20 border border-red-800/30">
                <h4 className="text-xs font-semibold text-red-400 uppercase tracking-wide mb-1">
                  Why it&rsquo;s being replaced
                </h4>
                <p className="text-xs text-slate-400">
                  4 models = massive VRAM (often 4&times; the base model size). Complex
                  hyperparameter tuning. Training instability from reward model drift. Difficult to
                  reproduce results.
                </p>
              </div>
            </div>
          </div>

          {/* The Simplification Wave */}
          <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h3 className="text-base font-semibold text-pink-400 mb-1">
              The Simplification Wave (2024&ndash;2025)
            </h3>
            <p className="text-xs text-slate-500 mb-4">
              Each new algorithm removes models from the training pipeline
            </p>

            <div className="space-y-3">
              {/* PPO row */}
              <div className="flex items-center gap-3">
                <span className="w-16 text-right text-xs font-semibold text-slate-400">PPO</span>
                <div className="flex-1 flex gap-2">
                  <span className="px-2 py-1 rounded text-xs bg-green-950/40 border border-green-800/40 text-green-400">
                    Policy
                  </span>
                  <span className="px-2 py-1 rounded text-xs bg-slate-800/50 border border-slate-700/30 text-slate-400">
                    Reference
                  </span>
                  <span className="px-2 py-1 rounded text-xs bg-orange-950/40 border border-orange-800/40 text-orange-400">
                    Reward
                  </span>
                  <span className="px-2 py-1 rounded text-xs bg-blue-950/40 border border-blue-800/40 text-blue-400">
                    Critic
                  </span>
                </div>
                <span className="text-xs text-slate-600">4 models</span>
              </div>

              {/* DPO row */}
              <div className="flex items-center gap-3">
                <span className="w-16 text-right text-xs font-semibold text-pink-400">DPO</span>
                <div className="flex-1 flex gap-2">
                  <span className="px-2 py-1 rounded text-xs bg-green-950/40 border border-green-800/40 text-green-400">
                    Policy
                  </span>
                  <span className="px-2 py-1 rounded text-xs bg-slate-800/50 border border-slate-700/30 text-slate-400">
                    Reference
                  </span>
                  <span className="px-2 py-1 rounded text-xs bg-slate-900/30 border border-slate-800/30 text-slate-600 line-through">
                    Reward
                  </span>
                  <span className="px-2 py-1 rounded text-xs bg-slate-900/30 border border-slate-800/30 text-slate-600 line-through">
                    Critic
                  </span>
                </div>
                <span className="text-xs text-pink-400">2 models</span>
              </div>

              {/* GRPO row */}
              <div className="flex items-center gap-3">
                <span className="w-16 text-right text-xs font-semibold text-emerald-400">GRPO</span>
                <div className="flex-1 flex gap-2">
                  <span className="px-2 py-1 rounded text-xs bg-green-950/40 border border-green-800/40 text-green-400">
                    Policy
                  </span>
                  <span className="px-2 py-1 rounded text-xs bg-slate-800/50 border border-slate-700/30 text-slate-400">
                    Reference
                  </span>
                  <span className="px-2 py-1 rounded text-xs bg-orange-950/40 border border-orange-800/40 text-orange-400">
                    Reward
                  </span>
                  <span className="px-2 py-1 rounded text-xs bg-slate-900/30 border border-slate-800/30 text-slate-600 line-through">
                    Critic
                  </span>
                </div>
                <span className="text-xs text-emerald-400">3 models*</span>
              </div>

              {/* RLOO/ReMax row */}
              <div className="flex items-center gap-3">
                <span className="w-16 text-right text-xs font-semibold text-violet-400">RLOO</span>
                <div className="flex-1 flex gap-2">
                  <span className="px-2 py-1 rounded text-xs bg-green-950/40 border border-green-800/40 text-green-400">
                    Policy
                  </span>
                  <span className="px-2 py-1 rounded text-xs bg-slate-800/50 border border-slate-700/30 text-slate-400">
                    Reference
                  </span>
                  <span className="px-2 py-1 rounded text-xs bg-orange-950/40 border border-orange-800/40 text-orange-400">
                    Reward
                  </span>
                  <span className="px-2 py-1 rounded text-xs bg-slate-900/30 border border-slate-800/30 text-slate-600 line-through">
                    Critic
                  </span>
                </div>
                <span className="text-xs text-violet-400">3 models*</span>
              </div>
            </div>
            <p className="text-xs text-slate-600 mt-3 italic">
              * GRPO and RLOO replace the Critic with group statistics or leave-one-out baselines
              &mdash; no learned value network needed. With verifiable rewards, the Reward model can
              also be a simple function (e.g., &ldquo;does the code pass the test?&rdquo;), bringing
              the effective count to 2.
            </p>
          </div>

          {/* The Convergence */}
          <div className="p-5 rounded-lg bg-gradient-to-r from-pink-950/30 to-emerald-950/30 border border-pink-800/30">
            <h3 className="text-base font-semibold text-pink-400 mb-3">The Convergence</h3>
            <p className="text-sm text-slate-300 leading-relaxed mb-3">
              Despite taking very different paths, DPO and GRPO are mathematically closer than
              initially thought. Both optimize the same{' '}
              <strong className="text-white">KL-regularized objective</strong>
              &mdash; they just approach it differently:
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="p-3 rounded bg-pink-950/20 border border-pink-800/30">
                <h4 className="text-xs font-semibold text-pink-300 mb-1">DPO&rsquo;s Approach</h4>
                <p className="text-xs text-slate-400">
                  Reparameterize the reward model into a closed-form expression over policy and
                  reference log-probabilities. Learn directly from preference pairs &mdash; no
                  explicit reward signal.
                </p>
              </div>
              <div className="p-3 rounded bg-emerald-950/20 border border-emerald-800/30">
                <h4 className="text-xs font-semibold text-emerald-300 mb-1">
                  GRPO&rsquo;s Approach
                </h4>
                <p className="text-xs text-slate-400">
                  Keep explicit rewards but replace the learned critic with group statistics (mean
                  and standard deviation across a batch of generations). Simpler, more stable, and
                  naturally suited to verifiable tasks.
                </p>
              </div>
            </div>
          </div>

          {/* Timeline */}
          <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h3 className="text-base font-semibold text-pink-400 mb-3">Timeline</h3>
            <div className="space-y-2 text-xs">
              <div className="flex gap-3 items-center">
                <span className="w-12 text-right font-mono text-slate-500">2017</span>
                <div className="flex-1 p-2 rounded bg-slate-800/50 border border-slate-700/30 text-slate-300">
                  <strong className="text-slate-200">PPO</strong> &mdash; OpenAI introduces Proximal
                  Policy Optimization
                </div>
              </div>
              <div className="flex gap-3 items-center">
                <span className="w-12 text-right font-mono text-slate-500">2022</span>
                <div className="flex-1 p-2 rounded bg-slate-800/50 border border-slate-700/30 text-slate-300">
                  <strong className="text-slate-200">RLHF + PPO</strong> &mdash; InstructGPT /
                  ChatGPT demonstrates RLHF at scale
                </div>
              </div>
              <div className="flex gap-3 items-center">
                <span className="w-12 text-right font-mono text-pink-500">2023</span>
                <div className="flex-1 p-2 rounded bg-pink-950/20 border border-pink-800/30 text-pink-300">
                  <strong className="text-pink-200">DPO</strong> &mdash; Stanford removes the reward
                  model entirely
                </div>
              </div>
              <div className="flex gap-3 items-center">
                <span className="w-12 text-right font-mono text-emerald-500">2024</span>
                <div className="flex-1 p-2 rounded bg-emerald-950/20 border border-emerald-800/30 text-emerald-300">
                  <strong className="text-emerald-200">GRPO</strong> &mdash; DeepSeek removes the
                  critic; group statistics suffice
                </div>
              </div>
              <div className="flex gap-3 items-center">
                <span className="w-12 text-right font-mono text-violet-500">2025</span>
                <div className="flex-1 p-2 rounded bg-violet-950/20 border border-violet-800/30 text-violet-300">
                  <strong className="text-violet-200">RLOO, ReMax, Dr-GRPO, DAPO</strong> &mdash;
                  Rapid iteration and refinement
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Tab 2: The Algorithms ── */}
      {section === 'algorithms' && (
        <div className="space-y-4">
          <p className="text-sm text-slate-400">
            Each card below follows a consistent format: core idea, mechanism, what it removes vs
            PPO, best use cases, and compute profile.
          </p>

          {/* PPO */}
          <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <div className="flex items-start justify-between mb-2">
              <div>
                <h3 className="text-base font-semibold text-slate-300">PPO</h3>
                <p className="text-xs text-slate-500">Proximal Policy Optimization</p>
              </div>
              <span className="px-2 py-0.5 rounded text-xs bg-slate-700/50 text-slate-400">
                Baseline
              </span>
            </div>
            <p className="text-sm text-slate-300 leading-relaxed mb-3">
              <strong className="text-slate-200">Core idea:</strong> Clip the policy gradient update
              to prevent destructive large steps.
            </p>
            <p className="text-xs text-slate-400 leading-relaxed mb-3">
              The policy generates responses, the reward model scores them, the critic estimates a
              baseline, and the policy is updated using a clipped surrogate objective that prevents
              it from changing too much in a single step. The reference model applies a KL penalty
              to prevent catastrophic forgetting.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
              <div className="p-2 rounded bg-slate-900/50 border border-slate-700/30">
                <div className="text-slate-500 mb-0.5">Removes vs PPO</div>
                <div className="text-slate-300">&mdash; (is the baseline)</div>
              </div>
              <div className="p-2 rounded bg-slate-900/50 border border-slate-700/30">
                <div className="text-slate-500 mb-0.5">Best for</div>
                <div className="text-slate-300">General alignment, chat</div>
              </div>
              <div className="p-2 rounded bg-slate-900/50 border border-slate-700/30">
                <div className="text-slate-500 mb-0.5">Compute</div>
                <div className="text-slate-300">1.0&times; (baseline)</div>
              </div>
              <div className="p-2 rounded bg-slate-900/50 border border-slate-700/30">
                <div className="text-slate-500 mb-0.5">Models in VRAM</div>
                <div className="text-slate-300">4</div>
              </div>
            </div>
          </div>

          {/* DPO */}
          <div className="p-5 rounded-lg bg-pink-950/10 border border-pink-800/30">
            <div className="flex items-start justify-between mb-2">
              <div>
                <h3 className="text-base font-semibold text-pink-300">DPO</h3>
                <p className="text-xs text-pink-400/70">Direct Preference Optimization</p>
              </div>
              <span className="px-2 py-0.5 rounded text-xs bg-pink-900/30 text-pink-400">
                DPO Family
              </span>
            </div>
            <p className="text-sm text-slate-300 leading-relaxed mb-3">
              <strong className="text-pink-200">Core idea:</strong> Skip the reward model &mdash;
              learn directly from human preference pairs.
            </p>
            <p className="text-xs text-slate-400 leading-relaxed mb-3">
              DPO shows the policy pairs of responses (chosen vs rejected) and adjusts the
              log-probabilities so the chosen response becomes more likely and the rejected one
              becomes less likely. This reparameterization absorbs the reward model into a
              closed-form expression over log-probs, eliminating two models entirely.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
              <div className="p-2 rounded bg-pink-950/20 border border-pink-800/20">
                <div className="text-slate-500 mb-0.5">Removes vs PPO</div>
                <div className="text-pink-300">Reward + Critic</div>
              </div>
              <div className="p-2 rounded bg-pink-950/20 border border-pink-800/20">
                <div className="text-slate-500 mb-0.5">Best for</div>
                <div className="text-pink-300">Preference alignment, style</div>
              </div>
              <div className="p-2 rounded bg-pink-950/20 border border-pink-800/20">
                <div className="text-slate-500 mb-0.5">Compute</div>
                <div className="text-pink-300">~0.4&times;</div>
              </div>
              <div className="p-2 rounded bg-pink-950/20 border border-pink-800/20">
                <div className="text-slate-500 mb-0.5">Models in VRAM</div>
                <div className="text-pink-300">2</div>
              </div>
            </div>
          </div>

          {/* GRPO */}
          <div className="p-5 rounded-lg bg-emerald-950/10 border border-emerald-800/30">
            <div className="flex items-start justify-between mb-2">
              <div>
                <h3 className="text-base font-semibold text-emerald-300">GRPO</h3>
                <p className="text-xs text-emerald-400/70">Group Relative Policy Optimization</p>
              </div>
              <span className="px-2 py-0.5 rounded text-xs bg-emerald-900/30 text-emerald-400">
                GRPO Family
              </span>
            </div>
            <p className="text-sm text-slate-300 leading-relaxed mb-3">
              <strong className="text-emerald-200">Core idea:</strong> Replace the critic with group
              statistics &mdash; compare each response to the group mean.
            </p>
            <p className="text-xs text-slate-400 leading-relaxed mb-3">
              For each prompt, generate a group of responses (e.g., 16). Score them all with a
              reward function, compute the group mean and standard deviation, then normalize each
              reward into an advantage. Responses above the mean get reinforced; below get
              suppressed. No learned critic needed. Used by DeepSeek R1.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
              <div className="p-2 rounded bg-emerald-950/20 border border-emerald-800/20">
                <div className="text-slate-500 mb-0.5">Removes vs PPO</div>
                <div className="text-emerald-300">Critic</div>
              </div>
              <div className="p-2 rounded bg-emerald-950/20 border border-emerald-800/20">
                <div className="text-slate-500 mb-0.5">Best for</div>
                <div className="text-emerald-300">Verifiable tasks (math, code)</div>
              </div>
              <div className="p-2 rounded bg-emerald-950/20 border border-emerald-800/20">
                <div className="text-slate-500 mb-0.5">Compute</div>
                <div className="text-emerald-300">~0.5&times;</div>
              </div>
              <div className="p-2 rounded bg-emerald-950/20 border border-emerald-800/20">
                <div className="text-slate-500 mb-0.5">Models in VRAM</div>
                <div className="text-emerald-300">2</div>
              </div>
            </div>
          </div>

          {/* RLOO */}
          <div className="p-5 rounded-lg bg-violet-950/10 border border-violet-800/30">
            <div className="flex items-start justify-between mb-2">
              <div>
                <h3 className="text-base font-semibold text-violet-300">RLOO</h3>
                <p className="text-xs text-violet-400/70">REINFORCE Leave-One-Out</p>
              </div>
              <span className="px-2 py-0.5 rounded text-xs bg-violet-900/30 text-violet-400">
                Emerging
              </span>
            </div>
            <p className="text-sm text-slate-300 leading-relaxed mb-3">
              <strong className="text-violet-200">Core idea:</strong> Compare each sample&rsquo;s
              reward to the mean reward of all <em>other</em> samples.
            </p>
            <p className="text-xs text-slate-400 leading-relaxed mb-3">
              Generate K completions per prompt. For each completion, compute its advantage as the
              difference between its reward and the mean reward of the other K&minus;1 completions.
              This leave-one-out baseline reduces variance without needing a learned critic. Simpler
              and more stable than PPO for tasks with verifiable rewards.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
              <div className="p-2 rounded bg-violet-950/20 border border-violet-800/20">
                <div className="text-slate-500 mb-0.5">Removes vs PPO</div>
                <div className="text-violet-300">Critic</div>
              </div>
              <div className="p-2 rounded bg-violet-950/20 border border-violet-800/20">
                <div className="text-slate-500 mb-0.5">Best for</div>
                <div className="text-violet-300">Verifiable tasks, reasoning</div>
              </div>
              <div className="p-2 rounded bg-violet-950/20 border border-violet-800/20">
                <div className="text-slate-500 mb-0.5">Compute</div>
                <div className="text-violet-300">~0.5&times;</div>
              </div>
              <div className="p-2 rounded bg-violet-950/20 border border-violet-800/20">
                <div className="text-slate-500 mb-0.5">Models in VRAM</div>
                <div className="text-violet-300">2</div>
              </div>
            </div>
          </div>

          {/* ReMax */}
          <div className="p-5 rounded-lg bg-violet-950/10 border border-violet-800/30">
            <div className="flex items-start justify-between mb-2">
              <div>
                <h3 className="text-base font-semibold text-violet-300">ReMax</h3>
                <p className="text-xs text-violet-400/70">Reward-Maximization</p>
              </div>
              <span className="px-2 py-0.5 rounded text-xs bg-violet-900/30 text-violet-400">
                Emerging
              </span>
            </div>
            <p className="text-sm text-slate-300 leading-relaxed mb-3">
              <strong className="text-violet-200">Core idea:</strong> Convert RL into
              reward-weighted language model training &mdash; no advantage estimation needed.
            </p>
            <p className="text-xs text-slate-400 leading-relaxed mb-3">
              Generate K completions, score them, and weight the language modeling loss by the
              reward. High-reward completions get high weight; low-reward get low weight. This
              converts the RL problem into a weighted supervised learning problem, eliminating the
              need for any advantage estimation or critic.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
              <div className="p-2 rounded bg-violet-950/20 border border-violet-800/20">
                <div className="text-slate-500 mb-0.5">Removes vs PPO</div>
                <div className="text-violet-300">Critic</div>
              </div>
              <div className="p-2 rounded bg-violet-950/20 border border-violet-800/20">
                <div className="text-slate-500 mb-0.5">Best for</div>
                <div className="text-violet-300">Simple reward functions</div>
              </div>
              <div className="p-2 rounded bg-violet-950/20 border border-violet-800/20">
                <div className="text-slate-500 mb-0.5">Compute</div>
                <div className="text-violet-300">~0.5&times;</div>
              </div>
              <div className="p-2 rounded bg-violet-950/20 border border-violet-800/20">
                <div className="text-slate-500 mb-0.5">Models in VRAM</div>
                <div className="text-violet-300">2</div>
              </div>
            </div>
          </div>

          {/* Dr-GRPO */}
          <div className="p-5 rounded-lg bg-violet-950/10 border border-violet-800/30">
            <div className="flex items-start justify-between mb-2">
              <div>
                <h3 className="text-base font-semibold text-violet-300">Dr-GRPO</h3>
                <p className="text-xs text-violet-400/70">Decoupled Regularized GRPO</p>
              </div>
              <span className="px-2 py-0.5 rounded text-xs bg-violet-900/30 text-violet-400">
                Emerging
              </span>
            </div>
            <p className="text-sm text-slate-300 leading-relaxed mb-3">
              <strong className="text-violet-200">Core idea:</strong> Separate reward into
              independent components &mdash; e.g., correctness vs reasoning quality.
            </p>
            <p className="text-xs text-slate-400 leading-relaxed mb-3">
              Extends GRPO by decoupling the reward signal into multiple independent components,
              each with its own normalization and weighting. This prevents one reward signal (e.g.,
              format compliance) from drowning out another (e.g., factual correctness). An academic
              refinement that improves training stability on multi-objective tasks.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
              <div className="p-2 rounded bg-violet-950/20 border border-violet-800/20">
                <div className="text-slate-500 mb-0.5">Removes vs PPO</div>
                <div className="text-violet-300">Critic</div>
              </div>
              <div className="p-2 rounded bg-violet-950/20 border border-violet-800/20">
                <div className="text-slate-500 mb-0.5">Best for</div>
                <div className="text-violet-300">Multi-objective tasks</div>
              </div>
              <div className="p-2 rounded bg-violet-950/20 border border-violet-800/20">
                <div className="text-slate-500 mb-0.5">Compute</div>
                <div className="text-violet-300">~0.5&times;</div>
              </div>
              <div className="p-2 rounded bg-violet-950/20 border border-violet-800/20">
                <div className="text-slate-500 mb-0.5">Models in VRAM</div>
                <div className="text-violet-300">2</div>
              </div>
            </div>
          </div>

          {/* DAPO */}
          <div className="p-5 rounded-lg bg-violet-950/10 border border-violet-800/30">
            <div className="flex items-start justify-between mb-2">
              <div>
                <h3 className="text-base font-semibold text-violet-300">DAPO</h3>
                <p className="text-xs text-violet-400/70">Direct Advantage Policy Optimization</p>
              </div>
              <span className="px-2 py-0.5 rounded text-xs bg-violet-900/30 text-violet-400">
                Emerging
              </span>
            </div>
            <p className="text-sm text-slate-300 leading-relaxed mb-3">
              <strong className="text-violet-200">Core idea:</strong> Advantage-weighted
              log-likelihood within a DPO-style objective, with KL regularization for stability.
            </p>
            <p className="text-xs text-slate-400 leading-relaxed mb-3">
              Combines the best of both worlds: uses advantage estimates (like PPO/GRPO) but
              optimizes them within a DPO-style log-likelihood framework. Adds explicit KL
              regularization to prevent the policy from drifting too far, achieving better stability
              than either pure DPO or pure GRPO on long-horizon reasoning tasks.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
              <div className="p-2 rounded bg-violet-950/20 border border-violet-800/20">
                <div className="text-slate-500 mb-0.5">Removes vs PPO</div>
                <div className="text-violet-300">Critic + Reward</div>
              </div>
              <div className="p-2 rounded bg-violet-950/20 border border-violet-800/20">
                <div className="text-slate-500 mb-0.5">Best for</div>
                <div className="text-violet-300">Long-horizon reasoning</div>
              </div>
              <div className="p-2 rounded bg-violet-950/20 border border-violet-800/20">
                <div className="text-slate-500 mb-0.5">Compute</div>
                <div className="text-violet-300">~0.4&times;</div>
              </div>
              <div className="p-2 rounded bg-violet-950/20 border border-violet-800/20">
                <div className="text-slate-500 mb-0.5">Models in VRAM</div>
                <div className="text-violet-300">2</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Tab 3: Enterprise Impact ── */}
      {section === 'enterprise' && (
        <div className="space-y-6">
          {/* RLVR */}
          <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h3 className="text-base font-semibold text-pink-400 mb-1">
              RLVR &mdash; Reinforcement Learning with Verifiable Rewards
            </h3>
            <p className="text-xs text-slate-500 mb-3">
              The pattern behind DeepSeek R1, Qwen 2.5, and most reasoning breakthroughs
            </p>
            <p className="text-sm text-slate-300 leading-relaxed mb-4">
              Instead of training a reward model from human preferences, use a{' '}
              <strong className="text-white">verifier</strong> that can programmatically check if
              the answer is correct. Binary rewards: correct = 1, incorrect = 0.
            </p>

            <div className="space-y-3 mb-4">
              <div className="flex gap-3 items-start">
                <span className="text-sm font-bold text-pink-400 w-6 shrink-0">1.</span>
                <div>
                  <p className="text-sm font-semibold text-slate-200">Generate candidates</p>
                  <p className="text-xs text-slate-400 mt-1">
                    For each prompt, the model generates 16&ndash;64 candidate solutions at high
                    temperature.
                  </p>
                </div>
              </div>
              <div className="flex gap-3 items-start">
                <span className="text-sm font-bold text-pink-400 w-6 shrink-0">2.</span>
                <div>
                  <p className="text-sm font-semibold text-slate-200">Verify each candidate</p>
                  <p className="text-xs text-slate-400 mt-1">
                    A deterministic verifier checks each solution. Math: does the answer match?
                    Code: do the tests pass? Logic: is the proof valid?
                  </p>
                </div>
              </div>
              <div className="flex gap-3 items-start">
                <span className="text-sm font-bold text-pink-400 w-6 shrink-0">3.</span>
                <div>
                  <p className="text-sm font-semibold text-slate-200">
                    Reinforce correct solutions
                  </p>
                  <p className="text-xs text-slate-400 mt-1">
                    Correct solutions are reinforced (increase log-probability), incorrect ones
                    receive no reward. The model learns to generate more solutions like the correct
                    ones.
                  </p>
                </div>
              </div>
              <div className="flex gap-3 items-start">
                <span className="text-sm font-bold text-pink-400 w-6 shrink-0">4.</span>
                <div>
                  <p className="text-sm font-semibold text-slate-200">Reasoning emerges</p>
                  <p className="text-xs text-slate-400 mt-1">
                    Through selection pressure, chain-of-thought reasoning emerges naturally &mdash;
                    not because we trained it explicitly, but because solutions with reasoning steps
                    are more likely to be correct.
                  </p>
                </div>
              </div>
            </div>

            <div className="p-3 rounded bg-pink-950/20 border border-pink-800/30">
              <p className="text-xs text-pink-300 leading-relaxed">
                <strong>Key insight:</strong> No human labels needed, just a verifier. This makes
                RLVR scalable to domains where checking is cheap but generating is hard &mdash;
                math, code, logic, and structured data classification (like our storage workload
                task).
              </p>
            </div>
          </div>

          {/* KDRL */}
          <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h3 className="text-base font-semibold text-pink-400 mb-1">
              KDRL &mdash; Knowledge Distillation + Reinforcement Learning
            </h3>
            <p className="text-xs text-slate-500 mb-3">
              The most enterprise-relevant hybrid approach
            </p>
            <p className="text-sm text-slate-300 leading-relaxed mb-4">
              Combines two complementary training signals: a teacher model provides{' '}
              <strong className="text-white">&ldquo;how to do it&rdquo;</strong> (knowledge
              distillation) while RL provides{' '}
              <strong className="text-white">&ldquo;what outcome to optimize&rdquo;</strong>{' '}
              (rewards).
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
              <div className="p-3 rounded bg-blue-950/20 border border-blue-800/30">
                <h4 className="text-xs font-semibold text-blue-400 uppercase tracking-wide mb-1">
                  Knowledge Distillation (KD)
                </h4>
                <p className="text-xs text-slate-400">
                  Student imitates teacher&rsquo;s behavior: tone, reasoning steps, tool usage
                  patterns, response structure. Learns <em>how</em> to respond.
                </p>
              </div>
              <div className="p-3 rounded bg-orange-950/20 border border-orange-800/30">
                <h4 className="text-xs font-semibold text-orange-400 uppercase tracking-wide mb-1">
                  Reinforcement Learning (RL)
                </h4>
                <p className="text-xs text-slate-400">
                  Reward for resolution, penalty for policy violations, bonus for first-contact
                  resolution. Learns <em>what outcomes</em> matter.
                </p>
              </div>
            </div>

            {/* Example */}
            <div className="p-4 rounded-lg bg-slate-900/50 border border-slate-700/30 mb-4">
              <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
                Example: Enterprise Support Agent
              </h4>
              <div className="space-y-2 text-xs">
                <div className="flex gap-3 items-start">
                  <span className="text-blue-400 w-8 shrink-0 font-semibold">KD</span>
                  <p className="text-slate-400">
                    Student learns teacher&rsquo;s tone, step-by-step troubleshooting, and tool
                    usage patterns
                  </p>
                </div>
                <div className="flex gap-3 items-start">
                  <span className="text-orange-400 w-8 shrink-0 font-semibold">RL</span>
                  <p className="text-slate-400">
                    Reward for ticket resolution, penalty for policy violations, bonus for customer
                    satisfaction
                  </p>
                </div>
                <div className="flex gap-3 items-start">
                  <span className="text-green-400 w-8 shrink-0 font-semibold">Result</span>
                  <p className="text-slate-400">
                    Cheaper model with teacher-quality behavior and better compliance than either
                    approach alone
                  </p>
                </div>
              </div>
            </div>

            <div className="p-3 rounded bg-pink-950/20 border border-pink-800/30">
              <p className="text-xs text-pink-300 leading-relaxed">
                <strong>Why enterprises care:</strong> KDRL lets you deploy a small, cheap model
                (e.g., 7B) that behaves like a large, expensive teacher (e.g., 70B) while also
                optimizing for business-specific outcomes that the teacher was never trained on.
              </p>
            </div>
          </div>

          {/* Infrastructure comparison table */}
          <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h3 className="text-base font-semibold text-pink-400 mb-3">
              Infrastructure Comparison
            </h3>
            <p className="text-sm text-slate-400 mb-4">
              The practical bottom line: how each algorithm impacts your GPU budget and I/O
              pipeline.
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-slate-700/50">
                    <th className="text-left py-2 pr-3 text-slate-400 font-semibold">Algorithm</th>
                    <th className="text-left py-2 pr-3 text-slate-400 font-semibold">
                      Models in Memory
                    </th>
                    <th className="text-center py-2 pr-3 text-slate-400 font-semibold">
                      Relative Compute
                    </th>
                    <th className="text-center py-2 text-slate-400 font-semibold">Relative I/O</th>
                  </tr>
                </thead>
                <tbody className="text-slate-300">
                  <tr className="border-b border-slate-800/50">
                    <td className="py-2 pr-3 font-semibold text-slate-300">PPO</td>
                    <td className="py-2 pr-3 text-slate-400">4 (Policy, Ref, Reward, Critic)</td>
                    <td className="py-2 pr-3 text-center">1.0&times;</td>
                    <td className="py-2 text-center">1.0&times;</td>
                  </tr>
                  <tr className="border-b border-slate-800/50">
                    <td className="py-2 pr-3 font-semibold text-pink-300">DPO</td>
                    <td className="py-2 pr-3 text-slate-400">2 (Policy, Reference)</td>
                    <td className="py-2 pr-3 text-center text-pink-300">~0.4&times;</td>
                    <td className="py-2 text-center text-pink-300">~0.3&times;</td>
                  </tr>
                  <tr className="border-b border-slate-800/50">
                    <td className="py-2 pr-3 font-semibold text-emerald-300">GRPO</td>
                    <td className="py-2 pr-3 text-slate-400">2 (Policy, Reference)</td>
                    <td className="py-2 pr-3 text-center text-emerald-300">~0.5&times;</td>
                    <td className="py-2 text-center text-emerald-300">~0.8&times;</td>
                  </tr>
                  <tr className="border-b border-slate-800/50">
                    <td className="py-2 pr-3 font-semibold text-violet-300">RLOO</td>
                    <td className="py-2 pr-3 text-slate-400">2 (Policy, Reference)</td>
                    <td className="py-2 pr-3 text-center text-violet-300">~0.5&times;</td>
                    <td className="py-2 text-center text-violet-300">~0.6&times;</td>
                  </tr>
                  <tr>
                    <td className="py-2 pr-3 font-semibold text-blue-300">KDRL</td>
                    <td className="py-2 pr-3 text-slate-400">3 (Student, Teacher, Reference)</td>
                    <td className="py-2 pr-3 text-center text-blue-300">~0.7&times;</td>
                    <td className="py-2 text-center text-blue-300">~0.5&times;</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="text-xs text-slate-600 mt-3 italic">
              GRPO&rsquo;s higher I/O reflects multi-generation (16&ndash;64 completions per
              prompt). DPO&rsquo;s low I/O is because it only needs pre-computed preference pairs,
              not online generation.
            </p>
          </div>

          {/* Bottom callout */}
          <div className="p-4 rounded-lg bg-gradient-to-r from-pink-950/30 to-blue-950/30 border border-pink-800/30">
            <h4 className="text-sm font-semibold text-pink-400 mb-2">The Storage Angle</h4>
            <p className="text-sm text-slate-300 leading-relaxed mb-2">
              For storage and infrastructure engineers, the shift from PPO to GRPO/DPO isn&rsquo;t
              just an academic curiosity &mdash; it directly reduces the GPU memory, checkpoint I/O,
              and training time required to fine-tune models.
            </p>
            <p className="text-sm text-slate-400 leading-relaxed">
              Our demo uses GRPO with verifiable rewards (RLVR) because the storage workload
              classification task has a clear correctness criterion: does the model predict the
              right workload category? This makes it a natural fit for the simplest, most efficient
              RL approach.
            </p>
          </div>
        </div>
      )}

      <SectionTabs tabs={TABS} active={section} onSelect={setSection} color="pink" />
    </div>
  )
}
