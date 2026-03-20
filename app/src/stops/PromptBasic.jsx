import { useState } from 'react'
import ModelOutput from '../components/ModelOutput'
import TokenProbChart from '../components/TokenProbChart'

// ---------------------------------------------------------------------------
// Precomputed data
// ---------------------------------------------------------------------------

const EXAMPLE_INPUT =
  "IOPS: 45000 | Latency: 0.3ms | Block Size: 8K | Read/Write: 70/30 | Sequential: 15% | Queue Depth: 32"

const BASE_RESPONSE = `This appears to be some kind of storage performance metrics. The IOPS value is 45000 which is relatively high. The latency is 0.3ms which is quite low. The block size is 8K. There is a 70/30 read to write ratio with 15% sequential access and a queue depth of 32. These metrics could be from various types of storage workloads depending on the specific use case and configuration being measured.`

const BASE_TOKEN_PROBS = [
  { token: 'This', probability: 0.18 },
  { token: 'The', probability: 0.14 },
  { token: 'Based', probability: 0.09 },
  { token: 'OLTP', probability: 0.04 },
  { token: 'These', probability: 0.07 },
  { token: 'It', probability: 0.06 },
  { token: 'Storage', probability: 0.05 },
  { token: 'High', probability: 0.04 },
  { token: 'Looking', probability: 0.04 },
  { token: 'I', probability: 0.03 },
  { token: 'Database', probability: 0.03 },
  { token: 'A', probability: 0.03 },
  { token: 'With', probability: 0.02 },
  { token: 'VDI', probability: 0.02 },
  { token: 'Given', probability: 0.02 },
]

// ---------------------------------------------------------------------------
// Prompt variants for "What is Prompting?" tab
// ---------------------------------------------------------------------------

const PROMPT_VARIANTS = [
  {
    id: 'vague',
    style: 'Vague',
    tagColor: 'bg-red-900/50 text-red-400',
    prompt: `What is this?\n\n${EXAMPLE_INPUT}`,
    response: `This is a set of numerical parameters or specifications, possibly related to computer storage or disk performance benchmarking. IOPS stands for Input/Output Operations Per Second, which measures how fast a storage device can read and write. The other values describe characteristics of the workload being measured.`,
    verdict: 'The model has no idea what we actually want. It explains the acronyms like a dictionary.',
    verdictColor: 'text-red-300',
  },
  {
    id: 'structured',
    style: 'Structured',
    tagColor: 'bg-yellow-900/50 text-yellow-400',
    prompt: `You are a storage I/O expert. Given the following metrics, classify the workload into one of these categories: OLTP Database, OLAP Analytics, AI/ML Training, Video Streaming, VDI Virtual Desktop, Backup/Archive.\n\nMetrics:\n${EXAMPLE_INPUT}`,
    response: `Based on the provided metrics, this workload appears to be consistent with an OLTP Database pattern. The high IOPS (45,000) combined with very low latency (0.3ms) and small block size (8K) are characteristic of transactional database workloads. The predominantly random access pattern (only 15% sequential) further supports this classification.`,
    verdict: 'Much better! Giving it categories and a role narrows its focus. But it hedges with "appears to be."',
    verdictColor: 'text-yellow-300',
  },
  {
    id: 'instruction',
    style: 'Instruction-tuned',
    tagColor: 'bg-green-900/50 text-green-400',
    prompt: `Classify the following storage I/O pattern into exactly one category.\n\nCategories: OLTP Database, OLAP Analytics, AI/ML Training, Video Streaming, VDI Virtual Desktop, Backup/Archive\n\nRespond with ONLY the category name, then a confidence level (High/Medium/Low), then a one-sentence justification.\n\nPattern:\n${EXAMPLE_INPUT}`,
    response: `OLTP Database\nConfidence: High\nHigh IOPS (45K) with sub-millisecond latency and small random I/O blocks are the signature of transaction processing.`,
    verdict: 'Tight instructions produce tight output. The model follows the exact format we asked for.',
    verdictColor: 'text-green-300',
  },
]

// ---------------------------------------------------------------------------
// Full prompt text for the "See It Work" tab
// ---------------------------------------------------------------------------

const SYSTEM_MESSAGE = `You are a storage workload classification assistant. When given storage I/O metrics, classify the workload into one of these categories: OLTP Database, OLAP Analytics, AI/ML Training, Video Streaming, VDI Virtual Desktop, Backup/Archive.`

const USER_MESSAGE = `Classify this storage I/O pattern into a workload type:\n\n${EXAMPLE_INPUT}`

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function PromptBasic({ explore = false }) {
  const [section, setSection] = useState('problem')
  const [selectedVariant, setSelectedVariant] = useState(0)

  const tabs = [
    { id: 'problem', label: 'The Problem' },
    { id: 'concept', label: 'What is Prompting?' },
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
                ? 'bg-orange-600 text-white font-semibold'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* ==================== THE PROBLEM ==================== */}
      {section === 'problem' && (
        <div className="space-y-5">
          {/* Framing */}
          <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h3 className="text-base font-semibold text-orange-400 mb-3">
              A smart model with no domain knowledge
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed mb-3">
              SmolLM2-360M is a capable small language model. It can summarize text,
              answer general knowledge questions, and follow simple instructions. But it
              was trained on general internet text &mdash; it has never seen storage I/O
              patterns before and has no idea that workload categories like "OLTP Database"
              or "VDI Virtual Desktop" exist.
            </p>
            <p className="text-sm text-slate-400 leading-relaxed">
              So what happens when we show it a string of storage metrics and ask it to
              classify the workload? Let's find out.
            </p>
          </div>

          {/* The raw input */}
          <div>
            <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
              Storage I/O Pattern (our input)
            </label>
            <div className="bg-slate-800 border border-slate-600 rounded-lg p-3 font-mono text-sm text-slate-200">
              {EXAMPLE_INPUT}
            </div>
            <p className="text-xs text-slate-500 mt-1">
              This is a classic OLTP Database pattern. High IOPS, sub-millisecond latency,
              small blocks, mostly random reads.
            </p>
          </div>

          {/* Base model output */}
          <ModelOutput
            label="Base Model Response (SmolLM2-360M)"
            text={BASE_RESPONSE}
            variant="base"
            isCorrect={false}
          />

          {/* Commentary */}
          <div className="p-4 rounded-lg bg-red-950/20 border border-red-800/30">
            <h4 className="text-sm font-semibold text-red-400 mb-2">What went wrong?</h4>
            <p className="text-sm text-slate-300 leading-relaxed mb-2">
              The model isn't stupid &mdash; it correctly describes each metric. It knows
              45,000 IOPS is "relatively high" and 0.3ms latency is "quite low." But it
              never actually <em>classifies</em> the workload. It ends with a vague
              hedge: "could be from various types of storage workloads."
            </p>
            <p className="text-sm text-slate-400 leading-relaxed">
              This is the fundamental challenge: the model can read the numbers but has
              no framework for mapping them to storage workload categories. It's like
              showing someone a blood test result who has never studied medicine &mdash;
              they can read the numbers, but they can't tell you what's wrong.
            </p>
          </div>

          {/* The six categories */}
          <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/50">
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">
              The 6 workload categories the model needs to learn
            </h4>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
              {[
                { name: 'OLTP Database', desc: 'High IOPS, tiny blocks, sub-ms latency' },
                { name: 'OLAP Analytics', desc: 'Large sequential reads, high throughput' },
                { name: 'AI/ML Training', desc: 'Huge sequential reads, GPU-fed pipeline' },
                { name: 'Video Streaming', desc: 'Steady sequential reads, large blocks' },
                { name: 'VDI Virtual Desktop', desc: 'Mixed R/W, small blocks, many users' },
                { name: 'Backup/Archive', desc: 'Bulk sequential writes, high compression' },
              ].map((cat) => (
                <div key={cat.name} className="p-2 rounded bg-slate-900/50 border border-slate-700/30">
                  <div className="text-xs font-semibold text-orange-300">{cat.name}</div>
                  <div className="text-xs text-slate-500 mt-0.5">{cat.desc}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Transition */}
          <div className="p-3 rounded bg-orange-950/20 border border-orange-800/30">
            <p className="text-sm text-orange-300">
              Can we fix this without changing the model at all? The first technique to
              try is <strong>prompt engineering</strong> &mdash; changing <em>how we ask</em>
              the question. Head to the next tab to see how.
            </p>
          </div>
        </div>
      )}

      {/* ==================== WHAT IS PROMPTING? ==================== */}
      {section === 'concept' && (
        <div className="space-y-5">
          {/* Intro */}
          <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h3 className="text-base font-semibold text-orange-400 mb-3">
              Prompt Engineering: How you ask matters
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed mb-3">
              Imagine you're asking a new colleague for help. If you say "Hey, what do
              you think about this?" and hand them a spreadsheet, you'll get a vague
              answer. But if you say "I need you to categorize this into one of six
              buckets &mdash; here are the buckets, here's the format I want &mdash; go,"
              you'll get exactly what you need.
            </p>
            <p className="text-sm text-slate-400 leading-relaxed">
              Prompt engineering is the same idea applied to language models. The model
              doesn't change &mdash; the weights stay frozen, the architecture is the same.
              You're just changing the <strong className="text-orange-300">input text</strong> to
              guide the model toward the output you want.
            </p>
          </div>

          {/* Analogy box */}
          <div className="p-4 rounded-lg bg-blue-950/20 border border-blue-800/30">
            <p className="text-xs text-blue-300 leading-relaxed">
              <strong>Key insight:</strong> Prompt engineering does not modify the model.
              There is no training, no new data, no GPU time. You are only changing
              the text the model receives as input. Think of it as writing better
              instructions on the exam paper &mdash; the student is the same, but clearer
              questions produce clearer answers.
            </p>
          </div>

          {/* Three prompt variants */}
          <div>
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">
              Same question, three different prompts
            </h4>
            <div className="flex gap-2 mb-4">
              {PROMPT_VARIANTS.map((v, i) => (
                <button
                  key={v.id}
                  onClick={() => setSelectedVariant(i)}
                  className={`px-3 py-1.5 text-xs rounded-md transition-colors ${
                    i === selectedVariant
                      ? 'bg-orange-600 text-white font-semibold'
                      : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                  }`}
                >
                  {v.style}
                </button>
              ))}
            </div>

            {/* Selected variant */}
            {(() => {
              const v = PROMPT_VARIANTS[selectedVariant]
              return (
                <div className="space-y-3">
                  {/* Prompt text */}
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
                        Prompt
                      </span>
                      <span className={`text-xs px-2 py-0.5 rounded-full font-semibold ${v.tagColor}`}>
                        {v.style}
                      </span>
                    </div>
                    <pre className="bg-slate-800 border border-slate-600 rounded-lg p-3 font-mono text-sm text-slate-200 whitespace-pre-wrap leading-relaxed">
                      {v.prompt}
                    </pre>
                  </div>

                  {/* Model response */}
                  <div>
                    <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide block mb-1">
                      Model Response
                    </span>
                    <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-3">
                      <pre className="text-sm text-slate-200 whitespace-pre-wrap font-mono leading-relaxed">
                        {v.response}
                      </pre>
                    </div>
                  </div>

                  {/* Verdict */}
                  <div className="p-3 rounded bg-slate-800/30 border border-slate-700/30">
                    <p className={`text-sm ${v.verdictColor}`}>
                      {v.verdict}
                    </p>
                  </div>
                </div>
              )
            })()}
          </div>

          {/* Progression summary */}
          <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/50">
            <h4 className="text-sm font-semibold text-orange-400 mb-3">
              What changed across those three prompts?
            </h4>
            <div className="space-y-2">
              <div className="flex gap-3 items-start">
                <span className="text-xs font-mono text-red-400 w-24 shrink-0 pt-0.5">Vague</span>
                <p className="text-xs text-slate-400">
                  No role, no categories, no format. The model treats it like a general
                  knowledge question and explains what the acronyms mean.
                </p>
              </div>
              <div className="flex gap-3 items-start">
                <span className="text-xs font-mono text-yellow-400 w-24 shrink-0 pt-0.5">Structured</span>
                <p className="text-xs text-slate-400">
                  We give the model a role ("storage I/O expert"), list the six categories,
                  and provide the metrics clearly. The model now attempts classification
                  but still hedges its language.
                </p>
              </div>
              <div className="flex gap-3 items-start">
                <span className="text-xs font-mono text-green-400 w-24 shrink-0 pt-0.5">Instruction</span>
                <p className="text-xs text-slate-400">
                  We specify the exact output format: category name, confidence, one
                  sentence. The model follows instructions precisely and produces a
                  clean, actionable answer.
                </p>
              </div>
            </div>
          </div>

          {/* Limitations teaser */}
          <div className="p-3 rounded bg-orange-950/20 border border-orange-800/30">
            <p className="text-sm text-orange-300">
              Prompt engineering is powerful &mdash; and it's free. No training, no GPUs,
              no data pipelines. But it has limits: the model still doesn't truly
              <em> understand</em> storage I/O patterns. It's following instructions,
              not applying learned expertise. We'll explore those limits in later stops.
            </p>
          </div>
        </div>
      )}

      {/* ==================== SEE IT WORK ==================== */}
      {section === 'demo' && (
        <div className="space-y-5">
          {/* Input */}
          <div>
            <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
              Storage I/O Pattern
            </label>
            <div className="bg-slate-800 border border-slate-600 rounded-lg p-3 font-mono text-sm text-slate-200">
              {EXAMPLE_INPUT}
            </div>
          </div>

          {/* Show the actual prompt */}
          <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h4 className="text-xs font-semibold text-orange-400 uppercase tracking-wide mb-3">
              What the model actually sees
            </h4>
            <p className="text-xs text-slate-400 mb-3">
              Language models don't just receive your question &mdash; they receive a
              structured prompt. Here's the full text we send to SmolLM2-360M:
            </p>

            {/* System message */}
            <div className="mb-3">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
                  System Message
                </span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-slate-700 text-slate-400">
                  Sets the role
                </span>
              </div>
              <pre className="bg-slate-900 border border-slate-700 rounded-lg p-3 font-mono text-xs text-slate-300 whitespace-pre-wrap leading-relaxed">
                {SYSTEM_MESSAGE}
              </pre>
            </div>

            {/* User message */}
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
                  User Message
                </span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-slate-700 text-slate-400">
                  Your query
                </span>
              </div>
              <pre className="bg-slate-900 border border-slate-700 rounded-lg p-3 font-mono text-xs text-slate-300 whitespace-pre-wrap leading-relaxed">
                {USER_MESSAGE}
              </pre>
            </div>
          </div>

          {/* Model response */}
          <ModelOutput
            label="Base Model Response (SmolLM2-360M, no fine-tuning)"
            text={BASE_RESPONSE}
            variant="base"
            isCorrect={false}
          />

          {/* Explanation */}
          <div className="p-4 rounded-lg bg-red-950/20 border border-red-800/30">
            <h4 className="text-sm font-semibold text-red-400 mb-2">Result: Incorrect</h4>
            <p className="text-sm text-slate-300 leading-relaxed mb-2">
              Even with a system message and a clear instruction, the base model fails to
              classify. It describes the metrics accurately but never commits to a
              category. The response ends with "could be from various types of storage
              workloads" &mdash; which is not useful.
            </p>
            <p className="text-sm text-slate-400 leading-relaxed">
              The correct answer is <strong className="text-green-400">OLTP Database</strong>.
              The pattern &mdash; high IOPS, sub-ms latency, small random blocks, read-heavy
              &mdash; is textbook OLTP. But the model has no training data that maps
              these patterns to workload categories, so prompting alone isn't enough.
            </p>
          </div>

          {/* What prompting can vs. cannot do */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="p-3 rounded-lg bg-green-950/20 border border-green-800/30">
              <h4 className="text-xs font-semibold text-green-400 uppercase tracking-wide mb-2">
                What prompting can do
              </h4>
              <ul className="text-xs text-slate-400 space-y-1.5">
                <li>- Control output format (JSON, bullet points, single word)</li>
                <li>- Assign a role to focus the model's behavior</li>
                <li>- Provide categories to constrain possible answers</li>
                <li>- Include examples (few-shot) for the model to mimic</li>
              </ul>
            </div>
            <div className="p-3 rounded-lg bg-red-950/20 border border-red-800/30">
              <h4 className="text-xs font-semibold text-red-400 uppercase tracking-wide mb-2">
                What prompting cannot do
              </h4>
              <ul className="text-xs text-slate-400 space-y-1.5">
                <li>- Teach the model new domain knowledge</li>
                <li>- Make the model reliably distinguish similar categories</li>
                <li>- Guarantee consistent, structured outputs</li>
                <li>- Replace training when the task requires specialized reasoning</li>
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* ==================== UNDER THE COVERS ==================== */}
      {section === 'deepdive' && (
        <div className="space-y-5">
          {/* Intro to token probs */}
          <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h3 className="text-base font-semibold text-orange-400 mb-3">
              Token Probabilities: How the model "thinks"
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed mb-3">
              Every time a language model generates a word (or more precisely, a
              <strong className="text-orange-300"> token</strong>), it isn't just picking one
              answer. It's computing a probability distribution over its entire vocabulary
              &mdash; tens of thousands of possible next tokens &mdash; and then sampling
              from that distribution.
            </p>
            <p className="text-sm text-slate-400 leading-relaxed">
              The chart below shows the model's probability distribution for the
              <strong className="text-slate-200"> very first token</strong> of its response.
              This is the moment of truth: what does the model think it should say first?
            </p>
          </div>

          {/* Token prob chart */}
          <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/50">
            <TokenProbChart
              data={BASE_TOKEN_PROBS}
              label="Base Model: First Token Probabilities"
              highlightToken="OLTP"
            />
          </div>

          {/* Reading the chart */}
          <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <h4 className="text-sm font-semibold text-orange-400 mb-3">
              Reading the chart
            </h4>
            <div className="space-y-3">
              <div className="flex gap-3 items-start">
                <span className="text-lg leading-none mt-0.5">1.</span>
                <p className="text-sm text-slate-300 leading-relaxed">
                  <strong className="text-slate-200">The top tokens are generic words.</strong>{' '}
                  "This" (18%), "The" (14%), "Based" (9%). The model's strongest instinct
                  is to start a descriptive sentence &mdash; not to classify.
                </p>
              </div>
              <div className="flex gap-3 items-start">
                <span className="text-lg leading-none mt-0.5">2.</span>
                <p className="text-sm text-slate-300 leading-relaxed">
                  <strong className="text-green-400">"OLTP" has only 4% probability</strong>{' '}
                  (highlighted in green). The correct answer is in the model's vocabulary
                  &mdash; it's not impossible for the model to say it &mdash; but it's
                  buried under a pile of generic alternatives.
                </p>
              </div>
              <div className="flex gap-3 items-start">
                <span className="text-lg leading-none mt-0.5">3.</span>
                <p className="text-sm text-slate-300 leading-relaxed">
                  <strong className="text-slate-200">The distribution is flat.</strong>{' '}
                  No single token dominates. This is what uncertainty looks like in a
                  language model &mdash; it has no strong opinion about how to start, so
                  probability is spread thinly across many options.
                </p>
              </div>
            </div>
          </div>

          {/* Why this matters */}
          <div className="p-4 rounded-lg bg-orange-950/20 border border-orange-800/30">
            <h4 className="text-sm font-semibold text-orange-400 mb-2">
              Why this matters for post-training
            </h4>
            <p className="text-sm text-slate-300 leading-relaxed mb-2">
              When we later apply fine-tuning techniques (SFT, DPO, GRPO), the goal is
              to reshape this probability distribution. A well-trained model for this
              task should put 80%+ probability on "OLTP" and near-zero on generic words
              like "This" or "The."
            </p>
            <p className="text-sm text-slate-400 leading-relaxed">
              That's the core idea of post-training: <strong className="text-orange-300">
              shifting the model's probability distribution</strong> so that the correct,
              domain-specific answer becomes the most likely output &mdash; not just a
              long-shot possibility buried at 4%.
            </p>
          </div>

          {/* Technical detail */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/50">
              <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wide mb-2">
                What is a token?
              </h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                Language models don't work with whole words. They split text into
                "tokens" &mdash; subword pieces. "OLTP" might be one token, "Database"
                another. SmolLM2-360M has a vocabulary of ~49,152 tokens. Every generation
                step picks from this full set based on the probability distribution.
              </p>
            </div>
            <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/50">
              <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wide mb-2">
                Temperature and sampling
              </h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                The probabilities shown are the model's raw "logits" after softmax. In
                practice, a "temperature" parameter can sharpen or flatten this
                distribution before sampling. Lower temperature = more deterministic
                (picks the top token more often). But even at temperature 0, a flat
                distribution like this produces unreliable output because "This" and
                "The" are so close in probability.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
