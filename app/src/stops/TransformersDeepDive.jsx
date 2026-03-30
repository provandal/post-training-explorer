// Bonus stop: Deep dive into Transformer architecture and attention layers.
// Accessible from Explore mode via the LoRA Weights aside in SFTComparison.

export default function TransformersDeepDive() {
  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <span className="text-xs font-semibold text-violet-400 uppercase tracking-wide">
          Bonus Deep Dive
        </span>
        <h2 className="text-xl font-bold text-white mt-1">
          Transformers and Attention: The Architecture Behind LLMs
        </h2>
        <p className="text-sm text-slate-400 mt-2">
          Every large language model &mdash; GPT, Claude, Llama, SmolLM2 &mdash; is built on the
          same core architecture: the <strong className="text-violet-300">Transformer</strong>.
          Understanding how it works explains why techniques like LoRA and fine-tuning are possible
          at all.
        </p>
      </div>

      {/* The big picture */}
      <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-violet-400 mb-3">What is a Transformer?</h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          A Transformer is a neural network architecture designed to process sequences of tokens
          (words, subwords, or characters). It was introduced in 2017 in the paper "Attention Is All
          You Need" and has since become the foundation of virtually all modern language models.
        </p>
        <p className="text-sm text-slate-400 leading-relaxed mb-4">
          The key innovation: instead of processing tokens one at a time (like older RNNs), a
          Transformer processes all tokens <em>in parallel</em> and uses a mechanism called{' '}
          <strong className="text-violet-300">attention</strong> to figure out which tokens are
          relevant to each other.
        </p>

        {/* Architecture diagram */}
        <div className="p-4 rounded-lg bg-slate-900/50 border border-slate-700/30">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">
            Simplified Transformer Architecture (decoder-only, like SmolLM2)
          </p>
          <div className="space-y-2 font-mono text-xs">
            <div className="flex items-center gap-3">
              <span className="w-24 text-right text-slate-500">Input</span>
              <div className="flex-1 p-2 rounded bg-slate-800 border border-slate-700 text-slate-300">
                "Classify this storage I/O workload: IOPS: 45000..."
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="w-24 text-right text-slate-500">&darr;</span>
              <span className="text-slate-600">Tokenize &rarr; convert to token IDs</span>
            </div>
            <div className="flex items-center gap-3">
              <span className="w-24 text-right text-blue-400">Embedding</span>
              <div className="flex-1 p-2 rounded bg-blue-950/30 border border-blue-800/30 text-blue-300">
                Each token &rarr; 960-dimensional vector (SmolLM2's hidden size)
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="w-24 text-right text-slate-500">&darr;</span>
              <span className="text-slate-600">
                + positional encoding (where each token is in the sequence)
              </span>
            </div>

            {/* Transformer layers block */}
            <div className="flex items-stretch gap-3">
              <span className="w-24 text-right text-violet-400 pt-2">
                Layers
                <br />
                (&times;32)
              </span>
              <div className="flex-1 p-3 rounded bg-violet-950/20 border-2 border-violet-800/40 space-y-2">
                <div className="p-2 rounded bg-violet-900/30 border border-violet-700/30 text-violet-300">
                  Self-Attention &mdash; "which other tokens should I pay attention to?"
                </div>
                <div className="text-center text-slate-600">&darr; add &amp; normalize</div>
                <div className="p-2 rounded bg-slate-800 border border-slate-700 text-slate-300">
                  Feed-Forward Network &mdash; process each token independently
                </div>
                <div className="text-center text-slate-600">&darr; add &amp; normalize</div>
                <p className="text-xs text-violet-400/60 italic">
                  This block repeats 32 times. Each is called a "layer."
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <span className="w-24 text-right text-slate-500">&darr;</span>
              <span className="text-slate-600">Final layer norm</span>
            </div>
            <div className="flex items-center gap-3">
              <span className="w-24 text-right text-green-400">Output</span>
              <div className="flex-1 p-2 rounded bg-green-950/30 border border-green-800/30 text-green-300">
                Probability distribution over ~49,152 possible next tokens
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Attention explained */}
      <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-violet-400 mb-3">
          Attention: how the model connects ideas
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          Attention is the mechanism that lets each token "look at" every other token in the
          sequence and decide how much to weight each one. This is how the model connects "IOPS:
          45000" with "OLTP Database" even when they're far apart in the text.
        </p>

        <div className="p-4 rounded-lg bg-slate-900/50 border border-slate-700/30 mb-4">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">
            How attention works in three steps
          </p>
          <div className="space-y-3">
            <div className="flex gap-3 items-start">
              <span className="text-sm font-bold text-violet-400 w-6 shrink-0">1.</span>
              <div>
                <p className="text-sm text-slate-300 leading-relaxed">
                  <strong className="text-violet-300">Query, Key, Value.</strong> Each token is
                  projected into three vectors: a <em>query</em> ("what am I looking for?"), a{' '}
                  <em>key</em> ("what do I contain?"), and a <em>value</em> ("what information
                  should I contribute?"). These projections are learned matrices &mdash;
                  <strong className="text-violet-300">
                    {' '}
                    this is where LoRA inserts its adapters.
                  </strong>
                </p>
              </div>
            </div>
            <div className="flex gap-3 items-start">
              <span className="text-sm font-bold text-violet-400 w-6 shrink-0">2.</span>
              <div>
                <p className="text-sm text-slate-300 leading-relaxed">
                  <strong className="text-violet-300">Attention scores.</strong> The model computes
                  dot products between each token's query and every other token's key. High score =
                  "these tokens are relevant to each other." The scores are normalized with softmax
                  so they sum to 1.
                </p>
              </div>
            </div>
            <div className="flex gap-3 items-start">
              <span className="text-sm font-bold text-violet-400 w-6 shrink-0">3.</span>
              <div>
                <p className="text-sm text-slate-300 leading-relaxed">
                  <strong className="text-violet-300">Weighted combination.</strong> Each token's
                  output is a weighted sum of all value vectors, weighted by the attention scores.
                  Tokens that scored high contribute more. This is how information flows between
                  distant parts of the sequence.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Concrete example */}
        <div className="p-4 rounded-lg bg-violet-950/20 border border-violet-800/30">
          <p className="text-xs font-semibold text-violet-400 uppercase tracking-wide mb-2">
            Example: What attention looks like for I/O classification
          </p>
          <p className="text-xs text-slate-300 leading-relaxed mb-3">
            When the model processes "IOPS: 45000 | Latency: 0.3ms | Block Size: 8K" and needs to
            generate "Classification:", attention helps it connect:
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-2 text-xs">
            <div className="p-2 rounded bg-slate-800/50 border border-slate-700/30">
              <span className="font-semibold text-green-400">"IOPS: 45000"</span>
              <p className="text-slate-500 mt-1">
                High attention weight &rarr; this is a strong discriminator between workload types
              </p>
            </div>
            <div className="p-2 rounded bg-slate-800/50 border border-slate-700/30">
              <span className="font-semibold text-green-400">"Latency: 0.3ms"</span>
              <p className="text-slate-500 mt-1">
                High attention weight &rarr; sub-ms latency strongly suggests OLTP
              </p>
            </div>
            <div className="p-2 rounded bg-slate-800/50 border border-slate-700/30">
              <span className="font-semibold text-slate-500">"Block Size: 8K"</span>
              <p className="text-slate-500 mt-1">
                Moderate attention &rarr; useful but less discriminating on its own
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Multi-head attention */}
      <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-violet-400 mb-3">
          Multi-head attention: looking at things from multiple angles
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          A single attention mechanism can only capture one type of relationship at a time.
          SmolLM2-360M uses{' '}
          <strong className="text-violet-300">15 attention heads per layer</strong>, each learning
          to focus on different patterns:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="p-3 rounded bg-slate-800/50 border border-slate-700/30">
            <h4 className="text-xs font-semibold text-violet-300 mb-1">Head A: Syntax</h4>
            <p className="text-xs text-slate-400">
              Might learn to connect "Classification:" with the format that follows &mdash; ensuring
              the output follows a structured pattern.
            </p>
          </div>
          <div className="p-3 rounded bg-slate-800/50 border border-slate-700/30">
            <h4 className="text-xs font-semibold text-violet-300 mb-1">
              Head B: Numeric reasoning
            </h4>
            <p className="text-xs text-slate-400">
              Might focus on the relationship between "IOPS: 45000" and "high" &mdash; connecting
              specific numbers to qualitative descriptions.
            </p>
          </div>
          <div className="p-3 rounded bg-slate-800/50 border border-slate-700/30">
            <h4 className="text-xs font-semibold text-violet-300 mb-1">Head C: Cross-metric</h4>
            <p className="text-xs text-slate-400">
              Might learn that high IOPS + low latency + small blocks together point to OLTP &mdash;
              combining evidence across multiple fields.
            </p>
          </div>
        </div>
        <p className="text-xs text-slate-500 mt-3 leading-relaxed">
          The outputs of all 15 heads are concatenated and projected through a final linear layer.
          Each layer in the model has its own set of 15 heads, for a total of 15 &times; 32 ={' '}
          <strong className="text-slate-300">480 attention heads</strong> across the entire model.
        </p>
      </div>

      {/* How layers build on each other */}
      <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-violet-400 mb-3">Layers build on each other</h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          SmolLM2-360M has 32 layers stacked on top of each other. Each layer takes the output of
          the previous layer as input. Research has shown that different layers tend to capture
          different levels of abstraction:
        </p>
        <div className="space-y-2">
          <div className="flex gap-3 items-center">
            <div className="w-20 text-right">
              <span className="text-xs font-mono text-blue-400">Layers 1-8</span>
            </div>
            <div className="flex-1 h-8 rounded bg-blue-950/30 border border-blue-800/30 flex items-center px-3">
              <span className="text-xs text-blue-300">
                Surface patterns: token identity, position, basic syntax
              </span>
            </div>
          </div>
          <div className="flex gap-3 items-center">
            <div className="w-20 text-right">
              <span className="text-xs font-mono text-violet-400">Layers 9-20</span>
            </div>
            <div className="flex-1 h-8 rounded bg-violet-950/30 border border-violet-800/30 flex items-center px-3">
              <span className="text-xs text-violet-300">
                Semantic meaning: "45000 IOPS is high", number-to-concept mapping
              </span>
            </div>
          </div>
          <div className="flex gap-3 items-center">
            <div className="w-20 text-right">
              <span className="text-xs font-mono text-green-400">Layers 21-32</span>
            </div>
            <div className="flex-1 h-8 rounded bg-green-950/30 border border-green-800/30 flex items-center px-3">
              <span className="text-xs text-green-300">
                Task-level reasoning: combining evidence &rarr; "this is OLTP"
              </span>
            </div>
          </div>
        </div>
        <p className="text-xs text-slate-500 mt-3 leading-relaxed">
          This is why LoRA can be targeted at specific layers. For a classification task, adapting
          the later layers (where task-level reasoning happens) often has more impact than adapting
          the early layers (which handle surface-level patterns that are already well-learned).
        </p>
      </div>

      {/* Connection to LoRA */}
      <div className="p-5 rounded-lg bg-gradient-to-r from-violet-950/30 to-blue-950/30 border border-violet-800/30">
        <h3 className="text-base font-semibold text-violet-400 mb-3">Where LoRA fits in</h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          Now you can see why the LoRA heatmap is labeled "for one attention layer." Each attention
          head has four weight matrices (Q, K, V, and output projection). LoRA inserts a small
          trainable adapter alongside these matrices:
        </p>
        <div className="p-4 rounded-lg bg-slate-900/50 border border-slate-700/30 font-mono text-xs space-y-1">
          <div className="text-slate-500"># Original attention computation:</div>
          <div className="text-slate-300">
            output = input &times; W<sub className="text-slate-500">original</sub>
            <span className="text-slate-600"> (360M params, frozen)</span>
          </div>
          <div className="text-slate-500 mt-2"># With LoRA:</div>
          <div className="text-violet-300">
            output = input &times; W<sub className="text-slate-500">original</sub> + input &times; A
            &times; B<span className="text-violet-500"> (432K params, trained)</span>
          </div>
          <div className="text-slate-500 mt-2">
            # A is rank&times;hidden, B is hidden&times;rank
          </div>
          <div className="text-slate-500">
            # rank=16 for our training &mdash; hence the 16 rows in the heatmap
          </div>
        </div>
        <p className="text-sm text-slate-400 leading-relaxed mt-3">
          The heatmap you saw on the SFT stop visualizes the product A &times; B for one layer's
          query projection. The structured patterns in those weights are the model's learned
          understanding of "how to map I/O metrics to workload categories" &mdash; compressed into a
          tiny 16-rank adapter matrix.
        </p>
      </div>

      {/* SmolLM2 spec card */}
      <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/50">
        <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">
          SmolLM2-360M architecture summary
        </h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-center text-xs">
          <div className="p-2 rounded bg-slate-900/50">
            <div className="text-lg font-bold text-slate-200">32</div>
            <div className="text-slate-500">layers</div>
          </div>
          <div className="p-2 rounded bg-slate-900/50">
            <div className="text-lg font-bold text-slate-200">15</div>
            <div className="text-slate-500">attention heads/layer</div>
          </div>
          <div className="p-2 rounded bg-slate-900/50">
            <div className="text-lg font-bold text-slate-200">960</div>
            <div className="text-slate-500">hidden dimension</div>
          </div>
          <div className="p-2 rounded bg-slate-900/50">
            <div className="text-lg font-bold text-slate-200">49,152</div>
            <div className="text-slate-500">vocabulary size</div>
          </div>
        </div>
      </div>
    </div>
  )
}
