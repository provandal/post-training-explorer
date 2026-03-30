// Bonus stop: Deep dive into LoRA — how parameters are selected, rank decomposition,
// and the training process. Accessible from the "How SFT Works" tab in SFTComparison.

export default function LoRADeepDive() {
  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <span className="text-xs font-semibold text-violet-400 uppercase tracking-wide">
          Bonus Deep Dive
        </span>
        <h2 className="text-xl font-bold text-white mt-1">LoRA: How It Decides What to Train</h2>
        <p className="text-sm text-slate-400 mt-2">
          LoRA trains less than 0.12% of a model's parameters and yet changes its behavior
          dramatically. How does it know <em>which</em> parameters to target, and why does such a
          small change have such a big effect?
        </p>
      </div>

      {/* The problem LoRA solves */}
      <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-violet-400 mb-3">
          Why not train the whole model?
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          Full fine-tuning updates every one of the model's 360 million parameters. This requires
          storing a complete copy of the model in GPU memory (for gradients and optimizer states),
          which triples the memory footprint. For SmolLM2 that's manageable, but for a 70B-parameter
          model it's prohibitive.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="p-3 rounded bg-red-950/20 border border-red-800/30">
            <h4 className="text-xs font-semibold text-red-400 uppercase tracking-wide mb-2">
              Full fine-tuning
            </h4>
            <div className="text-xs text-slate-400 space-y-1">
              <p>Model weights: 720 MB</p>
              <p>Gradients: 720 MB</p>
              <p>Optimizer states (Adam): 1,440 MB</p>
              <p className="font-semibold text-red-300 pt-1 border-t border-red-800/30">
                Total: ~2.9 GB GPU memory
              </p>
            </div>
          </div>
          <div className="p-3 rounded bg-green-950/20 border border-green-800/30">
            <h4 className="text-xs font-semibold text-green-400 uppercase tracking-wide mb-2">
              LoRA fine-tuning
            </h4>
            <div className="text-xs text-slate-400 space-y-1">
              <p>Model weights: 720 MB (frozen, no gradients)</p>
              <p>Adapter weights: 1.7 MB</p>
              <p>Adapter gradients + optimizer: ~5 MB</p>
              <p className="font-semibold text-green-300 pt-1 border-t border-green-800/30">
                Total: ~727 MB GPU memory
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Which parameters and why */}
      <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-violet-400 mb-3">
          Which parameters does LoRA target?
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          LoRA doesn't randomly pick parameters. It targets specific{' '}
          <strong className="text-violet-300">
            weight matrices inside the attention mechanism
          </strong>{' '}
          of each transformer layer. Every attention layer has four key projections:
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-4">
          <div className="p-3 rounded bg-violet-950/30 border border-violet-800/40 text-center">
            <div className="text-sm font-bold font-mono text-violet-400">q_proj</div>
            <div className="text-xs text-slate-500 mt-1">Query projection</div>
            <div className="text-xs text-slate-600 mt-0.5">"What am I looking for?"</div>
          </div>
          <div className="p-3 rounded bg-violet-950/30 border border-violet-800/40 text-center">
            <div className="text-sm font-bold font-mono text-violet-400">k_proj</div>
            <div className="text-xs text-slate-500 mt-1">Key projection</div>
            <div className="text-xs text-slate-600 mt-0.5">"What do I contain?"</div>
          </div>
          <div className="p-3 rounded bg-violet-950/30 border border-violet-800/40 text-center">
            <div className="text-sm font-bold font-mono text-violet-400">v_proj</div>
            <div className="text-xs text-slate-500 mt-1">Value projection</div>
            <div className="text-xs text-slate-600 mt-0.5">"What info do I carry?"</div>
          </div>
          <div className="p-3 rounded bg-slate-800/50 border border-slate-700/30 text-center">
            <div className="text-sm font-bold font-mono text-slate-400">o_proj</div>
            <div className="text-xs text-slate-500 mt-1">Output projection</div>
            <div className="text-xs text-slate-600 mt-0.5">"Combine all heads"</div>
          </div>
        </div>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          In our training configuration, we attach LoRA adapters to{' '}
          <strong className="text-violet-300">q_proj and v_proj</strong> in every layer. This is the
          most common default because research has shown these two projections have the most impact
          on model behavior:
        </p>
        <div className="space-y-2 text-xs text-slate-400">
          <div className="flex gap-3 items-start">
            <span className="font-mono text-violet-400 w-14 shrink-0 pt-0.5">q_proj</span>
            <p>
              Controls what each token <em>searches for</em>. Adapting this changes which
              relationships the model considers important &mdash; e.g., learning that "IOPS" and
              "Latency" together are more informative than either alone.
            </p>
          </div>
          <div className="flex gap-3 items-start">
            <span className="font-mono text-violet-400 w-14 shrink-0 pt-0.5">v_proj</span>
            <p>
              Controls what <em>information flows forward</em> from each token. Adapting this
              changes what the model extracts &mdash; e.g., learning to carry "45000 = high" rather
              than just the raw number.
            </p>
          </div>
        </div>
        <div className="mt-3 p-3 rounded bg-blue-950/20 border border-blue-800/30">
          <p className="text-xs text-blue-300 leading-relaxed">
            <strong>Why not all four projections?</strong> You can &mdash; and sometimes it helps.
            But each additional target doubles the adapter size. For our task (6-class
            classification with clear patterns), q_proj + v_proj is enough. For more complex tasks,
            you might target all four, or even include the feed-forward layers.
          </p>
        </div>
      </div>

      {/* Rank decomposition */}
      <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-violet-400 mb-3">
          Rank decomposition: why it's so small
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          A normal weight matrix in SmolLM2's attention is 960 &times; 960 = 921,600 parameters.
          LoRA replaces the update to this matrix with two much smaller matrices:
        </p>
        <div className="p-4 rounded-lg bg-slate-900/50 border border-slate-700/30 mb-4">
          <div className="font-mono text-xs space-y-3">
            <div className="text-slate-500">
              # Instead of learning a full 960&times;960 update matrix:
            </div>
            <div className="text-red-300">
              &Delta;W = <span className="text-slate-400">[960 &times; 960]</span> = 921,600
              parameters
            </div>
            <div className="text-slate-500 mt-2"># LoRA decomposes it into two small matrices:</div>
            <div className="text-violet-300">
              A = <span className="text-slate-400">[960 &times; 16]</span> = 15,360 parameters
            </div>
            <div className="text-violet-300">
              B = <span className="text-slate-400">[16 &times; 960]</span> = 15,360 parameters
            </div>
            <div className="text-green-300 mt-2">
              &Delta;W &asymp; A &times; B{' '}
              <span className="text-slate-400">
                &mdash; total: 30,720 parameters (3.3% of the full matrix)
              </span>
            </div>
          </div>
        </div>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          The number <strong className="text-violet-300">16</strong> is the{' '}
          <strong className="text-violet-300">rank</strong> &mdash; a hyperparameter you choose
          before training. It controls the capacity of the adapter:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="p-3 rounded bg-slate-800/50 border border-slate-700/30">
            <h4 className="text-xs font-semibold text-slate-300 mb-1">Rank 4 (tiny)</h4>
            <p className="text-xs text-slate-400">
              Very few parameters. Fast training, tiny adapter. Works for simple behavioral changes
              (output format tweaks).
            </p>
          </div>
          <div className="p-3 rounded bg-violet-950/30 border border-violet-800/30">
            <h4 className="text-xs font-semibold text-violet-300 mb-1">Rank 16 (our choice)</h4>
            <p className="text-xs text-slate-400">
              Good balance of capacity and efficiency. Enough to learn a 6-class classification task
              with reasoning. This is the most common default.
            </p>
          </div>
          <div className="p-3 rounded bg-slate-800/50 border border-slate-700/30">
            <h4 className="text-xs font-semibold text-slate-300 mb-1">Rank 64+ (large)</h4>
            <p className="text-xs text-slate-400">
              Approaches full fine-tuning capacity. Useful for complex tasks or multilingual
              adaptation. Larger adapter, slower training, more GPU memory.
            </p>
          </div>
        </div>
      </div>

      {/* Why low rank works */}
      <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-violet-400 mb-3">
          Why does such a small change work?
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          This is the key insight from the original LoRA paper: when you fine-tune a large model on
          a specific task, the weight changes live in a{' '}
          <strong className="text-violet-300">low-dimensional subspace</strong>. In plain English:
        </p>
        <div className="p-4 rounded-lg bg-violet-950/20 border border-violet-800/30 mb-4">
          <p className="text-sm text-violet-300 italic leading-relaxed">
            The base model already knows how to process language, reason about numbers, and follow
            instructions. Teaching it to classify storage I/O patterns doesn't require changing
            everything &mdash; it just needs a small, specific adjustment to redirect existing
            capabilities toward the new task. Like adjusting the steering on a car that already
            knows how to drive.
          </p>
        </div>
        <p className="text-sm text-slate-400 leading-relaxed">
          The rank-16 adapter can only represent changes in 16 "directions" in the weight space. For
          our task, 16 directions is more than enough because the classification task has 6
          categories, each defined by a handful of metric patterns. A rank of 16 gives comfortable
          headroom to learn the category boundaries, the output format, and the reasoning style
          simultaneously.
        </p>
      </div>

      {/* The training process */}
      <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-violet-400 mb-3">
          The training loop: what actually happens
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-4">
          Here's what happens for each training example, step by step:
        </p>
        <div className="space-y-3">
          <div className="flex gap-3 items-start">
            <span className="text-sm font-bold text-violet-400 w-6 shrink-0">1.</span>
            <div>
              <p className="text-sm font-semibold text-slate-200">Forward pass</p>
              <p className="text-xs text-slate-400 mt-1">
                The input ("IOPS: 45000...") flows through all 32 layers. At each attention layer,
                the computation is: output = input &times; W<sub>frozen</sub> + input &times; A
                &times; B. The frozen weights do the heavy lifting; the adapter nudges the result.
              </p>
            </div>
          </div>
          <div className="flex gap-3 items-start">
            <span className="text-sm font-bold text-violet-400 w-6 shrink-0">2.</span>
            <div>
              <p className="text-sm font-semibold text-slate-200">Compute loss</p>
              <p className="text-xs text-slate-400 mt-1">
                The model's output is compared to the desired output ("Classification: OLTP
                Database..."). The loss function measures how different they are &mdash; token by
                token, using cross-entropy loss. Lower loss = closer to the desired output.
              </p>
            </div>
          </div>
          <div className="flex gap-3 items-start">
            <span className="text-sm font-bold text-violet-400 w-6 shrink-0">3.</span>
            <div>
              <p className="text-sm font-semibold text-slate-200">Backward pass (gradients)</p>
              <p className="text-xs text-slate-400 mt-1">
                The loss is backpropagated through the network. But here's the key: gradients are
                only computed for the A and B matrices (the adapter), <strong>not</strong> for the
                frozen W matrices. This is what makes LoRA fast and memory-efficient &mdash; you
                skip gradient computation for 99.88% of the parameters.
              </p>
            </div>
          </div>
          <div className="flex gap-3 items-start">
            <span className="text-sm font-bold text-violet-400 w-6 shrink-0">4.</span>
            <div>
              <p className="text-sm font-semibold text-slate-200">Update adapter weights</p>
              <p className="text-xs text-slate-400 mt-1">
                The optimizer (AdamW) uses the gradients to adjust A and B slightly, pushing the
                model's output closer to the desired answer. This repeats for each training example,
                across 3 epochs (3 full passes through the 1,400 examples).
              </p>
            </div>
          </div>
          <div className="flex gap-3 items-start">
            <span className="text-sm font-bold text-violet-400 w-6 shrink-0">5.</span>
            <div>
              <p className="text-sm font-semibold text-slate-200">Save the adapter</p>
              <p className="text-xs text-slate-400 mt-1">
                After training, only the A and B matrices are saved (1.7 MB). To use the fine-tuned
                model, you load the base model and then apply the adapter on top. You can swap
                adapters for different tasks without re-downloading the base model.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Parameter count breakdown */}
      <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-violet-400 mb-3">
          Where do the 432K parameters come from?
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          Let's count. For each of the 32 layers, we adapt 2 projections (q_proj, v_proj), each with
          two low-rank matrices:
        </p>
        <div className="p-4 rounded-lg bg-slate-900/50 border border-slate-700/30 font-mono text-xs space-y-1">
          <div className="text-slate-400">Per projection:</div>
          <div className="text-slate-300 ml-4">A: 960 &times; 16 = 15,360 params</div>
          <div className="text-slate-300 ml-4">B: 16 &times; 960 = 15,360 params</div>
          <div className="text-slate-400 ml-4">Subtotal: 30,720 params</div>
          <div className="text-slate-400 mt-2">Per layer (2 projections):</div>
          <div className="text-slate-300 ml-4">30,720 &times; 2 = 61,440 params</div>
          <div className="text-slate-400 mt-2">Total (32 layers):</div>
          <div className="text-violet-300 ml-4 font-semibold">
            61,440 &times; 32 = <span className="text-violet-400">1,966,080 params</span>
          </div>
          <div className="text-slate-500 mt-2 italic">
            Note: the actual count varies slightly depending on implementation details (bias terms,
            layer norm adaptations, etc.)
          </div>
        </div>
      </div>

      {/* Practical implications */}
      <div className="p-4 rounded-lg bg-gradient-to-r from-violet-950/30 to-blue-950/30 border border-violet-800/30">
        <h4 className="text-sm font-semibold text-violet-400 mb-2">
          Why this matters for infrastructure teams
        </h4>
        <p className="text-sm text-slate-300 leading-relaxed mb-2">
          LoRA's efficiency is what makes fine-tuning practical at enterprise scale. A single
          accelerator with 8 GB VRAM can fine-tune a 360M-parameter model. A T4 (15 GB) can handle
          models up to ~7B parameters with LoRA. Without LoRA, you'd need 4&times; the training VRAM
          and the training time would increase proportionally.
        </p>
        <p className="text-sm text-slate-400 leading-relaxed">
          From a storage perspective, LoRA also means you can keep one base model on disk (720 MB)
          and maintain dozens of task-specific adapters (1-2 MB each). Swapping tasks means loading
          a different adapter file &mdash; not a different multi-gigabyte model.
        </p>
      </div>
    </div>
  )
}
