// Bonus stop: Deep dive into LoRA — how parameters are selected, rank decomposition,
// and the training process. Accessible from the "How SFT Works" tab in SFTComparison.

import { useTranslation } from 'react-i18next'

export default function LoRADeepDive() {
  const { t } = useTranslation()

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <span className="text-xs font-semibold text-violet-400 uppercase tracking-wide">
          {t('deepdive.lora.bonusLabel')}
        </span>
        <h2 className="text-xl font-bold text-white mt-1">{t('deepdive.lora.title')}</h2>
        <p className="text-sm text-slate-400 mt-2">
          {t('deepdive.lora.subtitle', {
            defaultValue:
              "LoRA trains less than 0.12% of a model's parameters and yet changes its behavior dramatically. How does it know which parameters to target, and why does such a small change have such a big effect?",
          })}
        </p>
      </div>

      {/* The problem LoRA solves */}
      <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-violet-400 mb-3">
          {t('deepdive.lora.whyNotWhole')}
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          {t('deepdive.lora.whyNotWholeP')}
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="p-3 rounded bg-red-950/20 border border-red-800/30">
            <h4 className="text-xs font-semibold text-red-400 uppercase tracking-wide mb-2">
              {t('deepdive.lora.fullFinetuning')}
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
              {t('deepdive.lora.loraFinetuning')}
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
          {t('deepdive.lora.whichParams')}
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          {t('deepdive.lora.whichParamsP', {
            defaultValue:
              "LoRA doesn't randomly pick parameters. It targets specific weight matrices inside the attention mechanism of each transformer layer. Every attention layer has four key projections:",
          })}
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-4">
          <div className="p-3 rounded bg-violet-950/30 border border-violet-800/40 text-center">
            <div className="text-sm font-bold font-mono text-violet-400">q_proj</div>
            <div className="text-xs text-slate-500 mt-1">{t('deepdive.lora.qProj')}</div>
            <div className="text-xs text-slate-600 mt-0.5">{t('deepdive.lora.qProjSub')}</div>
          </div>
          <div className="p-3 rounded bg-violet-950/30 border border-violet-800/40 text-center">
            <div className="text-sm font-bold font-mono text-violet-400">k_proj</div>
            <div className="text-xs text-slate-500 mt-1">{t('deepdive.lora.kProj')}</div>
            <div className="text-xs text-slate-600 mt-0.5">{t('deepdive.lora.kProjSub')}</div>
          </div>
          <div className="p-3 rounded bg-violet-950/30 border border-violet-800/40 text-center">
            <div className="text-sm font-bold font-mono text-violet-400">v_proj</div>
            <div className="text-xs text-slate-500 mt-1">{t('deepdive.lora.vProj')}</div>
            <div className="text-xs text-slate-600 mt-0.5">{t('deepdive.lora.vProjSub')}</div>
          </div>
          <div className="p-3 rounded bg-slate-800/50 border border-slate-700/30 text-center">
            <div className="text-sm font-bold font-mono text-slate-400">o_proj</div>
            <div className="text-xs text-slate-500 mt-1">{t('deepdive.lora.oProj')}</div>
            <div className="text-xs text-slate-600 mt-0.5">{t('deepdive.lora.oProjSub')}</div>
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
          {t('deepdive.lora.rankDecomp')}
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          {t('deepdive.lora.rankDecompP')}
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
            <h4 className="text-xs font-semibold text-slate-300 mb-1">
              {t('deepdive.lora.rank4')}
            </h4>
            <p className="text-xs text-slate-400">{t('deepdive.lora.rank4P')}</p>
          </div>
          <div className="p-3 rounded bg-violet-950/30 border border-violet-800/30">
            <h4 className="text-xs font-semibold text-violet-300 mb-1">
              {t('deepdive.lora.rank16')}
            </h4>
            <p className="text-xs text-slate-400">{t('deepdive.lora.rank16P')}</p>
          </div>
          <div className="p-3 rounded bg-slate-800/50 border border-slate-700/30">
            <h4 className="text-xs font-semibold text-slate-300 mb-1">
              {t('deepdive.lora.rank64')}
            </h4>
            <p className="text-xs text-slate-400">{t('deepdive.lora.rank64P')}</p>
          </div>
        </div>
      </div>

      {/* Why low rank works */}
      <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-violet-400 mb-3">
          {t('deepdive.lora.whySmallWorks')}
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          This is the key insight from the original LoRA paper: when you fine-tune a large model on
          a specific task, the weight changes live in a{' '}
          <strong className="text-violet-300">low-dimensional subspace</strong>. In plain English:
        </p>
        <div className="p-4 rounded-lg bg-violet-950/20 border border-violet-800/30 mb-4">
          <p className="text-sm text-violet-300 italic leading-relaxed">
            {t('deepdive.lora.whySmallWorksAnalogy')}
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
          {t('deepdive.lora.trainingLoop')}
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-4">
          {t('deepdive.lora.trainingLoopP')}
        </p>
        <div className="space-y-3">
          <div className="flex gap-3 items-start">
            <span className="text-sm font-bold text-violet-400 w-6 shrink-0">1.</span>
            <div>
              <p className="text-sm font-semibold text-slate-200">{t('deepdive.lora.step1')}</p>
              <p className="text-xs text-slate-400 mt-1">{t('deepdive.lora.step1P')}</p>
            </div>
          </div>
          <div className="flex gap-3 items-start">
            <span className="text-sm font-bold text-violet-400 w-6 shrink-0">2.</span>
            <div>
              <p className="text-sm font-semibold text-slate-200">{t('deepdive.lora.step2')}</p>
              <p className="text-xs text-slate-400 mt-1">{t('deepdive.lora.step2P')}</p>
            </div>
          </div>
          <div className="flex gap-3 items-start">
            <span className="text-sm font-bold text-violet-400 w-6 shrink-0">3.</span>
            <div>
              <p className="text-sm font-semibold text-slate-200">{t('deepdive.lora.step3')}</p>
              <p className="text-xs text-slate-400 mt-1">
                {t('deepdive.lora.step3P', {
                  defaultValue:
                    "The loss is backpropagated through the network. But here's the key: gradients are only computed for the A and B matrices (the adapter), not for the frozen W matrices. This is what makes LoRA fast and memory-efficient — you skip gradient computation for 99.88% of the parameters.",
                })}
              </p>
            </div>
          </div>
          <div className="flex gap-3 items-start">
            <span className="text-sm font-bold text-violet-400 w-6 shrink-0">4.</span>
            <div>
              <p className="text-sm font-semibold text-slate-200">{t('deepdive.lora.step4')}</p>
              <p className="text-xs text-slate-400 mt-1">{t('deepdive.lora.step4P')}</p>
            </div>
          </div>
          <div className="flex gap-3 items-start">
            <span className="text-sm font-bold text-violet-400 w-6 shrink-0">5.</span>
            <div>
              <p className="text-sm font-semibold text-slate-200">{t('deepdive.lora.step5')}</p>
              <p className="text-xs text-slate-400 mt-1">{t('deepdive.lora.step5P')}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Parameter count breakdown */}
      <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-violet-400 mb-3">
          {t('deepdive.lora.paramCountHeading')}
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          {t('deepdive.lora.paramCountP')}
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
          {t('deepdive.lora.infraImplications')}
        </h4>
        <p className="text-sm text-slate-300 leading-relaxed mb-2">{t('deepdive.lora.infraP1')}</p>
        <p className="text-sm text-slate-400 leading-relaxed">{t('deepdive.lora.infraP2')}</p>
      </div>
    </div>
  )
}
