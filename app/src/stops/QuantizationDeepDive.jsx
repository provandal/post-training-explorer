// Bonus stop: Deep dive into quantization and deployment optimization.
// Covers precision formats, alignment layers, MX formats, and practical pipeline impact.

import { useTranslation } from 'react-i18next'

const precisionLevels = [
  {
    label: 'FP32',
    repr: '0.23456789012345678',
    bits: 32,
    bytes: '4 bytes',
    color: 'bg-red-500',
    barWidth: 'w-full',
  },
  {
    label: 'FP16',
    repr: '0.2346',
    bits: 16,
    bytes: '2 bytes',
    color: 'bg-orange-500',
    barWidth: 'w-1/2',
  },
  {
    label: 'INT8',
    repr: '60/255 \u2248 0.235',
    bits: 8,
    bytes: '1 byte',
    color: 'bg-yellow-500',
    barWidth: 'w-1/4',
  },
  {
    label: 'INT4',
    repr: '4/15 \u2248 0.267',
    bits: 4,
    bytes: '0.5 bytes',
    color: 'bg-green-500',
    barWidth: 'w-[12.5%]',
  },
]

const modelSizes = [
  { label: 'FP32', size: '28 GB', value: 28, color: 'bg-red-500/70' },
  { label: 'FP16 / BF16', size: '14 GB', value: 14, color: 'bg-orange-500/70' },
  { label: 'INT8', size: '7 GB', value: 7, color: 'bg-yellow-500/70' },
  { label: 'INT4', size: '3.5 GB', value: 3.5, color: 'bg-green-500/70' },
]

const hardwareSupport = [
  { name: 'NVIDIA H100', formats: 'Native FP8, INT8, FP16', hasAll: true },
  { name: 'NVIDIA T4', formats: 'Native INT8, FP16 (no FP8)', hasAll: false },
  { name: 'CPU', formats: 'Usually INT8 only, INT4 via software', hasAll: false },
]

const softwareFrameworks = ['PyTorch', 'ONNX Runtime', 'TensorRT', 'vLLM']

const mxFormats = ['MXFP8', 'MXFP6', 'MXFP4', 'MXINT8']

export default function QuantizationDeepDive() {
  const { t } = useTranslation()

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <span className="text-xs font-semibold text-cyan-400 uppercase tracking-wide">
          {t('deepdive.quantization.bonusLabel', { defaultValue: 'Bonus Deep Dive' })}
        </span>
        <h2 className="text-xl font-bold text-white mt-1">
          {t('deepdive.quantization.title', {
            defaultValue: 'Quantization: Trading Precision for Efficiency',
          })}
        </h2>
        <p className="text-sm text-slate-400 mt-2">
          {t('deepdive.quantization.subtitle', {
            defaultValue:
              'Quantization reduces the numeric precision of model weights \u2014 storing numbers with fewer bits to shrink model size, reduce memory bandwidth, and accelerate inference. But smaller numbers only help when the hardware, software, and model all agree on the format.',
          })}
        </p>
      </div>

      {/* Section 1: What is Quantization? */}
      <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-cyan-400 mb-3">What is Quantization?</h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-4">
          Neural network weights are just numbers. During training, those numbers are stored as
          32-bit floating point values for maximum precision. But at inference time, we can
          represent them with fewer bits &mdash; trading a small amount of precision for a large
          reduction in size and speed.
        </p>
        <p className="text-xs text-slate-500 mb-3">
          The same weight value represented at different precisions:
        </p>
        <div className="space-y-3">
          {precisionLevels.map((level) => (
            <div key={level.label} className="flex items-center gap-3">
              <span className="font-mono text-xs text-cyan-300 w-10 shrink-0">{level.label}</span>
              <div className="flex-1">
                <div
                  className={`${level.barWidth} h-6 ${level.color}/20 rounded border ${level.color.replace('bg-', 'border-')}/40 flex items-center px-2`}
                >
                  <span className="font-mono text-xs text-slate-300 truncate">{level.repr}</span>
                </div>
              </div>
              <span className="text-xs text-slate-500 w-24 text-right">
                {level.bits} bits, {level.bytes}
              </span>
            </div>
          ))}
        </div>
        <p className="text-xs text-slate-500 mt-3 italic">
          Each step roughly halves the storage. The precision loss is small for most weights, but
          accumulates across billions of parameters.
        </p>
      </div>

      {/* Section 2: The Precision Ladder */}
      <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-cyan-400 mb-3">The Precision Ladder</h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-4">
          Model sizes at different precisions for a{' '}
          <strong className="text-cyan-300">7B parameter</strong> model:
        </p>
        <div className="space-y-3">
          {modelSizes.map((item) => (
            <div key={item.label} className="flex items-center gap-3">
              <span className="text-xs text-slate-300 w-20 shrink-0 text-right font-medium">
                {item.label}
              </span>
              <div className="flex-1 h-7 bg-slate-900/50 rounded overflow-hidden">
                <div
                  className={`h-full ${item.color} rounded flex items-center px-2 transition-all`}
                  style={{ width: `${(item.value / 28) * 100}%` }}
                >
                  <span className="text-xs font-semibold text-white whitespace-nowrap">
                    {item.size}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4 p-3 rounded bg-cyan-950/20 border border-cyan-800/30">
          <p className="text-xs text-cyan-300 leading-relaxed">
            <strong>SmolLM2-360M in our demo:</strong> 1.4 GB (FP32) &rarr; 720 MB (FP16) &rarr; 180
            MB (ONNX INT8). An 8x reduction from full precision to deployed model.
          </p>
        </div>
      </div>

      {/* Section 3: The Three-Layer Alignment Problem */}
      <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-cyan-400 mb-3">
          The Three-Layer Alignment Problem
        </h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-4">
          The key insight: <strong className="text-cyan-300">smaller does not mean faster</strong>{' '}
          unless three layers are aligned. If any layer doesn&rsquo;t natively support your chosen
          format, you lose the benefit.
        </p>

        {/* Three stacked layers */}
        <div className="space-y-0">
          {/* Layer 3: Model Accuracy (top) */}
          <div className="p-4 rounded-t-lg bg-emerald-950/20 border border-emerald-800/30 border-b-0">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-lg">3</span>
              <h4 className="text-sm font-semibold text-emerald-400">Model Accuracy</h4>
              <span className="text-xs text-slate-500 ml-auto">top layer</span>
            </div>
            <ul className="text-xs text-slate-400 space-y-1 ml-6 list-disc">
              <li>Aggressive quantization can degrade accuracy</li>
              <li>Some models are designed for quantization (GPTQ, AWQ techniques)</li>
              <li>
                <strong className="text-emerald-300">QLoRA</strong>: 4-bit base + FP16 adapters =
                best of both worlds
              </li>
            </ul>
          </div>

          {/* Layer 2: Software (middle) */}
          <div className="p-4 bg-blue-950/20 border-x border-blue-800/30">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-lg">2</span>
              <h4 className="text-sm font-semibold text-blue-400">Software</h4>
              <span className="text-xs text-slate-500 ml-auto">middle layer</span>
            </div>
            <div className="flex flex-wrap gap-2 mb-2 ml-6">
              {softwareFrameworks.map((fw) => (
                <span
                  key={fw}
                  className="px-2 py-0.5 rounded text-xs font-mono bg-blue-900/30 text-blue-300 border border-blue-800/40"
                >
                  {fw}
                </span>
              ))}
            </div>
            <p className="text-xs text-slate-400 ml-6">
              If the runtime doesn&rsquo;t support INT4, it dequantizes back to FP16 at runtime
              &mdash; making it <em>slower</em> than running in FP16 directly.
            </p>
          </div>

          {/* Layer 1: Hardware (bottom) */}
          <div className="p-4 rounded-b-lg bg-amber-950/20 border border-amber-800/30 border-t-0">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-lg">1</span>
              <h4 className="text-sm font-semibold text-amber-400">Hardware</h4>
              <span className="text-xs text-slate-500 ml-auto">bottom layer</span>
            </div>
            <div className="space-y-1.5 ml-6">
              {hardwareSupport.map((hw) => (
                <div key={hw.name} className="flex items-center gap-2 text-xs">
                  <span className="font-mono text-slate-300 w-28 shrink-0">{hw.name}</span>
                  <span className="text-slate-400">{hw.formats}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Aligned vs misaligned */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-4">
          <div className="p-3 rounded bg-green-950/20 border border-green-800/30">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-green-400 text-sm font-bold">&#10003; Aligned</span>
            </div>
            <div className="flex gap-1 mb-2">
              <span className="px-1.5 py-0.5 rounded text-xs bg-green-900/30 text-green-300 border border-green-800/40">
                HW &#10003;
              </span>
              <span className="px-1.5 py-0.5 rounded text-xs bg-green-900/30 text-green-300 border border-green-800/40">
                SW &#10003;
              </span>
              <span className="px-1.5 py-0.5 rounded text-xs bg-green-900/30 text-green-300 border border-green-800/40">
                Model &#10003;
              </span>
            </div>
            <p className="text-xs text-green-300">Fast inference, small model</p>
          </div>
          <div className="p-3 rounded bg-red-950/20 border border-red-800/30">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-red-400 text-sm font-bold">&#10007; Misaligned</span>
            </div>
            <div className="flex gap-1 mb-2">
              <span className="px-1.5 py-0.5 rounded text-xs bg-red-900/30 text-red-300 border border-red-800/40">
                HW &#10007;
              </span>
              <span className="px-1.5 py-0.5 rounded text-xs bg-green-900/30 text-green-300 border border-green-800/40">
                SW &#10003;
              </span>
              <span className="px-1.5 py-0.5 rounded text-xs bg-green-900/30 text-green-300 border border-green-800/40">
                Model &#10003;
              </span>
            </div>
            <p className="text-xs text-red-300">
              4-bit model on FP16-only GPU = dequantize at runtime = SLOWER than FP16
            </p>
          </div>
        </div>
      </div>

      {/* Section 4: Microscaling (MX) Formats */}
      <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-cyan-400 mb-3">Microscaling (MX) Formats</h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-3">
          The OCP (Open Compute Project) Microscaling standard introduces block-level scaling for
          sub-8-bit formats &mdash; a more principled approach to aggressive quantization.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
          <div className="p-3 rounded bg-slate-800/50 border border-slate-700/30">
            <h4 className="text-xs font-semibold text-cyan-300 uppercase tracking-wide mb-2">
              How it works
            </h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              32 elements share one scale factor. Instead of each weight having its own exponent, a
              block of weights shares scaling metadata &mdash; reducing overhead while preserving
              dynamic range within each block.
            </p>
          </div>
          <div className="p-3 rounded bg-slate-800/50 border border-slate-700/30">
            <h4 className="text-xs font-semibold text-cyan-300 uppercase tracking-wide mb-2">
              Sub-8-bit formats
            </h4>
            <div className="flex flex-wrap gap-2">
              {mxFormats.map((fmt) => (
                <span
                  key={fmt}
                  className="px-2 py-0.5 rounded text-xs font-mono bg-cyan-900/30 text-cyan-300 border border-cyan-800/40"
                >
                  {fmt}
                </span>
              ))}
            </div>
            <p className="text-xs text-slate-400 mt-2">
              Finer-grained scaling = better accuracy at low precision compared to naive
              quantization.
            </p>
          </div>
        </div>
        <div className="p-3 rounded bg-cyan-950/20 border border-cyan-800/30">
          <p className="text-xs text-cyan-300 leading-relaxed">
            <strong>Industry adoption:</strong> AMD, Intel, NVIDIA, Qualcomm, and Microsoft are all
            backing the MX specification &mdash; signaling that sub-8-bit inference is the direction
            the industry is moving.
          </p>
        </div>
      </div>

      {/* Section 5: Quantization in Practice */}
      <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
        <h3 className="text-base font-semibold text-cyan-400 mb-3">Quantization in Practice</h3>
        <p className="text-sm text-slate-300 leading-relaxed mb-4">
          Where quantization fits in the post-training pipeline:
        </p>
        <div className="space-y-3">
          <div className="flex gap-3 items-start">
            <span className="text-sm font-bold text-cyan-400 w-6 shrink-0">1.</span>
            <div>
              <p className="text-sm font-semibold text-slate-200">Training: QLoRA</p>
              <p className="text-xs text-slate-400 mt-1">
                Uses a 4-bit quantized base model during fine-tuning. The LoRA adapter stays in FP16
                &mdash; only the frozen base model is quantized. This reduces GPU memory
                requirements by ~75% compared to full-precision training without sacrificing adapter
                quality.
              </p>
            </div>
          </div>
          <div className="flex gap-3 items-start">
            <span className="text-sm font-bold text-cyan-400 w-6 shrink-0">2.</span>
            <div>
              <p className="text-sm font-semibold text-slate-200">Deployment: ONNX Export</p>
              <p className="text-xs text-slate-400 mt-1">
                After training, the merged model (base + adapter) is exported via ONNX with INT8
                quantization. This produces a single optimized inference artifact that is 4&ndash;8x
                smaller than the FP32 original.
              </p>
            </div>
          </div>
        </div>
        <div className="mt-4 p-3 rounded bg-blue-950/20 border border-blue-800/30">
          <p className="text-xs text-blue-300 leading-relaxed">
            <strong>Key insight:</strong> Memory bandwidth, not compute, is usually the inference
            bottleneck. Quantization helps because smaller models move through memory faster &mdash;
            you&rsquo;re limited by how fast you can feed weights to the GPU cores, not by the math
            itself.
          </p>
        </div>
      </div>

      {/* Section 6: Connection to Infrastructure */}
      <div className="p-4 rounded-lg bg-gradient-to-r from-cyan-950/30 to-blue-950/30 border border-cyan-800/30">
        <h4 className="text-sm font-semibold text-cyan-400 mb-3">Connection to Infrastructure</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="space-y-2">
            <div className="flex gap-2 items-start">
              <span className="text-cyan-400 text-xs mt-0.5 shrink-0">&#9632;</span>
              <div>
                <p className="text-xs font-semibold text-slate-200">Storage impact</p>
                <p className="text-xs text-slate-400">
                  Quantized models are 2&ndash;8x smaller on disk, reducing storage costs and
                  checkpoint overhead.
                </p>
              </div>
            </div>
            <div className="flex gap-2 items-start">
              <span className="text-cyan-400 text-xs mt-0.5 shrink-0">&#9632;</span>
              <div>
                <p className="text-xs font-semibold text-slate-200">Network impact</p>
                <p className="text-xs text-slate-400">
                  Faster model distribution across clusters, faster container startup, and reduced
                  registry bandwidth.
                </p>
              </div>
            </div>
          </div>
          <div className="space-y-2">
            <div className="flex gap-2 items-start">
              <span className="text-cyan-400 text-xs mt-0.5 shrink-0">&#9632;</span>
              <div>
                <p className="text-xs font-semibold text-slate-200">Memory impact</p>
                <p className="text-xs text-slate-400">
                  More models per GPU, or larger batch sizes on the same hardware. Multi-tenant
                  serving becomes practical.
                </p>
              </div>
            </div>
            <div className="flex gap-2 items-start">
              <span className="text-cyan-400 text-xs mt-0.5 shrink-0">&#9632;</span>
              <div>
                <p className="text-xs font-semibold text-slate-200">Edge deployment</p>
                <p className="text-xs text-slate-400">
                  INT4 models can run on mobile devices, IoT hardware, and laptops &mdash; enabling
                  inference without cloud connectivity.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
