// Service for opt-in client-side inference using transformers.js
// Two models only: base (untrained) and grpo (final trained)
// Models hosted on HuggingFace Hub, cached in browser Cache API

const MODEL_CONFIGS = {
  base: {
    id: 'YOUR_HF_USERNAME/smollm2-360m-storage-io-base-onnx',  // Update after running convert_to_onnx.py
    label: 'Base Model (untrained)',
  },
  grpo: {
    id: 'YOUR_HF_USERNAME/smollm2-360m-storage-io-grpo-onnx',  // Update after running convert_to_onnx.py
    label: 'GRPO Fine-tuned',
  },
}

let loadedModels = {}
let loadedTokenizers = {}

export async function loadModel(variant, onProgress = () => {}) {
  // Check if already loaded
  if (loadedModels[variant]) return true

  try {
    // Dynamic import so app works without the package installed.
    // Using a variable to prevent Vite from statically resolving the import.
    const moduleName = '@huggingface/transformers'
    const { pipeline, env } = await import(/* @vite-ignore */ moduleName)

    // Configure for browser usage
    env.allowLocalModels = false
    env.useBrowserCache = true

    const config = MODEL_CONFIGS[variant]
    if (!config) throw new Error(`Unknown model variant: ${variant}`)

    onProgress({ status: 'loading', variant, progress: 0 })

    const generator = await pipeline('text-generation', config.id, {
      progress_callback: (data) => {
        if (data.status === 'progress') {
          onProgress({ status: 'downloading', variant, progress: data.progress || 0 })
        }
      },
      dtype: 'q4',  // INT4 quantization
    })

    loadedModels[variant] = generator
    onProgress({ status: 'ready', variant, progress: 100 })
    return true
  } catch (err) {
    onProgress({ status: 'error', variant, error: err.message })
    console.error(`Failed to load model ${variant}:`, err)
    return false
  }
}

export async function classify(variant, inputText) {
  const generator = loadedModels[variant]
  if (!generator) {
    throw new Error(`Model ${variant} not loaded. Call loadModel first.`)
  }

  const prompt = `Classify the following storage I/O workload based on these metrics:\n${inputText}\n\nProvide the workload classification and a brief reason.`

  const startTime = performance.now()

  const result = await generator(prompt, {
    max_new_tokens: 80,
    do_sample: false,
    temperature: 1.0,
  })

  const genTime = performance.now() - startTime
  const generatedText = result[0].generated_text.slice(prompt.length).trim()

  return {
    generated_text: generatedText,
    generation_time_ms: Math.round(genTime),
    variant,
  }
}

export function isModelLoaded(variant) {
  return !!loadedModels[variant]
}

export function isTransformersAvailable() {
  // Check if the package can be dynamically imported
  // This is a lightweight check that doesn't actually import the package
  return true // Will fail gracefully at import time if not installed
}

export function getModelConfig(variant) {
  return MODEL_CONFIGS[variant] || null
}

export function getAvailableVariants() {
  return Object.keys(MODEL_CONFIGS)
}
