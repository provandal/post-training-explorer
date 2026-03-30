import { useState, useCallback } from 'react'
import { loadModel, classify, isModelLoaded, getModelConfig } from '../services/inference'

const DEFAULT_INPUT =
  'IOPS: 45000 | Latency: 0.3ms | Block Size: 8K | Read/Write: 70/30 | Sequential: 15% | Queue Depth: 32'

const MODEL_VARIANTS = [
  { key: 'base', label: 'Base', color: 'red' },
  { key: 'grpo', label: 'GRPO', color: 'emerald' },
]

export default function LiveInferencePanel() {
  const [selectedModel, setSelectedModel] = useState('base')
  const [inputText, setInputText] = useState(DEFAULT_INPUT)
  const [output, setOutput] = useState(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [downloadProgress, setDownloadProgress] = useState({})
  const [error, setError] = useState(null)

  const selectedConfig = getModelConfig(selectedModel)

  const handleProgress = useCallback((data) => {
    setDownloadProgress((prev) => ({ ...prev, [data.variant]: data }))
    if (data.status === 'error') {
      setError(data.error)
    }
  }, [])

  const handleDownload = async () => {
    setError(null)
    const success = await loadModel(selectedModel, handleProgress)
    if (!success) {
      setError(
        downloadProgress[selectedModel]?.error ||
          'Failed to download model. Make sure @huggingface/transformers is installed (npm install @huggingface/transformers).',
      )
    }
  }

  const handleGenerate = async () => {
    if (!isModelLoaded(selectedModel)) {
      setError('Model not downloaded yet. Click "Download Model" first.')
      return
    }

    setError(null)
    setIsGenerating(true)
    setOutput(null)

    try {
      const result = await classify(selectedModel, inputText)
      setOutput(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsGenerating(false)
    }
  }

  const progress = downloadProgress[selectedModel]
  const isDownloading =
    progress && (progress.status === 'loading' || progress.status === 'downloading')
  const isReady = isModelLoaded(selectedModel)
  const progressPercent = progress?.status === 'downloading' ? Math.round(progress.progress) : 0

  const variantMeta = MODEL_VARIANTS.find((v) => v.key === selectedModel)
  const accentColor = variantMeta?.color || 'cyan'

  // Tailwind color maps for dynamic accent
  const accentMap = {
    red: {
      text: 'text-red-400',
      bg: 'bg-red-600',
      bgHover: 'hover:bg-red-500',
      border: 'border-red-800/50',
      bgFaint: 'bg-red-950/20',
      ring: 'ring-red-500',
    },
    emerald: {
      text: 'text-emerald-400',
      bg: 'bg-emerald-600',
      bgHover: 'hover:bg-emerald-500',
      border: 'border-emerald-800/50',
      bgFaint: 'bg-emerald-950/20',
      ring: 'ring-emerald-500',
    },
  }
  const accent = accentMap[accentColor] || accentMap.emerald

  return (
    <div className="p-5 rounded-lg bg-slate-800/30 border border-slate-700/50">
      <h3 className="text-base font-semibold text-cyan-400 mb-1">Live Inference (Optional)</h3>
      <p className="text-xs text-slate-400 mb-4">
        Download a model to your browser and run inference locally. ~180MB per model, cached for
        offline use.
      </p>

      {/* ---- Model selector toggle ---- */}
      <div className="mb-4">
        <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
          Select Model
        </label>
        <div className="flex gap-1 bg-slate-800 rounded-lg p-1 w-fit">
          {MODEL_VARIANTS.map((v) => {
            const isActive = selectedModel === v.key
            const activeClass = v.color === 'red' ? 'bg-red-600' : 'bg-emerald-600'
            return (
              <button
                key={v.key}
                onClick={() => {
                  setSelectedModel(v.key)
                  setOutput(null)
                  setError(null)
                }}
                className={`px-4 py-2 text-sm rounded-md transition-colors ${
                  isActive
                    ? `${activeClass} text-white font-semibold`
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {v.label}
                {isModelLoaded(v.key) && <span className="ml-1.5 text-xs opacity-70">Ready</span>}
              </button>
            )
          })}
        </div>
        {selectedConfig && (
          <p className="text-xs text-slate-500 mt-1.5">
            {selectedConfig.label}{' '}
            {selectedConfig.id.includes('YOUR_HF_USERNAME') ? (
              <span className="text-amber-500">(not configured — see setup steps)</span>
            ) : (
              <span className="text-slate-600">({selectedConfig.id})</span>
            )}
          </p>
        )}
      </div>

      {/* ---- Download button + progress ---- */}
      <div className="mb-4">
        {!isReady ? (
          <div>
            <button
              onClick={handleDownload}
              disabled={isDownloading}
              className={`px-4 py-2 text-sm rounded-md font-medium transition-colors ${
                isDownloading
                  ? 'bg-slate-700 text-slate-400 cursor-wait'
                  : `${accent.bg} ${accent.bgHover} text-white`
              }`}
            >
              {isDownloading ? `Downloading... ${progressPercent}%` : 'Download Model (~180MB)'}
            </button>

            {/* Progress bar */}
            {isDownloading && (
              <div className="mt-2 h-2 bg-slate-700 rounded-full overflow-hidden">
                <div
                  className={`h-full ${accent.bg} rounded-full transition-all duration-300`}
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
            )}

            {progress?.status === 'loading' && (
              <p className="text-xs text-slate-500 mt-1.5">Initializing model pipeline...</p>
            )}
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-emerald-950/30 border border-emerald-800/30 text-xs font-semibold text-emerald-400">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              Model ready
            </span>
          </div>
        )}
      </div>

      {/* ---- Input textarea ---- */}
      <div className="mb-4">
        <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
          I/O Pattern Input
        </label>
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          rows={2}
          className="w-full bg-slate-800 border border-slate-600 rounded-lg px-3 py-2.5 text-sm text-slate-200 font-mono placeholder-slate-600 focus:outline-none focus:border-cyan-600 focus:ring-1 focus:ring-cyan-600 resize-none"
          placeholder="IOPS: ... | Latency: ... | Block Size: ... | Read/Write: ... | Sequential: ... | Queue Depth: ..."
        />
        <p className="text-xs text-slate-600 mt-1">
          Format: IOPS | Latency | Block Size | Read/Write ratio | Sequential % | Queue Depth
        </p>
      </div>

      {/* ---- Generate button ---- */}
      <div className="mb-4">
        <button
          onClick={handleGenerate}
          disabled={!isReady || isGenerating || !inputText.trim()}
          className={`px-5 py-2.5 text-sm rounded-md font-medium transition-colors ${
            !isReady || isGenerating || !inputText.trim()
              ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
              : 'bg-cyan-600 hover:bg-cyan-500 text-white'
          }`}
        >
          {isGenerating ? 'Generating...' : 'Classify'}
        </button>
      </div>

      {/* ---- Output display ---- */}
      {output && (
        <div className={`border rounded-lg p-4 ${accent.border} ${accent.bgFaint}`}>
          <div className="flex items-center justify-between mb-2">
            <span className={`text-xs font-semibold uppercase tracking-wide ${accent.text}`}>
              {selectedModel === 'base' ? 'Base Model Output' : 'GRPO Model Output'}
            </span>
            <span className="text-xs text-slate-500">{output.generation_time_ms}ms</span>
          </div>
          <pre className="text-sm text-slate-200 whitespace-pre-wrap font-mono leading-relaxed">
            {output.generated_text}
          </pre>
        </div>
      )}

      {/* ---- Error display ---- */}
      {error && (
        <div className="mt-3 p-3 rounded-lg bg-red-950/20 border border-red-800/30">
          {error.includes('SETUP_NEEDED') ? (
            <>
              <p className="text-sm font-semibold text-amber-400 mb-2">Model Not Yet Published</p>
              <p className="text-xs text-slate-300 mb-2">
                The ONNX models haven't been uploaded to HuggingFace Hub yet. To enable live
                inference:
              </p>
              <ol className="text-xs text-slate-400 list-decimal list-inside space-y-1">
                <li>
                  Run the training pipeline in Colab (or{' '}
                  <code className="text-slate-300 bg-slate-800 px-1 py-0.5 rounded">scripts/</code>)
                </li>
                <li>
                  Export to ONNX with{' '}
                  <code className="text-slate-300 bg-slate-800 px-1 py-0.5 rounded">
                    convert_to_onnx.py
                  </code>
                </li>
                <li>Push the ONNX model to a HuggingFace Hub repo</li>
                <li>
                  Update the model IDs in{' '}
                  <code className="text-slate-300 bg-slate-800 px-1 py-0.5 rounded">
                    app/src/services/inference.js
                  </code>
                </li>
              </ol>
            </>
          ) : (
            <>
              <p className="text-xs text-red-400">
                <strong>Error:</strong> {error}
              </p>
              {error.includes('@huggingface/transformers') && (
                <p className="text-xs text-slate-500 mt-1">
                  Install with:{' '}
                  <code className="text-slate-400 bg-slate-800 px-1.5 py-0.5 rounded">
                    npm install @huggingface/transformers
                  </code>
                </p>
              )}
            </>
          )}
        </div>
      )}

      {/* ---- Not-installed hint (always visible at bottom) ---- */}
      <p className="text-xs text-slate-600 mt-4 leading-relaxed">
        Requires{' '}
        <code className="text-slate-500 bg-slate-800 px-1 py-0.5 rounded text-xs">
          @huggingface/transformers
        </code>{' '}
        (not bundled by default). Models run entirely in-browser via WebAssembly &mdash; no data
        leaves your machine.
      </p>
    </div>
  )
}
