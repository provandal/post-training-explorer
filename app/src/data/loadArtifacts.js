// =============================================================================
// Data loader for precomputed training artifacts
// =============================================================================
// Fetches precomputed_results.json (produced by scripts/export_artifacts.py)
// and exposes typed accessors for each stop component.
//
// When real training data is not yet available, components should use their
// inline fallback data. Check `isLoaded()` before calling accessors.
// =============================================================================

let artifacts = null
let loadPromise = null
let loadError = null

/**
 * Fetch and parse precomputed_results.json.
 * Safe to call multiple times — deduplicates the fetch.
 * @returns {Promise<boolean>} true if loaded successfully
 */
export async function loadArtifacts() {
  if (artifacts) return true
  if (loadPromise) return loadPromise

  loadPromise = (async () => {
    try {
      const res = await fetch(`${import.meta.env.BASE_URL}data/precomputed_results.json`)
      if (!res.ok) {
        loadError = `HTTP ${res.status}`
        return false
      }
      artifacts = await res.json()
      loadError = null
      return true
    } catch (err) {
      loadError = err.message
      console.warn('[loadArtifacts] precomputed_results.json not available:', err.message)
      return false
    } finally {
      loadPromise = null
    }
  })()

  return loadPromise
}

/** Whether artifacts have been loaded. */
export function isLoaded() {
  return artifacts !== null
}

/** Last load error message, or null. */
export function getLoadError() {
  return loadError
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

export function getMetadata() {
  return artifacts?.metadata ?? null
}

export function getCategories() {
  return artifacts?.metadata?.categories ?? [
    'OLTP Database', 'OLAP Analytics', 'AI ML Training',
    'Video Streaming', 'VDI Virtual Desktop', 'Backup Archive',
  ]
}

export function getModelVariants() {
  return artifacts?.metadata?.model_variants ?? ['base']
}

// ---------------------------------------------------------------------------
// Test prompts
// ---------------------------------------------------------------------------

/** Get all 20 test prompts. */
export function getTestPrompts() {
  return artifacts?.test_prompts ?? []
}

/** Get a single test prompt by id. */
export function getTestPrompt(id) {
  return artifacts?.test_prompts?.find(p => p.id === id) ?? null
}

// ---------------------------------------------------------------------------
// Model outputs
// ---------------------------------------------------------------------------

/**
 * Get the generated output for a specific model variant and test prompt.
 * @param {'base'|'sft'|'dpo'|'grpo'} variant
 * @param {number} promptId - test prompt id (0-19)
 * @returns {{ generated_text, generation_time_ms, num_tokens_generated, token_probabilities } | null}
 */
export function getModelOutput(variant, promptId) {
  const outputs = artifacts?.model_results?.[variant]?.outputs
  if (!outputs || promptId < 0 || promptId >= outputs.length) return null
  return outputs[promptId]
}

/**
 * Get all outputs for a model variant.
 * @param {'base'|'sft'|'dpo'|'grpo'} variant
 */
export function getAllOutputs(variant) {
  return artifacts?.model_results?.[variant]?.outputs ?? []
}

/**
 * Get accuracy summary for a model variant.
 * @param {'base'|'sft'|'dpo'|'grpo'} variant
 * @returns {{ accuracy, correct, total, avg_generation_time_ms } | null}
 */
export function getModelSummary(variant) {
  return artifacts?.model_results?.[variant]?.summary ?? null
}

/** Get accuracy summary for all variants. */
export function getAccuracySummary() {
  const variants = getModelVariants()
  const result = {}
  for (const v of variants) {
    result[v] = getModelSummary(v)
  }
  return result
}

// ---------------------------------------------------------------------------
// Token probabilities
// ---------------------------------------------------------------------------

/**
 * Get token probability snapshots for a specific output.
 * @param {'base'|'sft'|'dpo'|'grpo'} variant
 * @param {number} promptId
 * @returns {Array<{ position, actual_token, top_tokens, top_probs }>}
 */
export function getTokenProbs(variant, promptId) {
  const output = getModelOutput(variant, promptId)
  return output?.token_probabilities ?? []
}

/**
 * Get token probs formatted for TokenProbChart component.
 * Returns the first token position's probs in { token, probability } format.
 * @param {'base'|'sft'|'dpo'|'grpo'} variant
 * @param {number} promptId
 * @param {number} [position=0] - which token position to get (0 = first generated token)
 */
export function getTokenProbsForChart(variant, promptId, position = 0) {
  const snapshots = getTokenProbs(variant, promptId)
  const snapshot = snapshots.find(s => s.position === position) ?? snapshots[0]
  if (!snapshot) return []
  return snapshot.top_tokens.map((token, i) => ({
    token,
    probability: snapshot.top_probs[i],
  }))
}

// ---------------------------------------------------------------------------
// Training data (loss curves, etc.)
// ---------------------------------------------------------------------------

/**
 * Get SFT training loss curve.
 * @returns {Array<{ step, loss }>}
 */
export function getSFTLossCurve() {
  return artifacts?.training_data?.sft?.loss_curve ?? []
}

/**
 * Get DPO training loss curve.
 * @returns {Array<{ step, loss }>}
 */
export function getDPOLossCurve() {
  return artifacts?.training_data?.dpo?.loss_curve ?? []
}

/**
 * Get GRPO accuracy curve.
 * @returns {Array<{ step, accuracy }>}
 */
export function getGRPOAccuracyCurve() {
  return artifacts?.training_data?.grpo?.accuracy_curve ??
         artifacts?.grpo_group_statistics?.accuracy_curve ?? []
}

/**
 * Get GRPO reward curve.
 * @returns {Array<{ step, mean_reward }>}
 */
export function getGRPORewardCurve() {
  return artifacts?.training_data?.grpo?.reward_curve ??
         artifacts?.grpo_group_statistics?.reward_curve ?? []
}

/**
 * Get training time for a technique.
 * @param {'sft'|'dpo'|'grpo'} technique
 * @returns {number|null} seconds
 */
export function getTrainingTime(technique) {
  return artifacts?.training_data?.[technique]?.training_time_seconds ?? null
}

// ---------------------------------------------------------------------------
// LoRA weights (for heatmap visualization)
// ---------------------------------------------------------------------------

/**
 * Get LoRA weight visualization data.
 * @returns {{ layer, lora_A, lora_B, rank } | null}
 */
export function getLoRAWeights() {
  return artifacts?.lora_weight_visualization ?? null
}

// ---------------------------------------------------------------------------
// SFT before/after comparison
// ---------------------------------------------------------------------------

/**
 * Get SFT before/after comparison examples.
 * @returns {Array<{ input, base_output, sft_output, true_label, base_correct, sft_correct }>}
 */
export function getSFTBeforeAfter() {
  return artifacts?.sft_before_after ?? []
}

// ---------------------------------------------------------------------------
// DPO
// ---------------------------------------------------------------------------

/**
 * Get DPO probability shift data.
 * @returns {{ examples: Array<{ input, chosen_style, rejected_style, before, after }> } | null}
 */
export function getDPOProbabilityShifts() {
  return artifacts?.dpo_probability_shifts ?? null
}

/**
 * Get DPO preference pair examples.
 * @returns {Array}
 */
export function getDPOPreferenceExamples() {
  return artifacts?.dpo_preference_examples ?? []
}

// ---------------------------------------------------------------------------
// GRPO
// ---------------------------------------------------------------------------

/**
 * Get GRPO generation log examples.
 * @returns {{ examples: Array<{ input, true_label, generations }> } | null}
 */
export function getGRPOGenerationLogs() {
  return artifacts?.grpo_generation_logs ?? null
}

/**
 * Get GRPO group statistics.
 * @returns {{ accuracy_curve, reward_curve, training_time_seconds } | null}
 */
export function getGRPOGroupStatistics() {
  return artifacts?.grpo_group_statistics ?? null
}

// ---------------------------------------------------------------------------
// Resource utilization
// ---------------------------------------------------------------------------

/**
 * Get resource utilization info from training.
 * @returns {{ gpu_available, gpu_name, gpu_memory_total_gb, sft_*, dpo_*, grpo_* } | null}
 */
export function getResourceUtilization() {
  return artifacts?.resource_utilization ?? null
}

// ---------------------------------------------------------------------------
// Convenience: Format a test prompt's metrics into the compact display string
// ---------------------------------------------------------------------------

export function formatPromptMetrics(testPrompt) {
  if (!testPrompt?.metrics) return testPrompt?.prompt ?? ''
  const m = testPrompt.metrics
  return `IOPS: ${m.iops?.toLocaleString()} | Throughput: ${m.throughput_mb?.toLocaleString()} MB/s | Latency: ${m.avg_latency_us?.toLocaleString()} us | Read/Write: ${m.read_pct}/${m.write_pct} | Random/Sequential: ${m.random_pct}/${m.sequential_pct} | Block Size: ${m.block_size_kb} KB | Queue Depth: ${m.queue_depth}`
}

// ---------------------------------------------------------------------------
// Model Size Comparison (multi-model)
// ---------------------------------------------------------------------------

/**
 * Get the full model size comparison data.
 * @returns {{ models, accuracy_by_technique, training_time_minutes, gpu_memory_gb, head_to_head } | null}
 */
export function getModelSizeComparison() {
  return artifacts?.model_size_comparison ?? null
}

/**
 * Get list of model sizes that have comparison data.
 * @returns {string[]} e.g. ["360M", "1.7B"]
 */
export function getModelSizes() {
  return artifacts?.model_size_comparison?.models ?? []
}

/**
 * Get accuracy by technique and model size.
 * @param {string} technique - 'base', 'sft', 'dpo', 'grpo'
 * @param {string} size - '360M', '1.7B'
 * @returns {number|null}
 */
export function getAccuracyByTechniqueAndSize(technique, size) {
  return artifacts?.model_size_comparison?.accuracy_by_technique?.[technique]?.[size] ?? null
}

/**
 * Get training time by technique and model size.
 * @param {string} technique - 'sft', 'dpo', 'grpo'
 * @param {string} size - '360M', '1.7B'
 * @returns {number|null} minutes
 */
export function getTrainingTimeBySize(technique, size) {
  return artifacts?.model_size_comparison?.training_time_minutes?.[technique]?.[size] ?? null
}

/**
 * Get GPU memory usage by model size.
 * @param {string} size - '360M', '1.7B'
 * @returns {number|null} GB
 */
export function getGPUMemoryBySize(size) {
  return artifacts?.model_size_comparison?.gpu_memory_gb?.[size] ?? null
}

/**
 * Get head-to-head comparison outputs for a specific test prompt.
 * @param {number} promptIndex
 * @returns {{ prompt_id, prompt_snippet, true_label, results } | null}
 */
export function getHeadToHeadOutputs(promptIndex) {
  const h2h = artifacts?.model_size_comparison?.head_to_head
  if (!h2h || promptIndex < 0 || promptIndex >= h2h.length) return null
  return h2h[promptIndex]
}
