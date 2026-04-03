// Hardcoded data for the interactive Transformers Deep Dive.
// All attention matrices are hand-crafted to be pedagogically clear
// and demonstrate realistic attention patterns for a storage I/O classification task.
// Rows sum to ~1 (softmaxed). Upper triangle is 0 (causal mask for decoder-only models).

// --- Tokens ---
// BPE tokenization of: "Classify this I/O workload: IOPS: 45000 | Latency: 0.3ms | Block Size: 8K → Classification:"
// 18 tokens total — small enough for a readable heatmap

export const TOKENS = [
  { id: 0, text: 'Class', tokenId: 8834, type: 'task' },
  { id: 1, text: 'ify', tokenId: 1420, type: 'task' },
  { id: 2, text: ' this', tokenId: 436, type: 'task' },
  { id: 3, text: ' I/O', tokenId: 2178, type: 'context' },
  { id: 4, text: ' workload', tokenId: 30122, type: 'context' },
  { id: 5, text: ':', tokenId: 29, type: 'punct' },
  { id: 6, text: ' IOPS', tokenId: 22834, type: 'metric' },
  { id: 7, text: ':', tokenId: 29, type: 'punct' },
  { id: 8, text: ' 45', tokenId: 4153, type: 'value' },
  { id: 9, text: '000', tokenId: 830, type: 'value' },
  { id: 10, text: ' |', tokenId: 930, type: 'punct' },
  { id: 11, text: ' Latency', tokenId: 43234, type: 'metric' },
  { id: 12, text: ':', tokenId: 29, type: 'punct' },
  { id: 13, text: ' 0.3', tokenId: 470, type: 'value' },
  { id: 14, text: 'ms', tokenId: 1093, type: 'value' },
  { id: 15, text: ' |', tokenId: 930, type: 'punct' },
  { id: 16, text: ' Block', tokenId: 9414, type: 'metric' },
  { id: 17, text: ':', tokenId: 29, type: 'punct' },
]

export const TOKEN_COLORS = {
  task: { bg: 'bg-blue-900/40', border: 'border-blue-600/60', text: 'text-blue-300' },
  context: { bg: 'bg-slate-800/50', border: 'border-slate-600/50', text: 'text-slate-300' },
  metric: { bg: 'bg-violet-900/40', border: 'border-violet-600/60', text: 'text-violet-300' },
  value: { bg: 'bg-green-900/40', border: 'border-green-600/60', text: 'text-green-300' },
  punct: { bg: 'bg-slate-800/60', border: 'border-slate-700/40', text: 'text-slate-500' },
}

export const TOKEN_TYPE_LABELS = {
  task: 'Task instruction',
  context: 'Context',
  metric: 'Metric label',
  value: 'Numeric value',
  punct: 'Punctuation',
}

export function getTokenTypeLabels(t) {
  return {
    task: t('transformer.tokenType.task'),
    context: t('transformer.tokenType.context'),
    metric: t('transformer.tokenType.metric'),
    value: t('transformer.tokenType.value'),
    punct: t('transformer.tokenType.punct'),
  }
}

// --- Attention Heads ---
export const ATTENTION_HEADS = [
  {
    id: 'syntax',
    label: 'Position / Syntax',
    description:
      'This head tracks structural patterns — colons that separate labels from values, pipes that delimit metrics, and positional relationships between tokens.',
  },
  {
    id: 'value-link',
    label: 'Value Linking',
    description:
      'This head connects numeric values back to their metric labels — linking "45000" to "IOPS" and "0.3ms" to "Latency". It learns which numbers belong to which metrics.',
  },
  {
    id: 'cross-metric',
    label: 'Cross-Metric',
    description:
      'This head combines evidence across multiple metrics. When deciding the classification, it attends to IOPS, Latency, and Block Size simultaneously — looking at the full picture.',
  },
]

export function getAttentionHeads(t) {
  return [
    {
      id: 'syntax',
      label: t('transformer.headSyntax'),
      description: t('transformer.headSyntaxDesc'),
    },
    {
      id: 'value-link',
      label: t('transformer.headValueLink'),
      description: t('transformer.headValueLinkDesc'),
    },
    {
      id: 'cross-metric',
      label: t('transformer.headCrossMetric'),
      description: t('transformer.headCrossMetricDesc'),
    },
  ]
}

export const ATTENTION_LAYERS = [
  {
    id: 0,
    num: 4,
    label: 'Layer 4 (Early)',
    description: 'Surface patterns — local position, token adjacency',
  },
  {
    id: 1,
    num: 16,
    label: 'Layer 16 (Middle)',
    description: 'Semantic patterns — connecting meaning across tokens',
  },
  {
    id: 2,
    num: 28,
    label: 'Layer 28 (Late)',
    description: 'Task reasoning — combining evidence for classification',
  },
]

export function getAttentionLayers(t) {
  return [
    {
      id: 0,
      num: 4,
      label: t('transformer.layerEarly'),
      description: t('transformer.layerEarlyDesc'),
    },
    {
      id: 1,
      num: 16,
      label: t('transformer.layerMiddle'),
      description: t('transformer.layerMiddleDesc'),
    },
    {
      id: 2,
      num: 28,
      label: t('transformer.layerLate'),
      description: t('transformer.layerLateDesc'),
    },
  ]
}

// --- Attention Weight Matrices ---
// Key format: "layerIndex_headId"
// Each is an 18x18 matrix. Row i = attention FROM token i. Column j = attention TO token j.
// Upper triangle (j > i) must be 0 (causal mask). Each row sums to ~1.

// Helper: create a zeroed NxN matrix
const N = TOKENS.length
const zeros = () => Array.from({ length: N }, () => new Array(N).fill(0))

// Normalize rows to sum to 1 (only lower triangle + diagonal)
function normalizeRows(matrix) {
  return matrix.map((row, i) => {
    // Zero out upper triangle (causal mask)
    const masked = row.map((v, j) => (j > i ? 0 : v))
    const sum = masked.reduce((a, b) => a + b, 0)
    return sum > 0 ? masked.map((v) => v / sum) : masked
  })
}

// Layer 0 (Early) — Syntax head: strong local/positional attention
function buildEarlySyntax() {
  const m = zeros()
  for (let i = 0; i < N; i++) {
    // Strong self-attention + adjacent token attention
    m[i][i] = 0.4
    if (i > 0) m[i][i - 1] = 0.3
    if (i > 1) m[i][i - 2] = 0.1
    // Colons (5,7,12,17) get moderate attention from nearby tokens
    for (const c of [5, 7, 12, 17]) {
      if (c <= i && Math.abs(c - i) <= 3) m[i][c] += 0.15
    }
  }
  return normalizeRows(m)
}

// Layer 0 (Early) — Value-link head: mostly local, just starting to form associations
function buildEarlyValueLink() {
  const m = zeros()
  for (let i = 0; i < N; i++) {
    m[i][i] = 0.5
    if (i > 0) m[i][i - 1] = 0.25
    if (i > 1) m[i][i - 2] = 0.1
    // Faint attention from values to nearby metrics
    if (i === 8 || i === 9) m[i][6] += 0.08 // 45/000 → IOPS (weak)
    if (i === 13 || i === 14) m[i][11] += 0.06 // 0.3/ms → Latency (weak)
  }
  return normalizeRows(m)
}

// Layer 0 (Early) — Cross-metric: uniform local attention, no cross-metric yet
function buildEarlyCrossMetric() {
  const m = zeros()
  for (let i = 0; i < N; i++) {
    m[i][i] = 0.45
    if (i > 0) m[i][i - 1] = 0.25
    if (i > 1) m[i][i - 2] = 0.15
    if (i > 2) m[i][i - 3] = 0.05
  }
  return normalizeRows(m)
}

// Layer 1 (Middle) — Syntax head: attention to structure tokens across the sequence
function buildMiddleSyntax() {
  const m = zeros()
  for (let i = 0; i < N; i++) {
    m[i][i] = 0.2
    // Strong attention to all colons and pipes in the sequence
    for (const c of [5, 7, 10, 12, 15, 17]) {
      if (c <= i) m[i][c] += 0.12
    }
    // Moderate attention to task prefix
    if (i > 2) {
      m[i][0] += 0.05
      m[i][1] += 0.03
    }
  }
  return normalizeRows(m)
}

// Layer 1 (Middle) — Value-link: strong value→metric connections forming
function buildMiddleValueLink() {
  const m = zeros()
  for (let i = 0; i < N; i++) {
    m[i][i] = 0.15
    if (i > 0) m[i][i - 1] = 0.08
  }
  // IOPS values → IOPS metric
  m[8][6] = 0.45
  m[9][6] = 0.4
  m[9][8] = 0.25
  // Latency values → Latency metric
  m[13][11] = 0.42
  m[14][11] = 0.35
  m[14][13] = 0.2
  // Block → Block metric (self-referencing the label)
  m[16][16] = 0.3
  // Colon after metric → metric label
  m[7][6] = 0.4
  m[12][11] = 0.38
  m[17][16] = 0.35
  return normalizeRows(m)
}

// Layer 1 (Middle) — Cross-metric: beginning to look across metrics
function buildMiddleCrossMetric() {
  const m = zeros()
  for (let i = 0; i < N; i++) {
    m[i][i] = 0.15
  }
  // Later tokens attend to both IOPS and Latency values
  for (let i = 10; i < N; i++) {
    m[i][6] += 0.1 // IOPS label
    m[i][8] += 0.08 // 45
    m[i][9] += 0.08 // 000
  }
  for (let i = 15; i < N; i++) {
    m[i][11] += 0.08 // Latency label
    m[i][13] += 0.07 // 0.3
  }
  // Task prefix gets attention from later tokens
  for (let i = 5; i < N; i++) {
    m[i][0] += 0.06
    m[i][1] += 0.04
  }
  return normalizeRows(m)
}

// Layer 2 (Late) — Syntax head: clean structural pattern for output formatting
function buildLateSyntax() {
  const m = zeros()
  for (let i = 0; i < N; i++) {
    m[i][i] = 0.1
    // Heavy attention to task instruction prefix (formatting cue)
    m[i][0] += 0.12
    m[i][1] += 0.08
    // All colons (structural rhythm)
    for (const c of [5, 7, 12, 17]) {
      if (c <= i) m[i][c] += 0.08
    }
    // The final colon (17) gets extra attention — it's the output trigger
    if (17 <= i) m[i][17] += 0.12
  }
  return normalizeRows(m)
}

// Layer 2 (Late) — Value-link: very strong value→metric connections
function buildLateValueLink() {
  const m = zeros()
  for (let i = 0; i < N; i++) {
    m[i][i] = 0.1
  }
  // Rock-solid value→metric links
  m[8][6] = 0.55
  m[9][6] = 0.5
  m[9][8] = 0.15
  m[13][11] = 0.5
  m[14][11] = 0.4
  m[14][13] = 0.15
  // Later tokens maintain these links
  for (let i = 10; i < N; i++) {
    m[i][6] += 0.08
    m[i][8] += 0.06
  }
  for (let i = 15; i < N; i++) {
    m[i][11] += 0.07
    m[i][13] += 0.05
  }
  return normalizeRows(m)
}

// Layer 2 (Late) — Cross-metric: the "decision maker" — broad attention across all evidence
function buildLateCrossMetric() {
  const m = zeros()
  // Early tokens: mostly self-attention
  for (let i = 0; i < 6; i++) {
    m[i][i] = 0.6
    if (i > 0) m[i][i - 1] = 0.2
  }
  // Middle and late tokens: spread attention across all metrics and values
  for (let i = 6; i < N; i++) {
    m[i][0] = 0.03 // Classify
    m[i][6] = 0.12 // IOPS
    m[i][8] = 0.14 // 45
    m[i][9] = 0.1 // 000
    m[i][11] = 0.1 // Latency
    m[i][13] = 0.12 // 0.3
    m[i][14] = 0.06 // ms
    m[i][16] = 0.08 // Block
    m[i][i] = 0.08 // self
  }
  // The final token (17, ":") is the decision point — strongest cross-metric attention
  m[17][6] = 0.15 // IOPS
  m[17][8] = 0.16 // 45
  m[17][9] = 0.12 // 000
  m[17][11] = 0.12 // Latency
  m[17][13] = 0.14 // 0.3
  m[17][14] = 0.08 // ms
  m[17][16] = 0.1 // Block
  m[17][0] = 0.05 // Classify
  m[17][17] = 0.04 // self
  return normalizeRows(m)
}

export const ATTENTION_WEIGHTS = {
  '0_syntax': buildEarlySyntax(),
  '0_value-link': buildEarlyValueLink(),
  '0_cross-metric': buildEarlyCrossMetric(),
  '1_syntax': buildMiddleSyntax(),
  '1_value-link': buildMiddleValueLink(),
  '1_cross-metric': buildMiddleCrossMetric(),
  '2_syntax': buildLateSyntax(),
  '2_value-link': buildLateValueLink(),
  '2_cross-metric': buildLateCrossMetric(),
}

// --- Attention Insights ---
// Key: "headId_tokenIndex" — explanation shown when user selects that combination
export const ATTENTION_INSIGHTS = {
  // Syntax head insights
  syntax_5:
    'The first colon after "workload" — this head recognizes it as the boundary between the task instruction and the data fields.',
  syntax_7:
    'Colon after "IOPS" — the syntax head sees this as a label:value separator and uses it to track the structure of the input.',
  syntax_10:
    'The pipe character "|" — recognized as a field delimiter, helping the model understand where one metric ends and another begins.',
  syntax_17:
    'The final colon — this is the output trigger. The syntax head pays heavy attention here because it signals "now generate the classification."',

  // Value-link head insights
  'value-link_8':
    '"45" attends strongly back to "IOPS" — the value-link head has learned that this number belongs to the IOPS metric.',
  'value-link_9':
    '"000" attends to both "IOPS" and "45" — completing the number 45000 and linking it to its metric label.',
  'value-link_13':
    '"0.3" attends strongly to "Latency" — the model connects this sub-millisecond value to the latency metric, which is a key OLTP indicator.',
  'value-link_14':
    '"ms" reinforces the Latency connection — the unit confirms this is a time measurement, strengthening the link.',

  // Cross-metric head insights
  'cross-metric_6':
    'From "IOPS", attention spreads to other metric values — the cross-metric head is already comparing IOPS against latency and block size.',
  'cross-metric_8':
    'From "45", the model looks at latency (0.3ms) and block size — combining high IOPS + low latency = classic OLTP pattern.',
  'cross-metric_11':
    'From "Latency", attention reaches back to IOPS values and forward to block size — building a holistic view of the workload.',
  'cross-metric_17':
    'The final token is the decision point — it gathers evidence from ALL metrics simultaneously: high IOPS (45000), low latency (0.3ms), and block size, then passes this combined signal to generate "OLTP".',
}

export function getAttentionInsights(t) {
  return {
    syntax_5: t('transformer.insight.syntax_5'),
    syntax_7: t('transformer.insight.syntax_7'),
    syntax_10: t('transformer.insight.syntax_10'),
    syntax_17: t('transformer.insight.syntax_17'),
    'value-link_8': t('transformer.insight.valueLink_8'),
    'value-link_9': t('transformer.insight.valueLink_9'),
    'value-link_13': t('transformer.insight.valueLink_13'),
    'value-link_14': t('transformer.insight.valueLink_14'),
    'cross-metric_6': t('transformer.insight.crossMetric_6'),
    'cross-metric_8': t('transformer.insight.crossMetric_8'),
    'cross-metric_11': t('transformer.insight.crossMetric_11'),
    'cross-metric_17': t('transformer.insight.crossMetric_17'),
  }
}

// --- Forward Pass Step Definitions ---
export const FORWARD_PASS_STEPS = [
  {
    id: 'tokenize',
    title: 'Tokenization',
    subtitle: 'Text → Tokens → IDs',
    formula: '"Classify this I/O..." → ["Class", "ify", " this", ...] → [8834, 1420, 436, ...]',
    explanation:
      'The tokenizer splits text into subword pieces using Byte Pair Encoding (BPE). Common words stay whole; rare words get split. "Classify" becomes "Class" + "ify" because the model learned these subwords are more efficient than storing every possible word.',
  },
  {
    id: 'embed',
    title: 'Embedding',
    subtitle: 'Token IDs → 960-dimensional Vectors',
    formula: 'embedding = E[token_id]    (lookup in 49,152 × 960 table)',
    explanation:
      'Each token ID is used to look up a 960-dimensional vector from the embedding table. This vector is the token\'s "meaning" in the model\'s learned representation space. Similar tokens end up with similar vectors.',
  },
  {
    id: 'position',
    title: 'Positional Encoding',
    subtitle: 'Adding Position Information',
    formula: 'input = embedding + position_encoding[position]',
    explanation:
      'Transformers process all tokens in parallel, so they have no built-in sense of order. Positional encodings are added to tell the model where each token sits in the sequence. Without this, "IOPS: 45000" and "45000: IOPS" would look identical.',
  },
  {
    id: 'attention',
    title: 'Self-Attention',
    subtitle: 'Connecting Tokens to Each Other',
    formula: 'Attention(Q, K, V) = softmax(Q · K^T / √64) · V',
    explanation:
      'Each token is projected into Query (what am I looking for?), Key (what do I contain?), and Value (what information should I share?) vectors. The dot product of Q and K produces attention scores — how much each token should "pay attention to" every other token. LoRA adapters insert into Q and V projections.',
  },
  {
    id: 'ffn',
    title: 'Feed-Forward Network',
    subtitle: 'Transform Each Token Independently',
    formula: 'FFN(x) = GELU(x · W₁ + b₁) · W₂ + b₂    (960 → 3840 → 960)',
    explanation:
      'After attention mixes information between tokens, the feed-forward network processes each token independently. It first expands the representation to 4× width (3840 dims), applies a nonlinearity (GELU), then compresses back to 960. This is where the model "thinks" about each token\'s updated meaning.',
  },
  {
    id: 'predict',
    title: 'Next Token Prediction',
    subtitle: 'Hidden State → Vocabulary Probabilities',
    formula: 'probs = softmax(hidden · W_vocab)    (960 → 49,152 logits)',
    explanation:
      'The final hidden state is projected to a score for every token in the 49,152-word vocabulary. Softmax converts these scores to probabilities. The token with the highest probability is selected as the output. Fine-tuning changes which token gets the highest probability — from generic words to the correct classification label.',
  },
]

export function getForwardPassSteps(t) {
  const ids = ['tokenize', 'embed', 'position', 'attention', 'ffn', 'predict']
  return FORWARD_PASS_STEPS.map((step, i) => ({
    ...step,
    title: t(`forwardPass.${ids[i]}.title`),
    subtitle: t(`forwardPass.${ids[i]}.subtitle`),
    explanation: t(`forwardPass.${ids[i]}.explanation`),
  }))
}

// --- Embedding Visualization Data ---
// 8 representative dimensions (out of 960) for 5 tokens
// Values are normalized to [-1, 1] range for visualization
export const EMBEDDING_DIMS = {
  labels: ['dim 12', 'dim 87', 'dim 203', 'dim 341', 'dim 502', 'dim 619', 'dim 744', 'dim 890'],
  tokens: [
    { text: 'Class', values: [0.72, -0.31, 0.15, 0.88, -0.44, 0.21, -0.67, 0.53] },
    { text: 'IOPS', values: [-0.18, 0.84, -0.62, 0.33, 0.71, -0.29, 0.45, -0.11] },
    { text: ' 45', values: [-0.22, 0.79, -0.58, 0.28, 0.65, -0.33, 0.41, -0.15] },
    { text: ' Latency', values: [-0.15, 0.81, -0.55, 0.37, 0.68, -0.25, 0.48, -0.08] },
    { text: ' 0.3', values: [0.11, 0.42, -0.73, 0.19, 0.55, 0.38, -0.22, 0.64] },
  ],
}

// --- Output Probabilities ---
// Base model vs fine-tuned model, compatible with TokenProbChart props
export const OUTPUT_PROBS = {
  base: [
    { token: 'The', probability: 0.18 },
    { token: '\n', probability: 0.14 },
    { token: 'This', probability: 0.11 },
    { token: ' I', probability: 0.08 },
    { token: 'Based', probability: 0.07 },
    { token: ' OLTP', probability: 0.04 },
    { token: 'Class', probability: 0.03 },
    { token: ' It', probability: 0.03 },
    { token: 'OLTP', probability: 0.03 },
    { token: ' Work', probability: 0.02 },
  ],
  finetuned: [
    { token: 'OLTP', probability: 0.73 },
    { token: ' OLTP', probability: 0.12 },
    { token: 'Class', probability: 0.04 },
    { token: 'The', probability: 0.02 },
    { token: '\n', probability: 0.02 },
    { token: 'This', probability: 0.01 },
    { token: ' VDI', probability: 0.01 },
    { token: 'Based', probability: 0.01 },
    { token: ' I', probability: 0.01 },
    { token: ' Work', probability: 0.01 },
  ],
}
