// =============================================================================
// Pre-computed data for Post-Training Explorer demo
// Task: Storage I/O workload classification into 6 categories
// Categories: OLTP Database, OLAP Analytics, AI ML Training,
//             Video Streaming, VDI Virtual Desktop, Backup Archive
//
// This file will be replaced with real training outputs later.
// =============================================================================

// ---------------------------------------------------------------------------
// Helper: generate a realistic SFT loss curve (~525 steps)
// Starts ~2.8, drops steeply, then tapers to ~0.3 with noise
// ---------------------------------------------------------------------------
function generateSFTLossCurve() {
  const points = [];
  const totalSteps = 525;
  for (let i = 0; i <= totalSteps; i += 1) {
    // Exponential-ish decay with a bit of noise
    const progress = i / totalSteps;
    const base = 2.8 * Math.exp(-4.5 * progress) + 0.28;
    // Add Gaussian-ish noise that shrinks as loss decreases
    const noise = (Math.sin(i * 7.3) * 0.04 + Math.cos(i * 13.1) * 0.03) * (1 - progress * 0.6);
    points.push({ step: i, loss: parseFloat(Math.max(0.15, base + noise).toFixed(4)) });
  }
  return points;
}

// ---------------------------------------------------------------------------
// Helper: generate a DPO loss curve (~100 steps)
// DPO loss starts near 0.69 (ln2) and drops to ~0.35
// ---------------------------------------------------------------------------
function generateDPOLossCurve() {
  const points = [];
  const totalSteps = 100;
  for (let i = 0; i <= totalSteps; i += 1) {
    const progress = i / totalSteps;
    const base = 0.693 * Math.exp(-2.0 * progress) + 0.33;
    const noise = Math.sin(i * 5.7) * 0.015 + Math.cos(i * 11.3) * 0.01;
    points.push({ step: i, loss: parseFloat(Math.max(0.28, base + noise).toFixed(4)) });
  }
  return points;
}

// ---------------------------------------------------------------------------
// Helper: generate a GRPO reward/loss curve (~200 steps)
// Reward improves from ~0.35 to ~0.82
// ---------------------------------------------------------------------------
function generateGRPOLossCurve() {
  const points = [];
  const totalSteps = 200;
  for (let i = 0; i <= totalSteps; i += 1) {
    const progress = i / totalSteps;
    // Policy gradient loss starts high and decreases
    const base = 1.2 * Math.exp(-3.0 * progress) + 0.25;
    const noise = Math.sin(i * 4.9) * 0.03 + Math.cos(i * 9.7) * 0.02;
    points.push({ step: i, loss: parseFloat(Math.max(0.18, base + noise).toFixed(4)) });
  }
  return points;
}

// ---------------------------------------------------------------------------
// Helper: generate a small random matrix (LoRA weights)
// ---------------------------------------------------------------------------
function generateMatrix(rows, cols, scale = 0.02) {
  const m = [];
  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < cols; c++) {
      // Pseudorandom but deterministic using trig
      const v = Math.sin(r * 31.7 + c * 17.3) * Math.cos(r * 11.1 + c * 23.9) * scale;
      row.push(parseFloat(v.toFixed(6)));
    }
    m.push(row);
  }
  return m;
}

function generateDeltaMatrix(matA, matB) {
  // matA is 16x32, matB is 32x16 -> result is 16x16
  const rows = matA.length;
  const inner = matA[0].length;
  const cols = matB[0].length;
  const result = [];
  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < cols; c++) {
      let sum = 0;
      for (let k = 0; k < inner; k++) {
        sum += matA[r][k] * matB[k][c];
      }
      row.push(parseFloat(sum.toFixed(6)));
    }
    result.push(row);
  }
  return result;
}

// Pre-generate the matrices
const loraMatrixA = generateMatrix(16, 32, 0.02);
const loraMatrixB = generateMatrix(32, 16, 0.02);
const loraDelta = generateDeltaMatrix(loraMatrixA, loraMatrixB);

// =============================================================================
// MAIN DATA OBJECT
// =============================================================================

export default {

  // ===========================================================================
  // 10 Example I/O patterns covering all 6 categories
  // ===========================================================================
  examples: [
    {
      id: 1,
      input: "IOPS: 45000 | Latency: 0.3ms | Block Size: 8K | Read/Write: 70/30 | Sequential: 15% | Queue Depth: 32",
      correctLabel: "OLTP Database",
      difficulty: "easy"
    },
    {
      id: 2,
      input: "IOPS: 12000 | Latency: 4.5ms | Block Size: 128K | Read/Write: 92/8 | Sequential: 78% | Queue Depth: 8",
      correctLabel: "OLAP Analytics",
      difficulty: "easy"
    },
    {
      id: 3,
      input: "IOPS: 8500 | Latency: 1.2ms | Block Size: 256K | Read/Write: 85/15 | Sequential: 91% | Queue Depth: 16",
      correctLabel: "AI ML Training",
      difficulty: "easy"
    },
    {
      id: 4,
      input: "IOPS: 2200 | Latency: 8.0ms | Block Size: 1M | Read/Write: 99/1 | Sequential: 98% | Queue Depth: 4",
      correctLabel: "Video Streaming",
      difficulty: "easy"
    },
    {
      id: 5,
      input: "IOPS: 18000 | Latency: 1.5ms | Block Size: 16K | Read/Write: 60/40 | Sequential: 25% | Queue Depth: 24",
      correctLabel: "VDI Virtual Desktop",
      difficulty: "easy"
    },
    {
      id: 6,
      input: "IOPS: 3500 | Latency: 12ms | Block Size: 512K | Read/Write: 5/95 | Sequential: 95% | Queue Depth: 2",
      correctLabel: "Backup Archive",
      difficulty: "easy"
    },
    {
      id: 7,
      input: "IOPS: 5200 | Latency: 3.8ms | Block Size: 256K | Read/Write: 88/12 | Sequential: 82% | Queue Depth: 12",
      correctLabel: "OLAP Analytics",
      difficulty: "medium"
    },
    {
      id: 8,
      input: "IOPS: 32000 | Latency: 0.8ms | Block Size: 8K | Read/Write: 55/45 | Sequential: 22% | Queue Depth: 48",
      correctLabel: "VDI Virtual Desktop",
      difficulty: "hard"
    },
    {
      id: 9,
      input: "IOPS: 6800 | Latency: 2.1ms | Block Size: 64K | Read/Write: 80/20 | Sequential: 65% | Queue Depth: 16",
      correctLabel: "AI ML Training",
      difficulty: "hard"
    },
    {
      id: 10,
      input: "IOPS: 28000 | Latency: 0.5ms | Block Size: 4K | Read/Write: 65/35 | Sequential: 18% | Queue Depth: 64",
      correctLabel: "OLTP Database",
      difficulty: "hard"
    }
  ],

  // ===========================================================================
  // Prompt Engineering results
  // ===========================================================================
  promptEngineering: {
    // Basic zero-shot prompting
    basic: {
      1: {
        prompt: "Classify the following storage I/O pattern into one of these categories: OLTP Database, OLAP Analytics, AI ML Training, Video Streaming, VDI Virtual Desktop, Backup Archive.\n\nPattern: IOPS: 45000 | Latency: 0.3ms | Block Size: 8K | Read/Write: 70/30 | Sequential: 15% | Queue Depth: 32",
        response: "This storage I/O pattern shows high IOPS with small block sizes. It could be a database workload or perhaps a virtual desktop environment. The relatively low latency suggests fast storage. I would guess this is likely related to some kind of database or transactional system, but it's hard to say definitively without more context.",
        correct: false
      },
      2: {
        prompt: "Classify the following storage I/O pattern into one of these categories: OLTP Database, OLAP Analytics, AI ML Training, Video Streaming, VDI Virtual Desktop, Backup Archive.\n\nPattern: IOPS: 12000 | Latency: 4.5ms | Block Size: 128K | Read/Write: 92/8 | Sequential: 78% | Queue Depth: 8",
        response: "Looking at this pattern, the large block size of 128K and high sequential percentage of 78% along with the read-heavy nature suggests this is some form of data processing workload. The moderate IOPS count combined with the sequential access could indicate analytics or possibly streaming. I think this could be OLAP Analytics or Video Streaming.",
        correct: false
      },
      3: {
        prompt: "Classify the following storage I/O pattern into one of these categories: OLTP Database, OLAP Analytics, AI ML Training, Video Streaming, VDI Virtual Desktop, Backup Archive.\n\nPattern: IOPS: 8500 | Latency: 1.2ms | Block Size: 256K | Read/Write: 85/15 | Sequential: 91% | Queue Depth: 16",
        response: "This pattern has large block sizes and very high sequential access which reminds me of a streaming or backup workload. The read-heavy nature with 85% reads could indicate data being loaded from storage. It might be Video Streaming or some kind of batch processing workload.",
        correct: false
      },
      4: {
        prompt: "Classify the following storage I/O pattern into one of these categories: OLTP Database, OLAP Analytics, AI ML Training, Video Streaming, VDI Virtual Desktop, Backup Archive.\n\nPattern: IOPS: 2200 | Latency: 8.0ms | Block Size: 1M | Read/Write: 99/1 | Sequential: 98% | Queue Depth: 4",
        response: "Very large block size of 1MB with nearly all reads and almost 100% sequential access. Low IOPS. This is clearly a streaming-type workload, most likely Video Streaming.",
        correct: true
      },
      5: {
        prompt: "Classify the following storage I/O pattern into one of these categories: OLTP Database, OLAP Analytics, AI ML Training, Video Streaming, VDI Virtual Desktop, Backup Archive.\n\nPattern: IOPS: 18000 | Latency: 1.5ms | Block Size: 16K | Read/Write: 60/40 | Sequential: 25% | Queue Depth: 24",
        response: "This shows a mixed read/write pattern with moderate IOPS and small-to-medium block sizes. The random nature (only 25% sequential) and balanced read/write could suggest a desktop-type workload or possibly a database. I think this might be some kind of mixed-use system, perhaps a database or virtualized environment.",
        correct: false
      },
      6: {
        prompt: "Classify the following storage I/O pattern into one of these categories: OLTP Database, OLAP Analytics, AI ML Training, Video Streaming, VDI Virtual Desktop, Backup Archive.\n\nPattern: IOPS: 3500 | Latency: 12ms | Block Size: 512K | Read/Write: 5/95 | Sequential: 95% | Queue Depth: 2",
        response: "This is overwhelmingly write-heavy with large blocks and very sequential access. Low IOPS and low queue depth. This is a Backup Archive workload.",
        correct: true
      },
      7: {
        prompt: "Classify the following storage I/O pattern into one of these categories: OLTP Database, OLAP Analytics, AI ML Training, Video Streaming, VDI Virtual Desktop, Backup Archive.\n\nPattern: IOPS: 5200 | Latency: 3.8ms | Block Size: 256K | Read/Write: 88/12 | Sequential: 82% | Queue Depth: 12",
        response: "Large block sequential reads with moderate IOPS. Could be analytics scanning large datasets or possibly streaming. The block size and access pattern suggest some form of data processing. I'd say either OLAP Analytics or Video Streaming, leaning towards Video Streaming given the sequential reads.",
        correct: false
      },
      8: {
        prompt: "Classify the following storage I/O pattern into one of these categories: OLTP Database, OLAP Analytics, AI ML Training, Video Streaming, VDI Virtual Desktop, Backup Archive.\n\nPattern: IOPS: 32000 | Latency: 0.8ms | Block Size: 8K | Read/Write: 55/45 | Sequential: 22% | Queue Depth: 48",
        response: "Very high IOPS with small blocks, low latency, and random access. The balanced read/write ratio is interesting. This looks like it could be OLTP Database given the high IOPS and small block random I/O, or possibly a highly active database system.",
        correct: false
      },
      9: {
        prompt: "Classify the following storage I/O pattern into one of these categories: OLTP Database, OLAP Analytics, AI ML Training, Video Streaming, VDI Virtual Desktop, Backup Archive.\n\nPattern: IOPS: 6800 | Latency: 2.1ms | Block Size: 64K | Read/Write: 80/20 | Sequential: 65% | Queue Depth: 16",
        response: "Moderate IOPS with medium block sizes and mixed sequential/random access. This is somewhat ambiguous -- the block size suggests something between transactional and analytical. Could be OLAP Analytics or possibly VDI. I'll say OLAP Analytics.",
        correct: false
      },
      10: {
        prompt: "Classify the following storage I/O pattern into one of these categories: OLTP Database, OLAP Analytics, AI ML Training, Video Streaming, VDI Virtual Desktop, Backup Archive.\n\nPattern: IOPS: 28000 | Latency: 0.5ms | Block Size: 4K | Read/Write: 65/35 | Sequential: 18% | Queue Depth: 64",
        response: "Very high IOPS with tiny blocks, extremely low latency, and highly random access. This is consistent with a transactional workload. Could be OLTP Database or perhaps a busy VDI environment. The high queue depth and very small blocks lean towards OLTP Database.",
        correct: true
      }
    },

    // Few-shot prompting (with examples in the prompt)
    fewShot: {
      1: {
        prompt: "Classify storage I/O patterns into one of: OLTP Database, OLAP Analytics, AI ML Training, Video Streaming, VDI Virtual Desktop, Backup Archive.\n\nExample 1:\nPattern: IOPS: 52000 | Latency: 0.2ms | Block Size: 4K | Read/Write: 80/20 | Sequential: 10% | Queue Depth: 64\nClassification: OLTP Database\n\nExample 2:\nPattern: IOPS: 1800 | Latency: 15ms | Block Size: 1M | Read/Write: 2/98 | Sequential: 99% | Queue Depth: 1\nClassification: Backup Archive\n\nExample 3:\nPattern: IOPS: 9500 | Latency: 3.0ms | Block Size: 256K | Read/Write: 90/10 | Sequential: 85% | Queue Depth: 8\nClassification: OLAP Analytics\n\nNow classify:\nPattern: IOPS: 45000 | Latency: 0.3ms | Block Size: 8K | Read/Write: 70/30 | Sequential: 15% | Queue Depth: 32",
        response: "Classification: OLTP Database",
        correct: true,
        fewShotExamples: [
          { input: "IOPS: 52000 | Latency: 0.2ms | Block Size: 4K | Read/Write: 80/20 | Sequential: 10% | Queue Depth: 64", label: "OLTP Database" },
          { input: "IOPS: 1800 | Latency: 15ms | Block Size: 1M | Read/Write: 2/98 | Sequential: 99% | Queue Depth: 1", label: "Backup Archive" },
          { input: "IOPS: 9500 | Latency: 3.0ms | Block Size: 256K | Read/Write: 90/10 | Sequential: 85% | Queue Depth: 8", label: "OLAP Analytics" }
        ]
      },
      2: {
        prompt: "Classify storage I/O patterns into one of: OLTP Database, OLAP Analytics, AI ML Training, Video Streaming, VDI Virtual Desktop, Backup Archive.\n\n[3 few-shot examples]\n\nNow classify:\nPattern: IOPS: 12000 | Latency: 4.5ms | Block Size: 128K | Read/Write: 92/8 | Sequential: 78% | Queue Depth: 8",
        response: "Classification: OLAP Analytics",
        correct: true,
        fewShotExamples: [
          { input: "IOPS: 52000 | Latency: 0.2ms | Block Size: 4K | Read/Write: 80/20 | Sequential: 10% | Queue Depth: 64", label: "OLTP Database" },
          { input: "IOPS: 1800 | Latency: 15ms | Block Size: 1M | Read/Write: 2/98 | Sequential: 99% | Queue Depth: 1", label: "Backup Archive" },
          { input: "IOPS: 9500 | Latency: 3.0ms | Block Size: 256K | Read/Write: 90/10 | Sequential: 85% | Queue Depth: 8", label: "OLAP Analytics" }
        ]
      },
      3: {
        prompt: "[few-shot prompt]\n\nPattern: IOPS: 8500 | Latency: 1.2ms | Block Size: 256K | Read/Write: 85/15 | Sequential: 91% | Queue Depth: 16",
        response: "Classification: OLAP Analytics",
        correct: false,
        fewShotExamples: [
          { input: "IOPS: 52000 | Latency: 0.2ms | Block Size: 4K | Read/Write: 80/20 | Sequential: 10% | Queue Depth: 64", label: "OLTP Database" },
          { input: "IOPS: 1800 | Latency: 15ms | Block Size: 1M | Read/Write: 2/98 | Sequential: 99% | Queue Depth: 1", label: "Backup Archive" },
          { input: "IOPS: 9500 | Latency: 3.0ms | Block Size: 256K | Read/Write: 90/10 | Sequential: 85% | Queue Depth: 8", label: "OLAP Analytics" }
        ]
      },
      4: {
        prompt: "[few-shot prompt]\n\nPattern: IOPS: 2200 | Latency: 8.0ms | Block Size: 1M | Read/Write: 99/1 | Sequential: 98% | Queue Depth: 4",
        response: "Classification: Video Streaming",
        correct: true,
        fewShotExamples: [
          { input: "IOPS: 52000 | Latency: 0.2ms | Block Size: 4K | Read/Write: 80/20 | Sequential: 10% | Queue Depth: 64", label: "OLTP Database" },
          { input: "IOPS: 1800 | Latency: 15ms | Block Size: 1M | Read/Write: 2/98 | Sequential: 99% | Queue Depth: 1", label: "Backup Archive" },
          { input: "IOPS: 9500 | Latency: 3.0ms | Block Size: 256K | Read/Write: 90/10 | Sequential: 85% | Queue Depth: 8", label: "OLAP Analytics" }
        ]
      },
      5: {
        prompt: "[few-shot prompt]\n\nPattern: IOPS: 18000 | Latency: 1.5ms | Block Size: 16K | Read/Write: 60/40 | Sequential: 25% | Queue Depth: 24",
        response: "Classification: OLTP Database",
        correct: false,
        fewShotExamples: [
          { input: "IOPS: 52000 | Latency: 0.2ms | Block Size: 4K | Read/Write: 80/20 | Sequential: 10% | Queue Depth: 64", label: "OLTP Database" },
          { input: "IOPS: 1800 | Latency: 15ms | Block Size: 1M | Read/Write: 2/98 | Sequential: 99% | Queue Depth: 1", label: "Backup Archive" },
          { input: "IOPS: 9500 | Latency: 3.0ms | Block Size: 256K | Read/Write: 90/10 | Sequential: 85% | Queue Depth: 8", label: "OLAP Analytics" }
        ]
      },
      6: {
        prompt: "[few-shot prompt]\n\nPattern: IOPS: 3500 | Latency: 12ms | Block Size: 512K | Read/Write: 5/95 | Sequential: 95% | Queue Depth: 2",
        response: "Classification: Backup Archive",
        correct: true,
        fewShotExamples: [
          { input: "IOPS: 52000 | Latency: 0.2ms | Block Size: 4K | Read/Write: 80/20 | Sequential: 10% | Queue Depth: 64", label: "OLTP Database" },
          { input: "IOPS: 1800 | Latency: 15ms | Block Size: 1M | Read/Write: 2/98 | Sequential: 99% | Queue Depth: 1", label: "Backup Archive" },
          { input: "IOPS: 9500 | Latency: 3.0ms | Block Size: 256K | Read/Write: 90/10 | Sequential: 85% | Queue Depth: 8", label: "OLAP Analytics" }
        ]
      },
      7: {
        prompt: "[few-shot prompt]\n\nPattern: IOPS: 5200 | Latency: 3.8ms | Block Size: 256K | Read/Write: 88/12 | Sequential: 82% | Queue Depth: 12",
        response: "Classification: OLAP Analytics",
        correct: true,
        fewShotExamples: [
          { input: "IOPS: 52000 | Latency: 0.2ms | Block Size: 4K | Read/Write: 80/20 | Sequential: 10% | Queue Depth: 64", label: "OLTP Database" },
          { input: "IOPS: 1800 | Latency: 15ms | Block Size: 1M | Read/Write: 2/98 | Sequential: 99% | Queue Depth: 1", label: "Backup Archive" },
          { input: "IOPS: 9500 | Latency: 3.0ms | Block Size: 256K | Read/Write: 90/10 | Sequential: 85% | Queue Depth: 8", label: "OLAP Analytics" }
        ]
      },
      8: {
        prompt: "[few-shot prompt]\n\nPattern: IOPS: 32000 | Latency: 0.8ms | Block Size: 8K | Read/Write: 55/45 | Sequential: 22% | Queue Depth: 48",
        response: "Classification: OLTP Database",
        correct: false,
        fewShotExamples: [
          { input: "IOPS: 52000 | Latency: 0.2ms | Block Size: 4K | Read/Write: 80/20 | Sequential: 10% | Queue Depth: 64", label: "OLTP Database" },
          { input: "IOPS: 1800 | Latency: 15ms | Block Size: 1M | Read/Write: 2/98 | Sequential: 99% | Queue Depth: 1", label: "Backup Archive" },
          { input: "IOPS: 9500 | Latency: 3.0ms | Block Size: 256K | Read/Write: 90/10 | Sequential: 85% | Queue Depth: 8", label: "OLAP Analytics" }
        ]
      },
      9: {
        prompt: "[few-shot prompt]\n\nPattern: IOPS: 6800 | Latency: 2.1ms | Block Size: 64K | Read/Write: 80/20 | Sequential: 65% | Queue Depth: 16",
        response: "Classification: OLAP Analytics",
        correct: false,
        fewShotExamples: [
          { input: "IOPS: 52000 | Latency: 0.2ms | Block Size: 4K | Read/Write: 80/20 | Sequential: 10% | Queue Depth: 64", label: "OLTP Database" },
          { input: "IOPS: 1800 | Latency: 15ms | Block Size: 1M | Read/Write: 2/98 | Sequential: 99% | Queue Depth: 1", label: "Backup Archive" },
          { input: "IOPS: 9500 | Latency: 3.0ms | Block Size: 256K | Read/Write: 90/10 | Sequential: 85% | Queue Depth: 8", label: "OLAP Analytics" }
        ]
      },
      10: {
        prompt: "[few-shot prompt]\n\nPattern: IOPS: 28000 | Latency: 0.5ms | Block Size: 4K | Read/Write: 65/35 | Sequential: 18% | Queue Depth: 64",
        response: "Classification: OLTP Database",
        correct: true,
        fewShotExamples: [
          { input: "IOPS: 52000 | Latency: 0.2ms | Block Size: 4K | Read/Write: 80/20 | Sequential: 10% | Queue Depth: 64", label: "OLTP Database" },
          { input: "IOPS: 1800 | Latency: 15ms | Block Size: 1M | Read/Write: 2/98 | Sequential: 99% | Queue Depth: 1", label: "Backup Archive" },
          { input: "IOPS: 9500 | Latency: 3.0ms | Block Size: 256K | Read/Write: 90/10 | Sequential: 85% | Queue Depth: 8", label: "OLAP Analytics" }
        ]
      }
    },

    // Token probabilities for example 1 with basic prompt (spread out, uncertain)
    tokenProbsBasic: [
      { token: "The", probability: 0.12 },
      { token: "This", probability: 0.11 },
      { token: "OLTP", probability: 0.08 },
      { token: "Based", probability: 0.07 },
      { token: "I", probability: 0.06 },
      { token: "High", probability: 0.055 },
      { token: "It", probability: 0.05 },
      { token: "Looking", probability: 0.045 },
      { token: "Given", probability: 0.04 },
      { token: "With", probability: 0.035 },
      { token: "OL", probability: 0.03 },
      { token: "VDI", probability: 0.028 },
      { token: "Database", probability: 0.025 },
      { token: "These", probability: 0.022 },
      { token: "Analyzing", probability: 0.02 },
      { token: "Storage", probability: 0.018 },
      { token: "From", probability: 0.015 },
      { token: "Considering", probability: 0.013 },
      { token: "Let", probability: 0.012 },
      { token: "After", probability: 0.01 }
    ],

    // Token probabilities for example 1 with few-shot prompt (more focused)
    tokenProbsFewShot: [
      { token: "OLTP", probability: 0.31 },
      { token: "Classification", probability: 0.18 },
      { token: "OL", probability: 0.09 },
      { token: "This", probability: 0.06 },
      { token: "VDI", probability: 0.05 },
      { token: "Database", probability: 0.04 },
      { token: "The", probability: 0.035 },
      { token: "Based", probability: 0.03 },
      { token: "It", probability: 0.025 },
      { token: "High", probability: 0.02 },
      { token: "I", probability: 0.018 },
      { token: "Trans", probability: 0.015 },
      { token: "AI", probability: 0.012 },
      { token: "Backup", probability: 0.01 },
      { token: "Video", probability: 0.008 },
      { token: "Given", probability: 0.007 },
      { token: "With", probability: 0.006 },
      { token: "Looking", probability: 0.005 },
      { token: "Pattern", probability: 0.004 },
      { token: "Storage", probability: 0.003 }
    ]
  },

  // ===========================================================================
  // RAG results
  // ===========================================================================
  rag: {
    knowledgeBase: [
      { id: 1, pattern: "High IOPS (>30K), small blocks (4-8K), low latency (<1ms), random I/O (sequential <20%)", label: "OLTP Database", description: "Transaction processing workloads like MySQL, PostgreSQL, Oracle with high-frequency small random reads/writes for row-level operations." },
      { id: 2, pattern: "Moderate IOPS (5K-15K), large blocks (64K-256K), read-heavy (>80% reads), sequential (>70%)", label: "OLAP Analytics", description: "Analytical query workloads scanning large datasets. Tools like Spark, Presto, Redshift performing columnar scans and aggregations." },
      { id: 3, pattern: "Moderate IOPS (5K-10K), large blocks (128K-512K), read-heavy, highly sequential (>85%), moderate queue depth", label: "AI ML Training", description: "Deep learning training data loading. GPU-bound workloads with periodic large sequential reads to feed training pipelines. Frameworks like PyTorch DataLoader, TensorFlow tf.data." },
      { id: 4, pattern: "Low IOPS (<5K), very large blocks (512K-2M), almost all reads (>95%), nearly 100% sequential, low queue depth", label: "Video Streaming", description: "Media streaming workloads serving pre-encoded video chunks. Low IOPS but high throughput with large sequential reads." },
      { id: 5, pattern: "Moderate-high IOPS (15K-25K), small-medium blocks (8-32K), balanced read/write (50-65% read), mostly random, moderate-high queue depth", label: "VDI Virtual Desktop", description: "Virtual desktop infrastructure aggregating I/O from many user desktops. Mix of boot storms, application launches, and document access." },
      { id: 6, pattern: "Low IOPS (<5K), large blocks (256K-1M), write-heavy (>90% writes), highly sequential, low queue depth (1-4)", label: "Backup Archive", description: "Backup and archival workloads writing large sequential streams to tape or object storage. Deduplication and compression stages." },
      { id: 7, pattern: "Very high IOPS (>50K), tiny blocks (4K), ultra-low latency (<0.5ms), random, very high queue depth (>64)", label: "OLTP Database", description: "High-performance OLTP with NVMe storage. In-memory databases with write-ahead logging creating intense small random I/O." },
      { id: 8, pattern: "Low-moderate IOPS (3K-8K), medium blocks (64K-128K), read-heavy, mixed sequential/random (50-70% sequential)", label: "OLAP Analytics", description: "Ad-hoc analytical queries with partial table scans. Less sequential than full scans due to predicate pushdown and index usage." },
      { id: 9, pattern: "Variable IOPS (2K-15K), large blocks (256K-1M), read-heavy during training, write-heavy during checkpointing, highly sequential", label: "AI ML Training", description: "ML training with periodic checkpoint writes. Training phases show sequential reads; checkpoint phases show bursty large sequential writes." },
      { id: 10, pattern: "Low IOPS (1K-3K), very large blocks (1-4M), read-only, sequential, very low queue depth (1-2)", label: "Video Streaming", description: "4K/8K video streaming requiring very large sequential reads. Minimal random access due to pre-fetching and buffering." },
      { id: 11, pattern: "High IOPS (20K-40K), small blocks (4-16K), write-heavy during boot storms (30-70% write), random during steady state", label: "VDI Virtual Desktop", description: "VDI boot storm pattern where hundreds of desktops boot simultaneously, creating heavy random write I/O from OS and application loading." },
      { id: 12, pattern: "Very low IOPS (<2K), very large blocks (1M+), write-only, sequential, minimal queue depth", label: "Backup Archive", description: "Full backup streams writing at maximum throughput. Single-threaded sequential writes optimized for tape drive streaming." },
      { id: 13, pattern: "Moderate IOPS (10K-20K), small blocks (4-8K), read-heavy, random, moderate latency (1-3ms)", label: "OLTP Database", description: "Read-replica database workloads serving queries. Lower write ratio than primary but same random small-block read pattern." },
      { id: 14, pattern: "High throughput, large blocks (128K-512K), sequential reads with periodic random writes, moderate IOPS", label: "AI ML Training", description: "Distributed training with data sharding. Each GPU reads its shard sequentially while coordination writes are random." },
      { id: 15, pattern: "Bimodal IOPS (peaks at boot, moderate steady-state), small blocks, mixed read/write, random", label: "VDI Virtual Desktop", description: "VDI steady-state pattern after boot storm subsides. Application I/O from office apps, browsers, creating moderate random mixed I/O." },
      { id: 16, pattern: "Moderate IOPS (8K-15K), medium-large blocks (64K-256K), read-heavy (85%+), mostly sequential, moderate queue depth", label: "OLAP Analytics", description: "Data warehouse queries scanning partitioned tables. Sequential within partitions but may skip between them." },
      { id: 17, pattern: "Extremely low IOPS (<1K), maximum block size, write-dominant, sequential, single queue", label: "Backup Archive", description: "Archive tier cold storage writes. Infrequent but sustained sequential write streams for long-term retention." },
      { id: 18, pattern: "High IOPS (25K-45K), small blocks (4-8K), low latency, predominantly random, high queue depth", label: "OLTP Database", description: "Mixed OLTP workload with both transactional queries and batch updates. Write-ahead log and data file I/O combined." }
    ],

    // RAG retrieval results and responses for each example
    results: {
      1: {
        retrievedDocs: [1, 7, 18],
        augmentedPrompt: "Using the following reference information about storage I/O patterns:\n\n1. High IOPS (>30K), small blocks (4-8K), low latency (<1ms), random I/O -> OLTP Database\n2. Very high IOPS (>50K), tiny blocks (4K), ultra-low latency -> OLTP Database\n3. High IOPS (25K-45K), small blocks (4-8K), low latency, random, high queue depth -> OLTP Database\n\nClassify: IOPS: 45000 | Latency: 0.3ms | Block Size: 8K | Read/Write: 70/30 | Sequential: 15% | Queue Depth: 32",
        response: "Based on the retrieved reference patterns, this matches OLTP Database. The 45000 IOPS with 8K block size, 0.3ms latency, and 15% sequential (highly random) access aligns closely with documented OLTP Database patterns showing high IOPS, small blocks, low latency, and random I/O.",
        correct: true
      },
      2: {
        retrievedDocs: [2, 8, 16],
        augmentedPrompt: "[RAG prompt with OLAP references]\n\nClassify: IOPS: 12000 | Latency: 4.5ms | Block Size: 128K | Read/Write: 92/8 | Sequential: 78% | Queue Depth: 8",
        response: "This pattern matches OLAP Analytics. The 128K block size with 92% reads and 78% sequential access aligns with analytical query workloads scanning large datasets, consistent with the reference patterns for OLAP Analytics.",
        correct: true
      },
      3: {
        retrievedDocs: [3, 9, 14],
        augmentedPrompt: "[RAG prompt with AI ML references]\n\nClassify: IOPS: 8500 | Latency: 1.2ms | Block Size: 256K | Read/Write: 85/15 | Sequential: 91% | Queue Depth: 16",
        response: "This matches AI ML Training. The 256K block size with 91% sequential reads at moderate IOPS is consistent with deep learning data loading pipelines that read training data in large sequential batches.",
        correct: true
      },
      4: {
        retrievedDocs: [4, 10, 12],
        augmentedPrompt: "[RAG prompt with streaming/archive references]\n\nClassify: IOPS: 2200 | Latency: 8.0ms | Block Size: 1M | Read/Write: 99/1 | Sequential: 98% | Queue Depth: 4",
        response: "This is Video Streaming. The 1MB block size with 99% reads and 98% sequential access at low IOPS matches the Video Streaming reference pattern of large sequential reads with minimal random access.",
        correct: true
      },
      5: {
        retrievedDocs: [5, 11, 15],
        augmentedPrompt: "[RAG prompt with VDI references]\n\nClassify: IOPS: 18000 | Latency: 1.5ms | Block Size: 16K | Read/Write: 60/40 | Sequential: 25% | Queue Depth: 24",
        response: "This matches VDI Virtual Desktop. The 18000 IOPS with 16K blocks, 60/40 read/write balance, and 25% sequential (mostly random) access aligns with VDI steady-state patterns aggregating desktop I/O.",
        correct: true
      },
      6: {
        retrievedDocs: [6, 12, 17],
        augmentedPrompt: "[RAG prompt with backup references]\n\nClassify: IOPS: 3500 | Latency: 12ms | Block Size: 512K | Read/Write: 5/95 | Sequential: 95% | Queue Depth: 2",
        response: "This is Backup Archive. The 512K blocks with 95% writes and 95% sequential access at low queue depth matches backup workload patterns writing large sequential streams.",
        correct: true
      },
      7: {
        retrievedDocs: [2, 8, 16],
        augmentedPrompt: "[RAG prompt with OLAP references]\n\nClassify: IOPS: 5200 | Latency: 3.8ms | Block Size: 256K | Read/Write: 88/12 | Sequential: 82% | Queue Depth: 12",
        response: "This matches OLAP Analytics. The 256K block size with 88% reads and 82% sequential access is consistent with analytical queries performing large dataset scans with some predicate-based filtering.",
        correct: true
      },
      8: {
        retrievedDocs: [5, 11, 1],
        augmentedPrompt: "[RAG prompt with VDI and OLTP references]\n\nClassify: IOPS: 32000 | Latency: 0.8ms | Block Size: 8K | Read/Write: 55/45 | Sequential: 22% | Queue Depth: 48",
        response: "This pattern shows characteristics of both VDI Virtual Desktop and OLTP Database. The 32000 IOPS and 8K blocks could indicate either. However, the nearly balanced 55/45 read/write ratio is more typical of VDI aggregated desktop I/O than typical OLTP which tends to be more read-heavy. Classification: VDI Virtual Desktop.",
        correct: true
      },
      9: {
        retrievedDocs: [3, 8, 14],
        augmentedPrompt: "[RAG prompt with AI ML and OLAP references]\n\nClassify: IOPS: 6800 | Latency: 2.1ms | Block Size: 64K | Read/Write: 80/20 | Sequential: 65% | Queue Depth: 16",
        response: "This is somewhat ambiguous. The 64K block size falls between typical OLAP Analytics (64K-256K) and AI ML Training (128K-512K). The 65% sequential rate is lower than typical ML training (>85%). Based on the reference patterns, this could be either OLAP Analytics with partial table scans or AI ML Training with distributed data sharding. I'll classify as OLAP Analytics due to the moderate sequential percentage.",
        correct: false
      },
      10: {
        retrievedDocs: [1, 7, 18],
        augmentedPrompt: "[RAG prompt with OLTP references]\n\nClassify: IOPS: 28000 | Latency: 0.5ms | Block Size: 4K | Read/Write: 65/35 | Sequential: 18% | Queue Depth: 64",
        response: "This matches OLTP Database. The 28000 IOPS with 4K blocks, 0.5ms latency, and 18% sequential (highly random) access aligns with OLTP workloads. The high queue depth of 64 indicates many concurrent transactions.",
        correct: true
      }
    },

    // Limitation example: RAG gets the right answer but with poor format/behavior
    limitationExample: {
      input: "IOPS: 38000 | Latency: 0.4ms | Block Size: 8K | Read/Write: 72/28 | Sequential: 12% | Queue Depth: 56",
      retrievedDocs: [
        "High IOPS (>30K), small blocks (4-8K), low latency (<1ms), random I/O -> OLTP Database: Transaction processing workloads like MySQL, PostgreSQL...",
        "High IOPS (25K-45K), small blocks (4-8K), low latency, predominantly random, high queue depth -> OLTP Database: Mixed OLTP workload...",
        "Moderate-high IOPS (15K-25K), small-medium blocks (8-32K), balanced read/write -> VDI Virtual Desktop: Virtual desktop infrastructure..."
      ],
      response: "Based on the retrieved information, this COULD be an OLTP Database workload, although there's a possibility it might also be a VDI Virtual Desktop pattern. The IOPS of 38000 falls within the OLTP Database range (>30K) mentioned in the reference documents, and the 8K block size with 0.4ms latency are consistent with that classification. However, I should note that some VDI patterns can also exhibit high IOPS with small blocks. Looking at the queue depth of 56 and the read/write ratio of 72/28, these lean more towards OLTP. Overall, I would classify this as OLTP Database with moderate-to-high confidence, though I cannot rule out the possibility of it being VDI or another similar high-IOPS workload.",
      desiredResponse: "OLTP Database"
    }
  },

  // ===========================================================================
  // SFT (Supervised Fine-Tuning) results
  // ===========================================================================
  sft: {
    trainingConfig: {
      baseModel: "HuggingFaceTB/SmolLM2-360M",
      method: "LoRA",
      loraRank: 16,
      loraAlpha: 32,
      trainableParams: "0.12%",
      totalParams: "360M",
      trainableParamsCount: "432K",
      epochs: 3,
      learningRate: 2e-4,
      batchSize: 8,
      trainingExamples: 1400
    },

    // SFT Loss curve (~525 steps)
    lossCurve: generateSFTLossCurve(),

    // Before/after comparison for each example
    comparisons: {
      1: {
        baseOutput: "This storage I/O pattern shows high IOPS with small block sizes and low latency. It could be a database or VDI workload. I think it might be related to a transactional system.",
        sftOutput: "OLTP Database",
        baseCorrect: false,
        sftCorrect: true
      },
      2: {
        baseOutput: "Large block reads with sequential access suggest some form of data processing. Could be analytics or streaming.",
        sftOutput: "OLAP Analytics",
        baseCorrect: false,
        sftCorrect: true
      },
      3: {
        baseOutput: "Large sequential reads with moderate IOPS. This looks like it could be streaming or analytical workload.",
        sftOutput: "AI ML Training",
        baseCorrect: false,
        sftCorrect: true
      },
      4: {
        baseOutput: "Very large sequential reads with low IOPS. This is clearly some kind of streaming workload, likely video.",
        sftOutput: "Video Streaming",
        baseCorrect: true,
        sftCorrect: true
      },
      5: {
        baseOutput: "Mixed read/write with moderate IOPS and small blocks. Could be a database or virtual desktop environment.",
        sftOutput: "VDI Virtual Desktop",
        baseCorrect: false,
        sftCorrect: true
      },
      6: {
        baseOutput: "Write-heavy sequential access with large blocks. This is backup or archival activity.",
        sftOutput: "Backup Archive",
        baseCorrect: true,
        sftCorrect: true
      },
      7: {
        baseOutput: "Sequential reads with large blocks. Possibly analytics or some kind of data pipeline.",
        sftOutput: "OLAP Analytics",
        baseCorrect: false,
        sftCorrect: true
      },
      8: {
        baseOutput: "High IOPS random access with small blocks. Looks like OLTP Database to me.",
        sftOutput: "OLTP Database",
        baseCorrect: false,
        sftCorrect: false
      },
      9: {
        baseOutput: "Medium blocks with moderate sequential access. Could be analytics or ML training.",
        sftOutput: "OLAP Analytics",
        baseCorrect: false,
        sftCorrect: false
      },
      10: {
        baseOutput: "Very high IOPS with tiny blocks and random access. Database workload, possibly OLTP.",
        sftOutput: "OLTP Database",
        baseCorrect: true,
        sftCorrect: true
      }
    },

    // Token probability comparison for example 1
    tokenProbs: {
      1: {
        base: [
          { token: "The", probability: 0.12 },
          { token: "This", probability: 0.11 },
          { token: "OLTP", probability: 0.08 },
          { token: "Based", probability: 0.07 },
          { token: "I", probability: 0.06 },
          { token: "High", probability: 0.055 },
          { token: "It", probability: 0.05 },
          { token: "Looking", probability: 0.045 },
          { token: "Given", probability: 0.04 },
          { token: "With", probability: 0.035 },
          { token: "OL", probability: 0.03 },
          { token: "VDI", probability: 0.028 },
          { token: "Database", probability: 0.025 },
          { token: "These", probability: 0.022 },
          { token: "Analyzing", probability: 0.02 },
          { token: "Storage", probability: 0.018 },
          { token: "From", probability: 0.015 },
          { token: "Considering", probability: 0.013 },
          { token: "Let", probability: 0.012 },
          { token: "After", probability: 0.01 }
        ],
        sft: [
          { token: "OLTP", probability: 0.82 },
          { token: "VDI", probability: 0.06 },
          { token: "OL", probability: 0.03 },
          { token: "Database", probability: 0.02 },
          { token: "The", probability: 0.012 },
          { token: "Trans", probability: 0.008 },
          { token: "This", probability: 0.006 },
          { token: "AI", probability: 0.005 },
          { token: "High", probability: 0.004 },
          { token: "Based", probability: 0.003 },
          { token: "Backup", probability: 0.0025 },
          { token: "Video", probability: 0.002 },
          { token: "It", probability: 0.0018 },
          { token: "I", probability: 0.0015 },
          { token: "Storage", probability: 0.0012 },
          { token: "Looking", probability: 0.001 },
          { token: "With", probability: 0.0008 },
          { token: "Given", probability: 0.0006 },
          { token: "From", probability: 0.0005 },
          { token: "Let", probability: 0.0004 }
        ]
      },
      8: {
        base: [
          { token: "This", probability: 0.13 },
          { token: "High", probability: 0.10 },
          { token: "The", probability: 0.09 },
          { token: "OLTP", probability: 0.07 },
          { token: "VDI", probability: 0.065 },
          { token: "Based", probability: 0.05 },
          { token: "I", probability: 0.045 },
          { token: "It", probability: 0.04 },
          { token: "Looking", probability: 0.035 },
          { token: "With", probability: 0.03 },
          { token: "Database", probability: 0.025 },
          { token: "Given", probability: 0.022 },
          { token: "These", probability: 0.02 },
          { token: "OL", probability: 0.018 },
          { token: "Random", probability: 0.015 },
          { token: "Small", probability: 0.012 },
          { token: "Analyzing", probability: 0.01 },
          { token: "From", probability: 0.008 },
          { token: "Storage", probability: 0.007 },
          { token: "Let", probability: 0.006 }
        ],
        sft: [
          { token: "OLTP", probability: 0.45 },
          { token: "VDI", probability: 0.38 },
          { token: "Database", probability: 0.04 },
          { token: "OL", probability: 0.025 },
          { token: "The", probability: 0.02 },
          { token: "Virtual", probability: 0.015 },
          { token: "This", probability: 0.012 },
          { token: "Trans", probability: 0.008 },
          { token: "High", probability: 0.006 },
          { token: "Based", probability: 0.005 },
          { token: "AI", probability: 0.004 },
          { token: "It", probability: 0.003 },
          { token: "Backup", probability: 0.0025 },
          { token: "Video", probability: 0.002 },
          { token: "I", probability: 0.0018 },
          { token: "Storage", probability: 0.0015 },
          { token: "Looking", probability: 0.001 },
          { token: "With", probability: 0.0008 },
          { token: "Given", probability: 0.0006 },
          { token: "From", probability: 0.0005 }
        ]
      }
    },

    // LoRA weight visualization data
    loraWeights: {
      layerName: "model.layers.8.self_attn.q_proj",
      matrixA: loraMatrixA,
      matrixB: loraMatrixB,
      deltaVisualization: loraDelta
    },

    infrastructure: {
      gpuMemoryGB: 4.2,
      trainingTimeMinutes: 12,
      checkpointSizeMB: 1.7,
      fullModelSizeMB: 720,
      peakGPUUtilization: 87,
      storageIOPattern: "Bursty writes every ~50 steps (checkpoint), steady reads"
    }
  },

  // ===========================================================================
  // DPO (Direct Preference Optimization) results
  // ===========================================================================
  dpo: {
    trainingConfig: {
      method: "DPO with LoRA",
      beta: 0.1,
      trainableParams: "0.12%",
      preferencePairs: 400,
      epochs: 2,
      learningRate: 5e-5
    },

    preferencePairs: [
      {
        input: "IOPS: 45000 | Latency: 0.3ms | Block Size: 8K | Read/Write: 70/30 | Sequential: 15% | Queue Depth: 32",
        chosen: "Classification: OLTP Database\nConfidence: High\nKey indicators: High IOPS with small block sizes and very low latency are hallmarks of transaction processing.",
        rejected: "Classification: OLTP Database\nThis could potentially be an OLTP Database workload. The IOPS are quite high at 45000, and the block size of 8K is relatively small. The latency is low at 0.3ms. However, it could also possibly be a VDI workload given the mixed read/write ratio. On balance, OLTP Database seems most likely but I'm not entirely certain.",
        whyChosen: "Concise, confident, highlights the key indicators"
      },
      {
        input: "IOPS: 12000 | Latency: 4.5ms | Block Size: 128K | Read/Write: 92/8 | Sequential: 78% | Queue Depth: 8",
        chosen: "Classification: OLAP Analytics\nConfidence: High\nKey indicators: Large sequential reads with moderate IOPS indicate analytical query workloads scanning datasets.",
        rejected: "OLAP Analytics\n\nI classified this as OLAP because the IOPS are 12000 and the block size is 128K and the read/write ratio is 92/8 and the sequential percentage is 78% and the queue depth is 8. All of these metrics together point to OLAP.",
        whyChosen: "Structured format with meaningful reasoning vs. listing metrics without insight"
      },
      {
        input: "IOPS: 8500 | Latency: 1.2ms | Block Size: 256K | Read/Write: 85/15 | Sequential: 91% | Queue Depth: 16",
        chosen: "Classification: AI ML Training\nConfidence: High\nKey indicators: Very high sequential reads with large blocks at moderate IOPS match GPU training data pipelines loading batches.",
        rejected: "Classification: Video Streaming\nConfidence: Medium\nKey indicators: Sequential reads with large blocks suggest media serving.",
        whyChosen: "Correct classification with accurate reasoning vs. wrong classification"
      },
      {
        input: "IOPS: 32000 | Latency: 0.8ms | Block Size: 8K | Read/Write: 55/45 | Sequential: 22% | Queue Depth: 48",
        chosen: "Classification: VDI Virtual Desktop\nConfidence: Medium\nKey indicators: Balanced read/write ratio with high random IOPS suggests aggregated desktop I/O. The near-even split distinguishes this from OLTP which skews read-heavy.",
        rejected: "Classification: VDI Virtual Desktop\nConfidence: Medium\nI think this is VDI because the read/write ratio is balanced.",
        whyChosen: "Provides comparative reasoning explaining what distinguishes VDI from similar patterns"
      },
      {
        input: "IOPS: 3500 | Latency: 12ms | Block Size: 512K | Read/Write: 5/95 | Sequential: 95% | Queue Depth: 2",
        chosen: "Classification: Backup Archive\nConfidence: Very High\nKey indicators: Extreme write dominance (95%) with large sequential blocks and minimal queue depth are textbook backup characteristics.",
        rejected: "Classification: Backup Archive\nConfidence: Very High\nKey indicators: Write-heavy, large blocks, sequential.\n\nAdditional analysis: The IOPS of 3500 combined with 512K blocks gives a throughput of approximately 1.75 GB/s which is consistent with modern LTO-9 tape drives rated at 400 MB/s native or network-attached backup targets. The 12ms latency is typical for spinning disk backup targets or tape libraries with robotic media changers. The queue depth of 2 suggests a single backup stream with double-buffering.",
        whyChosen: "Concise and sufficient vs. over-analyzing with unnecessary calculations for a classification task"
      },
      {
        input: "IOPS: 6800 | Latency: 2.1ms | Block Size: 64K | Read/Write: 80/20 | Sequential: 65% | Queue Depth: 16",
        chosen: "Classification: AI ML Training\nConfidence: Medium\nKey indicators: Moderate sequential reads with medium-large blocks suggest data pipeline loading, though the lower sequential rate may indicate distributed training with data sharding.",
        rejected: "Classification: AI ML Training\nConfidence: Low\nI'm not really sure about this one. It could be AI ML Training but it might also be OLAP Analytics. The block size is kind of in between. Maybe it's something else entirely. I'll go with AI ML Training but I wouldn't be surprised if I'm wrong.",
        whyChosen: "Acknowledges uncertainty through confidence level without undermining the classification"
      }
    ],

    lossCurve: generateDPOLossCurve(),

    // Probability shift showing how DPO changes model preferences
    probabilityShift: {
      example: "IOPS: 45000 | Latency: 0.3ms | Block Size: 8K | Read/Write: 70/30 | Sequential: 15% | Queue Depth: 32",
      beforeDPO: { chosenLogProb: -2.1, rejectedLogProb: -1.8 },
      afterDPO: { chosenLogProb: -0.9, rejectedLogProb: -3.2 }
    },

    // DPO style improvements for each example
    comparisons: {
      1: {
        sftOutput: "OLTP Database",
        dpoOutput: "Classification: OLTP Database\nConfidence: High\nKey indicators: High IOPS (45K) with small blocks (8K), sub-millisecond latency, and random access pattern are hallmarks of transaction processing."
      },
      2: {
        sftOutput: "OLAP Analytics",
        dpoOutput: "Classification: OLAP Analytics\nConfidence: High\nKey indicators: Large block (128K) sequential reads at moderate IOPS indicate analytical query workloads performing dataset scans."
      },
      3: {
        sftOutput: "AI ML Training",
        dpoOutput: "Classification: AI ML Training\nConfidence: High\nKey indicators: Very high sequential rate (91%) with large blocks (256K) and read-heavy access match GPU training data loading pipelines."
      },
      4: {
        sftOutput: "Video Streaming",
        dpoOutput: "Classification: Video Streaming\nConfidence: Very High\nKey indicators: Very large blocks (1M), nearly all reads (99%), near-total sequential access, and low queue depth are characteristic of media streaming."
      },
      5: {
        sftOutput: "VDI Virtual Desktop",
        dpoOutput: "Classification: VDI Virtual Desktop\nConfidence: High\nKey indicators: Balanced read/write (60/40) with moderate IOPS and small blocks suggest aggregated desktop I/O from multiple concurrent users."
      },
      6: {
        sftOutput: "Backup Archive",
        dpoOutput: "Classification: Backup Archive\nConfidence: Very High\nKey indicators: Extreme write dominance (95%), large blocks (512K), sequential access, and low queue depth are textbook backup stream characteristics."
      },
      7: {
        sftOutput: "OLAP Analytics",
        dpoOutput: "Classification: OLAP Analytics\nConfidence: High\nKey indicators: Large sequential reads (256K blocks, 82% sequential) with high read ratio indicate analytical queries scanning large datasets."
      },
      8: {
        sftOutput: "OLTP Database",
        dpoOutput: "Classification: VDI Virtual Desktop\nConfidence: Medium\nKey indicators: Near-balanced read/write (55/45) with high IOPS and random access distinguishes this from OLTP which typically skews more read-heavy."
      },
      9: {
        sftOutput: "OLAP Analytics",
        dpoOutput: "Classification: AI ML Training\nConfidence: Medium\nKey indicators: Read-heavy sequential access with medium blocks (64K) suggests data loading, though the moderate sequential rate may indicate distributed training with data sharding."
      },
      10: {
        sftOutput: "OLTP Database",
        dpoOutput: "Classification: OLTP Database\nConfidence: High\nKey indicators: Very high IOPS (28K) with tiny blocks (4K), sub-millisecond latency, and highly random access are definitive OLTP characteristics."
      }
    },

    infrastructure: {
      gpuMemoryGB: 5.1,
      trainingTimeMinutes: 8,
      checkpointSizeMB: 1.7,
      peakGPUUtilization: 82,
      storageIOPattern: "Similar to SFT but with paired examples loaded",
      vsRLHF: {
        rlhfGPUMemoryGB: 12.8,
        rlhfTrainingTimeMinutes: 45,
        rlhfModelsInMemory: 3,
        dpoModelsInMemory: 1,
        note: "DPO achieves similar alignment with ~60% less compute"
      }
    }
  },

  // ===========================================================================
  // GRPO (Group Relative Policy Optimization) results
  // ===========================================================================
  grpo: {
    trainingConfig: {
      method: "GRPO",
      groupSize: 8,
      rewardType: "Binary (correct classification = 1.0, incorrect = 0.0)",
      epochs: 3,
      learningRate: 1e-5,
      batchSize: 4
    },

    generationLogs: [
      {
        input: "IOPS: 12000 | Latency: 4.5ms | Block Size: 128K | Read/Write: 92/8 | Sequential: 78% | Queue Depth: 8",
        correctLabel: "OLAP Analytics",
        generations: [
          { text: "Classification: OLAP Analytics\nReason: High sequential reads with large blocks suggest analytical queries.", reward: 1.0, advantage: 0.625 },
          { text: "Classification: AI ML Training\nReason: Large block sequential reads often indicate training data loading.", reward: 0.0, advantage: -0.375 },
          { text: "Classification: OLAP Analytics\nReason: Moderate IOPS with 128K blocks and read-heavy access match data warehouse queries.", reward: 1.0, advantage: 0.625 },
          { text: "Classification: Video Streaming\nReason: Sequential reads with large blocks could be media serving.", reward: 0.0, advantage: -0.375 },
          { text: "Classification: OLAP Analytics\nReason: 78% sequential with 92% reads is characteristic of columnar scans.", reward: 1.0, advantage: 0.625 },
          { text: "Classification: OLAP Analytics\nReason: Read-dominant large-block sequential I/O matches analytical processing.", reward: 0.0, advantage: -0.375 },
          { text: "Classification: AI ML Training\nReason: Could be loading training batches from storage.", reward: 0.0, advantage: -0.375 },
          { text: "Classification: OLAP Analytics\nReason: Large sequential reads at moderate IOPS are typical of data analytics.", reward: 0.0, advantage: -0.375 }
        ],
        groupMeanReward: 0.375,
        groupStdReward: 0.484
      },
      {
        input: "IOPS: 45000 | Latency: 0.3ms | Block Size: 8K | Read/Write: 70/30 | Sequential: 15% | Queue Depth: 32",
        correctLabel: "OLTP Database",
        generations: [
          { text: "Classification: OLTP Database\nReason: Very high IOPS with small random blocks and sub-ms latency.", reward: 1.0, advantage: 0.375 },
          { text: "Classification: OLTP Database\nReason: High IOPS, small blocks, low latency indicate transactional workload.", reward: 1.0, advantage: 0.375 },
          { text: "Classification: VDI Virtual Desktop\nReason: High IOPS with mixed read/write could be VDI.", reward: 0.0, advantage: -0.625 },
          { text: "Classification: OLTP Database\nReason: Random 8K I/O at 45K IOPS with 0.3ms latency matches database.", reward: 1.0, advantage: 0.375 },
          { text: "Classification: OLTP Database\nReason: Classic OLTP pattern - small blocks, low latency, random access.", reward: 1.0, advantage: 0.375 },
          { text: "Classification: OLTP Database\nReason: Transaction processing signature: high random IOPS, small blocks.", reward: 1.0, advantage: 0.375 },
          { text: "Classification: VDI Virtual Desktop\nReason: High IOPS random I/O could be aggregated desktop access.", reward: 0.0, advantage: -0.625 },
          { text: "Classification: OLTP Database\nReason: Unmistakable OLTP: 45K IOPS, 8K blocks, 0.3ms, random.", reward: 0.0, advantage: -0.625 }
        ],
        groupMeanReward: 0.625,
        groupStdReward: 0.484
      },
      {
        input: "IOPS: 8500 | Latency: 1.2ms | Block Size: 256K | Read/Write: 85/15 | Sequential: 91% | Queue Depth: 16",
        correctLabel: "AI ML Training",
        generations: [
          { text: "Classification: OLAP Analytics\nReason: Large block sequential reads suggest analytical queries.", reward: 0.0, advantage: -0.25 },
          { text: "Classification: AI ML Training\nReason: Large sequential reads with 256K blocks match training data loading.", reward: 1.0, advantage: 0.75 },
          { text: "Classification: Video Streaming\nReason: Sequential reads with large blocks.", reward: 0.0, advantage: -0.25 },
          { text: "Classification: AI ML Training\nReason: High sequential rate with large blocks indicates GPU data pipeline.", reward: 1.0, advantage: 0.75 },
          { text: "Classification: OLAP Analytics\nReason: Read-heavy sequential large block access pattern.", reward: 0.0, advantage: -0.25 },
          { text: "Classification: AI ML Training\nReason: 91% sequential, 256K blocks, and moderate IOPS match ML training.", reward: 0.0, advantage: -0.25 },
          { text: "Classification: OLAP Analytics\nReason: Could be data warehouse full table scans.", reward: 0.0, advantage: -0.25 },
          { text: "Classification: AI ML Training\nReason: Training data loaders produce this sequential large-block read pattern.", reward: 0.0, advantage: -0.25 }
        ],
        groupMeanReward: 0.25,
        groupStdReward: 0.433
      },
      {
        input: "IOPS: 32000 | Latency: 0.8ms | Block Size: 8K | Read/Write: 55/45 | Sequential: 22% | Queue Depth: 48",
        correctLabel: "VDI Virtual Desktop",
        generations: [
          { text: "Classification: OLTP Database\nReason: High IOPS with small random blocks.", reward: 0.0, advantage: -0.125 },
          { text: "Classification: VDI Virtual Desktop\nReason: Balanced read/write with high random IOPS suggests many desktops.", reward: 1.0, advantage: 0.875 },
          { text: "Classification: OLTP Database\nReason: Small block random I/O at high IOPS.", reward: 0.0, advantage: -0.125 },
          { text: "Classification: OLTP Database\nReason: 32K IOPS with 8K random blocks looks transactional.", reward: 0.0, advantage: -0.125 },
          { text: "Classification: OLTP Database\nReason: Low latency random small-block I/O is OLTP.", reward: 0.0, advantage: -0.125 },
          { text: "Classification: VDI Virtual Desktop\nReason: The balanced 55/45 read/write ratio distinguishes this from OLTP.", reward: 0.0, advantage: -0.125 },
          { text: "Classification: OLTP Database\nReason: High IOPS, small blocks, low latency -> database.", reward: 0.0, advantage: -0.125 },
          { text: "Classification: VDI Virtual Desktop\nReason: Nearly even read/write split with random access = aggregated desktops.", reward: 0.0, advantage: -0.125 }
        ],
        groupMeanReward: 0.125,
        groupStdReward: 0.331
      },
      {
        input: "IOPS: 2200 | Latency: 8.0ms | Block Size: 1M | Read/Write: 99/1 | Sequential: 98% | Queue Depth: 4",
        correctLabel: "Video Streaming",
        generations: [
          { text: "Classification: Video Streaming\nReason: Very large sequential reads, almost read-only, low IOPS.", reward: 1.0, advantage: 0.25 },
          { text: "Classification: Video Streaming\nReason: 1M blocks with 99% reads and 98% sequential = media streaming.", reward: 1.0, advantage: 0.25 },
          { text: "Classification: Video Streaming\nReason: Classic streaming pattern: large blocks, sequential, read-only.", reward: 1.0, advantage: 0.25 },
          { text: "Classification: Backup Archive\nReason: Large sequential blocks could be backup reads.", reward: 0.0, advantage: -0.75 },
          { text: "Classification: Video Streaming\nReason: Low IOPS, huge blocks, sequential reads indicate media serving.", reward: 1.0, advantage: 0.25 },
          { text: "Classification: Video Streaming\nReason: Streaming media from storage: large sequential read-only access.", reward: 1.0, advantage: 0.25 },
          { text: "Classification: Video Streaming\nReason: 1MB sequential reads at low IOPS match video content delivery.", reward: 1.0, advantage: 0.25 },
          { text: "Classification: Video Streaming\nReason: Unmistakable streaming: huge sequential reads, near-zero writes.", reward: 0.0, advantage: -0.75 }
        ],
        groupMeanReward: 0.75,
        groupStdReward: 0.433
      },
      {
        input: "IOPS: 6800 | Latency: 2.1ms | Block Size: 64K | Read/Write: 80/20 | Sequential: 65% | Queue Depth: 16",
        correctLabel: "AI ML Training",
        generations: [
          { text: "Classification: OLAP Analytics\nReason: Medium-large block reads with sequential access.", reward: 0.0, advantage: -0.375 },
          { text: "Classification: AI ML Training\nReason: Sequential reads with moderate blocks could be training data.", reward: 1.0, advantage: 0.625 },
          { text: "Classification: OLAP Analytics\nReason: Read-heavy sequential access with 64K blocks.", reward: 0.0, advantage: -0.375 },
          { text: "Classification: AI ML Training\nReason: 64K sequential reads with moderate IOPS match distributed training data sharding.", reward: 1.0, advantage: 0.625 },
          { text: "Classification: OLAP Analytics\nReason: Analytical workload with partial sequential scans.", reward: 0.0, advantage: -0.375 },
          { text: "Classification: AI ML Training\nReason: Data loading for ML with some random access from data augmentation.", reward: 1.0, advantage: 0.625 },
          { text: "Classification: OLAP Analytics\nReason: Mixed sequential/random with large-ish blocks suggests analytics.", reward: 0.0, advantage: -0.375 },
          { text: "Classification: VDI Virtual Desktop\nReason: Moderate IOPS with mixed access pattern.", reward: 0.0, advantage: -0.375 }
        ],
        groupMeanReward: 0.375,
        groupStdReward: 0.484
      },
      {
        input: "IOPS: 3500 | Latency: 12ms | Block Size: 512K | Read/Write: 5/95 | Sequential: 95% | Queue Depth: 2",
        correctLabel: "Backup Archive",
        generations: [
          { text: "Classification: Backup Archive\nReason: Write-heavy sequential large blocks = backup stream.", reward: 1.0, advantage: 0.125 },
          { text: "Classification: Backup Archive\nReason: 95% writes, 512K blocks, sequential, low queue depth.", reward: 1.0, advantage: 0.125 },
          { text: "Classification: Backup Archive\nReason: Textbook backup: large sequential writes dominate.", reward: 1.0, advantage: 0.125 },
          { text: "Classification: Backup Archive\nReason: Low IOPS high-write sequential = archival write stream.", reward: 1.0, advantage: 0.125 },
          { text: "Classification: Backup Archive\nReason: Massive sequential writes with minimal reads.", reward: 1.0, advantage: 0.125 },
          { text: "Classification: Backup Archive\nReason: Writing backup data in 512K sequential chunks.", reward: 1.0, advantage: 0.125 },
          { text: "Classification: Video Streaming\nReason: Large sequential blocks.", reward: 0.0, advantage: -0.875 },
          { text: "Classification: Backup Archive\nReason: Sequential write-dominant pattern typical of backup jobs.", reward: 1.0, advantage: 0.125 }
        ],
        groupMeanReward: 0.875,
        groupStdReward: 0.331
      }
    ],

    lossCurve: generateGRPOLossCurve(),

    // Accuracy improvement over training
    accuracyOverTraining: [
      { step: 0, accuracy: 0.45 },
      { step: 10, accuracy: 0.47 },
      { step: 20, accuracy: 0.50 },
      { step: 30, accuracy: 0.53 },
      { step: 40, accuracy: 0.55 },
      { step: 50, accuracy: 0.58 },
      { step: 60, accuracy: 0.61 },
      { step: 70, accuracy: 0.63 },
      { step: 80, accuracy: 0.66 },
      { step: 90, accuracy: 0.68 },
      { step: 100, accuracy: 0.70 },
      { step: 110, accuracy: 0.71 },
      { step: 120, accuracy: 0.73 },
      { step: 130, accuracy: 0.74 },
      { step: 140, accuracy: 0.76 },
      { step: 150, accuracy: 0.77 },
      { step: 160, accuracy: 0.78 },
      { step: 170, accuracy: 0.79 },
      { step: 180, accuracy: 0.80 },
      { step: 190, accuracy: 0.81 },
      { step: 200, accuracy: 0.82 }
    ],

    comparisons: {
      1: {
        sftOutput: "OLTP Database",
        grpoOutput: "Classification: OLTP Database\nConfidence: High\nKey indicators: High random IOPS (45K) with small blocks (8K) and sub-millisecond latency are definitive OLTP transaction processing markers."
      },
      2: {
        sftOutput: "OLAP Analytics",
        grpoOutput: "Classification: OLAP Analytics\nConfidence: High\nKey indicators: Large block (128K) sequential reads dominating at 92% read ratio indicate analytical dataset scanning."
      },
      3: {
        sftOutput: "AI ML Training",
        grpoOutput: "Classification: AI ML Training\nConfidence: High\nKey indicators: 91% sequential access with 256K blocks and moderate IOPS match deep learning data pipeline patterns."
      },
      4: {
        sftOutput: "Video Streaming",
        grpoOutput: "Classification: Video Streaming\nConfidence: Very High\nKey indicators: 1M block sequential reads at near-100% read ratio with low queue depth are unmistakable media streaming characteristics."
      },
      5: {
        sftOutput: "VDI Virtual Desktop",
        grpoOutput: "Classification: VDI Virtual Desktop\nConfidence: High\nKey indicators: Balanced 60/40 read/write with moderate random IOPS and 16K blocks indicate aggregated desktop I/O from multiple concurrent users."
      },
      6: {
        sftOutput: "Backup Archive",
        grpoOutput: "Classification: Backup Archive\nConfidence: Very High\nKey indicators: 95% write dominance with large (512K) sequential blocks and minimal queue depth are textbook backup stream behavior."
      },
      7: {
        sftOutput: "OLAP Analytics",
        grpoOutput: "Classification: OLAP Analytics\nConfidence: High\nKey indicators: 256K block sequential reads at 88% read ratio match data warehouse analytical query patterns."
      },
      8: {
        sftOutput: "OLTP Database",
        grpoOutput: "Classification: VDI Virtual Desktop\nConfidence: Medium\nKey indicators: Near-balanced 55/45 read/write split with high random IOPS distinguishes this from OLTP, which typically shows 65%+ reads. The high queue depth (48) suggests many concurrent desktop sessions."
      },
      9: {
        sftOutput: "OLAP Analytics",
        grpoOutput: "Classification: AI ML Training\nConfidence: Medium\nKey indicators: 64K blocks with 65% sequential and 80% reads suggest distributed training data loading. Lower sequential rate than typical ML may indicate data augmentation or shuffled reading."
      },
      10: {
        sftOutput: "OLTP Database",
        grpoOutput: "Classification: OLTP Database\nConfidence: High\nKey indicators: 28K IOPS with 4K random blocks, 0.5ms latency, and high queue depth (64) indicate heavy transactional database workload."
      }
    },

    infrastructure: {
      gpuMemoryGB: 6.8,
      trainingTimeMinutes: 35,
      checkpointSizeMB: 1.7,
      peakGPUUtilization: 95,
      storageIOPattern: "Heavy compute bursts (generating 8 completions per prompt), periodic checkpoint writes",
      note: "8x generation per prompt means 8x the compute per training step vs SFT"
    }
  },

  // ===========================================================================
  // Combined pipeline results (RAG + fine-tuned model)
  // ===========================================================================
  combined: {
    results: {
      1: { response: "Classification: OLTP Database\nConfidence: Very High\nKey indicators: High random IOPS (45K), small blocks (8K), sub-ms latency. Reference patterns confirm transaction processing signature.", correct: true },
      2: { response: "Classification: OLAP Analytics\nConfidence: Very High\nKey indicators: Large block (128K) sequential reads at 92% read ratio match analytical scanning patterns in knowledge base.", correct: true },
      3: { response: "Classification: AI ML Training\nConfidence: Very High\nKey indicators: 256K sequential reads at 91% sequential rate. Retrieved references distinguish this from OLAP by the higher sequential rate and larger block size typical of data loaders.", correct: true },
      4: { response: "Classification: Video Streaming\nConfidence: Very High\nKey indicators: 1M block sequential reads, 99% read, matching streaming media reference patterns.", correct: true },
      5: { response: "Classification: VDI Virtual Desktop\nConfidence: Very High\nKey indicators: Balanced 60/40 read/write with 16K blocks and moderate random IOPS. Knowledge base confirms VDI aggregated desktop pattern.", correct: true },
      6: { response: "Classification: Backup Archive\nConfidence: Very High\nKey indicators: 95% write, 512K sequential blocks, low queue depth. Exact match with backup stream reference patterns.", correct: true },
      7: { response: "Classification: OLAP Analytics\nConfidence: Very High\nKey indicators: 256K sequential reads at 82% sequential. Retrieved context helps distinguish from AI ML Training by the moderate sequential rate and lower IOPS.", correct: true },
      8: { response: "Classification: VDI Virtual Desktop\nConfidence: High\nKey indicators: Near-balanced 55/45 read/write with high IOPS. Retrieved references show OLTP skews 65%+ reads while VDI shows more balanced I/O. Queue depth of 48 consistent with many concurrent desktop sessions.", correct: true },
      9: { response: "Classification: AI ML Training\nConfidence: High\nKey indicators: 64K blocks, 80% reads, 65% sequential. Retrieved context on distributed ML training with data sharding matches the moderate sequential rate. OLAP reference patterns show higher sequential rates for standard queries.", correct: true },
      10: { response: "Classification: OLTP Database\nConfidence: Very High\nKey indicators: 4K random blocks at 28K IOPS with 0.5ms latency. Knowledge base confirms high-IOPS small-block random pattern as OLTP.", correct: true }
    },

    // Progressive improvement: show how each technique improves the output
    progressiveComparison: [
      {
        exampleId: 1,
        label: "OLTP Database (Easy)",
        stages: {
          base: {
            output: "This storage I/O pattern shows high IOPS with small block sizes. It could be a database or VDI workload.",
            correct: false
          },
          fewShot: {
            output: "Classification: OLTP Database",
            correct: true
          },
          rag: {
            output: "Based on retrieved reference patterns, this matches OLTP Database. The 45000 IOPS with 8K blocks, 0.3ms latency match documented OLTP patterns.",
            correct: true
          },
          sft: {
            output: "OLTP Database",
            correct: true
          },
          dpo: {
            output: "Classification: OLTP Database\nConfidence: High\nKey indicators: High IOPS (45K) with small blocks (8K), sub-millisecond latency, and random access are hallmarks of transaction processing.",
            correct: true
          },
          grpo: {
            output: "Classification: OLTP Database\nConfidence: High\nKey indicators: High random IOPS (45K) with small blocks (8K) and sub-millisecond latency are definitive OLTP transaction processing markers.",
            correct: true
          },
          ragPlusGrpo: {
            output: "Classification: OLTP Database\nConfidence: Very High\nKey indicators: High random IOPS (45K), small blocks (8K), sub-ms latency. Reference patterns confirm transaction processing signature.",
            correct: true
          }
        }
      },
      {
        exampleId: 8,
        label: "VDI Virtual Desktop (Hard - ambiguous with OLTP)",
        stages: {
          base: {
            output: "High IOPS random access with small blocks. Looks like OLTP Database to me.",
            correct: false
          },
          fewShot: {
            output: "Classification: OLTP Database",
            correct: false
          },
          rag: {
            output: "This pattern shows characteristics of both VDI and OLTP. The balanced 55/45 read/write ratio is more typical of VDI. Classification: VDI Virtual Desktop.",
            correct: true
          },
          sft: {
            output: "OLTP Database",
            correct: false
          },
          dpo: {
            output: "Classification: VDI Virtual Desktop\nConfidence: Medium\nKey indicators: Near-balanced read/write (55/45) with high IOPS distinguishes from OLTP which skews read-heavy.",
            correct: true
          },
          grpo: {
            output: "Classification: VDI Virtual Desktop\nConfidence: Medium\nKey indicators: Near-balanced 55/45 read/write split with high random IOPS distinguishes from OLTP. High queue depth (48) suggests many concurrent desktop sessions.",
            correct: true
          },
          ragPlusGrpo: {
            output: "Classification: VDI Virtual Desktop\nConfidence: High\nKey indicators: Near-balanced 55/45 read/write with high IOPS. Retrieved references show OLTP skews 65%+ reads while VDI shows more balanced I/O. Queue depth of 48 consistent with many concurrent desktop sessions.",
            correct: true
          }
        }
      },
      {
        exampleId: 9,
        label: "AI ML Training (Hard - ambiguous with OLAP)",
        stages: {
          base: {
            output: "Medium blocks with moderate sequential access. Could be analytics or ML training.",
            correct: false
          },
          fewShot: {
            output: "Classification: OLAP Analytics",
            correct: false
          },
          rag: {
            output: "This is somewhat ambiguous between OLAP Analytics and AI ML Training. I'll classify as OLAP Analytics due to the moderate sequential percentage.",
            correct: false
          },
          sft: {
            output: "OLAP Analytics",
            correct: false
          },
          dpo: {
            output: "Classification: AI ML Training\nConfidence: Medium\nKey indicators: Moderate sequential reads with medium blocks suggest data pipeline loading with distributed training data sharding.",
            correct: true
          },
          grpo: {
            output: "Classification: AI ML Training\nConfidence: Medium\nKey indicators: 64K blocks with 65% sequential and 80% reads suggest distributed training data loading. Lower sequential rate may indicate data augmentation.",
            correct: true
          },
          ragPlusGrpo: {
            output: "Classification: AI ML Training\nConfidence: High\nKey indicators: 64K blocks, 80% reads, 65% sequential. Retrieved context on distributed ML training with data sharding matches the moderate sequential rate. OLAP reference patterns show higher sequential rates for standard queries.",
            correct: true
          }
        }
      }
    ]
  },

  // ===========================================================================
  // Infrastructure comparison across all techniques
  // ===========================================================================
  infrastructureSummary: {
    techniques: [
      { name: "Prompt Engineering", gpuMemory: "N/A (inference only)", trainingTime: "N/A", costRelative: 1, modelsInMemory: 1 },
      { name: "RAG", gpuMemory: "N/A + vector DB", trainingTime: "Index build: ~5min", costRelative: 1.2, modelsInMemory: 1 },
      { name: "SFT (LoRA)", gpuMemory: "4.2 GB", trainingTime: "12 min", costRelative: 3, modelsInMemory: 1 },
      { name: "DPO (LoRA)", gpuMemory: "5.1 GB", trainingTime: "8 min", costRelative: 4, modelsInMemory: 1 },
      { name: "RLHF (PPO)", gpuMemory: "12.8 GB", trainingTime: "45 min", costRelative: 10, modelsInMemory: 3 },
      { name: "GRPO", gpuMemory: "6.8 GB", trainingTime: "35 min", costRelative: 7, modelsInMemory: 1 }
    ],
    scalingNote: "All numbers above are for SmolLM2-360M. For a 7B model, multiply GPU memory by ~20x and training time by ~15x. For 70B, multiply by ~200x and ~150x."
  }
};
