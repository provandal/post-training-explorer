import { useState } from 'react'
import ModelOutput from '../components/ModelOutput'
import InfrastructureCard from '../components/InfrastructureCard'

const EXAMPLES = [
  {
    id: 1,
    input: "IOPS: 45000 | Latency: 0.3ms | Block Size: 8K | Read/Write: 70/30 | Sequential: 15% | Queue Depth: 32",
    label: "OLTP Database",
    difficulty: "Easy",
    baseOutput: `This appears to be some kind of storage performance metrics. The IOPS value is 45000 which is relatively high. The latency is 0.3ms which is quite low. The block size is 8K. There is a 70/30 read to write ratio with 15% sequential access and a queue depth of 32. These metrics could be from various types of storage workloads depending on the specific use case and configuration being measured.`,
    sftOutput: `Classification: OLTP Database
Confidence: High
Key indicators: High IOPS (45K) with very low latency (0.3ms) and small block size (8K) are hallmarks of transaction processing. Random-dominant access (85%) with moderate read bias confirms OLTP pattern.`,
    baseCorrect: false,
    sftCorrect: true,
  },
  {
    id: 2,
    input: "IOPS: 12000 | Latency: 4.5ms | Block Size: 128K | Read/Write: 92/8 | Sequential: 78% | Queue Depth: 8",
    label: "OLAP Analytics",
    difficulty: "Medium",
    baseOutput: `The storage metrics show 12000 IOPS with a latency of 4.5ms. The block size is 128K which is larger than usual. The read/write ratio is heavily skewed towards reads at 92/8, with 78% sequential access. Queue depth is 8. This looks like it could be some kind of read-heavy sequential workload, possibly related to data processing or analytics.`,
    sftOutput: `Classification: OLAP Analytics
Confidence: High
Key indicators: Large block sequential reads (128K, 78% sequential) with heavy read bias (92/8) indicate analytical query processing. Moderate IOPS with higher latency tolerance is typical of scan-heavy warehouse workloads.`,
    baseCorrect: false,
    sftCorrect: true,
  },
  {
    id: 3,
    input: "IOPS: 38000 | Latency: 0.5ms | Block Size: 8K | Read/Write: 55/45 | Sequential: 22% | Queue Depth: 64",
    label: "VDI Virtual Desktop",
    difficulty: "Hard",
    baseOutput: `These storage metrics indicate a workload with 38000 IOPS at 0.5ms latency. Block size is 8K, read/write ratio is nearly balanced at 55/45, with 22% sequential access and a queue depth of 64. The high IOPS and low latency suggest this is a performance-sensitive database application, possibly OLTP or a similar transactional workload.`,
    sftOutput: `Classification: VDI Virtual Desktop
Confidence: Medium
Key indicators: Nearly balanced R/W ratio (55/45) distinguishes this from OLTP despite similar IOPS and block size. High queue depth (64) and moderate sequential ratio (22%) are characteristic of multi-user desktop virtualization with mixed application workloads.`,
    baseCorrect: false,
    sftCorrect: true,
  },
]

const SFT_INFRA = {
  gpuMemoryGB: 4.2,
  trainingTimeMinutes: 12,
  checkpointSizeMB: 1.7,
  peakGPUUtilization: 87,
  storageIOPattern: "Bursty checkpoint writes every ~50 steps, steady data reads",
  note: "LoRA adapter is only 1.7 MB. The full model (720 MB) stays frozen. This is why PEFT changed everything — you can fine-tune on a single consumer GPU."
}

export default function SFTComparison({ explore = false }) {
  const [selectedExample, setSelectedExample] = useState(0)
  const [showInfra, setShowInfra] = useState(false)
  const ex = EXAMPLES[selectedExample]

  return (
    <div className="max-w-5xl mx-auto">
      {/* Example selector */}
      <div className="flex gap-2 mb-4">
        {EXAMPLES.map((e, i) => (
          <button
            key={i}
            onClick={() => setSelectedExample(i)}
            className={`px-3 py-1.5 text-xs rounded-md transition-colors ${
              i === selectedExample
                ? 'bg-violet-600 text-white'
                : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
            }`}
          >
            {e.label} ({e.difficulty})
          </button>
        ))}
      </div>

      {/* Input */}
      <div className="mb-4 bg-slate-800 border border-slate-600 rounded-lg p-3 font-mono text-sm text-slate-200">
        {ex.input}
      </div>

      {/* Side by side comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <ModelOutput
          label="Base Model (no training)"
          text={ex.baseOutput}
          variant="base"
          isCorrect={ex.baseCorrect}
        />
        <ModelOutput
          label="After SFT (1,400 examples)"
          text={ex.sftOutput}
          variant="sft"
          isCorrect={ex.sftCorrect}
        />
      </div>

      {/* Training config summary */}
      <div className="mt-4 p-3 rounded-lg bg-violet-950/20 border border-violet-800/30">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-center">
          <div>
            <div className="text-lg font-bold text-violet-400">1,400</div>
            <div className="text-xs text-slate-500">training examples</div>
          </div>
          <div>
            <div className="text-lg font-bold text-violet-400">3</div>
            <div className="text-xs text-slate-500">epochs</div>
          </div>
          <div>
            <div className="text-lg font-bold text-violet-400">0.12%</div>
            <div className="text-xs text-slate-500">params trained (LoRA)</div>
          </div>
          <div>
            <div className="text-lg font-bold text-violet-400">12 min</div>
            <div className="text-xs text-slate-500">training time</div>
          </div>
        </div>
      </div>

      {/* Analogy */}
      <p className="mt-4 text-sm text-slate-400 italic">
        Think of SFT like training a new team member with an example handbook: "When you see a pattern like this,
        classify it like that." After 1,400 examples, the model has internalized the patterns.
      </p>

      {/* Infrastructure */}
      <button
        onClick={() => setShowInfra(!showInfra)}
        className="mt-4 text-sm text-violet-400 hover:text-violet-300 underline underline-offset-4"
      >
        {showInfra ? 'Hide' : 'Show'} infrastructure profile
      </button>
      {showInfra && (
        <div className="mt-3">
          <InfrastructureCard data={SFT_INFRA} />
        </div>
      )}
    </div>
  )
}
