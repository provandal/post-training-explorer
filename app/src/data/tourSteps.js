// Tour step definitions - each step maps to a quadrant position,
// narration text, and which interactive component to show.

const tourSteps = [
  // === INTRO ===
  {
    id: 'welcome',
    quadrant: null,
    title: 'Welcome to the Post-Training Explorer',
    narration: `You have an LLM. It knows a lot about the world, but it doesn't know anything about YOUR world — your storage patterns, your infrastructure, your team's needs. This guided tour will walk you through the techniques used to change that, from the simplest (prompting) to the most advanced (reinforcement learning). At each stop, you'll try the technique yourself and see what's happening under the covers.`,
    component: 'Welcome',
  },

  // === PROMPT ENGINEERING (Lower-Left) ===
  {
    id: 'prompt-basic',
    quadrant: 'prompt',
    subStop: null,
    zigzagPosition: 0,
    title: 'Prompt Engineering: Basic Prompt',
    narration: `Let's start with the simplest approach: just asking the model a question. Here's a storage I/O pattern — can the model classify it? Try typing your own prompt or use the default one. This is a base model with no special training on storage workloads.`,
    component: 'PromptBasic',
  },
  {
    id: 'prompt-fewshot',
    quadrant: 'prompt',
    subStop: null,
    zigzagPosition: 1,
    title: 'Prompt Engineering: Few-Shot Learning',
    narration: `Better idea — what if we show the model some examples first? This is "few-shot" prompting: we include labeled examples right in the prompt. Watch how the model's accuracy improves. But also watch the token probabilities — the model is more confident, but still not fully convinced.`,
    component: 'PromptFewShot',
  },
  {
    id: 'prompt-limitation',
    quadrant: 'prompt',
    subStop: null,
    zigzagPosition: 1,
    title: 'The Ceiling of Prompting',
    narration: `Here's where prompting breaks down. This I/O pattern is ambiguous — it could be VDI or OLTP. Even with examples in the prompt, the model hedges. And there's a bigger problem: those few-shot examples disappear after every conversation. You're burning context window space to teach the model something it should just KNOW. We need something more permanent.`,
    component: 'PromptLimitation',
  },

  // === RAG (Upper-Left) ===
  {
    id: 'rag-simple',
    quadrant: 'rag',
    subStop: null,
    zigzagPosition: 2,
    title: 'RAG: Simple Retrieval',
    narration: `Instead of cramming examples into the prompt, what if we gave the model a reference library? RAG (Retrieval-Augmented Generation) searches a knowledge base for patterns similar to your input, then includes the most relevant matches in the prompt. The model now has expert-level context — without using up your few-shot budget.`,
    component: 'RAGSimple',
  },
  {
    id: 'rag-limitation',
    quadrant: 'rag',
    subStop: null,
    zigzagPosition: 3,
    title: 'The Ceiling of RAG',
    narration: `RAG gave the model the right information. It correctly identified this as an OLTP workload. But look at the output — it's verbose, hedging, and doesn't follow the format your team needs. The model KNOWS the answer but doesn't ACT the way you need. RAG changes what the model sees. To change how the model behaves, we need to change the model itself.`,
    component: 'RAGLimitation',
  },

  // === POST-TRAINING: SFT (Lower-Right) ===
  {
    id: 'sft',
    quadrant: 'posttraining',
    subStop: 'sft',
    zigzagPosition: 4,
    title: 'SFT: Teaching by Example',
    narration: `Supervised Fine-Tuning is like giving the model a training manual. We showed it 1,400 examples of "here's an I/O pattern, here's the correct classification." Use the tabs to explore the problem, learn how SFT works, see before/after comparisons, and dig into the token probabilities and LoRA weights under the covers.`,
    component: 'SFTComparison',
  },

  // === POST-TRAINING: DPO (Lower-Right, sub-stop) ===
  {
    id: 'dpo',
    quadrant: 'posttraining',
    subStop: 'dpo',
    zigzagPosition: 5,
    title: 'DPO: Learning from Your Preferences',
    narration: `SFT taught the model WHAT to say. DPO teaches it HOW to say it. Both outputs might be correct, but one is better — more concise, more confident. Pick your preferred response and see how DPO uses pairs like these to reshape the model's style directly — no reward model needed.`,
    component: 'DPOPreferences',
  },

  // === POST-TRAINING: GRPO (Lower-Right, sub-stop) ===
  {
    id: 'grpo',
    quadrant: 'posttraining',
    subStop: 'grpo',
    zigzagPosition: 6,
    title: 'GRPO: Self-Improving Reasoning',
    narration: `This is the frontier — the technique behind DeepSeek R1. Instead of human preferences, GRPO uses verifiable rewards. The model generates 8 attempts, scores each one, and learns from the group statistics. No human in the loop. The answer itself is the teacher. Explore the tabs to see how it works.`,
    component: 'GRPOGenerations',
  },

  // === ALL OPTIONS (Upper-Right) ===
  {
    id: 'combined',
    quadrant: 'alloptions',
    subStop: null,
    zigzagPosition: 7,
    title: 'Putting It All Together',
    narration: `The upper-right quadrant is the destination: a model that's been fine-tuned AND gets the right context at inference time. Watch the progressive improvement: base model → few-shot → RAG → SFT → DPO → GRPO → RAG + GRPO. Each technique built on the last. This is what a mature enterprise AI deployment looks like.`,
    component: 'CombinedResults',
  },
  {
    id: 'infrastructure',
    quadrant: 'alloptions',
    subStop: null,
    zigzagPosition: 7,
    title: 'The Infrastructure Story',
    narration: `Every technique you just experienced has a different infrastructure footprint. Prompting costs nothing extra. RAG needs a vector database. SFT needs a few GB of GPU memory for a few minutes. GRPO needs sustained GPU bursts. And RLHF — the technique we skipped over with DPO — needs three models in memory simultaneously. This is where YOUR expertise matters: understanding these tradeoffs is how storage and infrastructure teams create real value.`,
    component: 'InfrastructureSummary',
  },

  // === EPILOGUE ===
  {
    id: 'epilogue',
    quadrant: null,
    title: 'Your Turn',
    narration: `Everything you just saw is real. The models are on HuggingFace. The training code is in Colab notebooks. The dataset is open source. You can run this yourself on a single GPU in under an hour. Click "Explore Freely" to go back to any section, try your own I/O patterns, and dig deeper. Or follow the links below to start your own post-training journey.`,
    component: 'Epilogue',
  },
]

export default tourSteps
