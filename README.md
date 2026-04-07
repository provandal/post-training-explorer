# Post-Training Explorer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

An interactive educational tool that teaches LLM post-training techniques through a hands-on storage I/O workload classification task. Built for the [SNIA](https://www.snia.org/) DSN Post-Training Webinar.

**[Live Demo](https://provandal.github.io/post-training-explorer/)**

## What You'll Learn

The guided tour walks through a progression of techniques, each building on the last:

1. **Prompt Engineering** -- basic prompts, few-shot learning, and their limitations
2. **RAG** -- retrieval-augmented generation with vector search
3. **SFT** -- supervised fine-tuning with LoRA adapters
4. **DPO** -- direct preference optimization
5. **GRPO** -- group relative policy optimization (reinforcement learning)

Each stop includes interactive demos, visualizations, and "under the covers" deep dives into how the techniques work.

## Project Structure

```
app/          React frontend (Vite + Tailwind + Zustand + D3)
scripts/      Python training pipeline (SFT, DPO, GRPO)
notebooks/    Jupyter notebooks for Colab
```

## Quick Start

```bash
cd app
npm install
npm run dev
```

The app runs entirely in the browser with precomputed data -- no backend or GPU required.

## Training Pipeline

To train your own models, see [`scripts/README.md`](scripts/README.md) for the full pipeline:

1. Generate synthetic dataset
2. SFT with LoRA
3. DPO alignment
4. GRPO reinforcement learning
5. Export artifacts
6. Convert to ONNX for browser inference

All scripts run on a free Colab T4 GPU. See [`WALKTHROUGH.md`](WALKTHROUGH.md) for detailed instructions.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| [Post_Training_Pipeline.ipynb](notebooks/Post_Training_Pipeline.ipynb) | Full SFT, DPO, GRPO pipeline |
| [Traditional_ML_Comparison.ipynb](notebooks/Traditional_ML_Comparison.ipynb) | XGBoost/RF baseline comparison |
| [Realistic_LLM_Use_Case.ipynb](notebooks/Realistic_LLM_Use_Case.ipynb) | Unstructured error log classification |

## Acknowledgments

The four-quadrant optimization framework used throughout this app is adapted from ["A Survey of Techniques for Maximizing LLM Performance"](https://www.youtube.com/watch?v=ahnGLM-RC1Y) by Colin Jarvis and John Allard, presented at OpenAI DevDay (November 2023). Also available in the [OpenAI documentation](https://developers.openai.com/api/docs/guides/optimizing-llm-accuracy).

## License

[MIT](./LICENSE)
