# Post-Training Explorer: End-to-End Walkthrough

Complete instructions to go from code to working demo. The arc is:
**Tour (learn) → Explore (browse real data) → Try It (run inference in your browser)**

---

## Prerequisites

- Node.js 18+ installed locally
- A Google account (for Colab)
- A HuggingFace account (free — for hosting ONNX models)

---

## Phase 1: Verify the App Works in Fallback Mode

No training data needed yet. This confirms the app is structurally sound.

### 1.1 Start the dev server

```bash
cd app
npm install
npm run dev
```

Open `http://localhost:5173`.

### 1.2 Test the landing page

You should see:
- A quadrant map visualization
- Two buttons: "Start Guided Tour" and "Explore Freely"
- SNIA branding at the bottom

### 1.3 Walk the tour

Click **"Start Guided Tour"** and navigate through all stops using the **Next** button or **→** arrow key.

What to check:
- All 13+ stops render without white screens or crashes
- Each stop shows hardcoded fallback data (charts, text, animations)
- The **PatternPicker** component is hidden (it only appears when real data is loaded)
- The console shows `[loadArtifacts] precomputed_results.json not available` — this is expected

### 1.4 Test explore mode

Click **"Explore Freely"** (from the landing page or the Epilogue stop). Click different quadrants in the sidebar. Components should load for each section.

The **LiveInferencePanel** at the bottom will show a message about needing models — this is expected until Phase 3.

### 1.5 Verify the production build

```bash
npm run build
```

Should complete successfully. Expected output: ~450KB JS, ~50KB CSS.

**Phase 1 pass criteria:** No crashes, no white screens, all stops render, navigation works, build succeeds.

---

## Phase 2: Train Models on Google Colab

This runs the SFT → DPO → GRPO training pipeline on a free GPU and produces the `precomputed_results.json` file that bridges the training world to the web app.

### 2.1 Open the notebook

Upload or open `notebooks/Post_Training_Pipeline.ipynb` in [Google Colab](https://colab.research.google.com/).

You'll need the project files accessible in Colab. Options:
- **Zip upload:** Zip the project, upload to Colab, uncomment the `unzip` line in Cell 1
- **GitHub clone:** Push to a repo, uncomment the `git clone` line in Cell 1
- **Manual upload:** Use Colab's file browser to upload the `scripts/` directory

### 2.2 Set runtime to GPU

Go to **Runtime → Change runtime type → T4 GPU**.

### 2.3 Run cells 1-2: Setup + GPU check

- **Cell 1** installs dependencies, sets `PROJECT_ROOT`, verifies scripts are present
- **Cell 2** confirms GPU availability — you should see "Tesla T4, ~15.0 GB"

If GPU is not detected, check that you changed the runtime type.

### 2.4 Run Cell 4: SFT Training (~12 min)

This is the foundation step. Watch for:
- Training loss dropping from ~2.8 → ~0.3
- "Before" and "after" model outputs showing clear improvement
- Output saved to `scripts/outputs/sft/adapter/`

### 2.5 Run Cell 6: DPO Training (~8 min)

Refines the model's response style. Watch for:
- Reward margins increasing over training
- Output saved to `scripts/outputs/dpo/adapter/`

### 2.6 Run Cell 8: GRPO Training (~35 min)

This is the longest step — online RL with a binary reward function. Watch for:
- Accuracy climbing from ~45% → ~85%+
- Reward scores increasing across training steps
- Output saved to `scripts/outputs/grpo/adapter/`

> **If Colab disconnects during GRPO:** Your SFT and DPO adapters are already saved.
> Reconnect, re-run Cell 1 (setup only), then run Cell 8 (GRPO) directly.
> The script detects the existing SFT adapter automatically.

### 2.7 Run Cell 10: Export Artifacts (~3 min)

This loads all four model variants (base, SFT, DPO, GRPO), runs inference on 20 standardized test prompts, and packages everything into one JSON file.

Output: `app/public/data/precomputed_results.json`

### 2.8 Run Cell 14: Validate

Check the printed output. You should see:
```
Accuracy summary:
    base: ~15%  (3/20)
     sft: ~65%+ (13/20)
     dpo: ~65%+ (13/20)
    grpo: ~85%+ (17/20)
```

All training artifact sections should show "yes" (not "MISSING").

### 2.9 Run Cell 15: Download

This triggers a browser download of `precomputed_results.json`. Save it.

### 2.10 Place the file in your local project

Copy the downloaded file to:

```
app/public/data/precomputed_results.json
```

**Phase 2 pass criteria:** Validation cell shows 4 model variants, accuracy progression (base < sft ≤ dpo < grpo), all artifact sections present.

---

## Phase 3: Verify the App With Real Data

### 3.1 Restart the dev server

```bash
cd app
npm run dev
```

### 3.2 Walk the tour again

This time, every stop shows **real model outputs** instead of fallback data. Stop-by-stop:

| Stop | What changes with real data |
|------|----------------------------|
| PromptBasic | Real base model token probabilities |
| SFTComparison | Real loss curve, real LoRA weight heatmap, real before/after outputs |
| DPOPreferences | Real chosen/rejected log-prob shifts |
| GRPOGenerations | Real 8-completion groups with rewards, real accuracy/reward curves |
| CombinedResults | Real outputs for all stages, real accuracy summary |
| InfrastructureSummary | Real training times and checkpoint sizes from your GPU run |

### 3.3 Test PatternPicker

The **PatternPicker** component now appears — a dropdown with 20 test prompts. Selecting different prompts should instantly update the displayed model outputs. This component was hidden in Phase 1.

### 3.4 Verify the build

```bash
npm run build
```

The `precomputed_results.json` file gets auto-copied into `dist/data/`.

**Phase 3 pass criteria:** All stops show real data. PatternPicker works. No console errors. Build succeeds.

---

## Phase 4: ONNX Conversion for Browser Inference

This is what makes the capstone experience possible. You'll convert the base and GRPO models to ONNX format and host them on HuggingFace Hub so browsers can download and run them directly.

### 4.1 Get a HuggingFace access token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token with **Write** access
3. Copy the token — you'll paste it in the next step

### 4.2 Run the ONNX conversion cell in Colab

Back in the Colab notebook, run **Cell 12** (Step 5: Convert to ONNX).

This cell:
1. Installs `optimum` and `onnxruntime`
2. Prompts you to paste your HuggingFace token
3. Runs `scripts/convert_to_onnx.py`, which:
   - Saves the unmodified base model
   - Merges SFT + GRPO adapters into a single model
   - Converts both to ONNX format
   - Pushes both to HuggingFace Hub

When complete, it prints the two repo IDs:

```
  Base ONNX model: https://huggingface.co/<your-username>/smollm2-360m-storage-io-base-onnx
  GRPO ONNX model: https://huggingface.co/<your-username>/smollm2-360m-storage-io-grpo-onnx
```

### 4.3 Verify the repos exist

Open both URLs in your browser. Each repo should contain ONNX model files.

### 4.4 Update inference.js with your real repo IDs

Open `app/src/services/inference.js` and replace the placeholder model IDs (lines 7 and 11):

```javascript
const MODEL_CONFIGS = {
  base: {
    id: '<your-username>/smollm2-360m-storage-io-base-onnx',  // ← your actual HF username
    label: 'Base Model (untrained)',
  },
  grpo: {
    id: '<your-username>/smollm2-360m-storage-io-grpo-onnx',  // ← your actual HF username
    label: 'GRPO Fine-tuned',
  },
}
```

**Phase 4 pass criteria:** Both HuggingFace repos exist with ONNX model files. `inference.js` has real repo IDs.

---

## Phase 5: The Capstone — Live Browser Inference

This is the moment everything builds to.

### 5.1 Start the dev server

```bash
cd app
npm run dev
```

### 5.2 Navigate to LiveInferencePanel

Either:
- Walk the full tour → reach Epilogue → click "Explore Freely" → scroll to bottom
- Or click "Explore Freely" from landing → scroll to bottom

The **LiveInferencePanel** appears with two model buttons: Base Model and GRPO Fine-tuned.

### 5.3 Download the base model

1. Select **"Base Model"**
2. Click **"Download Model (~180MB)"**
3. Watch the progress bar as the ONNX model downloads
4. Status changes to **"Ready"** when complete

The model is cached in the browser's Cache API. Future visits skip the download.

### 5.4 Run inference with the base model

1. Enter a custom I/O pattern in the text input, or use the default
2. Click **"Classify"**
3. See the base model's output — it should be confused, rambling, or incorrect

### 5.5 Download and run the GRPO model

1. Switch to **"GRPO Fine-tuned"**
2. Download that model (also ~180MB, also cached)
3. Enter the **same I/O pattern** as before
4. Click **"Classify"**
5. See a crisp, correct classification with concise reasoning

**That's the demo.** The user just experienced the difference that post-training makes — running entirely in their browser, no GPU required.

### 5.6 Verify caching

Reload the page. Select a model you already downloaded. It should load near-instantly from cache (no re-download).

**Phase 5 pass criteria:**
- Both models download successfully with progress bar
- Base model produces incoherent/incorrect output
- GRPO model produces correct classification with concise reasoning
- Models load from cache on reload (near-instant)

---

## Phase 6: Final Polish

### 6.1 Update Epilogue links

Open `app/src/stops/Epilogue.jsx` and replace the three placeholder URLs (lines 9-11):

```javascript
{ label: 'Training Notebooks (Colab)', url: '<YOUR COLAB LINK>', description: '...', placeholder: false },
{ label: 'Storage I/O Dataset', url: 'https://huggingface.co/datasets/<your-username>/storage-io-workload', description: '...', placeholder: false },
{ label: 'Pre-trained Models', url: 'https://huggingface.co/<your-username>', description: '...', placeholder: false },
```

Remove `placeholder: true` from each item so the links become clickable instead of grayed out.

### 6.2 Final build + preview

```bash
cd app
npm run build
npx vite preview
```

Open the preview URL and do a full end-to-end walkthrough:
1. Landing page loads
2. Tour navigates through all stops with real data
3. PatternPicker switches between prompts
4. Epilogue links are clickable
5. Explore mode works for all quadrants
6. LiveInferencePanel downloads models and runs inference

### 6.3 Deploy

Deploy the `app/dist/` directory to your hosting platform (GitHub Pages, Netlify, Vercel, etc.).

---

## Troubleshooting

### "precomputed_results.json not available" in console
Expected before Phase 2. The app uses fallback data. After training, place the file at `app/public/data/precomputed_results.json`.

### GRPO training times out on Colab
SFT and DPO adapters are already saved. Reconnect, re-run Cell 1 (setup), then run only Cell 8 (GRPO) and Cell 10 (export).

### ONNX conversion fails
Make sure `optimum` and `onnxruntime` installed correctly. Check that both SFT and GRPO adapters exist in `scripts/outputs/`.

### LiveInferencePanel shows "model not found" error
Verify the repo IDs in `inference.js` match real HuggingFace repos. The repos must be public.

### Model download hangs or fails in browser
Check browser console for CORS errors. HuggingFace Hub serves CORS headers by default, but some corporate proxies block them. Try a different network.

### Build fails after adding @huggingface/transformers
Run `npm install` again. The package should be in `package.json` dependencies. `inference.js` uses a dynamic import so the package is only loaded when the user actually clicks to download a model.

---

## File Reference

| File | Purpose |
|------|---------|
| `scripts/train_sft.py` | Step 1: Supervised fine-tuning with LoRA |
| `scripts/train_dpo.py` | Step 2: Direct preference optimization |
| `scripts/train_grpo.py` | Step 3: Group relative policy optimization |
| `scripts/export_artifacts.py` | Step 4: Export all results to JSON |
| `scripts/convert_to_onnx.py` | Step 5: Convert to ONNX + push to HF Hub |
| `notebooks/Post_Training_Pipeline.ipynb` | Colab notebook running all 5 steps |
| `app/public/data/precomputed_results.json` | Bridge from training to web app (generated) |
| `app/src/data/loadArtifacts.js` | Loads precomputed_results.json, 30+ accessor functions |
| `app/src/services/inference.js` | Client-side ONNX model loading + inference |
| `app/src/components/LiveInferencePanel.jsx` | The capstone inference UI |
| `app/src/stops/Epilogue.jsx` | Final tour stop with resource links |
