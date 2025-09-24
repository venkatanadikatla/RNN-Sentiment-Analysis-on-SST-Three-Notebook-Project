# RNN-Sentiment-Analysis-on-SST-Three-Notebook-Project
Three related notebooks for Stanford Sentiment Treebank sentiment classification: (1) baseline RNN, (2) reusable training/evaluation utilities, and (3) improved BiLSTM achieving ~62–63% test accuracy (vs ~56–57% baseline). Colab-ready; uses legacy TorchText (Field/BucketIterator).
# Build a consolidated README.md for the three uploaded notebooks and save it for download.
repo_title = "RNN Sentiment Analysis on SST — Three-Notebook Project"
short_description = (
    "Three related notebooks covering a baseline RNN, reusable training/evaluation utilities, "
    "and an improved BiLSTM for sentiment analysis on the Stanford Sentiment Treebank (SST)."
)

readme_md = f"""# {repo_title}

> {short_description}

This repo combines **three sub-projects** into a single GitLab project. Each notebook can be run on its own,
but the README ties them together and explains how they build on one another.

---

## Repository Structure
├── 01_baseline/
│ └── RNN_Baseline_Model.ipynb
├── 02_utils/
│ └── RNN_Building_Training_and_Evaluation_Functions.ipynb
├── 03_better_accuracy/
│ └── RNN_with_SST_Better_Accuracy.ipynb
└── README.md ← you are here

> **Tip:** On GitLab, keep the notebooks in the folders shown above (or similar) so it’s obvious there are
> three distinct sub-projects while still living in one repo.

---

## What’s Inside (TL;DR)

| Sub-Project | Notebook | Goal | Key Ideas | Result (Test Acc.)* |
|---|---|---|---|---|
| 1. Baseline RNN | `RNN_Baseline_Model.ipynb` | Establish a minimal sentiment model on **SST** | PyTorch + TorchText (legacy `Field`/`BucketIterator`), vanilla LSTM | **~56.6%** |
| 2. Training Utils | `RNN_Building_Training_and_Evaluation_Functions.ipynb` | Factor out training/eval boilerplate | `train()` loop, metrics, early-stopping scaffolding, reproducibility helpers | N/A (utility notebook) |
| 3. Better Accuracy | `RNN_with_SST_Better_Accuracy.ipynb` | Improve baseline performance | **BiLSTM**, dropout, tuned hparams/optimizer | **~62.6–62.8%** |

\\* Reported numbers are taken from the notebooks’ own output cells. Expect small variation run-to-run.

---

## Dataset

- **Stanford Sentiment Treebank (SST)**. The notebooks are set up to work with TorchText’s SST loader
  (legacy API) and are configured for *sentence-level* sentiment.
- Labels: the baseline notes **positive / negative / neutral** (3-class). If you prefer binary SST-2,
  you can adapt the TorchText dataset flags in the loading cell.

> If SST download fails behind a firewall, download locally and point the TorchText dataset root to your path.

---

## Environments & Requirements

These notebooks use the **legacy TorchText API** (`Field`, `BucketIterator`), which pairs best with older
PyTorch/TorchText versions. They run well on **Google Colab** with a few pins.

### Option A — Google Colab (recommended)
Just open the notebooks and run the setup cells. If you need to pin versions explicitly, use:
```bash
pip install torch==1.7.1 torchtext==0.6.0 spacy==2.3.9
python -m spacy download en_core_web_sm


For utils, skim and run to register functions in the kernel.

For better-accuracy, ensure the utils have been executed (if shared in session), then train.

Compare your test accuracy with the numbers above. Small fluctuations are normal.


## How the Three Notebooks Fit Together

### 1) Baseline — `RNN_Baseline_Model.ipynb`
**Goal:** Minimal sentiment classifier to establish a performance floor.

- **Model:** Single-layer LSTM with embedding → LSTM → linear classifier.  
- **Data:** TorchText legacy pipeline (`Field`, `LabelField`, `BucketIterator`) on SST.  
- **Training:** Straightforward loop, CrossEntropy loss, Adam optimizer.  
- **Result:** ~56.6% test accuracy (3-class as configured).  

**Why it matters:** Provides a clean starting point and a reference implementation for the rest.

---

### 2) Utilities — `RNN_Building_Training_and_Evaluation_Functions.ipynb`
**Goal:** Remove repetition and standardize the workflow.

**What it adds:**
- `train()` and evaluation scaffolding (epoch loop, running loss/accuracy).  
- Metric helpers (accuracy) and clean logging.  
- Reproducibility (seed-setting) and small quality-of-life helpers.  

**Outcome:** A drop-in training/eval utility you can reuse in the baseline and improved models.  

**Why it matters:** Makes experiments repeatable and easier to compare.

---

### 3) Better Accuracy — `RNN_with_SST_Better_Accuracy.ipynb`
**Goal:** Beat the baseline with principled architectural & training changes.

- **Model:** BiLSTM (bidirectional LSTM) with dropout; linear head over concatenated hidden states.  
- **Training:** Tuned learning rate/optimizer; reuses standardized loops from the utils notebook.  
- **Result:** ~62.6–62.8% test accuracy, a clear gain over baseline.  

**What changed vs baseline (high-level):**
- Bidirectionality adds backward context → better sentence representation.  
- Regularization via dropout to curb overfitting.  
- Minor hyperparameter/optimizer improvements (learning rate, scheduler choice).  

---

## Reproducing Results
1. Open the notebook in **Google Colab** (GPU runtime recommended).  
2. Run the setup cell(s) to install the pinned versions (if needed).  
3. Execute all cells top-to-bottom:  
   - For the baseline, verify SST downloads and vocabulary build complete.  

---

## Summary
This project demonstrates the evolution of an RNN-based sentiment analysis pipeline on the SST dataset:  
- From a **baseline LSTM (~56.6%)** → to **refactored training utilities** → to a **BiLSTM with dropout (~62.8%)**.  
It highlights how principled architectural tweaks and standardized workflows can significantly improve performance.
"""
