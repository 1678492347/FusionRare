# FusionRare: A Unified Multiparadigm Fusion Intelligence System for Rare Disease Diagnosis

[![Web Application](https://img.shields.io/badge/Web_App-Available-blue)](http://rdx.nbscn.org/) 
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


> [NOTE]
> **Project Status Update (October 2026)**:  
> The core functionalities of FusionRare are now fully available in this repository. The development team is currently conducting final-stage environment configuration and dependency debugging to streamline local deployment and testing for external researchers. An optimized release is scheduled for **October 31, 2026**.

## 📖 Overview

**FusionRare** is a unified multiparadigm fusion framework designed to prioritize and diagnose rare diseases (RDs). Rare diseases afflict an estimated 300 million people globally, often leaving patients to navigate a protracted "diagnostic odyssey". To address clinical scarcity and fragmented medical knowledge, FusionRare integrates heterogeneous artificial intelligence paradigms under strict anti-leakage protocols. By synthesizing complementary reasoning modalities, the system overcomes the structural limitations inherent in individual AI paradigms, offering a robust, interpretable, and scalable pathway toward shortening the diagnostic trajectory for RDs.

---

## ✨ Key Features

* **Likelihood Ratio (LR) Engine**: Utilizes knowledge-driven symbolic reasoning based on the Human Phenotype Ontology (HPO) and statistical priors.
* **Residual Network Classifier (RNC)**: Applies deep-learning pattern recognition to capture high-dimensional clinical and phenotypic associations.
* **Patient Bank Retriever (PBR)**: Executes case-based similarity retrieval against a dynamic repository of over 30,000 historical and literature-derived rare disease cases.
* **Zero-shot LLM Diagnosis**: Leverages Large Language Models (LLMs) for heuristic inference and contextual medical reasoning.
* **Statistical Meta-Aggregation & Agentic Reflection**: Synthesizes the outputs of the four source paradigms through statistical meta-aggregation to establish a deterministic boundary, which is subsequently refined by a knowledge-enhanced agentic reflection layer using Chain-of-Thought (CoT) reasoning.

---

## 📁 Project Structure

```
FusionRare/
├── config.py                    # Global configuration: paths, hyperparameters, API keys, device
├── demo.py                      # End-to-end single-case demo (orchestrates all paradigms + fusion)
├── train.py                     # Training entry point for the Residual Network Classifier (RNC)
├── rra.py                       # Robust Rank Aggregation (RRA) statistical meta-aggregation
├── patient_base.py              # Patient Bank Retriever (PBR): case-based similarity retrieval
├── llm_diag.py                  # Zero-shot LLM diagnosis (source paradigm)
├── llm_reflection.py            # Knowledge-enhanced agentic reflection layer (Chain-of-Thought)
│
├── model/                       # Deep-learning diagnostic model
│   ├── main.py                  # DiagDL model + dynamic patient simulator
│   ├── output.py                # ResidualNetworkClassifier with LR knowledge priors
│   └── dataset.py               # PatientDataset and collate functions
│
├── disease_retriever/           # Disease name vectorization & retrieval
│   ├── vectorization.py         # Build the disease-name vector database (BioLORD embeddings)
│   └── retriever.py             # FAISS cosine-similarity index for disease retrieval
│
├── preprocess/                  # Knowledge-base preprocessing
│   └── calc_lr.py               # Compute Likelihood Ratio (LR) priors from HPO annotations
│
├── utils/                       # Shared helper functions
│   └── utils.py                 # JSON I/O, grouping/merging, HPO vocab utilities
│
└── data/                        # Ontology mappings, sample cases, and reference data
    ├── checkpoint/              # Trained RNC model weights
    ├── dise_synonyms.json       # Synonyms for rare diseases
    ├── hpo_ids.txt              # HPO term identifiers
    └── RPB_sample.json          # Sample patient-bank records
```

---

## 🚀 Step-by-Step Setup & Data Preparation

### 1. Create the Python environment

The pinned dependencies were recorded with Python **3.9.13**. The commands below assume Windows PowerShell and a working Conda installation:

```powershell
conda create -n fusionrare python=3.9.13 -y
conda activate fusionrare
python --version

python -m pip install --upgrade pip
python -m pip install torch==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

`requirements.txt` targets the CUDA 12.1 build of PyTorch. An NVIDIA driver compatible with CUDA 12.1 is required when `device` is set to `cuda`. Verify the installation before continuing:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### 2. Configure the runtime

Open `config.py` and set the values for the local machine. Paths are relative to the repository root:

```python
device = "cuda"       # Use "cpu" when no CUDA GPU is available
GPT_MODEL = "<model-name>"
OPENAI_API_BASE = "https://<api-host>/v1"
OPENAI_API_KEY = "<api-key>"
HF_CACHE_FOLDER = ""  # Optional Hugging Face model cache directory
```

The LLM settings are required by `llm_diag.py` and `llm_reflection.py`. Do not commit a real API key. The default `RECOMMEND_TOP_N = 5` can be adjusted if needed.

### 3. Download Ontology & Pretrained Assets

Download the specific HPO release files and place them into the `data/` directory:
* **HPO Ontology (`hp.obo`)**: [Download v2023-01-27](https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2023-01-27/hp.obo)
* **Phenotype Annotations (`phenotype.hpoa`)**: [Download v2023-01-27](https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2023-01-27/phenotype.hpoa)

Download the model weights and core annotation databases from the official [Hugging Face Repository](https://huggingface.co/yj1678492347/FusionRare):
* **`integrated_anno.json`**: Disease-phenotype annotation frequency database (save to `data/`).
* **`rnc.pth`**: Pretrained RNC model weights (save to `data/checkpoint/`).
* **`23559858-Ajmal-2013-BBS1-IV-5_family_A.json`** *(Optional)*: Example VCF-parsed case for genomic/phenotypic joint analysis.

Before preprocessing, confirm that these runtime inputs exist:

```text
data/hp.obo
data/phenotype.hpoa
data/dise_synonyms.json
data/integrated_anno.json
data/checkpoint/rnc.pth
data/RPB_sample.json
```

---

### 4. Generate derived knowledge-base files

Run the offline preprocessing scripts to calculate prior likelihood ratios and build the semantic vector index for disease names:

```powershell
# 1. Compute propagated frequencies and Likelihood Ratio (LR) prior matrices
python preprocess/calc_lr.py

# 2. Build the disease-name BioLORD embedding vector database
python disease_retriever/vectorization.py
```

The resulting derived files should be:

```text
data/integrated_propagated_freq.json
data/integrated_lr.json
data/dise_synonyms_BioLORD_metadata.npy
```

The vectorization step downloads the `FremyCompany/BioLORD-2023` model on its first run. Make sure the machine can access Hugging Face, or set `HF_CACHE_FOLDER` to an existing model cache.

---

### 5. Patient Bank Construction (PBR)

The Patient Bank Retriever (PBR) leverages real-world historical clinical cases. Researchers can collect public rare disease cases from the authoritative sources listed below:

| Source Dataset | Access URL / Identifier |
| :--- | :--- |
| **RareArena** | [https://github.com/zhao-zy15/RareArena](https://github.com/zhao-zy15/RareArena)|
| **Phenopacket** | [https://github.com/monarch-initiative/phenopacket-store](https://github.com/monarch-initiative/phenopacket-store) |
| **LIRICAL** | [https://zenodo.org/record/3905420](https://zenodo.org/record/3905420) |
| **RDmaster** | [https://github.com/1678492347/rdmaster-test](https://github.com/1678492347/rdmaster-test) |
| **PhenoDP** | [10.1186/s13073-025-01496-8#Sec37](https://link.springer.com/article/10.1186/s13073-025-01496-8#Sec37) |
| **MME** | [https://github.com/ga4gh/mme-apis/tree/master/testing](https://github.com/ga4gh/mme-apis/tree/master/testing) |
| **VPI** | [https://www.nature.com/articles/ejhg2017123#Sec9](https://www.nature.com/articles/ejhg2017123#Sec9) |
| **ADX** | [10.1186/s13023-019-1040-6#Sec37](https://link.springer.com/article/10.1186/s13023-019-1040-6#Sec37) |
| **DECIPHER** | [https://www.deciphergenomics.org/](https://www.deciphergenomics.org/) |
| **RAMEDIS** | [http://www.ramedis.de](http://www.ramedis.de) *(Backup: [Zenodo 10774650](https://zenodo.org/records/10774650))* |

#### Standardized Patient Format
All retrieved cases should be parsed and organized into a standardized JSON list:

```python
[
  {
    "patient_id": "P0001",
    "term_id": [
      "HP:0001250",
      "HP:0000252",
      "HP:0001263"
    ],
    "diagnosis": [
      "OMIM:219700", # or ORPHA code
    ]
  }
]
```

> **📌 Note on Sample Data (`RPB_sample.json`)**:  
> For immediate pipeline testing, `data/RPB_sample.json` provides **1,000 de-identified sample cases**. This allows you to verify the entire pipeline (`demo.py`) out of the box. However, replicating the **full benchmark diagnostic accuracy** requires constructing the comprehensive patient library using the sources above.

### 6. Run the demo

From the repository root, run:

```powershell
conda activate fusionrare
python demo.py
```

The demo loads the RNC checkpoint, the generated FAISS disease index, and the sample patient bank. It also calls the configured LLM endpoint for the zero-shot diagnosis and reflection stages, so valid LLM settings in `config.py` are required for the complete pipeline.


---

## 🌐 Web Platform

The interactive web application for clinical demonstration is accessible at:
👉 **[http://rdx.nbscn.org/](http://rdx.nbscn.org/)**

---

## 📂 Supplementary Documents

* **Benchmark Challenge Profiles**: 300 de-identified challenge evaluation profiles and their benchmark performance data are archived on [Zenodo (Record: 20697776)](https://zenodo.org/records/20697776).

