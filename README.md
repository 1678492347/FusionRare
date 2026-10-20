# FusionRare: A Unified Multiparadigm Fusion Intelligence System for Rare Disease Diagnosis

[![Web Application](https://img.shields.io/badge/Web_App-Available-blue)](http://rdx.nbscn.org/) 
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

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

### 1. Download Ontology & Pretrained Assets

Download the specific HPO release files and place them into the `data/` directory:
* **HPO Ontology (`hp.obo`)**: [Download v2023-01-27](https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2023-01-27/hp.obo)
* **Phenotype Annotations (`phenotype.hpoa`)**: [Download v2023-01-27](https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2023-01-27/phenotype.hpoa)

Download the model weights and core annotation databases from the official [Hugging Face Repository](https://huggingface.co/yj1678492347/FusionRare):
* **`integrated_anno.json`**: Disease-phenotype annotation frequency database (save to `data/`).
* **`rnc.pth`**: Pretrained RNC model weights (save to `data/checkpoint/`).
* **`23559858-Ajmal-2013-BBS1-IV-5_family_A.json`** *(Optional)*: Example VCF-parsed case for genomic/phenotypic joint analysis.

---

### 2. Knowledge Base Preprocessing

Run the offline preprocessing scripts to calculate prior likelihood ratios and build the semantic vector index for disease names:

```bash
# 1. Compute propagated frequencies and Likelihood Ratio (LR) prior matrices
python preprocess/calc_lr.py

# 2. Build the disease-name BioLORD embedding vector database
python disease_retriever/vectorization.py
```

---

### 3. Patient Bank Construction (PBR)

The Patient Bank Retriever (PBR) leverages real-world historical clinical cases. Researchers can collect public rare disease cases from the authoritative sources listed below:

| Source Dataset | Access URL / Identifier | Reference / Notes |
| :--- | :--- | :--- |
| **RareArena** | [https://github.com/zhao-zy15/RareArena](https://github.com/zhao-zy15/RareArena) | Benchmark clinical cohort |
| **Phenopacket** | [https://github.com/monarch-initiative/phenopacket-store](https://github.com/monarch-initiative/phenopacket-store) | GA4GH standard patient store |
| **LIRICAL** | [https://zenodo.org/record/3905420](https://zenodo.org/record/3905420) | Phenotype-driven benchmark |
| **RDmaster** | [https://github.com/1678492347/rdmaster-test](https://github.com/1678492347/rdmaster-test) | Diagnostic evaluation set |
| **PhenoDP** | [10.1186/s13073-025-01496-8#Sec37](https://link.springer.com/article/10.1186/s13073-025-01496-8#Sec37) | Phenotype benchmark |
| **MME** | [https://github.com/ga4gh/mme-apis/tree/master/testing](https://github.com/ga4gh/mme-apis/tree/master/testing) | Matchmaker Exchange API suite |
| **VPI** | [https://www.nature.com/articles/ejhg2017123#Sec9](https://www.nature.com/articles/ejhg2017123#Sec9) | Variant-phenotype cohort |
| **ADX** | [10.1186/s13023-019-1040-6#Sec37](https://link.springer.com/article/10.1186/s13023-019-1040-6#Sec37) | Diagnostic validation data |
| **DECIPHER** | [https://www.deciphergenomics.org/](https://www.deciphergenomics.org/) | Microdeletion / CNV cohort |
| **RAMEDIS** | [http://www.ramedis.de](http://www.ramedis.de) *(Backup: [Zenodo 10774650](https://zenodo.org/records/10774650))* | Rare disease case registry |

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


---

## 🌐 Web Platform

The interactive web application for clinical demonstration is accessible at:
👉 **[http://rdx.nbscn.org/](http://rdx.nbscn.org/)**

---

## 📂 Supplementary Documents

* **Benchmark Challenge Profiles**: 300 de-identified challenge evaluation profiles and their benchmark performance data are archived on [Zenodo (Record: 20697776)](https://zenodo.org/records/20697776).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
```
