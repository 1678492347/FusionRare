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
    ├── hpo_ids.txt              # HPO term identifiers
    └── RPB_sample.json          # Sample patient-bank records
```

---

## 🌐 Access

* **Web Platform**: The extended web application developed for this research is hosted at [http://rdx.nbscn.org/](http://rdx.nbscn.org/).

---

## 📂 Data Availability

* **Ontologies & Knowledge Bases**: The Human Phenotype Ontology (HPO), Orphanet (ORPHA), and the Mondo Disease Ontology are publicly available for download from their official portals.
* **Benchmark Datasets**: 300 de-identified case profiles from the challenge subset, along with their benchmark performance data, are publicly available on Zenodo at [https://zenodo.org/records/20697776](https://zenodo.org/records/20697776).

---

## 💡 Replication Notice

This repository contains the official implementation framework of **FusionRare** to demonstrate the architectural design and multi-paradigm fusion methodology. Users can conceptualize the system flow from this codebase. To comply with institutional intellectual property restrictions and data privacy regulations, the full system performance can be reproduced through the official online portal.