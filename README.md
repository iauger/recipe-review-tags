# Structuring Recipe Reviews Through Qualitative Tagging Assignment

**Author:** Ian Auger  
**Course:** DSCI 632

## Project Abstract
Online recipe platforms suffer from "information density" and a severe optimism bias, where critical qualitative signals (e.g., taste issues, prep difficulty) are buried within millions of uniformly positive 5-star free-text reviews. 

This project develops a scalable, multi-label classification (MLC) tagging system using a PySpark Lakehouse architecture. By utilizing a Transformer-based Zero-Shot baseline to generate a "Silver Standard" ground truth, the pipeline trains a highly efficient, distributed Word2Vec centroid classifier. This custom 128-dimensional culinary vector space allows for the deterministic, high-throughput extraction of minority instructional critiques from over 1.07 million reviews.

The resulting structured "Gold Dataset" serves as the foundational feature space for downstream multi-head deep learning systems predicting recipe quality.

---

## Pipeline Architecture
The system is built on a modular Lakehouse design (Bronze ➔ Silver ➔ Gold) to ensure data immutability and distributed execution across PySpark workers.

1. **Phase 1: Ingestion (Bronze)**
   * Automated Kaggle API extraction of raw Food.com interactions.
   * Strict schema enforcement and Parquet serialization.
2. **Phase 2: Density Filtering (Silver)**
   * Aggressive regex normalization and syntactic "glue word" removal.
   * Semantic bounds filtering: Reviews are restricted to an "Informative Range" of 15–120 tokens to prevent vector dilution.
3. **Phase 3: Semantic Structuring & Calibration (Gold)**
   * **Zero-Shot Baseline:** A `distilbart-mnli-12-1` transformer generates multi-label ground truth for a 65,000-sample subset.
   * **Vector Space:** A Window-5, 128-D Word2Vec model maps the entire 1.07M review corpus.
   * **Calibration:** Diagonal extraction of cosine similarities generates automated, data-driven thresholds for each of the 17 culinary tags.
   * **Logical Pruning:** Post-processing semantic routing automatically resolves contradictory labels (e.g., `dry` vs. `moist_tender`).

---

## Repository Structure

The codebase is organized into cleanly scoped, single-responsibility modules:

```text
├── src/
│   ├── config.py                 # OS-agnostic path resolution & environment settings
│   ├── spark/
│   │   ├── session.py            # WSL-optimized PySpark session builder
│   │   ├── ingestion/            # Phase 1 & 2: Lakehouse builders
│   │   │   ├── bronze_ingest.py
│   │   │   ├── interactions_spark.py
│   │   │   ├── recipes_spark.py
│   │   │   └── merge_spark.py
│   │   ├── labeling/             # Phase 3: Zero-shot baseline & Inference
│   │   │   ├── taxonomy.py       # 17-tag culinary dictionary & hypotheses
│   │   │   ├── zero_shot.py      # HuggingFace Transformer integration
│   │   │   ├── inference.py      # Scale-out Word2Vec projection
│   │   │   └── postprocessing.py # Negation conflict resolution
│   │   └── features/             # Phase 3: Feature Engineering
│   │       ├── build_features.py # Orchestrator for vector training
│   │       ├── dataset_prep.py   # Train/Val/Test deterministic splitting
│   │       ├── nlp_pipeline.py   # Tokenization & Word2Vec configuration
│   │       ├── semantic_space.py # Centroid math and Algorithm 5 thresholding
│   │       └── io.py             # Parquet & JSON manifest writers
│   └── utils/
│       ├── text_cleaning.py      # Regex and string normalization
│       └── notebook_visualizations.py # Seaborn & Mermaid reporting utilities
├── notebooks/
│   ├── 00_project_report.ipynb         # Formal project report
│   ├── 01_ingestion_and_labeling.ipynb # EDA, Data Prep, and Transformer Labeling
│   └── 02_fe_and_scaling.ipynb         # Semantic Benchmarking and Scale-out Analytics
└── data/                         # .gitignored raw/processed Lakehouse layers
```

---

## Execution Flow
The pipeline execution is segmented into two notebooks. 

1. `01_ingestion_and_labeling.ipynb`:
- Download the Kaggle dataset, run the Bronze/Silver ingestion, and generate the zero-shot ground truth dataset.

2. `02_fe_and_scaling.ipynb`: 
- Train the global Word2Vec model and extract the calibration thresholds.
- Project the entire Silver corpus into the semantic space to assign final binary tags.
- Analyze results and benchmark the performance against baseline.

---

## Future Work

While statistical co-occurrence successfully maps broad culinary sentiment, conversational hubness (e.g., hedging words like "afraid" or "obviously") creates a ceiling for isolating high-resolution risk signals. Future iterations of this pipeline (V2) will pivot to Aspect-Based Sentiment Analysis via distributed Part-of-Speech (POS) tagging to achieve absolute semantic density before deep learning ingestion.

