# 🛡️ Marketplace Risk Intelligence Platform

> Built and tested on synthetic marketplace data.

A full-stack, multi-layered fraud and risk intelligence platform engineered to detect adversarial behaviour within high-throughput online marketplace ecosystems. 

Unlike standard standalone ML models, this system implements a production-style **Risk Fusion Engine**. It orchestrates rule-based heuristics, supervised machine learning, unsupervised anomaly detection, graph network intelligence, and temporal burst analysis into a single, cohesive analyst investigation interface.

---

## 🏗️ System Architecture

This platform processes signals through a multi-tiered pipeline, culminating in an Explainable AI (XAI) Analyst Workbench.

```text
Business Data Sources (Transactions, Refunds, Logins, Ratings, Device/Payment Signals)
        ↓
Data Validation & Feature Engineering Pipeline (Polars / Pandas)
        ↓
Feature Store / Analytical Extraction Layer
        ↓
Detection Layer
 ├─ V1: Rule Engine (Heuristic Weak Labels)
 ├─ V2: Supervised Risk Model (Random Forest / XGBoost Classifier)
 ├─ V2: Anomaly Detection Model (Isolation Forest)
 ├─ V3: Graph / Entity Link Analysis (NetworkX / Neo4j)
 └─ V4: Temporal Intelligence (Burst / Velocity Detection)
        ↓
Risk Fusion & Decision Engine (Multi-Signal Calibration)
        ↓
V5: Investigation Queue & Analyst Workbench (Streamlit)
        ↓
V6: Governance, Drift Monitoring, & Audit Logs
```

## Core Engineering Components
1. Detection Vectors
Rule Engine (V1): Catch-all heuristics for abnormal order spikes, login anomalies, and refund irregularities. Generates weak-supervision labels to bootstrap the ML models.

Machine Learning Layer (V2): Ensembles an Isolation Forest (for zero-day anomaly detection) with a Random Forest classifier (for historic fraud patterns) to generate entity-level risk probabilities.

Graph Intelligence (V3): Analyzes topological relationships to detect organized fraud rings. Computes graph centrality, shared devices/regions, and connected component clusters.

Temporal Intelligence (V4): Detects behavioural bursts using time-windowed aggregates (e.g., login velocity, order bursts, and refund density).

2. Multi-Layer Risk Fusion (V4.1)
Signals from all detection layers are mathematically fused into a calibrated, final risk score. The baseline fusion algorithm applies the following weights:

Final Risk = 0.40(ML Risk) + 0.25(Graph Risk) + 0.20(Temporal Risk) + 0.15(Anomaly Signal)

Entities are then classified into operational risk bands: Critical, High, Medium, Low.

3. Explainability & Investigation (V5)
Every flagged entity includes an auto-generated narrative. The Streamlit-based Analyst Investigation Workbench provides SOC teams with:

Real-time investigation queues

Granular signal breakdown and engineered feature snapshots

Graph and temporal evidence visualization

Analyst action panels for immediate remediation

4. MLOps Governance & Drift Control (V6)
Operational safeguards ensure pipeline stability and model safety:

Automated feature drift monitoring (v6_feature_drift.csv)

Pipeline health reporting (v6_pipeline_health.json)

Model registry and artifact tracking (ops_registry.db)

## Repository Structure

```text
├── artifacts/                # Serialized models, feature columns, and importance matrices
├── data/                     # Synthetic marketplace dataset
├── models/                   # Pickled ML models (RF, Isolation Forest)
├── outputs/                  # Final fused risk scores, explanations, and governance logs
│
├── create_database.py        # SQLite hydration layer
├── generate_data.py          # Synthetic data generation (V1)
├── feature_eng.py            # Feature engineering pipeline (V2)
├── train_model.py            # Ensembled model training and evaluation
├── risk_scoring.py           # Legacy heuristic scoring
├── score_entities.py         # ML inference and risk fusion
├── explain_entities.py       # XAI reason code generation
├── intelligence_graph.py     # Graph network analysis (V3)
├── temporal_intelligence.py  # Time-series burst detection (V4)
├── risk_fusionv4.py          # Final signal calibration
├── operation_v6.py           # MLOps drift and health monitoring
├── v5_investigation_workbench.py # Streamlit Command Center
│
├── pipeline_run.py           # Master DAG/Orchestrator
└── README.md
```

## Execution Guide

### 1. Initialize the Environment
Activate your Python virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Execute the Full Intelligence Pipeline
Run the master orchestrator to hydrate the database, engineer features, train the models, run graph/temporal analysis, and output the fused risk reports:

```bash
python pipeline_run.py
```

### 3. Launch the SOC Command Center
Start the interactive analyst dashboard:

```bash
streamlit run v5_investigation_workbench.py
```

## Future Architecture Roadmap

Data Layer: Refactor core ETL from Pandas to Polars for memory-safe, out-of-core vectorized processing.
Algorithm Swap: Upgrade the Random Forest classifier to XGBoost/LightGBM for optimized sparse data handling and reduced memory footprint.
Containerization: Wrap the entire architecture in a docker-compose.yml network (Feature Store, Inference API, and Streamlit UI) for deployment immutability.
Generative AI Integration: Deploy a localized, quantized LLM (via Ollama) to automatically generate human-readable case summaries from tabular risk signals.