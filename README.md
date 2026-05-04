# 🧩 RAG Evaluation System

<p align="center">
  <img src="https://img.shields.io/badge/LLM-OpenRouter-3F51B5" />
  <img src="https://img.shields.io/badge/Embeddings-HuggingFace-27A162" />
  <img src="https://img.shields.io/badge/Vector%20DB-Chroma-8659B9" />
  <img src="https://img.shields.io/badge/Database-MongoAtlas-76B900" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688" />
  <img src="https://img.shields.io/badge/Frontend-Streamlit-FF7043" />
  <img src="https://img.shields.io/badge/Design-SOLID%20Principles-17256b" />
  <img src="https://img.shields.io/badge/Code%20Style-PEP8-045E1A" />
  <img src="https://img.shields.io/badge/License-MIT-45a5d7" />
</p>

## 🎯 Overview

The **RAG Evaluation System** is an open-source platform designed to systematically benchmark Retrieval-Augmented Generation (RAG) pipelines. It enables developers to evaluate different configurations, such as chunking strategies, embedding models, and retrievers without rebuilding pipelines from scratch.

The system abstracts the full RAG workflow and provides a unified interface to run experiments and compare results. By automating evaluation and centralizing results in a dashboard, it reduces development overhead and helps identify the most effective configuration for a given knowledge base.


## ✨ Core Features

- **Knowledge Base & Test Dataset Handling**  
  Upload and process document collections and evaluation datasets

- **Configurable RAG Pipelines**  
  Swap chunking, embedding, and retrieval strategies via factory-based design

- **Multi-Model Support**  
  Supports OpenAI, OpenRouter, and local LLM integrations

- **Automated Evaluation Engine**  
  Runs end-to-end evaluation across multiple test cases

- **Parallel Execution**  
  Efficient evaluation using thread-based processing

- **Experiment Tracking (MongoDB)**  
  Stores all runs for comparison and reproducibility

- **Interactive Dashboard (Streamlit)**  
  Visualize KPIs and compare performance across experiments



## 📈 RAG Metrics & KPIs

The system evaluates retrieval quality using multiple metrics, but the dashboard focuses on a small set of **high-signal KPIs** for clarity and decision-making.

### Why These KPIs?

RAG evaluation can produce many metrics, but too many signals create noise. These KPIs were selected to balance:

- **Ranking quality**
- **Early relevance**
- **Coverage of information**
- **Progress tracking over time**

### Selected KPIs

| KPI | Purpose |
|-----|--------|
| **nDCG (Latest)** | Measures overall ranking quality |
| **nDCG (vs Best)** | Compares current run to best historical performance |
| **MRR (Latest)** | Evaluates how early the first correct result appears |
| **Recall (Latest)** | Measures coverage of relevant information |


### Metric Explanation
---

- **MRR (Mean Reciprocal Rank)**  
  <code>Focuses on the *first correct result*.</code>  
  - “Did we get a good answer at the top?”

- **nDCG (Normalized Discounted Cumulative Gain)**  
  <code>Considers both relevance and ranking position.</code> 
   - “Is the overall ranking of results good?”

- **Recall**  
  <code>Measures how many relevant items were retrieved.</code>  
   - “Did we find all the important information?”

- **Hit@K (Support Metric)**  
  <code>Checks if at least one correct result appears in top-K.</code>  
   - “Did we get at least one correct result?”



### KPI Representation Strategy
---

- **Latest Metrics** → Evaluate current system performance  
- **vs Best** → Benchmark against best historical run  
- **Delta Metrics (internal use)** → Track improvements/regressions  

This ensures the dashboard remains **interpretable, stable, and decision-focused**.



## 📸 UI Preview


<img width="900" height="500" alt="image" src="https://github.com/user-attachments/assets/41d41c25-e15f-42d1-9a29-d62287d5cc79"/>


## 🏗️ Architecture

The system follows a layered, interface-driven architecture that ensures modularity, maintainability, and extensibility. Its design is guided by core principles such as dependency injection, interface-based abstraction, separation of concerns, and a plugin-oriented architecture.

<b>Key Architectural Highlights:</b>
- Centralized container injects dependencies, promoting loose coupling and easier testing.  
- Core modules define clear interfaces to enforce contracts and enable polymorphism.
- Factory Pattern used to dynamically initialize different chunking and embedding strategies, allowing flexible experimentation with RAG configurations.  
- Singleton patterns are used for global utilities like logging.
- The architecture defines clear layers: UI/API, Components (business logic), Infrastructure (integrations), Interfaces (contracts), and Container (dependency management).


### 🔹 RAG EVAL Workflow

```
Knowledge Base & Test Data
         │
         ▼
┌───────────────────────────┐
│ Data Preprocessing        │
│ - File extraction         │
│ - Folder handling         │
└───────────────────────────┘
         │
         ▼
┌───────────────────────────┐
│ Chunking Step             │
│ - Recursive chunking      │
│ - LLM-assisted chunking   │
└───────────────────────────┘
         │
         ▼
┌───────────────────────────┐
│ Embedding Step            │
│ - HuggingFace Embeddings  │
│ - OpenAI Embeddings       │
└───────────────────────────┘
         │
         ▼
┌───────────────────────────┐
│ Vector Store / Retriever  │        
│ - Query handling          │
└───────────────────────────┘
         │
         ▼
┌───────────────────────────┐
│ Evaluation Step           │
│ - MRR                     │
│ - nDCG                    │
│ - Recall                  │
│ - Hit@K                   │
└───────────────────────────┘
         │
         ▼
┌───────────────────────────┐
│ MongoDB Atlas             │        
│ - MongoDB Storage         │
└───────────────────────────┘
         │
         ▼
┌───────────────────────────┐
│ Insights & Visualization  │
│ - Streamlit Dashboard     │ 
└───────────────────────────┘
```

## 📌 Prerequisites

1. **Python 3.10+**  
2. **Conda** (for environment management)  
3. **MongoDB Atlas account**  
   - DB name: `rag_db`  
   - Collection: `rag_eval_2`  
4. **API Keys**  
   - `OPENAI_API_KEY`  
   - `OPEN_ROUTER_KEY`  
5. **Knowledge Base Files** (zip/rar)  
6. **Test Dataset** (JSONL format)  
   - Required keys: `question`, `keywords`, `reference_answer`, `category`  

Example test entry:
```json
{
  "question": "Are there subscription plans available for regular customers?",
  "keywords": ["subscription", "recurring revenue", "customer plans"],
  "reference_answer": "Yes, we offer subscription plans for regular customers, providing recurring revenue and flexible customer plans.",
  "category": "pricing"
}
```

## 🤝 Contributing

Contributions are welcome in the following areas:

- New metrics & evaluation strategies
- Additional chunking or embedding methods
- Backend & UI improvements
- AI prompt engineering
- Testing and validation

### Contribution Steps

1. 🍴 Fork the repository  
2. 🌿 Create a `feature/*` branch  
3. 🛠️ Commit changes with clear messages  
4. 📤 Open a Pull Request  


## 🔀 Git Flow Workflow

The project follows a Git Flow–inspired workflow:

- 🌿 `master` — Stable, production-ready releases  
- 🌱 `develop` — Active development branch  
- ✨ `feature/*` — New feature branches  

### Typical Workflow

1. Pull latest changes from `develop`  
2. Create a `feature/*` branch  
3. Implement and test changes  
4. Open PR → Merge into `develop`  
5. Release from `develop` → Merge into `master`  

This ensures stability while enabling safe feature development.

---
