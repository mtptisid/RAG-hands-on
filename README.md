# 🧠 RAG-hands-on

A practical hands-on repository for exploring Retrieval-Augmented Generation (RAG) using open-source tools and LLMs.

---

## ✅ What Has Been Done So Far

- ✅ Environment setup completed (`env-setup/`)
- ✅ Installed and configured necessary libraries (e.g., `transformers`, `faiss`, `langchain`, `llama-cpp`)
- ✅ Added quantized model support using `bitsandbytes` for 4-bit/8-bit loading
- ✅ Implemented document ingestion pipeline:
  - Document loading
  - Chunking using LangChain
  - Embedding generation with Hugging Face models
  - Vector store indexing (FAISS, Chroma)
- ✅ Created working RAG pipeline:
  - User query → Retriever → LLM → Final answer
- ✅ Demonstrated Use Case 1: Simple RAG Q&A on sample data (LLaMA3 + A4000 GPU)
- ✅ Explored concepts like memory, temperature, top-p, and prompt engineering
- ✅ Drafted multi-step theoretical insights and notes
- ✅ Visualized working flow (text + image)

---

## 🧭 What to Do Next

### 🔧 1. Modularize Codebase

- Convert exploratory notebooks/scripts into reusable Python modules:
  - `src/ingest.py`
  - `src/retrieve.py`
  - `src/generate.py`
- Add CLI interface for each stage (with `argparse`)

### 📦 2. Add `demo/` Folder for Full Pipeline

- Sample commands:
  ```bash
  python demo/ingest.py --input ./docs --output ./vectorstore
  python demo/query.py --question "What is RAG?" --model llama
  ```
  
  Include sample files under docs/ and queries/


### 🤖 3. Expand with Advanced RAG Features
- Reranker using a cross-encoder model
- Query rewriting / decomposition
- Context-aware retrieval using similarity + neighbors
- Multi-hop retrieval

### 📊 4. Benchmarking
-	Create evaluation set (20–50 questions)
-	Track:
  	- Accuracy (retrieved doc vs answer quality)
  	- Latency
	- Model comparison: GPT-3.5 vs LLaMA vs Mistral
- Optionally use metrics like ROUGE or BLEU
