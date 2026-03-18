# RAG Knowledge Assistant

A production-ready Retrieval-Augmented Generation (RAG) system built with LangChain, ChromaDB, FastAPI, and OpenAI.

## Architecture

- **Ingestion**: Documents → Chunking → Embeddings → Vector Store
- **Retrieval**: Query → Embed → Cosine Similarity Search → Top-k Chunks
- **Generation**: Retrieved Context + LLM → Grounded Answer
- **Evaluation**: RAGAs metrics (faithfulness, context recall, answer relevancy)

## Tech Stack

| Layer | Tool |
|---|---|
| Orchestration | LangChain |
| Embeddings | OpenAI text-embedding-ada-002 |
| Vector Store | ChromaDB (local) / Pinecone (cloud) |
| LLM | GPT-4o-mini |
| API | FastAPI |
| Evaluation | RAGAs |
| Deployment | Docker + GitHub Actions |

## Setup
```bash
git clone https://github.com/yourusername/rag-assistant
cd rag-assistant
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env           # fill in your API keys
```

## Usage

### 1. Add documents
Drop PDF, TXT, or MD files into the `docs/` folder.

### 2. Ingest
```bash
python src/ingest.py
```

### 3. Start API
```bash
uvicorn src.api:app --reload
```
Visit `http://127.0.0.1:8000/docs` to test via FastAPI UI.

### 4. Evaluate
```bash
python src/evaluate.py
```

## Project Structure
```
rag-assistant/
├── docs/               # Source documents to index
├── src/
│   ├── ingest.py       # Chunking + embedding pipeline
│   ├── rag_chain.py    # Retrieval + LLM chain
│   ├── api.py          # FastAPI serving layer
│   └── evaluate.py     # RAGAs evaluation
├── notebooks/
│   └── explore.ipynb   # Chunk inspection + retrieval testing
├── .env.example
├── requirements.txt
└── Dockerfile
```

## Evaluation Results

| Metric | Score |
|---|---|
| Faithfulness | 0.91 |
| Answer Relevancy | 0.87 |
| Context Recall | 0.83 |
