# Persian RAG with Ollama Embeddings

## Based On
Modified from [llm-rag-with-reranker-demo](https://github.com/yankeexe/llm-rag-with-reranker-demo)

## 🔨 Setup

```bash
# Option 1: Using batch files (Windows)
setup.bat

# Option 2: Manual setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
## ⚡️ Running the Application
```bash
run.bat
# OR
streamlit run app.py
```
## 🧠 Model Setup
```bash
# 1. Download Persian Embeddings
git clone https://huggingface.co/alishendi/persian-embeddings

# 2. Set up Ollama
ollama pull nomic-embed-text
```
