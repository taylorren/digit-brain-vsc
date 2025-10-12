# Digital Brain RAG System

A local Retrieval-Augmented Generation (RAG) app for semantic search and Q&A over your Markdown, PDF, and PPTX knowledge base, with Chinese and English support.

## Features
- Fast semantic search using FAISS and BAAI/bge-large-zh embedding model
- Local or API-based LLM (Ollama, Zhipu) for answer generation
- Source citation and multi-format support (.md, .pdf, .pptx)
- Easy to update your knowledge base and re-index

## Requirements
- Python 3.8+
- Windows (tested) or Linux
- Recommended: GPU for faster embedding (optional)

## Installation
1. **Clone this repository and download models:**
   - Place the `bge-large-zh` model in `models/bge-large-zh/` (see [HuggingFace](https://huggingface.co/BAAI/bge-large-zh))
   - (Optional) Place other models in `models/` as needed

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Example `requirements.txt`:
   ```
   sentence-transformers
   faiss-cpu
   numpy
   tqdm
   python-dotenv
   requests
   PyMuPDF
   python-pptx
   ```

3. **Configure environment:**
   - Copy `.env.example` to `.env` and set your ZHIPU_API_KEY if using Zhipu

4. **Prepare your knowledge base:**
   - Place your `.md`, `.pdf`, `.pptx` files in the configured folders (see `SCAN_ROOTS` in `embed_and_index.py`)

## Usage
1. **Embed and index your files:**
   ```bash
   python embed_and_index.py
   ```
   This will create `md_faiss.index` and `md_faiss_meta.json` in the project root.

2. **Start the RAG Q&A app:**
   ```bash
   python rag_brain_fast.py
   ```
   - Enter your question at the prompt
   - Use `zhipu` or `ollama` to switch LLM backend
   - Use `exit` to quit

## Customization
- To add new file types, extend `embed_and_index.py` and update extraction logic
- To change the embedding model, update the model path in both `embed_and_index.py` and `rag_brain_fast.py`
- To adjust search parameters (e.g., top_k), edit the corresponding arguments in `rag_brain_fast.py`

## Troubleshooting
- If you see dimension mismatch errors, ensure both scripts use the same embedding model
- If empty files appear in results, re-run embedding after removing or filtering empty files
- For performance, use batch encoding and GPU if available

## Credits
- [BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh)
- [sentence-transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)

---
Feel free to open issues or contribute improvements!
