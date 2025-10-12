import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the local paraphrase-multilingual-MiniLM-L12-v2 model for better Chinese support
model = SentenceTransformer(os.path.join(os.path.dirname(__file__), 'models', 'paraphrase-multilingual-MiniLM-L12-v2'))

# Load FAISS index and metadata
index = faiss.read_index(os.path.join(os.path.dirname(__file__), 'md_faiss.index'))
with open(os.path.join(os.path.dirname(__file__), 'md_faiss_meta.json'), 'r', encoding='utf-8') as f:
    meta = json.load(f)

def search(query, top_k=5):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec).astype('float32'), top_k)
    results = []
    for idx, score in zip(I[0], D[0]):
        results.append({
            'score': float(score),
            'path': meta[idx]['path'],
            'content': meta[idx]['content'][:500]  # Show first 500 chars
        })
    return results

if __name__ == '__main__':
    while True:
        query = input('\nEnter your question (or "exit" to quit): ')
        if query.lower() == 'exit':
            break
        results = search(query)
        print(f'\nTop {len(results)} results:')
        for i, r in enumerate(results, 1):
            print(f"\nResult {i} (Score: {r['score']:.2f})")
            print(f"File: {r['path']}")
            print(f"Excerpt: {r['content']}\n{'-'*40}")
