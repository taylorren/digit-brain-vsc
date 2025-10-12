import os
import requests

MODEL_FILES = [
    'config.json',
    'pytorch_model.bin',
    'modules.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'vocab.txt',
    'special_tokens_map.json',
]

BASE_URL = 'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/'
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'all-MiniLM-L6-v2')
os.makedirs(MODEL_DIR, exist_ok=True)

def download_file(url, dest):
    print(f'Downloading {url} ...')
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f'Saved to {dest}')

for filename in MODEL_FILES:
    url = BASE_URL + filename
    dest = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(dest):
        try:
            download_file(url, dest)
        except Exception as e:
            print(f'Failed to download {filename}: {e}')
    else:
        print(f'{filename} already exists, skipping.')

print('All model files downloaded.')
