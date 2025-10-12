import os
import json

# Set the root directory to search for markdown files
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

md_files = []
for root, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        if file.endswith('.md'):
            md_files.append(os.path.join(root, file))

# Extract text content from each markdown file
md_data = []
for path in md_files:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        md_data.append({
            'path': path,
            'content': content
        })
    except Exception as e:
        print(f"Error reading {path}: {e}")

# Save extracted data to a JSON file for later processing
with open(os.path.join(os.path.dirname(__file__), 'markdown_data.json'), 'w', encoding='utf-8') as f:
    json.dump(md_data, f, ensure_ascii=False, indent=2)

print(f"Extracted content from {len(md_data)} markdown files and saved to markdown_data.json.")
