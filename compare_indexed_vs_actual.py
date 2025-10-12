import json
import os

# Load indexed file paths from metadata
with open('md_faiss_meta.json', 'r', encoding='utf-8') as f:
    indexed = set(entry['path'] for entry in json.load(f))

# Scan all files as in verify_indexed_files.py
SCAN_ROOTS = [
    r"F:/My Books/Working/My Own Writings Managed by Obsidian",
    r"F:/My Books/Working/_各读书会",
    # r"D:/Some/Other/Path"
]

all_files = set()
for scan_root in SCAN_ROOTS:
    for root, dirs, files in os.walk(scan_root):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.md', '.pdf', '.pptx']:
                all_files.add(os.path.join(root, file))

not_indexed = all_files - indexed
indexed_but_missing = indexed - all_files

print(f"Total scanned: {len(all_files)}")
print(f"Total indexed: {len(indexed)}")
print(f"Not indexed: {len(not_indexed)}")
print(f"Indexed but missing: {len(indexed_but_missing)}")

if not_indexed:
    print("\nFiles not indexed:")
    for f in list(not_indexed)[:20]:
        print(f)
    if len(not_indexed) > 20:
        print(f"...and {len(not_indexed)-20} more.")
if indexed_but_missing:
    print("\nIndexed but missing from disk:")
    for f in list(indexed_but_missing)[:20]:
        print(f)
    if len(indexed_but_missing) > 20:
        print(f"...and {len(indexed_but_missing)-20} more.")
