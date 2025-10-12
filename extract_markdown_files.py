import os

# Set the root directory to search for markdown files
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(ROOT_DIR)

md_files = []
for root, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        if file.endswith('.md'):
            md_files.append(os.path.join(root, file))

print(f"Found {len(md_files)} markdown files.")
for f in md_files:
    print(f)
