import os

SCAN_ROOTS = [
    r"F:/My Books/Working/My Own Writings Managed by Obsidian",
    r"F:/My Books/Working/_各读书会",
    # r"D:/Some/Other/Path"
]

md_files = []
pdf_files = []
pptx_files = []
for scan_root in SCAN_ROOTS:
    for root, dirs, files in os.walk(scan_root):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext == '.md':
                md_files.append(os.path.join(root, file))
            elif ext == '.pdf':
                pdf_files.append(os.path.join(root, file))
            elif ext == '.pptx':
                pptx_files.append(os.path.join(root, file))

print(f"Markdown files: {len(md_files)}")
print(f"PDF files: {len(pdf_files)}")
print(f"PPTX files: {len(pptx_files)}")
print(f"Total: {len(md_files) + len(pdf_files) + len(pptx_files)}")

# List all files if total != 329
if len(md_files) + len(pdf_files) + len(pptx_files) != 329:
    print("\nFile list (showing up to 20 per type):")
    print("\nMarkdown:")
    for f in md_files[:20]:
        print(f)
    print("\nPDF:")
    for f in pdf_files[:20]:
        print(f)
    print("\nPPTX:")
    for f in pptx_files[:20]:
        print(f)
