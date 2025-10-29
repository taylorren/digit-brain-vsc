import json
from collections import Counter

with open('markdown_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

lengths = [len(doc['content'].strip()) for doc in data]

# Define bins for length distribution
bins = [0, 30, 100, 500, 1000, 2000, 5000, 10000, float('inf')]
bin_labels = [
    '<30', '30-99', '100-499', '500-999', '1000-1999', '2000-4999', '5000-9999', '10000+'
]

# Count files in each bin
bin_counts = Counter()
for l in lengths:
    for i in range(len(bins)-1):
        if bins[i] <= l < bins[i+1]:
            bin_counts[bin_labels[i]] += 1
            break

print('Markdown file length distribution:')
for label in bin_labels:
    print(f'{label:>8}: {bin_counts[label]}')

# Show the 10 shortest files
shortest = sorted([(len(doc['content'].strip()), doc['path']) for doc in data])[:10]
print('\n10 shortest files:')
for l, path in shortest:
    print(f'{l:>4} chars: {path}')
