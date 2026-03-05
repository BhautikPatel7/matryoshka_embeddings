import json

# Count lines in the file
print("Counting lines in processed_documents.jsonl...")
line_count = 0
with open("data/processed_documents.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        line_count += 1
        if line_count % 100000 == 0:
            print(f"  Counted {line_count:,} lines...")

print(f"\n✅ Total lines in file: {line_count:,}")

# Verify last few documents are valid JSON
print("\n🔍 Checking last 3 documents...")
with open("data/processed_documents.jsonl", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
for i in [-3, -2, -1]:
    try:
        doc = json.loads(lines[i])
        print(f"  Document {len(lines) + i + 1}: ✅ Valid - ID: {doc['id']}, Title: {doc['title'][:50]}...")
    except Exception as e:
        print(f"  Document {len(lines) + i + 1}: ❌ Error - {e}")

print("\n✅ File verification complete!")