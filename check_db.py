#!/usr/bin/env python3
import sqlite3
import json

db_path = "app_data/rag_meta.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

print("=== DOCUMENTS ===")
cursor.execute("SELECT * FROM documents")
docs = cursor.fetchall()
print(f"Total documents: {len(docs)}")
for doc in docs:
    print(f"\nHash: {doc['doc_hash'][:16]}...")
    print(f"  Name: {doc['original_name']}")
    print(f"  Status: {doc['status']}")
    print(f"  Size: {doc['size']}")
    print(f"  Error: {doc['error']}")

print("\n=== EXTRACTIONS ===")
cursor.execute("SELECT doc_hash, parser, LENGTH(text) as text_len FROM extractions")
extractions = cursor.fetchall()
print(f"Total extractions: {len(extractions)}")
for ext in extractions:
    print(f"  {ext['doc_hash'][:16]}... - {ext['parser']}: {ext['text_len']} chars")

print("\n=== CHUNKS ===")
cursor.execute("SELECT doc_hash, parser, COUNT(*) as count FROM chunks GROUP BY doc_hash, parser")
chunks = cursor.fetchall()
print(f"Total chunk groups: {len(chunks)}")
for chunk in chunks:
    print(f"  {chunk['doc_hash'][:16]}... - {chunk['parser']}: {chunk['count']} chunks")

print("\n=== EMBEDDINGS ===")
cursor.execute("SELECT doc_hash, COUNT(*) as count FROM embeddings GROUP BY doc_hash")
embeddings = cursor.fetchall()
print(f"Total embedding groups: {len(embeddings)}")
for emb in embeddings:
    print(f"  {emb['doc_hash'][:16]}...: {emb['count']} embeddings")

conn.close()
