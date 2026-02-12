#!/usr/bin/env python3
"""
CHROMADB INSPECTOR: See what was actually indexed
Shows chunk contents, metadata, and identifies what went wrong
"""
import os
import sys
import chromadb
from collections import Counter
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.storage_utils import CHROMA_DB_PATH

def main():
    print("🔬 CHROMADB CONTENT INSPECTOR")
    print("="*80)
    
    # Connect to ChromaDB
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    try:
        collection = db.get_collection("solar_ppa_collection")
        count = collection.count()
        print(f"✅ Collection: solar_ppa_collection ({count} vectors)")
    except Exception as e:
        print(f"❌ Failed to open collection: {e}")
        return
    
    # Get all documents
    print(f"\n🔄 Fetching all {count} documents...")
    results = collection.get(include=["documents", "metadatas"])
    
    docs = results['documents']
    metas = results['metadatas']
    ids = results['ids']
    
    print(f"✅ Retrieved {len(docs)} documents")
    
    # Analyze metadata
    print("\n" + "="*80)
    print("📊 METADATA ANALYSIS")
    print("="*80)
    
    # Check for enhancement flags
    has_table_count = sum(1 for m in metas if m.get('has_table', False))
    is_definition_count = sum(1 for m in metas if m.get('is_definition', False))
    chunk_types = Counter(m.get('chunk_type', 'unknown') for m in metas)
    
    print(f"\n🔍 Enhancement Status:")
    print(f"   Table-enhanced chunks: {has_table_count}")
    print(f"   Definition chunks: {is_definition_count}")
    print(f"   Chunk types: {dict(chunk_types)}")
    
    # Analyze clause titles
    clause_titles = [m.get('clause_title', 'Unknown') for m in metas]
    
    print(f"\n📋 All Clause Titles:")
    unique_titles = list(dict.fromkeys(clause_titles))  # Preserve order, remove dupes
    for i, title in enumerate(unique_titles, 1):
        count = clause_titles.count(title)
        marker = "⭐" if "Definition:" in title else "  "
        print(f"   {marker} {i:2d}. {title[:70]} ({count})")
    
    # Check for definition patterns in text
    print("\n" + "="*80)
    print("🔍 DEFINITION DETECTION TEST")
    print("="*80)
    
    # Find Clause 1 (DEFINITIONS)
    def_chunks = [(i, doc, meta) for i, (doc, meta) in enumerate(zip(docs, metas)) 
                  if meta.get('clause_number') == '1']
    
    if def_chunks:
        print(f"\n✅ Found {len(def_chunks)} chunk(s) for Clause 1 (DEFINITIONS)")
        
        for idx, (i, doc, meta) in enumerate(def_chunks[:2], 1):  # Show first 2
            print(f"\n   --- Chunk {idx} ---")
            print(f"   Title: {meta.get('clause_title', 'Unknown')}")
            print(f"   Length: {len(doc)} chars")
            print(f"   Has '[TABLE SUMMARY]': {'✅' if '[TABLE SUMMARY]' in doc else '❌'}")
            print(f"   Starts with 'Definition of': {'✅' if doc.startswith('Definition of') else '❌'}")
            
            # Check for quoted terms with "means"
            means_pattern = r'["\"]([^"\"]+)["\"]?\s+means'
            means_matches = re.findall(means_pattern, doc[:2000], re.IGNORECASE)
            if means_matches:
                print(f"\n   Found {len(means_matches)} 'X means' patterns:")
                for term in means_matches[:5]:
                    print(f"      • {term}")
            
            print(f"\n   Preview (first 500 chars):")
            print(f"   {'-'*76}")
            print(f"   {doc[:500]}")
            print(f"   {'-'*76}")
    else:
        print(f"\n❌ No chunks found for Clause 1!")
    
    # Check for table patterns
    print("\n" + "="*80)
    print("🔍 TABLE DETECTION TEST")
    print("="*80)
    
    # Check all chunks for table indicators
    chunks_with_pipes = [(i, doc, meta) for i, (doc, meta) in enumerate(zip(docs, metas)) 
                        if '|' in doc and doc.count('|') >= 5]
    
    print(f"\n📊 Chunks containing | (pipes): {len(chunks_with_pipes)}")
    
    if chunks_with_pipes:
        print(f"\n   Showing first 3 chunks with tables:")
        for idx, (i, doc, meta) in enumerate(chunks_with_pipes[:3], 1):
            print(f"\n   --- Chunk {idx} ---")
            print(f"   Clause: {meta.get('clause_number')} - {meta.get('clause_title', 'Unknown')[:50]}")
            print(f"   Has '[TABLE SUMMARY]': {'✅' if '[TABLE SUMMARY]' in doc else '❌'}")
            
            # Extract table-looking lines
            lines = doc.split('\n')
            table_lines = [line for line in lines if '|' in line and line.count('|') >= 2]
            
            print(f"   Table-like lines: {len(table_lines)}")
            if table_lines:
                print(f"\n   Sample table rows:")
                for line in table_lines[:5]:
                    print(f"      {line[:80]}")
    else:
        print(f"\n❌ No chunks with table patterns found")
        print(f"   This suggests tables either:")
        print(f"   1. Don't exist in the document")
        print(f"   2. Are formatted differently (not markdown tables)")
        print(f"   3. Were not preserved during parsing")
    
    # Summary
    print("\n" + "="*80)
    print("📋 DIAGNOSTIC SUMMARY")
    print("="*80)
    
    print(f"\n**Current Status:**")
    print(f"   Total chunks: {count}")
    print(f"   Table-enhanced: {has_table_count} (expected: 5-15)")
    print(f"   Definition-prefix: {is_definition_count} (expected: 10-30)")
    
    print(f"\n**Why enhancements didn't work:**")
    if has_table_count == 0:
        print(f"   ❌ No tables detected")
        print(f"      → Table detection regex didn't match")
        print(f"      → Check 'TABLE DETECTION TEST' above")
    
    if is_definition_count == 0:
        print(f"   ❌ No definition prefixes added")
        print(f"      → Definition extraction regex didn't match")
        print(f"      → Check 'DEFINITION DETECTION TEST' above")
    
    print(f"\n**Next steps:**")
    if has_table_count == 0 and is_definition_count == 0:
        print(f"   1. Review the patterns shown above")
        print(f"   2. Re-run ingestion to get cache file:")
        print(f"      rm -f cache/* ingestion_tracker.db")
        print(f"      python pipeline/ingest_01.py")
        print(f"   3. Run: python diagnose_structure.py")
        print(f"   4. Share the output → I'll fix the regex patterns")
    else:
        print(f"   Some enhancements worked! Check which ones and why.")

if __name__ == "__main__":
    main()
