#!/usr/bin/env python3
"""
CUSTOM INDEXER FOR SOLAR PPA FORMAT
- Handles " Term " format (spaces inside quotes) ← KEY FIX!
- No (a) (b) (c) prefixes required
- Table-aware chunking with synthetic sentences
- Definition prefix injection
"""
import sys
if 'utils.storage_utils' in sys.modules:
    del sys.modules['utils.storage_utils']

import os
os.environ['LLAMA_INDEX_LOGGING_LEVEL'] = 'WARNING'

import logging
logging.basicConfig(level=logging.WARNING)

import re
import json
import chromadb
from dotenv import load_dotenv

from llama_index.core import Settings
Settings.llm = None

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.storage_utils import CACHE_DIR, CHROMA_DB_PATH

os.makedirs(CACHE_DIR, exist_ok=True)

def detect_markdown_table(lines, start_idx):
    """Detect markdown table with separator line"""
    if start_idx >= len(lines):
        return False, start_idx
    
    # Check for table separator (|---|---|)
    potential_separator_idx = start_idx + 1
    if potential_separator_idx < len(lines):
        line = lines[potential_separator_idx].strip()
        if re.match(r'^\|[\s\-:|]+\|$', line):
            # Found separator, find end of table
            end_idx = potential_separator_idx + 1
            while end_idx < len(lines) and '|' in lines[end_idx]:
                end_idx += 1
            return True, end_idx - 1
    
    return False, start_idx

def extract_table_data(lines, start_idx, end_idx):
    """Extract key-value pairs from markdown table"""
    data = {}
    
    for i in range(start_idx, end_idx + 1):
        line = lines[i].strip()
        if not line or re.match(r'^\|[\s\-:|]+\|$', line):
            continue
        
        cells = [cell.strip() for cell in line.split('|') if cell.strip()]
        
        if len(cells) >= 2:
            key = re.sub(r'\*\*', '', cells[0].strip())
            value = re.sub(r'\*\*', '', cells[1].strip())
            
            # Skip placeholder values
            if key and value and value not in ['[ ● ]', '[●]', '[ ● ]']:
                data[key] = value
    
    return data

def create_synthetic_sentences_from_table(table_data, table_title=""):
    """Convert table to natural language sentences"""
    sentences = []
    
    for key, value in table_data.items():
        # Skip header rows
        if key.lower() in ['subject', 'clause', 'key information', 'item', 'description']:
            continue
        
        # Create natural sentence
        if value.lower() in ['yes', 'no', 'n/a', 'nil']:
            sentence = f"{key} is {value}."
        elif '%' in value or 'per' in value.lower():
            sentence = f"The {key} is {value}."
        elif any(word in key.lower() for word in ['date', 'period', 'time']):
            sentence = f"The {key} is {value}."
        elif any(word in key.lower() for word in ['rate', 'charge', 'amount', 'capacity', 'factor', 'limit']):
            sentence = f"The {key} is {value}."
        else:
            sentence = f"{key}: {value}."
        
        sentences.append(sentence)
    
    result = " ".join(sentences)
    if table_title:
        result = f"From {table_title}: {result}"
    
    return result

def extract_definitions_from_text(text):
    """
    Extract definitions from text using format: " Term " means
    
    CRITICAL: This PDF uses spaces inside quotes!
    Pattern: " Abandonment " means:
    NOT: "Abandonment" means
    """
    definitions = []
    
    # Split into lines for processing
    lines = text.split('\n')
    
    current_term = None
    current_def_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # Pattern: " TERM " means (with spaces inside quotes)
        # [\s]* allows any amount of whitespace inside the quotes
        # [A-Z][^"\"]* captures term starting with capital, up to closing quote
        match = re.search(r'["\"][\s]*([A-Z][^"\"]*?)[\s]*["\"][\s]*means', line_stripped, re.IGNORECASE)
        
        if match:
            # Save previous definition
            if current_term and current_def_lines:
                definition_text = '\n'.join(current_def_lines).strip()
                if definition_text:
                    definitions.append((current_term.strip(), definition_text))
            
            # Start new definition
            current_term = match.group(1).strip()
            current_def_lines = [line_stripped]
        elif current_term:
            # Continue current definition
            if line_stripped:
                current_def_lines.append(line_stripped)
    
    # Save last definition
    if current_term and current_def_lines:
        definition_text = '\n'.join(current_def_lines).strip()
        if definition_text:
            definitions.append((current_term.strip(), definition_text))
    
    return definitions

def split_into_enhanced_clauses(full_text, filename="document"):
    """Enhanced splitting with custom patterns"""
    lines = full_text.split('\n')
    clauses = []
    
    # Find all numbered sections (## 1. TITLE or ## 1 TITLE)
    section_indices = []
    for i, line in enumerate(lines):
        if re.match(r'^##\s+\d+\.?\s+[^\n]+', line.strip()):
            section_indices.append((i, line.strip()))
    
    if not section_indices:
        print(f"   ⚠️  No numbered sections - using semantic chunking")
        return semantic_fallback(full_text)
    
    print(f"   ✅ Detected pattern: ## N. TITLE ({len(section_indices)} sections)")
    
    # Process each section
    for idx, (start_line_num, header) in enumerate(section_indices):
        # Determine section boundaries
        if idx + 1 < len(section_indices):
            end_line_num = section_indices[idx + 1][0]
        else:
            end_line_num = len(lines)
        
        # Extract clause number and title
        number_match = re.search(r'\d+', header)
        clause_number = number_match.group(0) if number_match else str(idx + 1)
        
        title_match = re.search(r'##\s+\d+\.?\s+(.+)', header)
        clause_title = title_match.group(1).strip() if title_match else header
        
        # Get full clause text
        clause_lines = lines[start_line_num:end_line_num]
        clause_text = '\n'.join(clause_lines).strip()
        
        if len(clause_text) < 50:
            continue
        
        # --- TABLE DETECTION AND ENHANCEMENT ---
        has_table = False
        i = 0
        while i < len(clause_lines):
            is_table, table_end = detect_markdown_table(clause_lines, i)
            if is_table:
                has_table = True
                table_data = extract_table_data(clause_lines, i, table_end)
                if table_data:
                    synthetic_text = create_synthetic_sentences_from_table(table_data, clause_title)
                    clause_text = f"[TABLE SUMMARY] {synthetic_text}\n\n{clause_text}"
                i = table_end + 1
            else:
                i += 1
        
        # --- DEFINITION EXTRACTION ---
        if 'definition' in clause_title.lower():
            definitions = extract_definitions_from_text(clause_text)
            
            if definitions:
                print(f"   📖 Found {len(definitions)} definitions in Clause {clause_number}")
                
                # Create separate node for each definition
                for term, definition in definitions:
                    enhanced_text = f"Definition of {term}: {definition}"
                    clauses.append((
                        clause_number, 
                        f"Definition: {term}", 
                        enhanced_text
                    ))
                
                # Keep full clause too
                clauses.append((clause_number, clause_title, clause_text))
            else:
                # No definitions extracted, keep as-is
                clauses.append((clause_number, clause_title, clause_text))
        else:
            # Regular clause
            clauses.append((clause_number, clause_title, clause_text))
    
    return clauses

def semantic_fallback(text, chunk_size=1500, chunk_overlap=200):
    """Fallback to semantic chunking if no structure detected"""
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = Document(text=text)
    chunks = splitter.get_nodes_from_documents([doc])
    
    clauses = []
    for i, chunk in enumerate(chunks):
        first_line = chunk.text.split('\n')[0][:50]
        title = first_line if first_line else f"Chunk {i+1}"
        clauses.append((str(i+1), title, chunk.text))
    
    return clauses

def main():
    load_dotenv()
    
    print("="*80)
    print("🔧 CUSTOM INDEXER: Solar PPA Format")
    print("="*80)
    print(f"📁 Cache: {CACHE_DIR}")
    print(f"📁 ChromaDB: {CHROMA_DB_PATH}")
    
    # 1. Load embeddings
    print("\n🔄 Loading BGE-M3...")
    try:
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
        print("✅ Embedding model loaded")
    except Exception as e:
        print(f"❌ Failed: {e}")
        raise

    # 2. Connect to ChromaDB
    print("\n🔄 Connecting to ChromaDB...")
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection_name = "solar_ppa_collection"
    
    try:
        chroma_collection = db.get_collection(collection_name)
        print(f"📚 Using existing collection ({chroma_collection.count()} vectors)")
    except:
        chroma_collection = db.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ Created new collection")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. Process cache files
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.json')]
    
    if not cache_files:
        print("\n⏭️  No new cache files")
        return

    print(f"\n📦 Processing {len(cache_files)} document(s)")
    print("="*80)

    all_documents = []

    for cf in cache_files:
        cache_path = os.path.join(CACHE_DIR, cf)
        print(f"\n📄 {cf}")
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
        except Exception as e:
            print(f"   ❌ Failed to read: {e}")
            continue
        
        if not cache_data or not isinstance(cache_data, list):
            continue
            
        full_doc = cache_data[0]
        full_text = full_doc.get('text', '')
        base_metadata = full_doc.get('metadata', {})
        filename = base_metadata.get('filename', 'unknown.pdf')
        
        if len(full_text) < 100:
            continue
        
        # CUSTOM SPLITTING
        clauses = split_into_enhanced_clauses(full_text, filename)
        
        print(f"   📊 Generated {len(clauses)} total chunks")
        
        # Track enhancements
        table_chunks = sum(1 for _, _, text in clauses if '[TABLE SUMMARY]' in text)
        definition_chunks = sum(1 for _, title, _ in clauses if 'Definition:' in title)
        
        if table_chunks > 0:
            print(f"   📊 {table_chunks} table-enhanced chunks")
        if definition_chunks > 0:
            print(f"   📖 {definition_chunks} definition chunks with prefix")
        
        # Create Documents
        for clause_number, clause_title, clause_text in clauses:
            clause_metadata = base_metadata.copy()
            clause_metadata.update({
                'clause_number': clause_number,
                'clause_title': clause_title,
                'source_cache': cf,
                'filename': filename,
                'chunk_type': 'enhanced_clause',
                'has_table': '[TABLE SUMMARY]' in clause_text,
                'is_definition': 'Definition:' in clause_title
            })
            
            doc = Document(text=clause_text, metadata=clause_metadata)
            all_documents.append(doc)

    if not all_documents:
        print("\n⚠️  No documents to index")
        return

    # 4. Index
    print("\n" + "="*80)
    print(f"🚀 Indexing {len(all_documents)} enhanced chunks")
    print("="*80)
    
    try:
        index = VectorStoreIndex.from_documents(
            all_documents,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
            insert_batch_size=20
        )
        print("\n✅ Indexing complete")
    except Exception as e:
        print(f"\n❌ Indexing failed: {e}")
        raise
    
    # 5. Clean up
    print("\n🗑️  Cleaning up...")
    for cf in cache_files:
        try:
            os.remove(os.path.join(CACHE_DIR, cf))
            print(f"   ✓ {cf}")
        except Exception as e:
            print(f"   ⚠️  {e}")
    
    # 6. Stats
    final_count = chroma_collection.count()
    print("\n" + "="*80)
    print(f"✅ COMPLETE - {final_count} total vectors")
    print("   Enhancements:")
    print(f"   • Custom definition extraction (' Term ' format)")
    print(f"   • Table-aware chunking with synthetic sentences")
    print(f"   • Definition prefix: 'Definition of X:'")
    print("="*80)

if __name__ == "__main__":
    main()