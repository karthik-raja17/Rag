#!/usr/bin/env python3
import os
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from llama_index.core import Document

# Import storage utilities
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.storage_utils import (
    DATA_DIR, CACHE_DIR,
    calculate_file_hash,
    is_file_processed,
    save_to_cache,
    register_in_db,
    ensure_environment,
    init_tracker_db
)

def main():
    load_dotenv()
    ensure_environment()
    init_tracker_db()
    
    print("🐘 Initializing Local AI Ingestor...")
    
    # Get all PDF files
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
    print(f"🔍 Found {len(pdf_files)} files. Synchronizing...")
    
    converter = DocumentConverter()
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        
        # Check if already processed
        file_hash = calculate_file_hash(pdf_path)
        if is_file_processed(file_hash):
            print(f"⏭️  Skipping {pdf_file} (Already in Database).")
            continue
        
        print(f"📂 Parsing: {pdf_file}...")
        
        try:
            # Convert PDF to markdown
            result = converter.convert(pdf_path)
            markdown_text = result.document.export_to_markdown()
            
            # Create a LlamaIndex Document
            doc = Document(
                text=markdown_text,
                metadata={
                    'filename': pdf_file,
                    'source': pdf_path,
                    'file_hash': file_hash
                }
            )
            
            # Save to cache
            cache_path = save_to_cache(file_hash, [doc])
            
            # Register in tracker
            file_size = os.path.getsize(pdf_path)
            register_in_db(file_hash, pdf_file, cache_path, file_size=file_size)
            
            print(f"   ✅ Processed & Cached.")
            
        except Exception as e:
            print(f"   ❌ Error processing {pdf_file}: {e}")
    
    print("\n🏁 Pipeline complete.")

if __name__ == "__main__":
    main()