import os
import json
import sqlite3
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime

# --- ABSOLUTE PATH CONFIGURATION ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, "Dataset")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db")
TRACKER_DB = os.path.join(PROJECT_ROOT, "ingestion_tracker.db")

def ensure_environment():
    """Create all necessary directories if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    print(f"✅ Environment ready:")
    print(f"   📁 Data: {DATA_DIR}")
    print(f"   📁 Cache: {CACHE_DIR}")
    print(f"   📁 ChromaDB: {CHROMA_DB_PATH}")
    print(f"   🗄️  Tracker: {TRACKER_DB}")

def calculate_file_hash(file_path: str, chunk_size: int = 65536) -> str:
    """
    Generate SHA-256 hash of a file.
    
    Args:
        file_path: Path to the file
        chunk_size: Size of chunks to read (default 64KB)
    
    Returns:
        Hexadecimal SHA-256 hash string
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def init_tracker_db():
    """
    Initialize SQLite database for tracking processed files.
    Creates table if it doesn't exist.
    """
    conn = sqlite3.connect(TRACKER_DB)
    c = conn.cursor()
    c.execute(''' 
        CREATE TABLE IF NOT EXISTS parsed_files (
            file_hash TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            json_path TEXT NOT NULL,
            parsing_parameters TEXT DEFAULT 'markdown',
            processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_size INTEGER,
            page_count INTEGER DEFAULT 0
        )
    ''')
    
    # Create index for faster lookups
    c.execute('''
        CREATE INDEX IF NOT EXISTS idx_file_hash 
        ON parsed_files (file_hash)
    ''')
    
    conn.commit()
    conn.close()
    print(f"✅ Tracker database initialized at: {TRACKER_DB}")

def is_file_processed(file_hash: str) -> bool:
    """
    Check if a file has already been processed.
    
    Args:
        file_hash: SHA-256 hash of the file
    
    Returns:
        True if file exists in database, False otherwise
    """
    if not os.path.exists(TRACKER_DB):
        init_tracker_db()
        return False
    
    conn = sqlite3.connect(TRACKER_DB)
    c = conn.cursor()
    c.execute("SELECT 1 FROM parsed_files WHERE file_hash = ?", (file_hash,))
    result = c.fetchone()
    conn.close()
    return result is not None

def save_to_cache(file_hash: str, llama_documents: List[Any], metadata: Optional[Dict] = None) -> str:
    """
    Save processed documents to JSON cache.
    
    Args:
        file_hash: SHA-256 hash of the source file
        llama_documents: List of LlamaIndex Document objects
        metadata: Optional additional metadata to store
    
    Returns:
        Path to the saved cache file
    """
    ensure_environment()
    
    cache_file_name = f"{file_hash}.json"
    full_cache_path = os.path.join(CACHE_DIR, cache_file_name)
    
    # Convert documents to serializable format
    data_to_save = []
    for doc in llama_documents:
        doc_dict = {
            'text': doc.text,
            'metadata': {
                **doc.metadata,
                'file_hash': file_hash,
                'cached_date': datetime.now().isoformat()
            }
        }
        data_to_save.append(doc_dict)
    
    # Add any additional metadata
    if metadata:
        data_to_save[0]['metadata'].update(metadata)
    
    with open(full_cache_path, "w", encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"   💾 Cached: {os.path.basename(full_cache_path)}")
    return full_cache_path

def register_in_db(
    file_hash: str, 
    file_name: str, 
    cache_path: str, 
    parsing_parameters: str = "markdown",
    file_size: Optional[int] = None,
    page_count: int = 0
):
    """
    Register a processed file in the tracking database.
    
    Args:
        file_hash: SHA-256 hash of the file
        file_name: Original filename
        cache_path: Path to the cached JSON file
        parsing_parameters: Method used for parsing
        file_size: Size of the original file in bytes
        page_count: Number of pages in the document
    """
    if not os.path.exists(TRACKER_DB):
        init_tracker_db()
    
    conn = sqlite3.connect(TRACKER_DB)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO parsed_files 
        (file_hash, file_name, json_path, parsing_parameters, processed_date, file_size, page_count)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
    """, (file_hash, file_name, cache_path, parsing_parameters, file_size, page_count))
    conn.commit()
    conn.close()
    print(f"   🗄️ Registered in tracker: {file_name}")

def get_cached_path(file_hash: str) -> Optional[str]:
    """
    Get the cache file path for a given file hash.
    
    Args:
        file_hash: SHA-256 hash of the file
    
    Returns:
        Path to cache file if exists, None otherwise
    """
    cache_path = os.path.join(CACHE_DIR, f"{file_hash}.json")
    return cache_path if os.path.exists(cache_path) else None

def load_from_cache(file_hash: str) -> Optional[List[Dict]]:
    """
    Load cached documents from JSON file.
    
    Args:
        file_hash: SHA-256 hash of the source file
    
    Returns:
        List of document dictionaries if cache exists, None otherwise
    """
    cache_path = get_cached_path(file_hash)
    if not cache_path:
        return None
    
    with open(cache_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_all_processed_hashes() -> List[str]:
    """
    Get all file hashes that have been processed.
    
    Returns:
        List of file hash strings
    """
    if not os.path.exists(TRACKER_DB):
        init_tracker_db()
        return []
    
    conn = sqlite3.connect(TRACKER_DB)
    c = conn.cursor()
    c.execute("SELECT file_hash FROM parsed_files")
    results = [row[0] for row in c.fetchall()]
    conn.close()
    return results

def cleanup_cache(keep_hashes: Optional[List[str]] = None):
    """
    Remove orphaned cache files (files with no DB entry).
    
    Args:
        keep_hashes: Optional list of hashes to keep even if not in DB
    """
    ensure_environment()
    
    # Get all hashes in DB
    db_hashes = set(get_all_processed_hashes())
    if keep_hashes:
        db_hashes.update(keep_hashes)
    
    # Check all cache files
    removed = 0
    for filename in os.listdir(CACHE_DIR):
        if filename.endswith('.json'):
            file_hash = filename[:-5]  # Remove .json
            if file_hash not in db_hashes:
                file_path = os.path.join(CACHE_DIR, filename)
                os.remove(file_path)
                removed += 1
                print(f"   🗑️ Removed orphaned cache: {filename}")
    
    if removed > 0:
        print(f"   ✅ Cleaned {removed} orphaned cache files")
    else:
        print("   ✅ No orphaned cache files found")

def get_parsing_stats() -> Dict[str, Any]:
    """
    Get statistics about parsed files.
    
    Returns:
        Dictionary with statistics
    """
    if not os.path.exists(TRACKER_DB):
        init_tracker_db()
        return {"total_files": 0, "total_pages": 0, "total_size_mb": 0}
    
    conn = sqlite3.connect(TRACKER_DB)
    c = conn.cursor()
    
    # Total files
    c.execute("SELECT COUNT(*) FROM parsed_files")
    total_files = c.fetchone()[0]
    
    # Total pages
    c.execute("SELECT SUM(page_count) FROM parsed_files")
    total_pages = c.fetchone()[0] or 0
    
    # Total size
    c.execute("SELECT SUM(file_size) FROM parsed_files")
    total_size_bytes = c.fetchone()[0] or 0
    total_size_mb = total_size_bytes / (1024 * 1024)
    
    # Most recent file
    c.execute("SELECT file_name, processed_date FROM parsed_files ORDER BY processed_date DESC LIMIT 1")
    latest = c.fetchone()
    
    conn.close()
    
    return {
        "total_files": total_files,
        "total_pages": total_pages,
        "total_size_mb": round(total_size_mb, 2),
        "latest_file": latest[0] if latest else None,
        "latest_date": latest[1] if latest else None
    }

# Initialize everything when module is imported
ensure_environment()
init_tracker_db()

# Export commonly used functions
__all__ = [
    'PROJECT_ROOT', 'DATA_DIR', 'CACHE_DIR', 'CHROMA_DB_PATH', 'TRACKER_DB',
    'ensure_environment',
    'calculate_file_hash',
    'init_tracker_db',
    'is_file_processed',
    'save_to_cache',
    'register_in_db',
    'get_cached_path',
    'load_from_cache',
    'get_all_processed_hashes',
    'cleanup_cache',
    'get_parsing_stats'
]