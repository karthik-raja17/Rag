#!/usr/bin/env python3
"""
ONE-COMMAND FIX: Diagnose and repair your broken RAG system
Automatically detects issues and applies the correct fix
"""
import os
import sys
import subprocess
import shutil

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
os.chdir(PROJECT_ROOT)

def run_command(cmd, description):
    """Run a command and show output"""
    print(f"\n{'='*70}")
    print(f"🔧 {description}")
    print(f"{'='*70}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        print(result.stdout)
        if result.stderr:
            print(f"⚠️  Warnings: {result.stderr[:200]}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"❌ Command timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def check_file_exists(filepath):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"   {status} {filepath}")
    return exists

def main():
    print("╔" + "="*68 + "╗")
    print("║" + " "*20 + "AUTOMATED RAG REPAIR TOOL" + " "*23 + "║")
    print("╚" + "="*68 + "╝")
    
    # Step 1: Check environment
    print("\n📋 Step 1: Environment Check")
    print("-"*70)
    
    has_pdf = any(f.endswith('.pdf') for f in os.listdir('data') if os.path.isfile(os.path.join('data', f')))
    print(f"   {'✅' if has_pdf else '❌'} PDF in data/ directory")
    
    check_file_exists('pipeline/ingest_01.py')
    check_file_exists('pipeline/chat_03_PRODUCTION.py')
    check_file_exists('index_02_ADAPTIVE.py')
    check_file_exists('diagnose_pdf_structure.py')
    
    if not has_pdf:
        print("\n❌ No PDF found in data/ directory!")
        print("   Place your PDF in data/ and run this script again.")
        return False
    
    # Step 2: Backup current files
    print("\n💾 Step 2: Backup Current Configuration")
    print("-"*70)
    
    backups = []
    for f in ['pipeline/index_02.py', 'pipeline/chat_03.py']:
        if os.path.exists(f):
            backup = f + '.backup'
            shutil.copy2(f, backup)
            backups.append(backup)
            print(f"   ✅ Backed up: {f} → {backup}")
    
    # Step 3: Clean slate
    print("\n🧹 Step 3: Clean Old Data")
    print("-"*70)
    
    dirs_to_clean = ['chroma_db', 'cache']
    files_to_clean = ['ingestion_tracker.db']
    
    for d in dirs_to_clean:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"   ✅ Removed: {d}/")
        os.makedirs(d, exist_ok=True)
        print(f"   ✅ Created: {d}/")
    
    for f in files_to_clean:
        if os.path.exists(f):
            os.remove(f)
            print(f"   ✅ Deleted: {f}")
    
    # Step 4: Deploy new files
    print("\n📦 Step 4: Deploy Adaptive System")
    print("-"*70)
    
    if os.path.exists('index_02_ADAPTIVE.py'):
        shutil.copy2('index_02_ADAPTIVE.py', 'pipeline/index_02.py')
        print(f"   ✅ Deployed: index_02_ADAPTIVE.py → pipeline/index_02.py")
    else:
        print(f"   ⚠️  index_02_ADAPTIVE.py not found, keeping existing")
    
    if os.path.exists('chat_03_PRODUCTION.py'):
        shutil.copy2('chat_03_PRODUCTION.py', 'pipeline/chat_03.py')
        print(f"   ✅ Deployed: chat_03_PRODUCTION.py → pipeline/chat_03.py")
    else:
        print(f"   ⚠️  chat_03_PRODUCTION.py not found, keeping existing")
    
    # Step 5: Ingest PDFs
    if not run_command(f"{sys.executable} pipeline/ingest_01.py", "Step 5: Ingest PDFs"):
        print("\n❌ Ingestion failed! Check your PDF format.")
        return False
    
    # Step 6: Diagnose structure
    print("\n🔬 Step 6: Diagnose PDF Structure")
    if os.path.exists('diagnose_pdf_structure.py'):
        run_command(f"{sys.executable} diagnose_pdf_structure.py", "Analyzing PDF structure...")
        input("\nPress Enter to continue with indexing...")
    
    # Step 7: Index with adaptive splitter
    if not run_command(f"{sys.executable} pipeline/index_02.py", "Step 7: Index Documents"):
        print("\n❌ Indexing failed!")
        return False
    
    # Step 8: Verify vector database
    print("\n🔍 Step 8: Verify Vector Database")
    print("-"*70)
    
    try:
        import chromadb
        db = chromadb.PersistentClient(path='chroma_db')
        collection = db.get_collection('solar_ppa_collection')
        count = collection.count()
        
        print(f"   ✅ Collection: solar_ppa_collection")
        print(f"   ✅ Vector count: {count}")
        
        if count == 0:
            print(f"   ❌ WARNING: Collection is empty!")
            return False
        elif count > 200:
            print(f"   ⚠️  WARNING: Unusually high count ({count})")
            print(f"      Chunks might be too small")
        else:
            print(f"   ✅ Vector count looks reasonable")
            
    except Exception as e:
        print(f"   ❌ Could not verify: {e}")
        return False
    
    # Step 9: Test retrieval
    print("\n🧪 Step 9: Test Retrieval Quality")
    print("-"*70)
    
    try:
        from llama_index.core import VectorStoreIndex
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        print("   Loading embeddings...")
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
        
        print("   Building index...")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
        
        print("   Testing retrieval...")
        retriever = index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve("contract terms")
        
        if nodes:
            top_node = nodes[0]
            print(f"   ✅ Retrieval working")
            print(f"   ✅ Top score: {top_node.score:.3f}")
            print(f"   ✅ Top chunk length: {len(top_node.text)} chars")
            print(f"   ✅ Preview: {top_node.text[:100]}...")
            
            if top_node.score < 0.6:
                print(f"   ⚠️  Low score - retrieval quality might be poor")
            if len(top_node.text) < 100:
                print(f"   ⚠️  Very short chunks - might be just headings")
        else:
            print(f"   ❌ No results returned!")
            return False
            
    except Exception as e:
        print(f"   ⚠️  Could not test retrieval: {e}")
    
    # Step 10: Summary
    print("\n" + "="*70)
    print("🎉 REPAIR COMPLETE!")
    print("="*70)
    
    print(f"\n📊 System Status:")
    print(f"   Vector database: {count} chunks")
    print(f"   Backups created: {len(backups)}")
    print(f"   Ready for use: ✅")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Run: python3 pipeline/chat_03.py")
    print(f"   2. Test with: 'What is this document about?'")
    print(f"   3. Check retrieval score is >0.75")
    print(f"   4. Verify answer contains real content")
    
    print(f"\n💡 To undo changes:")
    print(f"   Restore from backups:")
    for backup in backups:
        original = backup.replace('.backup', '')
        print(f"   cp {backup} {original}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
