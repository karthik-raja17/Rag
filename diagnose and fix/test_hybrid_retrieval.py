#!/usr/bin/env python3
"""
VALIDATION SCRIPT: Test hybrid retrieval improvements
Compares vector-only vs hybrid retrieval for known problem queries
"""
import os
import sys
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
import chromadb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db")

def hybrid_retrieve(vector_retriever, bm25_retriever, query_str, top_k=5):
    """Hybrid retrieval with score normalization"""
    query_bundle = QueryBundle(query_str=query_str)
    
    vector_nodes = vector_retriever.retrieve(query_bundle)
    bm25_nodes = bm25_retriever.retrieve(query_bundle)
    
    node_scores = {}
    
    for node in vector_nodes:
        node_scores[node.node_id] = {
            'node': node,
            'vector_score': node.score,
            'bm25_score': 0.0
        }
    
    if bm25_nodes:
        max_bm25 = max(node.score for node in bm25_nodes)
        min_bm25 = min(node.score for node in bm25_nodes)
        bm25_range = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0
        
        for node in bm25_nodes:
            normalized_bm25 = (node.score - min_bm25) / bm25_range
            
            if node.node_id in node_scores:
                node_scores[node.node_id]['bm25_score'] = normalized_bm25
            else:
                node_scores[node.node_id] = {
                    'node': node,
                    'vector_score': 0.0,
                    'bm25_score': normalized_bm25
                }
    
    combined_nodes = []
    for node_id, scores in node_scores.items():
        combined_score = (0.6 * scores['vector_score']) + (0.4 * scores['bm25_score'])
        node = scores['node']
        node.score = combined_score
        combined_nodes.append(node)
    
    combined_nodes.sort(key=lambda x: x.score, reverse=True)
    return combined_nodes[:top_k]

def test_query(vector_retriever, bm25_retriever, query, expected_clause=None):
    """Test a single query with both methods"""
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}")
    
    # Vector-only
    print(f"\n📊 VECTOR-ONLY RETRIEVAL:")
    vector_nodes = vector_retriever.retrieve(QueryBundle(query_str=query))
    
    for i, node in enumerate(vector_nodes[:3]):
        clause_num = node.metadata.get('clause_number', '?')
        clause_title = node.metadata.get('clause_title', 'Unknown')[:50]
        print(f"   {i+1}. Clause {clause_num}: {clause_title}")
        print(f"      Score: {node.score:.3f}")
        print(f"      Preview: {node.text[:100]}...")
    
    # Hybrid
    print(f"\n🔬 HYBRID RETRIEVAL (BM25 + Vector):")
    try:
        hybrid_nodes = hybrid_retrieve(vector_retriever, bm25_retriever, query, top_k=5)
        
        for i, node in enumerate(hybrid_nodes[:3]):
            clause_num = node.metadata.get('clause_number', '?')
            clause_title = node.metadata.get('clause_title', 'Unknown')[:50]
            print(f"   {i+1}. Clause {clause_num}: {clause_title}")
            print(f"      Score: {node.score:.3f}")
            print(f"      Preview: {node.text[:100]}...")
        
        # Check if expected clause is in top 3
        if expected_clause:
            found = any(node.metadata.get('clause_number') == expected_clause 
                       for node in hybrid_nodes[:3])
            
            if found:
                print(f"\n   ✅ SUCCESS: Found expected Clause {expected_clause} in top 3")
            else:
                print(f"\n   ❌ MISS: Expected Clause {expected_clause} not in top 3")
                
                # Check if it's in top 5
                found_in_5 = any(node.metadata.get('clause_number') == expected_clause 
                                for node in hybrid_nodes[:5])
                if found_in_5:
                    print(f"      (But it's in top 5)")
    
    except Exception as e:
        print(f"   ⚠️  Hybrid retrieval failed: {e}")
        print(f"      This is normal on first run while BM25 index builds")

def main():
    load_dotenv()
    
    print("🧪 HYBRID RETRIEVAL VALIDATION TEST")
    print("="*80)
    print("This compares vector-only vs hybrid (BM25+vector) retrieval")
    print("for queries that previously failed.")
    print("="*80)
    
    # Load embeddings
    print("\n🔄 Loading BGE-M3...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    
    # Connect to ChromaDB
    print(f"📁 Chroma DB: {CHROMA_DB_PATH}")
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    try:
        collection = db.get_collection("solar_ppa_collection")
        print(f"✅ Collection: {collection.count()} vectors")
    except:
        print("❌ Collection not found!")
        return
    
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    
    # Create retrievers
    print("\n🔄 Creating retrievers...")
    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
    
    print("🔄 Building BM25 index (may take a moment)...")
    try:
        bm25_retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=10)
        print("✅ BM25 retriever ready")
    except Exception as e:
        print(f"❌ BM25 failed: {e}")
        print("   Will only test vector retrieval")
        bm25_retriever = None
    
    # Test queries
    print("\n" + "="*80)
    print("RUNNING TEST SUITE")
    print("="*80)
    
    test_cases = [
        {
            'query': "What is the Effective Date?",
            'expected': "1",  # Clause 1.1 - Definitions
            'description': "Definition query - should retrieve Clause 1.1, not usage"
        },
        {
            'query': "What is the Default Rate?",
            'expected': None,  # Unknown - depends on PDF structure
            'description': "Table query - hybrid should help find it"
        },
        {
            'query': "Who is responsible for the Grid?",
            'expected': "4",  # Clause 4.4
            'description': "Control test - should work with both methods"
        },
        {
            'query': "What does Abandonment mean?",
            'expected': "1",  # Clause 1.1 - Definitions
            'description': "Definition query - should work, hybrid might improve score"
        }
    ]
    
    results = {'improved': 0, 'same': 0, 'degraded': 0}
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n\nTEST {i}/{len(test_cases)}: {test['description']}")
        
        if bm25_retriever:
            test_query(vector_retriever, bm25_retriever, test['query'], test['expected'])
        else:
            print(f"\nQuery: {test['query']}")
            print("⚠️  BM25 not available, skipping hybrid test")
            nodes = vector_retriever.retrieve(QueryBundle(query_str=test['query']))
            for j, node in enumerate(nodes[:3]):
                print(f"{j+1}. {node.metadata.get('clause_title', 'Unknown')[:50]} (score: {node.score:.3f})")
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    if bm25_retriever:
        print("""
Expected improvements with hybrid retrieval:
1. Definition queries should retrieve Clause 1.1 (definitions)
2. Exact term queries should have higher scores
3. Table queries might work (depends on chunking)

If you see improvements, deploy hybrid chat:
   cp chat_03_HYBRID.py pipeline/chat_03.py

For even better results, deploy enhanced indexer:
   cp index_02_ENHANCED.py pipeline/index_02.py
   (requires re-indexing)
""")
    else:
        print("""
⚠️  BM25 retriever failed to build. This can happen if:
   - Nodes have no text content
   - Collection is empty
   - First-time indexing issue

Try running the test again. BM25 sometimes works on second try.
""")

if __name__ == "__main__":
    main()
