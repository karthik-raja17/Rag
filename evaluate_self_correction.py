#!/usr/bin/env python3
"""
EVALUATE SELF-CORRECTION
Compare baseline vs self-correcting system using RAGAS test set
"""
import os
import sys
import pandas as pd
from dotenv import load_dotenv
import chromadb
from groq import Groq

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import TextNode
from pathlib import Path
import hashlib

# Import components
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.adaptive_retriever import AdaptiveRetriever
from pipeline.self_correcting_engine import create_self_correcting_engine

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db")


def hybrid_retrieve(vector_retriever, bm25_retriever, query_str, top_k=5):
    """Hybrid retrieval function."""
    from llama_index.core import QueryBundle
    
    query_bundle = QueryBundle(query_str=query_str)
    vector_nodes = vector_retriever.retrieve(query_bundle)
    
    if bm25_retriever is None:
        return vector_nodes[:top_k]
    
    bm25_nodes = bm25_retriever.retrieve(query_bundle)
    
    node_scores = {}
    
    for node in vector_nodes:
        node_id = node.node_id
        node_scores[node_id] = {
            'node': node,
            'vector_score': node.score,
            'bm25_score': 0.0
        }
    
    if bm25_nodes:
        max_bm25 = max(node.score for node in bm25_nodes)
        min_bm25 = min(node.score for node in bm25_nodes)
        bm25_range = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0
        
        for node in bm25_nodes:
            node_id = node.node_id
            normalized_bm25 = (node.score - min_bm25) / bm25_range
            
            if node_id in node_scores:
                node_scores[node_id]['bm25_score'] = normalized_bm25
            else:
                node_scores[node_id] = {
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


def load_bm25_cache(cache_dir, doc_texts, ids, metadatas, similarity_top_k=10):
    """Load BM25 from cache."""
    cache_dir = Path(cache_dir)
    hash_file = cache_dir / "bm25_hash.txt"
    index_dir = cache_dir / "bm25_index"
    
    if not hash_file.exists() or not index_dir.exists():
        return None
    
    def compute_hash(texts):
        if not texts:
            return "empty"
        combined = "".join(texts).encode("utf-8")
        return hashlib.sha256(combined).hexdigest()
    
    with open(hash_file, "r") as f:
        saved_hash = f.read().strip()
    current_hash = compute_hash(doc_texts)
    
    if saved_hash != current_hash:
        return None
    
    try:
        import bm25s
        
        bm25 = bm25s.BM25.load(str(index_dir))
        
        nodes = [
            TextNode(id_=node_id, text=text, metadata=meta or {})
            for node_id, text, meta in zip(ids, doc_texts, metadatas)
        ]
        
        retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=similarity_top_k
        )
        retriever.bm25 = bm25
        
        return retriever
    except:
        return None


def setup_retrievers():
    """Setup retrievers for evaluation."""
    print("\n🔄 Setting up retrievers...")
    
    # Load embeddings
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    
    # Connect to ChromaDB
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = db.get_collection("solar_ppa_collection")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
        embed_model=embed_model
    )
    
    # Load BM25
    all_docs = chroma_collection.get(include=["documents", "metadatas"])
    ids = all_docs["ids"]
    doc_texts = all_docs["documents"]
    metadatas = all_docs["metadatas"]
    
    cache_dir = os.path.join(PROJECT_ROOT, "cache", "bm25_cache")
    bm25_retriever = load_bm25_cache(cache_dir, doc_texts, ids, metadatas, 10)
    
    if bm25_retriever is None:
        print("⚠️  BM25 cache not found, building...")
        nodes = [
            TextNode(id_=node_id, text=text, metadata=meta or {})
            for node_id, text, meta in zip(ids, doc_texts, metadatas)
        ]
        bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)
    
    print("✅ Retrievers ready")
    
    return vector_retriever, bm25_retriever


def run_evaluation(test_set_path: str, enable_correction: bool = True):
    """
    Run evaluation on test set.
    
    Args:
        test_set_path: Path to testset CSV
        enable_correction: Enable/disable self-correction
    
    Returns:
        DataFrame with results
    """
    load_dotenv()
    
    print("=" * 70)
    print(f"🧪 EVALUATION: {'Self-Correcting' if enable_correction else 'Baseline'} Mode")
    print("=" * 70)
    
    # Load test set
    print(f"\n📄 Loading test set: {test_set_path}")
    df = pd.read_csv(test_set_path)
    print(f"✅ Loaded {len(df)} test cases")
    
    # Setup retrievers
    vector_retriever, bm25_retriever = setup_retrievers()
    
    # Setup Groq
    groq_api_key = os.getenv("GROQ_API_KEY")
    analysis_llm = Groq(api_key=groq_api_key)
    
    # Create adaptive retriever
    print("\n🔄 Creating adaptive retriever...")
    adaptive_retriever = AdaptiveRetriever(
        hybrid_retrieve_fn=hybrid_retrieve,
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        analysis_llm=analysis_llm,
        verbose=False
    )
    
    # Create query engine
    print(f"🔄 Creating query engine (correction={'ON' if enable_correction else 'OFF'})...")
    
    system_prompt = """You are an expert legal consultant specializing in Solar Power Purchase Agreements (PPAs).

Answer based ONLY on the provided clauses.

RULES:
1. ALWAYS cite clause numbers (e.g., "As per Clause 1.1...")
2. If information not in clauses: "I cannot find this information in the provided clauses."
3. Be precise and professional

Answer concisely and clearly."""
    
    query_engine = create_self_correcting_engine(
        retriever=adaptive_retriever,
        groq_api_key=groq_api_key,
        system_prompt=system_prompt,
        enable_correction=enable_correction,
        max_correction_attempts=2,
        faithfulness_threshold=0.9,
        relevancy_threshold=0.8,
        verbose=False  # Silent for batch processing
    )
    
    print("✅ Query engine ready")
    
    # Run evaluation
    print(f"\n🚀 Running evaluation on {len(df)} questions...")
    print("=" * 70)
    
    results = []
    
    for idx, row in df.iterrows():
        question = row['user_input']
        
        print(f"\n[{idx+1}/{len(df)}] {question[:60]}...")
        
        try:
            # Query
            response = query_engine.query(question)
            
            answer = response.response
            metadata = response.metadata
            
            # Extract contexts
            contexts = [node.text for node in response.source_nodes]
            
            # Record result
            result = {
                'question': question,
                'answer': answer,
                'contexts': contexts,
                'ground_truth': row.get('reference', ''),
                'correction_enabled': enable_correction
            }
            
            # Add correction metadata if available
            if enable_correction and metadata.get('correction_enabled'):
                result['correction_attempts'] = metadata.get('correction_attempts', 0)
                result['corrections_made'] = ','.join(metadata.get('corrections_made', []))
                result['final_faithfulness'] = metadata.get('final_grading', {}).get('faithfulness', 0)
                result['final_relevancy'] = metadata.get('final_grading', {}).get('relevancy', 0)
                result['correction_success'] = metadata.get('correction_success', False)
            
            results.append(result)
            
            print(f"   ✓ Answer generated")
            if enable_correction:
                attempts = metadata.get('correction_attempts', 0)
                if attempts > 1:
                    print(f"   → {attempts} correction attempts")
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            results.append({
                'question': question,
                'answer': f"ERROR: {str(e)}",
                'contexts': [],
                'ground_truth': row.get('reference', ''),
                'correction_enabled': enable_correction,
                'error': str(e)
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    output_file = f"evaluation_results_{'corrected' if enable_correction else 'baseline'}.csv"
    results_df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 70)
    print(f"✅ Evaluation complete!")
    print(f"📁 Results saved to: {output_file}")
    print("=" * 70)
    
    # Summary statistics
    if enable_correction:
        total_corrections = results_df['correction_attempts'].sum()
        avg_corrections = results_df['correction_attempts'].mean()
        success_rate = results_df['correction_success'].mean() * 100 if 'correction_success' in results_df else 0
        
        print(f"\n📊 Self-Correction Statistics:")
        print(f"   Total correction attempts: {total_corrections}")
        print(f"   Average attempts per question: {avg_corrections:.2f}")
        print(f"   Success rate: {success_rate:.1f}%")
        
        if 'final_faithfulness' in results_df:
            avg_faith = results_df['final_faithfulness'].mean()
            avg_rel = results_df['final_relevancy'].mean()
            print(f"   Average final faithfulness: {avg_faith:.3f}")
            print(f"   Average final relevancy: {avg_rel:.3f}")
    
    return results_df


def main():
    """Run both baseline and self-correcting evaluations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate self-correction system')
    parser.add_argument('--test-set', type=str, default='testset_legal_simple.csv',
                       help='Path to test set CSV')
    parser.add_argument('--mode', type=str, choices=['baseline', 'corrected', 'both'],
                       default='both', help='Evaluation mode')
    
    args = parser.parse_args()
    
    if args.mode in ['baseline', 'both']:
        print("\n" + "="*70)
        print("📊 BASELINE EVALUATION (No Correction)")
        print("="*70)
        run_evaluation(args.test_set, enable_correction=False)
    
    if args.mode in ['corrected', 'both']:
        print("\n\n" + "="*70)
        print("📊 SELF-CORRECTING EVALUATION")
        print("="*70)
        run_evaluation(args.test_set, enable_correction=True)
    
    print("\n✅ All evaluations complete!")
    print("\nNext steps:")
    print("1. Compare evaluation_results_baseline.csv vs evaluation_results_corrected.csv")
    print("2. Run RAGAS metrics on both result sets")
    print("3. Analyze improvement in faithfulness and relevancy")


if __name__ == "__main__":
    main()
