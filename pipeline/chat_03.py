#!/usr/bin/env python3
"""
SELF-CORRECTING CHAT INTERFACE
Integrates self-correction loop into the adaptive retrieval pipeline
"""
import os
import sys
import chromadb
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path
import hashlib
import pickle

from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import TextNode

# Import our components
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.adaptive_retriever import AdaptiveRetriever
from pipeline.grader import AnswerGrader
from pipeline.corrector import CorrectionStrategies
from pipeline.self_correcting_engine import create_self_correcting_engine

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db")


def hybrid_retrieve(vector_retriever, bm25_retriever, query_str, top_k=5):
    """Hybrid BM25 + Vector retrieval."""
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


def compute_documents_hash(doc_texts):
    """SHA256 hash of document texts."""
    if not doc_texts:
        return "empty"
    combined = "".join(doc_texts).encode("utf-8")
    return hashlib.sha256(combined).hexdigest()


def save_bm25_cache(bm25_retriever, doc_texts, cache_dir):
    """Save BM25 index to cache."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    bm25_retriever.bm25.save(str(cache_dir / "bm25_index"))
    
    doc_hash = compute_documents_hash(doc_texts)
    with open(cache_dir / "bm25_hash.txt", "w") as f:
        f.write(doc_hash)
    
    with open(cache_dir / "bm25_retriever.pkl", "wb") as f:
        pickle.dump({"similarity_top_k": bm25_retriever.similarity_top_k}, f)
    
    print(f"💾 BM25 cache saved")


def load_bm25_cache(cache_dir, doc_texts, ids, metadatas, similarity_top_k=10):
    """Load BM25 index from cache."""
    cache_dir = Path(cache_dir)
    hash_file = cache_dir / "bm25_hash.txt"
    index_dir = cache_dir / "bm25_index"
    
    if not hash_file.exists() or not index_dir.exists():
        return None
    
    with open(hash_file, "r") as f:
        saved_hash = f.read().strip()
    current_hash = compute_documents_hash(doc_texts)
    
    if saved_hash != current_hash:
        print("🔄 Document hash changed – rebuilding BM25...")
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
        # Manually set the BM25 index
        retriever.bm25 = bm25
        
        print(f"✅ Loaded BM25 index from cache ({len(doc_texts)} documents)")
        return retriever
    except Exception as e:
        print(f"⚠️  Failed to load BM25 cache: {e}")
        return None


def detect_language(text):
    """Simple language detection."""
    french_indicators = [
        'quel', 'quelle', 'quoi', 'comment', 'pourquoi', 
        'qui', 'où', 'quand', 'combien', 'est-ce que'
    ]
    
    text_lower = text.lower()
    french_count = sum(1 for word in french_indicators if word in text_lower)
    
    return 'fr' if french_count >= 2 else 'en'


def get_system_prompt(language='en'):
    """Return language-specific system prompt."""
    if language == 'fr':
        return """Vous êtes un consultant juridique expert en contrats d'énergie solaire (PPA).

Répondez basé UNIQUEMENT sur les clauses fournies.

RÈGLES:
1. Citez TOUJOURS les numéros de clause (ex: "Selon la Clause 1.1...")
2. Si l'information n'est pas dans les clauses: "Je ne trouve pas cette information dans les clauses fournies."
3. Soyez précis et professionnel

Répondez en français de manière concise."""
    else:
        return """You are an expert legal consultant specializing in Solar Power Purchase Agreements (PPAs).

Answer based ONLY on the provided clauses.

RULES:
1. ALWAYS cite clause numbers (e.g., "As per Clause 1.1...")
2. If information not in clauses: "I cannot find this information in the provided clauses."
3. Be precise and professional

Answer concisely and clearly."""


def main():
    load_dotenv()
    
    print("=" * 70)
    print("☀️  SOLAR PPA LEGAL ASSISTANT - SELF-CORRECTING")
    print("=" * 70)
    print("🔬 Mode: Adaptive Retrieval + Self-Correction Loop")
    print("=" * 70)
    
    # 1. Load embedding model
    print("\n🔄 Loading BGE-M3...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    print("✅ Embedding model ready")
    
    # 2. Connect to ChromaDB
    print(f"📁 Chroma DB: {CHROMA_DB_PATH}")
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    try:
        chroma_collection = db.get_collection("solar_ppa_collection")
        count = chroma_collection.count()
        print(f"✅ Collection loaded: {count} vectors")
        
        if count == 0:
            print("⚠️  Collection is empty!")
            return
    except Exception as e:
        print(f"❌ Collection not found: {e}")
        return
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # 3. Build index and retrievers
    print("🔄 Building index and retrievers...")
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10, #reduced from 10 to 5 for testing
        embed_model=embed_model
    )
    
    # Build BM25
    print("🔄 Building BM25 index...")
    all_docs = chroma_collection.get(include=["documents", "metadatas"])
    ids = all_docs["ids"]
    doc_texts = all_docs["documents"]
    metadatas = all_docs["metadatas"]
    
    if not doc_texts:
        print("⚠️  No documents for BM25")
        bm25_retriever = None
    else:
        cache_dir = os.path.join(PROJECT_ROOT, "cache", "bm25_cache")
        bm25_retriever = load_bm25_cache(cache_dir, doc_texts, ids, metadatas, 10)
        
        if bm25_retriever is None:
            print("🔄 Building fresh BM25 index...")
            nodes = [
                TextNode(id_=node_id, text=text, metadata=meta or {})
                for node_id, text, meta in zip(ids, doc_texts, metadatas)
            ]
            
            bm25_retriever = BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=10,
                verbose=False
            )
            save_bm25_cache(bm25_retriever, doc_texts, cache_dir)
    
    # 4. Initialize Groq
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ GROQ_API_KEY not found")
        return
    
    analysis_llm = Groq(api_key=groq_api_key)
    
    # Test API
    try:
        test = analysis_llm.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=10
        )
        print(f"✅ Groq API: {test.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"❌ Groq failed: {e}")
        return
    
    # 5. Create adaptive retriever
    print("🔄 Creating adaptive retriever...")
    adaptive_retriever = AdaptiveRetriever(
        hybrid_retrieve_fn=hybrid_retrieve,
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        analysis_llm=analysis_llm,
        verbose=False  # Disable verbose for cleaner output
    )
    
    # 6. Create self-correcting query engine
    print("🔄 Creating self-correcting query engine...")
    
    # Ask user if they want correction enabled
    enable_correction_input = input("Enable self-correction? (y/n, default=y): ").strip().lower()
    enable_correction = enable_correction_input != 'n'
    
    if enable_correction:
        print("✅ Self-correction ENABLED (max 2 attempts)")
    else:
        print("⚠️  Self-correction DISABLED (baseline mode)")
    
    query_engine = create_self_correcting_engine(
        retriever=adaptive_retriever,
        groq_api_key=groq_api_key,
        system_prompt=get_system_prompt('en'),  # Default English
        enable_correction=enable_correction,
        max_correction_attempts=1,
        faithfulness_threshold=0.6,
        relevancy_threshold=0.5,
        verbose=True  # Show correction process
    )
    
    print("✅ Query engine ready")
    
    # 7. Chat loop
    print("\n" + "=" * 70)
    if enable_correction:
        print("💬 Ready! Self-correction active.")
        print("   Answers will be automatically graded and corrected if needed")
    else:
        print("💬 Ready! Baseline mode (no correction)")
    print("=" * 70 + "\n")
    
    while True:
        try:
            query = input("❓ Your question: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Query using self-correcting engine
            response = query_engine.query(query)
            
            answer = response.response
            metadata = response.metadata
            
            # Display answer
            print("\n" + "=" * 70)
            print("📢 ANSWER:")
            print("=" * 70)
            print(answer)
            print("=" * 70)
            
            # Display metadata
            if enable_correction and metadata.get('correction_enabled'):
                print(f"\n📊 Quality Metrics:")
                print(f"   Correction attempts: {metadata['correction_attempts']}")
                if metadata['corrections_made']:
                    print(f"   Corrections made: {', '.join(metadata['corrections_made'])}")
                print(f"   Final faithfulness: {metadata['final_grading']['faithfulness']:.3f}")
                print(f"   Final relevancy: {metadata['final_grading']['relevancy']:.3f}")
                
                if metadata['correction_success']:
                    print(f"   ✅ Answer meets quality standards")
                else:
                    print(f"   ⚠️  Answer did not fully meet standards after {metadata['correction_attempts']} attempts")
            
            # Display sources
            if response.source_nodes:
                print(f"\n📚 Sources ({len(response.source_nodes)} clauses):")
                for node in response.source_nodes[:3]:
                    clause_num = node.metadata.get('clause_number', '?')
                    clause_title = node.metadata.get('clause_title', 'Unknown')
                    print(f"   • Clause {clause_num}: {clause_title}")
                    print(f"     Score: {node.score:.3f}")
            
            print("")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()