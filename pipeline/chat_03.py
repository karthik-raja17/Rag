#!/usr/bin/env python3
"""
HYBRID RETRIEVAL: BM25 + Dense Vector Search
Catches exact-term matches that pure semantic search misses
Example: "Effective Date" now retrieves the definition, not just usage
"""
import hashlib
import pickle
from pathlib import Path
import os
import sys
import chromadb
from dotenv import load_dotenv
from groq import Groq
from llama_index.core.schema import TextNode

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import QueryBundle

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db")

def hybrid_retrieve(vector_retriever, bm25_retriever, query_str, top_k=5):
    query_bundle = QueryBundle(query_str=query_str)
    vector_nodes = vector_retriever.retrieve(query_bundle)
    
    # If BM25 is not available, just use vector results
    if bm25_retriever is None:
        return vector_nodes[:top_k]
    
    bm25_nodes = bm25_retriever.retrieve(query_bundle)
    
    # 🔧 Initialize the score dictionary
    node_scores = {}
    
    # Normalize vector scores (already 0-1 from cosine similarity)
    for node in vector_nodes:
        node_id = node.node_id
        node_scores[node_id] = {
            'node': node,
            'vector_score': node.score,
            'bm25_score': 0.0
        }
    
    # Normalize BM25 scores (need to scale to 0-1)
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
    
    # Combine scores (weighted average: 60% vector, 40% BM25)
    # Vector is better for semantic, BM25 is better for exact terms
    combined_nodes = []
    for node_id, scores in node_scores.items():
        combined_score = (0.6 * scores['vector_score']) + (0.4 * scores['bm25_score'])
        node = scores['node']
        node.score = combined_score  # Update score
        combined_nodes.append(node)
    
    # Sort by combined score and return top_k
    combined_nodes.sort(key=lambda x: x.score, reverse=True)
    return combined_nodes[:top_k]

def format_clauses_for_context(nodes, max_clauses=5):
    """Format retrieved nodes into clean context"""
    context_parts = []
    clause_info = []
    
    for i, node in enumerate(nodes[:max_clauses]):
        clause_number = node.metadata.get('clause_number', 'Unknown')
        clause_title = node.metadata.get('clause_title', 'Unknown Section')
        filename = node.metadata.get('filename', 'Unknown Document')
        score = node.score
        
        clause_header = f"[CLAUSE {clause_number}: {clause_title}]"
        clause_text = node.text
        
        context_parts.append(f"{clause_header}\n{clause_text}")
        
        clause_info.append({
            'number': clause_number,
            'title': clause_title,
            'filename': filename,
            'score': score,
            'text_preview': clause_text[:150] + "..." if len(clause_text) > 150 else clause_text
        })
    
    context_str = "\n\n---\n\n".join(context_parts)
    return context_str, clause_info

def detect_language(text):
    """Simple language detection"""
    french_indicators = [
        'quel', 'quelle', 'quoi', 'comment', 'pourquoi', 
        'qui', 'où', 'quand', 'combien', 'est-ce que'
    ]
    
    text_lower = text.lower()
    french_count = sum(1 for word in french_indicators if word in text_lower)
    
    return 'fr' if french_count >= 2 else 'en'

def get_system_prompt(language='en'):
    """Return language-specific system prompt"""
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

def compute_documents_hash(doc_texts):
    """Return SHA256 hash of concatenated document texts."""
    if not doc_texts:
        return "empty"
    combined = "".join(doc_texts).encode("utf-8")
    return hashlib.sha256(combined).hexdigest()

def save_bm25_cache(bm25_retriever, doc_texts, cache_dir):
    """Save BM25 index and document hash to cache."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the BM25 index using bm25s internal save
    bm25_retriever.bm25.save(str(cache_dir / "bm25_index"))
    
    # Save the hash
    doc_hash = compute_documents_hash(doc_texts)
    with open(cache_dir / "bm25_hash.txt", "w") as f:
        f.write(doc_hash)
    
    # Also pickle the retriever settings (optional, but helpful)
    with open(cache_dir / "bm25_retriever.pkl", "wb") as f:
        # We can't pickle the whole retriever because of the BM25 object,
        # but we can save parameters like similarity_top_k
        pickle.dump({"similarity_top_k": bm25_retriever.similarity_top_k}, f)
    
    print(f"💾 BM25 cache saved to {cache_dir}")

def load_bm25_cache(cache_dir, doc_texts, ids, metadatas, similarity_top_k=10):
    """Load BM25 index from cache if hash matches, else return None."""
    cache_dir = Path(cache_dir)
    hash_file = cache_dir / "bm25_hash.txt"
    index_dir = cache_dir / "bm25_index"
    
    if not hash_file.exists() or not index_dir.exists():
        return None
    
    # Check hash (based on texts only – you could include ids if desired)
    with open(hash_file, "r") as f:
        saved_hash = f.read().strip()
    current_hash = compute_documents_hash(doc_texts)
    
    if saved_hash != current_hash:
        print("🔄 Document hash changed – rebuilding BM25...")
        return None
    
    # Load BM25 index
    try:
        import bm25s
        
        bm25 = bm25s.BM25.load(str(index_dir))
        
        # Recreate the retriever using TextNode objects with preserved IDs
        from llama_index.core.retrievers import BM25Retriever
        from llama_index.core.schema import TextNode
        
        nodes = [
            TextNode(id_=node_id, text=text, metadata=meta or {})
            for node_id, text, meta in zip(ids, doc_texts, metadatas)
        ]
        
        retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=similarity_top_k,
            bm25=bm25  # Pass the pre‑loaded BM25 index
        )
        print(f"✅ Loaded BM25 index from cache ({len(doc_texts)} documents)")
        return retriever
    except Exception as e:
        print(f"⚠️  Failed to load BM25 cache: {e}")
        return None

def main():
    load_dotenv()
    
    print("=" * 70)
    print("☀️  SOLAR PPA LEGAL ASSISTANT - HYBRID RETRIEVAL")
    print("=" * 70)
    print("🔬 Mode: BM25 + Dense Vector Search")
    print("=" * 70)
    
    # 1. Load embedding model
    print("\n🔄 Loading BGE-M3 (1024-dim)...")
    try:
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
        print("✅ Embedding model ready")
    except Exception as e:
        print(f"❌ Failed to load embeddings: {e}")
        return
    
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
    
    # 3. Build index
    print("🔄 Building index...")
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    
    # 4. Create BOTH retrievers
    print("🔄 Creating hybrid retriever (BM25 + Vector)...")
    
    # Dense vector retriever
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
        embed_model=embed_model
    )
    
        # --- BM25 keyword retriever with caching ---
    print("🔄 Building BM25 index from stored documents...")
    
    # Fetch all documents from ChromaDB
    all_docs = chroma_collection.get(include=["documents", "metadatas"])
    ids = all_docs["ids"]
    doc_texts = all_docs["documents"]
    metadatas = all_docs["metadatas"]
    
    if not doc_texts:
        print("⚠️  No documents found for BM25 – using vector only")
        bm25_retriever = None
    else:
        cache_dir = os.path.join(PROJECT_ROOT, "cache", "bm25_cache")
        similarity_top_k = 10
        
        # Try to load from cache
        bm25_retriever = load_bm25_cache(cache_dir, doc_texts, ids, metadatas, similarity_top_k)
        
        # If cache miss or hash mismatch, rebuild
        if bm25_retriever is None:
            print("🔄 Building fresh BM25 index...")
            from llama_index.core.schema import TextNode
            
            nodes = [
                TextNode(id_=node_id, text=text, metadata=meta or {})
                for node_id, text, meta in zip(ids, doc_texts, metadatas)
            ]
            
            bm25_retriever = BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=similarity_top_k,
                verbose=True
            )
            # Save to cache
            save_bm25_cache(bm25_retriever, doc_texts, cache_dir)
            print(f"✅ BM25 index built and cached ({len(nodes)} nodes)")
        else:
            # Ensure the retriever has the correct metadata mapping (optional)
            pass
    
    print("✅ Hybrid retriever ready")
    print("   → Vector search: semantic similarity")
    print("   → BM25 search: exact term matching")
    
    # 5. Direct Groq client
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ GROQ_API_KEY not found")
        return
    
    try:
        client = Groq(api_key=groq_api_key)
        test = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=10
        )
        print(f"✅ Groq API: {test.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"❌ Groq failed: {e}")
        return
    
    # 6. Chat loop
    print("\n" + "=" * 70)
    print("💬 Ready! Hybrid retrieval active.")
    print("   Try queries with exact terms like 'Effective Date' or 'Default Rate'")
    print("=" * 70 + "\n")
    
    while True:
        try:
            query = input("❓ Your question: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # HYBRID RETRIEVAL
            print(f"\n🔍 Hybrid search (BM25 + Vector)...")
            
            try:
                nodes = hybrid_retrieve(
                    vector_retriever, 
                    bm25_retriever, 
                    query, 
                    top_k=10
                )
            except Exception as e:
                print(f"⚠️  Hybrid retrieval error: {e}")
                print("Falling back to vector-only...")
                nodes = vector_retriever.retrieve(QueryBundle(query_str=query))
            
            if len(nodes) == 0:
                print("❌ No relevant clauses found.")
                continue
            
            # Filter by threshold
            relevant_nodes = [n for n in nodes if n.score > 0.5]
            if not relevant_nodes:
                print("⚠️  No clauses meet relevance threshold (>0.5)")
                continue
            
            print(f"✅ Found {len(relevant_nodes)} relevant clause(s)")
            
            # Format context
            context, clause_info = format_clauses_for_context(relevant_nodes, max_clauses=5)
            
            top_clause = clause_info[0]
            print(f"\n📋 Top clause:")
            print(f"   Clause {top_clause['number']}: {top_clause['title']}")
            print(f"   Combined score: {top_clause['score']:.3f}")
            print(f"   Preview: {top_clause['text_preview']}")
            
            # Detect language and call Groq
            language = detect_language(query)
            system_prompt = get_system_prompt(language)
            
            print(f"\n🤖 Generating answer...")
            
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user", 
                        "content": f"PROVIDED CLAUSES:\n\n{context}\n\n---\n\nUSER QUESTION: {query}"
                    }
                ]
                
                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=800
                )
                
                answer = completion.choices[0].message.content.strip()
                
                if not answer:
                    print("⚠️  Empty response from LLM")
                    continue
                
                print("\n" + "=" * 70)
                print("📢 ANSWER:")
                print("=" * 70)
                print(answer)
                print("=" * 70)
                
                print(f"\n📚 Sources ({len(clause_info)} clauses):")
                for info in clause_info:
                    print(f"   • Clause {info['number']}: {info['title']}")
                    print(f"     Score: {info['score']:.3f}")
                
                print("")
                
            except Exception as e:
                print(f"❌ Groq API error: {e}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()