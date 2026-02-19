#!/usr/bin/env python3
"""
RAGAS Test Set Generator - FIXED VERSION
Handles embedding extraction properly and includes debugging
"""
import os
import warnings
import numpy as np
from llama_index.core import SimpleDirectoryReader
from langchain_core.documents import Document as LCDocument
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

warnings.filterwarnings("ignore", category=DeprecationWarning)

print("=" * 80)
print("📚 RAGAS Test Set Generator - FIXED VERSION")
print("=" * 80)

# 1. Load & chunk PDF
print("\n📄 Loading PDF...")
pdf_path = "Dataset/EN OSC V2 - PPA Final.pdf"
if not os.path.exists(pdf_path):
    print(f"❌ Error: File not found at {pdf_path}")
    exit()

docs = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
print(f"✅ Loaded {len(docs)} raw document(s)")

print("✂️  Chunking documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400,
    separators=["\n\n", "\n", ".", " ", ""]
)

chunked_docs = []
for doc in docs:
    for i, chunk in enumerate(text_splitter.split_text(doc.text)):
        if len(chunk) > 500:
            chunked_docs.append(LCDocument(
                page_content=chunk,
                metadata={**doc.metadata, "chunk_id": i}
            ))

max_chunks = 10
chunked_docs = chunked_docs[:max_chunks]
print(f"✅ Using {len(chunked_docs)} chunks for generation")

# 2. Wrap LLM & Embeddings
print("\n🦙 Initializing LLM + Embeddings...")
generator_llm = LangchainLLMWrapper(ChatOllama(
    model="qwen2.5:7b",
    temperature=0.0,
    format="json",
    num_predict=1024,
    num_ctx=8192,
))
critic_llm = LangchainLLMWrapper(ChatOllama(
    model="qwen2.5:7b",
    temperature=0.0,
    format="json",
    num_predict=512,
    num_ctx=4096,
))

print("🧠 Loading BGE-M3 embeddings...")
embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
)

# 3. Build KnowledgeGraph manually from chunks
print("🔧 Building knowledge graph...")
kg = KnowledgeGraph()
for doc in chunked_docs:
    kg.nodes.append(
        Node(
            type=NodeType.CHUNK,
            properties={
                "page_content": doc.page_content,
                "document_metadata": doc.metadata,
            }
        )
    )

print(f"   Created {len(kg.nodes)} nodes")

# 4. Apply transforms WITHOUT CosineSimilarityBuilder
print("\n⚙️  Getting default transforms...")
transforms = default_transforms(
    documents=chunked_docs,
    llm=generator_llm,
    embedding_model=embeddings,
)

# CRITICAL FIX: Remove CosineSimilarityBuilder if it causes issues
# Filter out the problematic transform
print("🔧 Filtering transforms to avoid embedding shape issues...")
safe_transforms = []
for transform in transforms:
    transform_name = transform.__class__.__name__
    # Skip CosineSimilarityBuilder - it's optional for test generation
    if "CosineSimilarity" not in transform_name:
        safe_transforms.append(transform)
        print(f"   ✅ Keeping: {transform_name}")
    else:
        print(f"   ⏭️  Skipping: {transform_name} (causes embedding shape errors)")

run_config = RunConfig(timeout=180, max_retries=1, max_workers=1)

print("\n⚙️  Applying safe transforms to knowledge graph...")
try:
    apply_transforms(kg, safe_transforms, run_config=run_config)
    print("✅ Transforms applied successfully")
except Exception as e:
    print(f"❌ Transform error: {e}")
    import traceback
    traceback.print_exc()
    exit()

# DEBUG: Inspect the knowledge graph after transforms
print("\n🔍 Inspecting knowledge graph after transforms...")
print(f"   Total nodes: {len(kg.nodes)}")

# Check if nodes have embeddings
nodes_with_embeddings = sum(1 for node in kg.nodes if 'embedding' in node.properties)
print(f"   Nodes with embeddings: {nodes_with_embeddings}")

if nodes_with_embeddings > 0:
    # Check embedding shape
    sample_node = next((n for n in kg.nodes if 'embedding' in n.properties), None)
    if sample_node:
        emb = sample_node.properties['embedding']
        print(f"   Sample embedding type: {type(emb)}")
        if isinstance(emb, (list, np.ndarray)):
            print(f"   Sample embedding shape/length: {len(emb) if isinstance(emb, list) else emb.shape}")

# Check for summaries
nodes_with_summary = sum(1 for node in kg.nodes if 'summary' in node.properties)
print(f"   Nodes with summary: {nodes_with_summary}")

# 5. Generate test set from the populated KG
print("\n🚀 Generating test questions...")
generator = TestsetGenerator(
    llm=generator_llm,
    critic_llm=critic_llm,
    embedding_model=embeddings,
    knowledge_graph=kg,
)

try:
    testset = generator.generate(testset_size=5, run_config=run_config)

    print("\n💾 Saving test set...")
    df = testset.to_pandas()
    df.to_csv("testset_legal.csv", index=False)
    print("✅ Saved to 'testset_legal.csv'")
    
    print("\n📋 Preview:")
    print(df[["question", "ground_truth"]].head())
    
    print("\n✅ Test generation complete!")

except Exception as e:
    print(f"\n❌ Generation failed: {e}")
    import traceback
    traceback.print_exc()