#!/usr/bin/env python3
"""
ADAPTIVE RETRIEVER: Query-Aware Dynamic Retrieval
- Analyzes query complexity using LLM
- Dynamically adjusts top_k (3-10 based on complexity)
- Expands complex queries into multiple variations
- Deduplicates and re-ranks results
"""
import json
from typing import List, Optional
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from groq import Groq


class AdaptiveRetriever(BaseRetriever):
    """
    Adaptive retriever that wraps a hybrid retriever (BM25 + vector).
    
    Features:
    1. Query Analysis: LLM classifies query complexity
    2. Dynamic top_k: Adjusts retrieval size based on complexity
    3. Multi-query Expansion: Generates variations for complex queries
    4. Deduplication: Merges results by node ID with score aggregation
    """
    
    def __init__(
        self,
        hybrid_retrieve_fn,  # The hybrid_retrieve function
        vector_retriever,    # Base vector retriever
        bm25_retriever,      # BM25 retriever (can be None)
        analysis_llm: Groq,  # Groq client for query analysis
        verbose: bool = True
    ):
        """
        Args:
            hybrid_retrieve_fn: Function(vector_ret, bm25_ret, query, top_k) -> nodes
            vector_retriever: VectorIndexRetriever
            bm25_retriever: BM25Retriever or None
            analysis_llm: Groq client for query analysis (temperature=0.0)
            verbose: Print analysis results
        """
        self._hybrid_retrieve = hybrid_retrieve_fn
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        self._llm = analysis_llm
        self._verbose = verbose
        super().__init__()
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Main retrieval method with adaptive logic.
        """
        query = query_bundle.query_str
        
        # 1. Analyze query
        analysis = self._analyze_query(query)
        
        if self._verbose:
            print(f"\n🔍 Query Analysis:")
            print(f"   Complexity: {analysis['complexity']}")
            print(f"   Suggested top_k: {analysis['suggested_top_k']}")
            print(f"   Multi-query: {analysis['use_multi_query']}")
        
        # 2. Retrieve with dynamic top_k
        top_k = analysis['suggested_top_k']
        
        if analysis['use_multi_query']:
            # Multi-query expansion
            variations = self._generate_variations(query)
            if self._verbose:
                print(f"   Generated {len(variations)} query variations")
            
            all_nodes = []
            
            # Retrieve for original query
            nodes = self._hybrid_retrieve(
                self._vector_retriever,
                self._bm25_retriever,
                query,
                top_k=top_k
            )
            all_nodes.extend(nodes)
            
            # Retrieve for each variation
            for var in variations:
                var_nodes = self._hybrid_retrieve(
                    self._vector_retriever,
                    self._bm25_retriever,
                    var,
                    top_k=top_k
                )
                all_nodes.extend(var_nodes)
            
            # Deduplicate and aggregate scores
            final_nodes = self._deduplicate(all_nodes)
            
            if self._verbose:
                print(f"   Retrieved {len(all_nodes)} total nodes")
                print(f"   After deduplication: {len(final_nodes)} unique nodes")
            
            return final_nodes[:top_k]  # Return top_k after dedup
        
        else:
            # Single query retrieval
            nodes = self._hybrid_retrieve(
                self._vector_retriever,
                self._bm25_retriever,
                query,
                top_k=top_k
            )
            
            if self._verbose:
                print(f"   Retrieved {len(nodes)} nodes")
            
            return nodes
    
    def _analyze_query(self, query: str) -> dict:
        """
        Analyze query using LLM to determine:
        - complexity: simple, medium, complex
        - suggested_top_k: 3-10
        - use_multi_query: boolean
        
        Returns dict with defaults on failure.
        """
        system_prompt = """You are a query analyzer for a legal document retrieval system.

Analyze the query and respond with JSON only (no markdown, no explanation):

{
  "complexity": "simple|medium|complex",
  "suggested_top_k": 3-10,
  "use_multi_query": true|false
}

Guidelines:
- SIMPLE: Direct factual question, single concept (e.g., "What is the Default Rate?")
  → top_k: 3, multi_query: false

- MEDIUM: Requires understanding of 1-2 clauses (e.g., "Who is responsible for the Grid?")
  → top_k: 5, multi_query: false

- COMPLEX: Multi-part question, comparison, or requires synthesis (e.g., "What are the termination conditions and notice periods?")
  → top_k: 7-10, multi_query: true

Respond with JSON only."""

        try:
            response = self._llm.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}"}
                ],
                temperature=0.0,
                max_tokens=150
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean potential markdown formatting
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            analysis = json.loads(content)
            
            # Validate and set defaults if needed
            if "complexity" not in analysis:
                analysis["complexity"] = "medium"
            if "suggested_top_k" not in analysis:
                analysis["suggested_top_k"] = 5
            if "use_multi_query" not in analysis:
                analysis["use_multi_query"] = False
            
            # Bounds checking
            analysis["suggested_top_k"] = max(3, min(10, analysis["suggested_top_k"]))
            
            return analysis
            
        except Exception as e:
            if self._verbose:
                print(f"   ⚠️  Query analysis failed: {e}")
                print(f"   Using defaults: medium complexity, top_k=5")
            
            # Fallback defaults
            return {
                "complexity": "medium",
                "suggested_top_k": 5,
                "use_multi_query": False
            }
    
    def _generate_variations(self, query: str) -> List[str]:
        """
        Generate alternative phrasings of the query.
        Returns list of 2-3 variations (not including original).
        """
        system_prompt = """You are a query variation generator for legal document retrieval.

Given a query, generate 3 alternative phrasings that ask the same question in different ways.
Focus on:
- Using synonyms
- Different sentence structures
- Alternative legal terminology

Respond with JSON only:
{
  "variations": ["variation 1", "variation 2", "variation 3"]
}

Keep each variation concise (< 20 words)."""

        try:
            response = self._llm.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Original query: {query}"}
                ],
                temperature=0.3,  # Slight creativity for variations
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean markdown
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            variations = result.get("variations", [])
            
            # Return up to 3 variations
            return variations[:3]
            
        except Exception as e:
            if self._verbose:
                print(f"   ⚠️  Variation generation failed: {e}")
            
            # Fallback: no variations
            return []
    
    def _deduplicate(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """
        Deduplicate nodes by ID, aggregating scores.
        Uses MAX score for each unique node.
        """
        node_map = {}  # node_id -> NodeWithScore
        
        for node in nodes:
            node_id = node.node_id
            
            if node_id not in node_map:
                node_map[node_id] = node
            else:
                # Keep node with higher score
                if node.score > node_map[node_id].score:
                    node_map[node_id] = node
        
        # Convert back to list and sort by score
        unique_nodes = list(node_map.values())
        unique_nodes.sort(key=lambda x: x.score, reverse=True)
        
        return unique_nodes