#!/usr/bin/env python3
"""
SELF-CORRECTING QUERY ENGINE
Integrates grading and correction into the LlamaIndex pipeline
"""
from typing import List, Optional
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.response import Response
from groq import Groq as GroqClient
import sys
import os

# Import our correction components
sys.path.append(os.path.dirname(__file__))
from grader import AnswerGrader
from corrector import CorrectionStrategies, CorrectionLoop


class SelfCorrectingQueryEngine(BaseQueryEngine):
    """
    Query engine with self-correction loop.
    
    Flow:
    1. Retrieve contexts using adaptive retriever
    2. Generate answer using LLM
    3. Grade answer (faithfulness + relevancy)
    4. If needed, correct (re-retrieve or refine)
    5. Return final answer
    """
    
    def __init__(
        self,
        retriever,  # AdaptiveRetriever instance
        answer_llm: GroqClient,  # Groq client for answer generation
        grader: AnswerGrader,
        corrector: CorrectionStrategies,
        system_prompt: str,
        max_correction_attempts: int = 2,
        faithfulness_threshold: float = 0.9,
        relevancy_threshold: float = 0.8,
        enable_correction: bool = True,
        verbose: bool = False
    ):
        """
        Args:
            retriever: Retriever (e.g., AdaptiveRetriever)
            answer_llm: Groq client for generating answers
            grader: AnswerGrader instance
            corrector: CorrectionStrategies instance
            system_prompt: System prompt for answer generation
            max_correction_attempts: Max correction loops
            faithfulness_threshold: Minimum faithfulness score
            relevancy_threshold: Minimum relevancy score
            enable_correction: If False, skip correction (for comparison)
            verbose: Print detailed process
        """
        self._retriever = retriever
        self._answer_llm = answer_llm
        self._grader = grader
        self._corrector = corrector
        self._system_prompt = system_prompt
        self._max_correction_attempts = max_correction_attempts
        self._faithfulness_threshold = faithfulness_threshold
        self._relevancy_threshold = relevancy_threshold
        self._enable_correction = enable_correction
        self._verbose = verbose
        
        # Create correction loop
        self._correction_loop = CorrectionLoop(
            grader=grader,
            corrector=corrector,
            max_attempts=max_correction_attempts,
            faithfulness_threshold=faithfulness_threshold,
            relevancy_threshold=relevancy_threshold,
            verbose=verbose
        )
        
        super().__init__(None)  # No callback manager needed
    
    def _query(self, query_bundle: QueryBundle) -> Response:
        """
        Main query method with self-correction.
        """
        query_str = query_bundle.query_str
        
        if self._verbose:
            print(f"\n{'='*70}")
            print(f"🤖 SELF-CORRECTING QUERY ENGINE")
            print(f"{'='*70}")
            print(f"Query: {query_str}")
        
        # 1. Retrieve contexts
        if self._verbose:
            print(f"\n🔍 Retrieving contexts...")
        
        try:
            nodes = self._retriever.retrieve(query_bundle)
        except Exception as e:
            if self._verbose:
                print(f"❌ Retrieval failed: {e}")
            return Response(
                response="I apologize, but I encountered an error retrieving relevant information.",
                source_nodes=[],
                metadata={'error': str(e)}
            )
        
        if not nodes:
            return Response(
                response="I cannot find relevant information to answer this question.",
                source_nodes=[],
                metadata={'no_retrieval': True}
            )
        
        # Extract context texts
        contexts = [node.text for node in nodes]
        
        if self._verbose:
            print(f"   Retrieved {len(nodes)} context chunks")
        
        # 2. Generate initial answer
        if self._verbose:
            print(f"\n🤖 Generating initial answer...")
        
        try:
            initial_answer = self._generate_answer(query_str, contexts)
        except Exception as e:
            if self._verbose:
                print(f"❌ Answer generation failed: {e}")
            return Response(
                response="I apologize, but I encountered an error generating an answer.",
                source_nodes=nodes,
                metadata={'error': str(e)}
            )
        
        if self._verbose:
            print(f"   Initial answer: {initial_answer[:100]}...")
        
        # 3. Self-correction (if enabled)
        if self._enable_correction:
            if self._verbose:
                print(f"\n🔍 Running self-correction...")
            
            # Define retrieval function for re-retrieval
            def retrieval_fn(reformulated_query: str) -> List[str]:
                """Retrieve contexts for reformulated query."""
                new_bundle = QueryBundle(query_str=reformulated_query)
                new_nodes = self._retriever.retrieve(new_bundle)
                return [node.text for node in new_nodes]
            
            # Run correction loop
            correction_result = self._correction_loop.correct_answer(
                question=query_str,
                answer=initial_answer,
                contexts=contexts,
                retrieval_fn=retrieval_fn
            )
            
            final_answer = correction_result['final_answer']
            final_contexts = correction_result['final_contexts']
            
            # Update nodes if contexts changed
            if final_contexts != contexts:
                # Re-retrieve to get NodeWithScore objects
                # (In practice, we'd need to track which nodes correspond to final_contexts)
                # For now, keep original nodes but note contexts changed
                pass
            
            metadata = {
                'correction_enabled': True,
                'correction_attempts': correction_result['attempts'],
                'corrections_made': correction_result['corrections_made'],
                'final_grading': {
                    'faithfulness': correction_result['final_grading']['faithfulness']['score'],
                    'relevancy': correction_result['final_grading']['relevancy']['score'],
                },
                'correction_success': correction_result['success']
            }
            
            if self._verbose:
                print(f"\n✅ Self-correction complete:")
                print(f"   Attempts: {correction_result['attempts']}")
                print(f"   Corrections: {correction_result['corrections_made']}")
                print(f"   Final faithfulness: {metadata['final_grading']['faithfulness']:.3f}")
                print(f"   Final relevancy: {metadata['final_grading']['relevancy']:.3f}")
        
        else:
            # No correction
            final_answer = initial_answer
            metadata = {'correction_enabled': False}
        
        # 4. Return response
        return Response(
            response=final_answer,
            source_nodes=nodes,
            metadata=metadata
        )
    
    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        """Async query (not implemented, falls back to sync)."""
        return self._query(query_bundle)
    
    def _generate_answer(
        self,
        question: str,
        contexts: List[str]
    ) -> str:
        """
        Generate answer using LLM.
        
        Args:
            question: User question
            contexts: Retrieved context chunks
        
        Returns:
            Generated answer string
        """
        #truncated_contexts = [ctx[:800] for ctx in contexts[:5]] # Added for testing - limit to 5 contexts and truncate to 800 chars each
        # Format contexts
        #context_str = "\n\n---\n\n".join(
            #f"[CONTEXT {i+1}]\n{ctx}" for i, ctx in enumerate(truncated_contexts) # changed from contexts to truncated_contexts for testing
        #)

        context_str = "\n\n---\n\n".join(
            f"[CONTEXT {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)
        )
        
        # Build messages
        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": f"PROVIDED CONTEXT:\n\n{context_str}\n\n---\n\nUSER QUESTION: {question}"
            }
        ]
        
        # Call Groq
        completion = self._answer_llm.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.1,
            max_tokens=800
        )
        
        answer = completion.choices[0].message.content.strip()
        return answer

    def _get_prompt_modules(self):
        """
        Required by LlamaIndex BaseQueryEngine to track promptable components.
        Since we handle prompts internally, we return an empty dict.
        """
        return {}

    def _get_prompts(self):
        """Optional: return any internal prompts used by the engine."""
        return {"system_prompt": self._system_prompt}

def create_self_correcting_engine(
    retriever,
    groq_api_key: str,
    system_prompt: str,
    enable_correction: bool = True,
    max_correction_attempts: int = 2,
    faithfulness_threshold: float = 0.9,
    relevancy_threshold: float = 0.8,
    verbose: bool = False
) -> SelfCorrectingQueryEngine:
    """
    Factory function to create self-correcting query engine.
    
    Args:
        retriever: Retriever instance (e.g., AdaptiveRetriever)
        groq_api_key: Groq API key for answer generation
        system_prompt: System prompt for answers
        enable_correction: Enable/disable self-correction
        max_correction_attempts: Max correction loops
        faithfulness_threshold: Minimum faithfulness score
        relevancy_threshold: Minimum relevancy score
        verbose: Print detailed process
    
    Returns:
        SelfCorrectingQueryEngine instance
    """
    # Initialize components
    answer_llm = GroqClient(api_key=groq_api_key)
    
    grader = AnswerGrader(
        model_name=groq_api_key,
        verbose=verbose
    )
    
    corrector = CorrectionStrategies(
        model_name="qwen2.5:7b",
        verbose=verbose
    )
    
    # Create engine
    engine = SelfCorrectingQueryEngine(
        retriever=retriever,
        answer_llm=answer_llm,
        grader=grader,
        corrector=corrector,
        system_prompt=system_prompt,
        max_correction_attempts=max_correction_attempts,
        faithfulness_threshold=faithfulness_threshold,
        relevancy_threshold=relevancy_threshold,
        enable_correction=enable_correction,
        verbose=verbose
    )
    
    return engine


# Example usage
if __name__ == "__main__":
    print("Self-correcting query engine module loaded successfully!")
    print("\nTo use:")
    print("1. Create your retriever (e.g., AdaptiveRetriever)")
    print("2. Call create_self_correcting_engine(retriever, groq_key, system_prompt)")
    print("3. Use engine.query('your question')")
