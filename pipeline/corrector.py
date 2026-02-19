#!/usr/bin/env python3
"""
CORRECTION STRATEGIES
Implements re-retrieval and answer refinement for self-correction
"""
import json
import re
from typing import List, Dict, Optional
from langchain_ollama import ChatOllama


class CorrectionStrategies:
    """
    Implements correction strategies for failed quality checks:
    1. Re-retrieval: Reformulate query to get better context
    2. Answer Refinement: Rewrite answer to be more relevant/faithful
    """
    
    def __init__(self, model_name: str = "qwen2.5:7b", verbose: bool = False):
        """
        Args:
            model_name: Ollama model for corrections
            verbose: Print correction details
        """
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.3,  # Slight creativity for reformulation
            num_predict=512,
            num_ctx=4096,
        )
        self.verbose = verbose
    
    def reformulate_query(
        self,
        original_question: str,
        grading_feedback: Dict[str, any]
    ) -> List[str]:
        """
        Generate alternative query formulations to retrieve better context.
        
        Used when faithfulness is low (answer not grounded in retrieved context).
        
        Args:
            original_question: Original user question
            grading_feedback: Output from grade_answer()
        
        Returns:
            List of 2-3 reformulated queries
        """
        faithfulness_info = grading_feedback.get('faithfulness', {})
        unsupported_claims = faithfulness_info.get('unsupported_claims', [])
        
        unsupported_str = "\n".join(f"- {claim}" for claim in unsupported_claims[:3])
        
        prompt = f"""You are reformulating a legal query to retrieve better context.

ORIGINAL QUESTION: {original_question}

PROBLEM: The answer contained claims not supported by retrieved context.
Unsupported claims:
{unsupported_str if unsupported_claims else "General lack of supporting evidence"}

TASK: Generate 2-3 alternative ways to phrase this question that might retrieve more relevant legal clauses.

GUIDELINES:
- Use more specific legal terminology
- Break down complex questions into sub-questions
- Add keywords related to missing information
- Keep queries focused and clear

Respond ONLY with valid JSON (no markdown):
{{
  "reformulated_queries": [
    "reformulated query 1",
    "reformulated query 2",
    "reformulated query 3"
  ]
}}
"""

        try:
            response = self.llm.invoke(prompt)
            content = self._clean_json(response.content.strip())
            
            result = json.loads(content)
            queries = result.get('reformulated_queries', [])
            
            # Ensure we have at least the original
            if not queries:
                queries = [original_question]
            
            if self.verbose:
                print(f"\n🔄 Reformulated Queries:")
                for i, q in enumerate(queries, 1):
                    print(f"   {i}. {q}")
            
            return queries[:3]  # Max 3
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Query reformulation failed: {e}")
            
            # Fallback: return original
            return [original_question]
    
    def refine_answer(
        self,
        question: str,
        original_answer: str,
        contexts: List[str],
        grading_feedback: Dict[str, any]
    ) -> str:
        """
        Refine answer to improve relevancy or faithfulness.
        
        Args:
            question: Original question
            original_answer: Answer that failed quality check
            contexts: Retrieved context chunks
            grading_feedback: Output from grade_answer()
        
        Returns:
            Refined answer string
        """
        correction_type = grading_feedback.get('correction_type', 'both')
        relevancy_info = grading_feedback.get('relevancy', {})
        faithfulness_info = grading_feedback.get('faithfulness', {})
        
        # Build guidance based on what failed
        guidance = []
        
        if correction_type in ['faithfulness', 'both']:
            unsupported = faithfulness_info.get('unsupported_claims', [])
            if unsupported:
                guidance.append(f"REMOVE these unsupported claims: {', '.join(unsupported[:2])}")
            guidance.append("ONLY use information directly stated in the provided context")
        
        if correction_type in ['relevancy', 'both']:
            missing = relevancy_info.get('missing_aspects', [])
            if missing:
                guidance.append(f"ADDRESS these missing aspects: {', '.join(missing[:2])}")
            guidance.append("Focus DIRECTLY on answering the specific question asked")
        
        guidance_str = "\n".join(f"- {g}" for g in guidance)
        
        context_str = "\n\n---\n\n".join(contexts)
        
        prompt = f"""You are refining a legal answer that failed quality checks.

QUESTION: {question}

RETRIEVED CONTEXT:
{context_str}

ORIGINAL ANSWER (has quality issues):
{original_answer}

REFINEMENT GUIDANCE:
{guidance_str}

TASK: Rewrite the answer to fix the quality issues while maintaining accuracy.

REQUIREMENTS:
1. Answer MUST be grounded in the provided context (cite clause numbers)
2. Answer MUST directly address the question
3. Keep the answer concise and professional
4. Remove any information not in the context
5. Ensure all aspects of the question are covered

Provide ONLY the refined answer (no preamble, no explanation):
"""

        try:
            response = self.llm.invoke(prompt)
            refined = response.content.strip()
            
            # Remove potential markdown or meta-text
            refined = re.sub(r'^(Refined answer:|Answer:)\s*', '', refined, flags=re.IGNORECASE)
            refined = refined.strip()
            
            if self.verbose:
                print(f"\n✏️  Refined Answer:")
                print(f"   {refined[:150]}...")
            
            return refined
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Answer refinement failed: {e}")
            
            # Fallback: return original
            return original_answer
    
    def _clean_json(self, text: str) -> str:
        """Remove markdown formatting from JSON response."""
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()
        return text


class CorrectionLoop:
    """
    Orchestrates the self-correction process with retry limits.
    """
    
    def __init__(
        self,
        grader,
        corrector,
        max_attempts: int = 2,
        faithfulness_threshold: float = 0.9,
        relevancy_threshold: float = 0.8,
        verbose: bool = False
    ):
        """
        Args:
            grader: AnswerGrader instance
            corrector: CorrectionStrategies instance
            max_attempts: Maximum correction attempts (prevents infinite loops)
            faithfulness_threshold: Minimum faithfulness score
            relevancy_threshold: Minimum relevancy score
            verbose: Print correction process
        """
        self.grader = grader
        self.corrector = corrector
        self.max_attempts = max_attempts
        self.faithfulness_threshold = faithfulness_threshold
        self.relevancy_threshold = relevancy_threshold
        self.verbose = verbose
    
    def correct_answer(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        retrieval_fn=None  # Optional: function to retrieve new contexts
    ) -> Dict[str, any]:
        """
        Run self-correction loop on an answer.
        
        Args:
            question: User question
            answer: Generated answer
            contexts: Retrieved context chunks
            retrieval_fn: Optional function(query) -> List[contexts] for re-retrieval
        
        Returns:
            {
                'final_answer': str,
                'final_contexts': List[str],
                'attempts': int,
                'corrections_made': List[str],
                'final_grading': Dict,
                'success': bool
            }
        """
        current_answer = answer
        current_contexts = contexts
        attempts = 0
        corrections_made = []
        
        while attempts < self.max_attempts:
            attempts += 1
            
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"🔄 Correction Attempt {attempts}/{self.max_attempts}")
                print(f"{'='*70}")
            
            # Grade current answer
            grading = self.grader.grade_answer(
                question,
                current_answer,
                current_contexts,
                self.faithfulness_threshold,
                self.relevancy_threshold
            )
            
            # Check if correction needed
            if not grading['needs_correction']:
                if self.verbose:
                    print(f"✅ Answer passes quality checks!")
                    print(f"   Faithfulness: {grading['faithfulness']['score']:.3f}")
                    print(f"   Relevancy: {grading['relevancy']['score']:.3f}")
                
                return {
                    'final_answer': current_answer,
                    'final_contexts': current_contexts,
                    'attempts': attempts,
                    'corrections_made': corrections_made,
                    'final_grading': grading,
                    'success': True
                }
            
            if self.verbose:
                print(f"⚠️  Answer needs correction ({grading['correction_type']})")
                print(f"   Faithfulness: {grading['faithfulness']['score']:.3f} (need {self.faithfulness_threshold})")
                print(f"   Relevancy: {grading['relevancy']['score']:.3f} (need {self.relevancy_threshold})")
            
            # Apply correction based on type
            correction_type = grading['correction_type']
            
            if correction_type in ['faithfulness', 'both'] and retrieval_fn:
                # Re-retrieval: reformulate query and get new contexts
                if self.verbose:
                    print(f"\n🔍 Re-retrieval: Reformulating query...")
                
                reformulated_queries = self.corrector.reformulate_query(question, grading)
                
                # Try to retrieve with reformulated queries
                new_contexts = []
                for query in reformulated_queries:
                    try:
                        retrieved = retrieval_fn(query)
                        new_contexts.extend(retrieved)
                    except Exception as e:
                        if self.verbose:
                            print(f"   ⚠️  Retrieval failed for '{query}': {e}")
                
                if new_contexts:
                    # Deduplicate contexts
                    current_contexts = list(dict.fromkeys(new_contexts))[:5]  # Top 5 unique
                    corrections_made.append('re-retrieval')
                    
                    if self.verbose:
                        print(f"   Retrieved {len(current_contexts)} new context chunks")
            
            # Answer refinement (always try if relevancy is low or if re-retrieval was skipped)
            if correction_type in ['relevancy', 'both'] or not retrieval_fn:
                if self.verbose:
                    print(f"\n✏️  Refining answer...")
                
                current_answer = self.corrector.refine_answer(
                    question,
                    current_answer,
                    current_contexts,
                    grading
                )
                corrections_made.append('refinement')
        
        # Max attempts reached
        if self.verbose:
            print(f"\n⚠️  Max attempts ({self.max_attempts}) reached")
        
        # Final grading
        final_grading = self.grader.grade_answer(
            question,
            current_answer,
            current_contexts,
            self.faithfulness_threshold,
            self.relevancy_threshold
        )
        
        return {
            'final_answer': current_answer,
            'final_contexts': current_contexts,
            'attempts': attempts,
            'corrections_made': corrections_made,
            'final_grading': final_grading,
            'success': not final_grading['needs_correction']
        }


# Example usage
if __name__ == "__main__":
    from grader import AnswerGrader
    
    print("=" * 80)
    print("🧪 TESTING CORRECTION STRATEGIES")
    print("=" * 80)
    
    grader = AnswerGrader(verbose=True)
    corrector = CorrectionStrategies(verbose=True)
    loop = CorrectionLoop(grader, corrector, max_attempts=2, verbose=True)
    
    # Test case: Answer with low relevancy
    question = "What is the Default Rate?"
    answer = "The Buyer must maintain the Grid connection."  # Irrelevant
    contexts = [
        "Definition of Default Rate: 'Default Rate' means the interest rate identified in the Key Information Table.",
        "The Buyer shall be responsible for the Grid connection."
    ]
    
    result = loop.correct_answer(question, answer, contexts)
    
    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    print(f"Success: {result['success']}")
    print(f"Attempts: {result['attempts']}")
    print(f"Corrections: {result['corrections_made']}")
    print(f"\nFinal Answer: {result['final_answer']}")
    print(f"\nFinal Grading:")
    print(f"  Faithfulness: {result['final_grading']['faithfulness']['score']:.3f}")
    print(f"  Relevancy: {result['final_grading']['relevancy']['score']:.3f}")
