#!/usr/bin/env python3
"""
FIXED CORRECTOR - Conservative refinement that avoids adding unsupported claims
Key fix: Emphasizes REMOVAL of bad content, not ADDITION of "better" content
Includes CorrectionLoop class (copied from original) to avoid circular import.
"""
import json
import re
from typing import List, Dict, Optional
from langchain_ollama import ChatOllama


class CorrectionStrategies:
    """
    Conservative correction strategies that fix the over-elaboration problem.
    """
    
    def __init__(self, model_name: str = "qwen2.5:7b", verbose: bool = False):
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.3,
            num_predict=512,
            num_ctx=4096,
        )
        self.verbose = verbose
    
    def reformulate_query(
        self,
        original_question: str,
        grading_feedback: Dict[str, any]
    ) -> List[str]:
        """Generate alternative query formulations (unchanged from original)."""
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
            
            if not queries:
                queries = [original_question]
            
            if self.verbose:
                print(f"\n🔄 Reformulated Queries:")
                for i, q in enumerate(queries, 1):
                    print(f"   {i}. {q}")
            
            return queries[:3]
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Reformulation failed: {e}")
            return [original_question]
    
    def refine_answer(
        self,
        question: str,
        original_answer: str,
        contexts: List[str],
        grading_feedback: Dict[str, any]
    ) -> str:
        """
        FIXED: Refine answer with emphasis on REMOVAL, not addition.
        """
        correction_type = grading_feedback.get('correction_type', 'both')
        relevancy_info = grading_feedback.get('relevancy', {})
        faithfulness_info = grading_feedback.get('faithfulness', {})
        
        # Build guidance
        guidance = []
        
        if correction_type in ['faithfulness', 'both']:
            unsupported = faithfulness_info.get('unsupported_claims', [])
            if unsupported:
                guidance.append(f"REMOVE these unsupported claims: {', '.join(unsupported[:2])}")
            guidance.append("ONLY use information EXPLICITLY stated in context")
            guidance.append("DO NOT add elaborations or interpretations")
        
        if correction_type in ['relevancy', 'both']:
            missing = relevancy_info.get('missing_aspects', [])
            if missing:
                guidance.append(f"ADDRESS these aspects if found in context: {', '.join(missing[:2])}")
            guidance.append("Focus on the SPECIFIC question asked")
        
        guidance_str = "\n".join(f"- {g}" for g in guidance)
        
        context_str = "\n\n---\n\n".join(contexts)
        
        # FIXED PROMPT - emphasizes removal, not addition
        prompt = f"""You are refining a legal answer that failed quality checks.

QUESTION: {question}

RETRIEVED CONTEXT:
{context_str}

ORIGINAL ANSWER (has quality issues):
{original_answer}

IDENTIFIED PROBLEMS:
{guidance_str}

CRITICAL REFINEMENT RULES:
1. ONLY use information EXPLICITLY stated in context
2. REMOVE any claims not in context - DO NOT REPLACE THEM
3. DO NOT add explanations, interpretations, or elaborations
4. DO NOT add details to "improve" the answer
5. Answer can be SHORTER than original - that's OK!
6. Cite exact clause numbers from context
7. If unsure or info missing: "I cannot find this information in the provided clauses"
8. CONSERVATIVE is better than COMPREHENSIVE

EXAMPLES OF GOOD REFINEMENT:

Bad Original: "The Default Rate is 12% per annum, compounded quarterly."
Context: "The Default Rate is identified in the Key Information Table."
Good Refinement: "The Default Rate is identified in the Key Information Table."
→ REMOVED unsupported details (12%, quarterly)

Bad Original: "The Buyer must design, install, operate, and maintain the Grid at their own expense."
Context: "The Buyer is responsible for the Grid."
Good Refinement: "The Buyer is responsible for the Grid."
→ REMOVED elaborations (design, install, expense)

TASK: Fix the answer by REMOVING problems. DO NOT make it "better" by adding content.

Provide ONLY the refined answer (no preamble, no explanation):
"""

        try:
            response = self.llm.invoke(prompt)
            refined = response.content.strip()
            
            # Remove meta-text
            refined = re.sub(r'^(Refined answer:|Answer:)\s*', '', refined, flags=re.IGNORECASE)
            refined = refined.strip()
            
            if self.verbose:
                print(f"\n✏️  Refined Answer:")
                print(f"   Original length: {len(original_answer)} chars")
                print(f"   Refined length: {len(refined)} chars")
                if len(refined) < len(original_answer):
                    print(f"   → Shortened (good! removed unsupported content)")
                elif len(refined) > len(original_answer) * 1.2:
                    print(f"   ⚠️  WARNING: Got longer (may have added content)")
            
            return refined
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Refinement failed: {e}")
            return original_answer
    
    def _clean_json(self, text: str) -> str:
        """Remove markdown formatting."""
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        return text.strip()


# ===== CorrectionLoop class (copied from original corrector.py) =====
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


# Test
if __name__ == "__main__":
    # Import the fixed grader – you need to create this file separately or adapt
    from grader import AnswerGrader  # Adjust import as needed
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    groq_key = os.getenv("GROQ_API_KEY")
    
    grader = AnswerGrader(groq_api_key=groq_key, verbose=True)
    corrector = CorrectionStrategies(verbose=True)
    loop = CorrectionLoop(grader, corrector, max_attempts=2, verbose=True,
                          faithfulness_threshold=0.95, relevancy_threshold=0.85)
    
    print("=" * 80)
    print("🧪 TESTING FIXED CORRECTOR")
    print("=" * 80)
    
    # Test: Answer with unsupported elaboration
    question = "What is the Default Rate?"
    answer = "The Default Rate is 12% per annum, compounded quarterly, as defined in the agreement."
    contexts = [
        "Definition of Default Rate: 'Default Rate' means the interest rate identified in the Key Information Table."
    ]
    
    result = loop.correct_answer(question, answer, contexts)
    
    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print(f"Original answer: {answer}")
    print(f"\nFinal answer: {result['final_answer']}")
    print(f"\nSuccess: {result['success']}")
    print(f"Attempts: {result['attempts']}")
    print(f"Corrections: {result['corrections_made']}")