#!/usr/bin/env python3
"""
FIXED GRADER - Uses Groq API for more accurate grading
Addresses the leniency problem with qwen2.5:7b
"""
import json
import re
from typing import Dict, List
from groq import Groq


class AnswerGrader:
    """
    Grader using Groq API (same model as answer generation) for consistency.
    Much stricter than the qwen2.5:7b version.
    """
    
    def __init__(self, groq_api_key: str, verbose: bool = False):
        """
        Args:
            groq_api_key: Groq API key
            verbose: Print grading details
        """
        self.llm = Groq(api_key=groq_api_key)
        self.verbose = verbose
    
    def grade_faithfulness(
        self, 
        question: str, 
        answer: str, 
        contexts: List[str]
    ) -> Dict[str, any]:
        """
        Grade faithfulness with STRICT evaluation.
        """
        context_str = "\n\n---\n\n".join(contexts)
        
        prompt = f"""You are a STRICT evaluator of legal answer faithfulness.

CRITICAL: Be very conservative. A claim is UNSUPPORTED unless explicitly stated in context.

QUESTION: {question}

RETRIEVED CONTEXT:
{context_str}

ANSWER TO EVALUATE:
{answer}

TASK: Check if EVERY claim in the answer is directly supported by context.

STRICT RULES:
1. Paraphrasing is OK ONLY if meaning is identical
2. Any elaboration, explanation, or interpretation = UNSUPPORTED
3. Legal terminology must match exactly
4. Clause numbers must be verified against context
5. If answer adds ANY detail not in context = UNSUPPORTED
6. General statements must have specific support

EXAMPLES:

Example 1 - UNSUPPORTED:
Context: "The Default Rate is identified in the Key Information Table."
Answer: "The Default Rate is 12% per annum."
→ UNSUPPORTED (adds specific 12% not in context)

Example 2 - SUPPORTED:
Context: "The Default Rate is identified in the Key Information Table."
Answer: "The Default Rate is identified in the Key Information Table."
→ SUPPORTED (exact match)

Example 3 - UNSUPPORTED:
Context: "The Buyer is responsible for the Grid."
Answer: "The Buyer must design, install, and maintain the Grid."
→ UNSUPPORTED (adds design, install, maintain not in context)

Be harsh. When in doubt, mark as UNSUPPORTED.

Respond ONLY with valid JSON (no markdown):
{{
  "verdict": "yes" or "no",
  "score": 0.0-1.0,
  "reason": "brief explanation",
  "unsupported_claims": ["claim 1", "claim 2", ...]
}}

If all claims supported: verdict="yes", score=1.0
If some unsupported: verdict="no", score=(supported/total), list ALL unsupported claims
"""

        try:
            response = self.llm.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a strict legal answer evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Deterministic
                max_tokens=512
            )
            
            content = response.choices[0].message.content.strip()
            content = self._clean_json(content)
            
            result = json.loads(content)
            
            # Validate
            if 'verdict' not in result:
                result['verdict'] = 'yes' if result.get('score', 0) >= 0.9 else 'no'
            if 'score' not in result:
                result['score'] = 1.0 if result['verdict'] == 'yes' else 0.5
            if 'reason' not in result:
                result['reason'] = 'Unknown'
            if 'unsupported_claims' not in result:
                result['unsupported_claims'] = []
            
            result['score'] = float(result['score'])
            
            if self.verbose:
                print(f"\n📊 Faithfulness (Groq):")
                print(f"   Score: {result['score']:.3f}")
                print(f"   Verdict: {result['verdict']}")
                if result['unsupported_claims']:
                    print(f"   Unsupported: {result['unsupported_claims'][:2]}")
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Faithfulness grading failed: {e}")
            
            return {
                'score': 0.5,
                'verdict': 'no',
                'reason': f'Error: {str(e)}',
                'unsupported_claims': []
            }
    
    def grade_relevancy(
        self, 
        question: str, 
        answer: str
    ) -> Dict[str, any]:
        """
        Grade answer relevancy with strict evaluation.
        """
        prompt = f"""You are a STRICT evaluator of answer relevancy for legal questions.

QUESTION: {question}

ANSWER TO EVALUATE:
{answer}

TASK: Determine if the answer DIRECTLY and COMPLETELY addresses the question.

STRICT RULES:
1. Answer must address the SPECIFIC question asked
2. ALL aspects of the question must be covered
3. Vague or general responses = LOW relevancy
4. Extra information is OK, but missing aspects = penalty
5. "I cannot find..." responses get relevancy based on appropriateness

EXAMPLES:

Example 1 - HIGH RELEVANCY:
Question: "What is the Default Rate?"
Answer: "The Default Rate is identified in the Key Information Table."
→ score=1.0 (directly answers)

Example 2 - LOW RELEVANCY:
Question: "What is the Default Rate?"
Answer: "Clause 1 defines various rates and charges."
→ score=0.3 (too vague, doesn't answer)

Example 3 - MEDIUM RELEVANCY:
Question: "What are the termination conditions and notice periods?"
Answer: "Termination requires 20 days notice."
→ score=0.5 (covers notice but missing conditions)

Be strict about completeness.

Respond ONLY with valid JSON (no markdown):
{{
  "verdict": "yes" or "no",
  "score": 0.0-1.0,
  "reason": "brief explanation",
  "missing_aspects": ["aspect 1", "aspect 2", ...]
}}

If answer fully addresses question: verdict="yes", score=1.0
If partial/off-topic: verdict="no", score=(addressed/total), list missing aspects
"""

        try:
            response = self.llm.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a strict answer relevancy evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=512
            )
            
            content = response.choices[0].message.content.strip()
            content = self._clean_json(content)
            
            result = json.loads(content)
            
            # Validate
            if 'verdict' not in result:
                result['verdict'] = 'yes' if result.get('score', 0) >= 0.8 else 'no'
            if 'score' not in result:
                result['score'] = 1.0 if result['verdict'] == 'yes' else 0.5
            if 'reason' not in result:
                result['reason'] = 'Unknown'
            if 'missing_aspects' not in result:
                result['missing_aspects'] = []
            
            result['score'] = float(result['score'])
            
            if self.verbose:
                print(f"\n📊 Relevancy (Groq):")
                print(f"   Score: {result['score']:.3f}")
                print(f"   Verdict: {result['verdict']}")
                if result['missing_aspects']:
                    print(f"   Missing: {result['missing_aspects'][:2]}")
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Relevancy grading failed: {e}")
            
            return {
                'score': 0.5,
                'verdict': 'no',
                'reason': f'Error: {str(e)}',
                'missing_aspects': []
            }
    
    def grade_answer(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        faithfulness_threshold: float = 0.95,  # RAISED (more lenient)
        relevancy_threshold: float = 0.85      # RAISED (more lenient)
    ) -> Dict[str, any]:
        """
        Grade both faithfulness and relevancy.
        
        NOTE: Thresholds are HIGHER (more lenient) because this grader is stricter.
        """
        faithfulness = self.grade_faithfulness(question, answer, contexts)
        relevancy = self.grade_relevancy(question, answer)
        
        needs_faithfulness_fix = faithfulness['score'] < faithfulness_threshold
        needs_relevancy_fix = relevancy['score'] < relevancy_threshold
        
        needs_correction = needs_faithfulness_fix or needs_relevancy_fix
        
        if needs_faithfulness_fix and needs_relevancy_fix:
            correction_type = 'both'
        elif needs_faithfulness_fix:
            correction_type = 'faithfulness'
        elif needs_relevancy_fix:
            correction_type = 'relevancy'
        else:
            correction_type = None
        
        return {
            'faithfulness': faithfulness,
            'relevancy': relevancy,
            'needs_correction': needs_correction,
            'correction_type': correction_type,
            'overall_quality': (faithfulness['score'] + relevancy['score']) / 2
        }
    
    def _clean_json(self, text: str) -> str:
        """Remove markdown formatting."""
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        return text.strip()


# Test
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("❌ GROQ_API_KEY not found")
        exit()
    
    grader = AnswerGrader(groq_api_key=groq_key, verbose=True)
    
    # Test 1: Good answer
    print("=" * 80)
    print("TEST 1: Good Answer")
    print("=" * 80)
    result = grader.grade_answer(
        question="What is the Default Rate?",
        answer="As per Clause 1, the Default Rate is the interest rate identified in the Key Information Table.",
        contexts=["Definition of Default Rate: 'Default Rate' means the interest rate identified in the Key Information Table."]
    )
    print(f"\nNeeds correction: {result['needs_correction']}")
    print(f"Correction type: {result['correction_type']}")
    
    # Test 2: Unfaithful answer
    print("\n" + "=" * 80)
    print("TEST 2: Unfaithful Answer (adds 12% not in context)")
    print("=" * 80)
    result = grader.grade_answer(
        question="What is the Default Rate?",
        answer="The Default Rate is 12% per annum, compounded quarterly.",
        contexts=["Definition of Default Rate: 'Default Rate' means the interest rate identified in the Key Information Table."]
    )
    print(f"\nNeeds correction: {result['needs_correction']}")
    print(f"Correction type: {result['correction_type']}")