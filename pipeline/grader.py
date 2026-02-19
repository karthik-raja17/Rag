#!/usr/bin/env python3
"""
SELF-CORRECTION GRADER
Real-time answer quality assessment using local Qwen2.5 7B
Evaluates Faithfulness and Answer Relevancy without API costs
"""
import json
import re
from typing import Dict, List, Optional
from langchain_ollama import ChatOllama


class AnswerGrader:
    """
    Grades answers using local LLM for:
    1. Faithfulness - Is answer grounded in retrieved context?
    2. Answer Relevancy - Does answer directly address the question?
    """
    
    def __init__(self, model_name: str = "qwen2.5:7b", verbose: bool = False):
        """
        Args:
            model_name: Ollama model to use for grading
            verbose: Print grading details
        """
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.0,  # Deterministic grading
            num_predict=512,
            num_ctx=4096,
        )
        self.verbose = verbose
    
    def grade_faithfulness(
        self, 
        question: str, 
        answer: str, 
        contexts: List[str]
    ) -> Dict[str, any]:
        """
        Grade faithfulness: Are all claims in the answer supported by context?
        
        Returns:
            {
                'score': float (0.0-1.0),
                'verdict': str ('yes'/'no'),
                'reason': str,
                'unsupported_claims': List[str]
            }
        """
        # Combine contexts
        context_str = "\n\n---\n\n".join(contexts)
        
        prompt = f"""You are evaluating the FAITHFULNESS of an answer to a legal question.

QUESTION: {question}

RETRIEVED CONTEXT:
{context_str}

ANSWER TO EVALUATE:
{answer}

TASK: Determine if ALL claims in the answer are directly supported by the retrieved context.

INSTRUCTIONS:
1. Identify each claim in the answer
2. Check if each claim has supporting evidence in the context
3. Mark claims as SUPPORTED or UNSUPPORTED
4. A claim is UNSUPPORTED if it introduces information not in the context

Respond ONLY with valid JSON (no markdown, no explanation):
{{
  "verdict": "yes" or "no",
  "score": 0.0-1.0,
  "reason": "brief explanation",
  "unsupported_claims": ["claim 1", "claim 2", ...]
}}

If all claims are supported, verdict="yes" and score=1.0.
If some claims are unsupported, verdict="no" and score = (supported_claims / total_claims).
"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Clean potential markdown
            content = self._clean_json(content)
            
            result = json.loads(content)
            
            # Validate and normalize
            if 'verdict' not in result:
                result['verdict'] = 'yes' if result.get('score', 0) >= 0.9 else 'no'
            if 'score' not in result:
                result['score'] = 1.0 if result['verdict'] == 'yes' else 0.5
            if 'reason' not in result:
                result['reason'] = 'Unknown'
            if 'unsupported_claims' not in result:
                result['unsupported_claims'] = []
            
            # Ensure score is float
            result['score'] = float(result['score'])
            
            if self.verbose:
                print(f"\n📊 Faithfulness Grading:")
                print(f"   Score: {result['score']:.3f}")
                print(f"   Verdict: {result['verdict']}")
                if result['unsupported_claims']:
                    print(f"   Unsupported claims: {len(result['unsupported_claims'])}")
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Faithfulness grading failed: {e}")
            
            # Conservative fallback
            return {
                'score': 0.5,
                'verdict': 'no',
                'reason': f'Grading error: {str(e)}',
                'unsupported_claims': []
            }
    
    def grade_relevancy(
        self, 
        question: str, 
        answer: str
    ) -> Dict[str, any]:
        """
        Grade answer relevancy: Does the answer directly address the question?
        
        Returns:
            {
                'score': float (0.0-1.0),
                'verdict': str ('yes'/'no'),
                'reason': str,
                'missing_aspects': List[str]
            }
        """
        prompt = f"""You are evaluating the RELEVANCY of an answer to a legal question.

QUESTION: {question}

ANSWER TO EVALUATE:
{answer}

TASK: Determine if the answer DIRECTLY and COMPLETELY addresses the question.

INSTRUCTIONS:
1. Identify what the question is asking for
2. Check if the answer provides that information
3. Check for completeness - are all aspects of the question addressed?
4. Ignore extra information (not penalized), focus on whether core question is answered

Respond ONLY with valid JSON (no markdown, no explanation):
{{
  "verdict": "yes" or "no",
  "score": 0.0-1.0,
  "reason": "brief explanation",
  "missing_aspects": ["aspect 1", "aspect 2", ...]
}}

If answer fully addresses question, verdict="yes" and score=1.0.
If answer is partial or off-topic, verdict="no" and score = (addressed_aspects / total_aspects).
"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Clean potential markdown
            content = self._clean_json(content)
            
            result = json.loads(content)
            
            # Validate and normalize
            if 'verdict' not in result:
                result['verdict'] = 'yes' if result.get('score', 0) >= 0.8 else 'no'
            if 'score' not in result:
                result['score'] = 1.0 if result['verdict'] == 'yes' else 0.5
            if 'reason' not in result:
                result['reason'] = 'Unknown'
            if 'missing_aspects' not in result:
                result['missing_aspects'] = []
            
            # Ensure score is float
            result['score'] = float(result['score'])
            
            if self.verbose:
                print(f"\n📊 Relevancy Grading:")
                print(f"   Score: {result['score']:.3f}")
                print(f"   Verdict: {result['verdict']}")
                if result['missing_aspects']:
                    print(f"   Missing aspects: {len(result['missing_aspects'])}")
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Relevancy grading failed: {e}")
            
            # Conservative fallback
            return {
                'score': 0.5,
                'verdict': 'no',
                'reason': f'Grading error: {str(e)}',
                'missing_aspects': []
            }
    
    def grade_answer(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        faithfulness_threshold: float = 0.9,
        relevancy_threshold: float = 0.8
    ) -> Dict[str, any]:
        """
        Grade both faithfulness and relevancy.
        
        Returns:
            {
                'faithfulness': {...},
                'relevancy': {...},
                'needs_correction': bool,
                'correction_type': str ('faithfulness'/'relevancy'/'both'/None)
            }
        """
        faithfulness = self.grade_faithfulness(question, answer, contexts)
        relevancy = self.grade_relevancy(question, answer)
        
        # Determine if correction is needed
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
        """Remove markdown formatting from JSON response."""
        # Remove ```json and ``` markers
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()
        return text


# Example usage
if __name__ == "__main__":
    grader = AnswerGrader(verbose=True)
    
    # Test case 1: Good answer
    question = "What is the Default Rate?"
    answer = "As per Clause 1, the Default Rate is the interest rate identified in the Key Information Table."
    contexts = [
        "Definition of Default Rate: 'Default Rate' means the interest rate identified in the Key Information Table."
    ]
    
    print("=" * 80)
    print("TEST 1: Good Answer (should pass)")
    print("=" * 80)
    result = grader.grade_answer(question, answer, contexts)
    print(f"\nNeeds correction: {result['needs_correction']}")
    print(f"Correction type: {result['correction_type']}")
    
    # Test case 2: Unfaithful answer
    question = "What is the Default Rate?"
    answer = "The Default Rate is 12% per annum, compounded quarterly."
    contexts = [
        "Definition of Default Rate: 'Default Rate' means the interest rate identified in the Key Information Table."
    ]
    
    print("\n" + "=" * 80)
    print("TEST 2: Unfaithful Answer (should fail faithfulness)")
    print("=" * 80)
    result = grader.grade_answer(question, answer, contexts)
    print(f"\nNeeds correction: {result['needs_correction']}")
    print(f"Correction type: {result['correction_type']}")
    
    # Test case 3: Irrelevant answer
    question = "What is the Default Rate?"
    answer = "The Buyer is responsible for maintaining the Grid connection."
    contexts = [
        "Definition of Default Rate: 'Default Rate' means the interest rate identified in the Key Information Table."
    ]
    
    print("\n" + "=" * 80)
    print("TEST 3: Irrelevant Answer (should fail relevancy)")
    print("=" * 80)
    result = grader.grade_answer(question, answer, contexts)
    print(f"\nNeeds correction: {result['needs_correction']}")
    print(f"Correction type: {result['correction_type']}")
