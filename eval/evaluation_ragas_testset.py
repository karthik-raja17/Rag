import pandas as pd
from datasets import Dataset
from ragas import evaluate
# FIX 1: Import the capitalized classes, not the lowercase modules
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import ast

print("=" * 60)
print("📊 RAGAS Evaluation Phase")
print("=" * 60)

# 1. Load the Data
csv_file = "testset_legal_simple.csv"
try:
    df = pd.read_csv(csv_file)
    
    # RAGAS specific: Convert string-represented lists back to actual lists
    context_columns = ['reference_contexts', 'retrieved_contexts', 'contexts']
    for col in context_columns:
        if col in df.columns:
            print(f"🔄 Converting {col} from string to list...")
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # NEW MAPPING BLOCK: satisfy the validation error requirements
    # 'Faithfulness' requires 'retrieved_contexts' and 'response'
    if 'reference_contexts' in df.columns and 'retrieved_contexts' not in df.columns:
        df['retrieved_contexts'] = df['reference_contexts']
    
    if 'reference' in df.columns and 'response' not in df.columns:
        # We use the reference/ground_truth as a placeholder response for the evaluation test
        df['response'] = df['reference']

    # Final RAGAS Schema Rename
    mapping = {
        'question': 'user_input',
        'ground_truth': 'reference'
    }
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    
    testset = Dataset.from_pandas(df)
    print(f"✅ Loaded {len(testset)} samples with columns: {df.columns.tolist()}")
except FileNotFoundError:
    print(f"❌ Error: {csv_file} not found.")
    exit()

# 2. Initialize Models
generator_llm = LangchainLLMWrapper(ChatOllama(model="qwen2.5:7b", temperature=0.0, num_ctx=8192))
embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="BAAI/bge-m3"))

# 3. FIX 2: Initialize the metrics as objects
# This avoids the "module is not callable" error by calling the class constructor
faith = Faithfulness(llm=generator_llm)
relevancy = AnswerRelevancy(llm=generator_llm, embeddings=embeddings)
precision = ContextPrecision(llm=generator_llm)

# 4. Run Evaluation
print("\n🚀 Running evaluation (this uses CUDA:0)...")
results = evaluate(
    dataset=testset,
    metrics=[faith, relevancy, precision],
)

# 5. Show results
print("\n" + "=" * 60)
print("📈 EVALUATION RESULTS")
print("=" * 60)
print(results)