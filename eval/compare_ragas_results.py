import pandas as pd
import ast
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

configs = [
    #("Simple", os.path.join(project_root, "evaluation_results_simple.csv")),
    #("Adaptive", os.path.join(project_root, "evaluation_results_adaptive.csv")),
    ("Adaptive+Correction", os.path.join(project_root, "evaluation_results_adaptive-correction.csv")),
]

def evaluate_file(csv_path):
    """Run RAGAS evaluation on a CSV file and return metrics dict."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")
        return None

    if len(df) == 0:
        print(f"⚠️  File empty: {csv_path}")
        return None

    # Convert contexts from string to list
    if 'contexts' in df.columns:
        df['contexts'] = df['contexts'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Rename columns to RAGAS expected names
    rename_map = {
        'question': 'user_input',
        'answer': 'response',
        'contexts': 'retrieved_contexts',
        'ground_truth': 'reference'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Ensure required columns exist
    required = ['user_input', 'response', 'retrieved_contexts', 'reference']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"❌ Missing columns in {csv_path}: {missing}")
        return None

    testset = Dataset.from_pandas(df[required])

    # Initialize models
    generator_llm = LangchainLLMWrapper(
        ChatOllama(model="qwen2.5:7b", temperature=0.0, num_ctx=8192, request_timeout=120)
    )
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    )

    faithfulness = Faithfulness(llm=generator_llm)
    answer_relevancy = AnswerRelevancy(llm=generator_llm, embeddings=embeddings)
    context_precision = ContextPrecision(llm=generator_llm)

    print(f"\n🚀 Running RAGAS evaluation for {os.path.basename(csv_path)}...")
    run_config = RunConfig(timeout=240, max_retries=3)
    results = evaluate(
        dataset=testset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        run_config=run_config
    )

    # Convert results to DataFrame and compute means only for metric columns
    results_df = results.to_pandas()
    metric_cols = ['faithfulness', 'answer_relevancy', 'context_precision']
    means = results_df[metric_cols].mean()

    return {
        'faithfulness': means['faithfulness'],
        'answer_relevancy': means['answer_relevancy'],
        'context_precision': means['context_precision']
    }

def main():
    print("=" * 70)
    print("📊 COMPARISON OF RAGAS METRICS FOR THREE CONFIGURATIONS")
    print("=" * 70)

    results = {}
    for name, path in configs:
        print(f"\n--- {name} ---")
        metrics = evaluate_file(path)
        if metrics:
            results[name] = metrics

    if not results:
        print("\n❌ No valid results to compare.")
        return

    # Print comparison table
    print("\n" + "=" * 70)
    print("📈 COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Configuration':<25} {'Faithfulness':>15} {'Answer Relevancy':>20} {'Context Precision':>20}")
    print("-" * 80)
    for name in results:
        m = results[name]
        print(f"{name:<25} {m['faithfulness']:>15.3f} {m['answer_relevancy']:>20.3f} {m['context_precision']:>20.3f}")

    # Improvement relative to simple baseline
    if "Simple" in results:
        simple = results["Simple"]
        print("\n" + "=" * 70)
        print("📊 IMPROVEMENT RELATIVE TO SIMPLE BASELINE")
        print("=" * 70)
        print(f"{'Configuration':<25} {'Faithfulness Δ%':>15} {'Answer Relevancy Δ%':>20} {'Context Precision Δ%':>20}")
        print("-" * 80)
        for name in results:
            if name == "Simple":
                continue
            m = results[name]
            faith_delta = ((m['faithfulness'] - simple['faithfulness']) / simple['faithfulness']) * 100
            rel_delta = ((m['answer_relevancy'] - simple['answer_relevancy']) / simple['answer_relevancy']) * 100
            prec_delta = ((m['context_precision'] - simple['context_precision']) / simple['context_precision']) * 100
            print(f"{name:<25} {faith_delta:>+15.1f}% {rel_delta:>+20.1f}% {prec_delta:>+20.1f}%")

    print("\n✅ Comparison complete.")

if __name__ == "__main__":
    main()