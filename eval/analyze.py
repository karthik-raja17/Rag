import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

baseline_path = os.path.join(project_root, 'evaluation_results_baseline.csv')
corrected_path = os.path.join(project_root, 'evaluation_results_corrected.csv')

try:
    corrected = pd.read_csv(corrected_path)

    print("Correction Statistics:")
    print(f"  Total corrections: {corrected['correction_attempts'].sum()}")
    print(f"  Avg corrections/question: {corrected['correction_attempts'].mean():.2f}")
    print(f"  Success rate: {corrected['correction_success'].mean()*100:.1f}%")

    print("\nQuality Metrics:")
    print(f"  Avg faithfulness: {corrected['final_faithfulness'].mean():.3f}")
    print(f"  Avg relevancy: {corrected['final_relevancy'].mean():.3f}")

except FileNotFoundError:
    print(f"❌ Could not find corrected CSV at: {corrected_path}")
except KeyError as e:
    print(f"❌ Missing column: {e}. Check that the corrected CSV has the expected columns.")