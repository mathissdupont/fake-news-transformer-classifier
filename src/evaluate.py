"""Evaluation script: aggregate metrics, generate figures, and prepare final results table."""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, 'results')


def load_all_metrics():
    """Load all metrics CSVs from results/."""
    metrics_files = list(Path(RESULTS_DIR).glob('*_metrics.csv'))
    dfs = []
    for f in metrics_files:
        df = pd.read_csv(f)
        dfs.append(df)
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        return combined
    else:
        print('No metrics CSV files found in', RESULTS_DIR)
        return None


def create_comparison_table(combined_df, output_path=None):
    """Create a formatted comparison table of all models."""
    if combined_df is None:
        return None
    
    # Select key columns for comparison
    comparison = combined_df[['Model', 'Classifier', 'Accuracy', 'Precision', 'Recall', 'F1-Score']]
    comparison = comparison.sort_values(['Model', 'Classifier'])
    
    # Round for readability
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        comparison[col] = comparison[col].round(4)
    
    if output_path:
        comparison.to_csv(output_path, index=False)
        print(f'Saved comparison table to {output_path}')
    
    return comparison


def plot_f1_comparison(combined_df, output_path=None):
    """Create F1-score comparison bar plot."""
    if combined_df is None:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    data = combined_df[['Model', 'Classifier', 'F1-Score']].copy()
    data['Name'] = data['Model'] + '_' + data['Classifier']
    
    ax.bar(range(len(data)), data['F1-Score'], color=['blue', 'green', 'red', 'orange'][:len(data)])
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data['Name'], rotation=45, ha='right')
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score Comparison: TF-IDF vs Embeddings')
    ax.set_ylim([0.8, 1.0])
    
    for i, v in enumerate(data['F1-Score']):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300)
        print(f'Saved comparison plot to {output_path}')
    plt.close()


def main():
    print('Loading all metrics...')
    combined = load_all_metrics()
    
    if combined is not None:
        print('\nAll metrics:')
        print(combined)
        
        # Save comparison table
        comparison_path = os.path.join(RESULTS_DIR, 'final_results_table.csv')
        comparison = create_comparison_table(combined, output_path=comparison_path)
        
        # Generate comparison plot
        plot_path = os.path.join(RESULTS_DIR, 'f1_comparison.png')
        plot_f1_comparison(combined, output_path=plot_path)
        
        print('\nEvaluation complete. Check results/ for outputs.')
    else:
        print('No metrics found. Ensure Person 1 and Person 2 have run their training scripts.')


if __name__ == '__main__':
    main()
