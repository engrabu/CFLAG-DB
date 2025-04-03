"""
Script to analyze convergence rates from existing experiment results
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict

def load_json_files(results_dir, dataset_filter=None):
    """Load all JSON files from results directory"""
    method_results = defaultdict(list)
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    for filename in json_files:
        if dataset_filter and not filename.startswith(dataset_filter):
            continue
            
        filepath = os.path.join(results_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Only process if it has both dataset and method
            if 'dataset' in data and 'method' in data:
                dataset = data['dataset']
                method = data['method']
                
                # Add to method results
                method_results[(dataset, method)].append(data)
                print(f"Loaded: {filename} - {dataset}/{method}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            
    print(f"Loaded results from {len(method_results)} different dataset/method combinations")
    return method_results

def calculate_convergence_rates(method_results):
    """Calculate convergence rates for all methods"""
    convergence_data = {}
    
    for (dataset, method), results_list in method_results.items():
        # Use the result with the most rounds
        result = max(results_list, key=lambda x: len(x.get('accuracy_history', [])))
        
        if 'accuracy_history' in result and len(result['accuracy_history']) > 1:
            accuracy_history = result['accuracy_history']
            
            # Calculate absolute change between rounds
            absolute_changes = [accuracy_history[i] - accuracy_history[i-1] for i in range(1, len(accuracy_history))]
            
            # Calculate percentage change relative to previous accuracy
            percentage_changes = []
            for i in range(1, len(accuracy_history)):
                prev_acc = accuracy_history[i-1]
                if prev_acc > 0.001:  # Avoid division by very small values
                    pct_change = (accuracy_history[i] - prev_acc) / prev_acc * 100
                else:
                    pct_change = float('inf') if accuracy_history[i] > 0 else 0
                percentage_changes.append(pct_change)
            
            # Calculate convergence rate as the average absolute improvement per round
            avg_improvement = np.mean(absolute_changes) if absolute_changes else 0
            
            # Track all data
            convergence_data[(dataset, method)] = {
                'accuracy_history': accuracy_history,
                'absolute_changes': absolute_changes,
                'percentage_changes': percentage_changes,
                'avg_absolute_improvement': avg_improvement,
                'final_accuracy': accuracy_history[-1] if accuracy_history else 0,
                'total_rounds': len(accuracy_history),
                'best_round_improvement': max(absolute_changes) if absolute_changes else 0,
                'best_round_index': np.argmax(absolute_changes) + 1 if absolute_changes else 0
            }
    
    return convergence_data

def create_comparison_table(convergence_data):
    """Create comparison table of convergence metrics"""
    rows = []
    
    for (dataset, method), data in convergence_data.items():
        row = {
            'Dataset': dataset,
            'Method': method,
            'Final Accuracy (%)': data['final_accuracy'] * 100,  # as percentage
            'Avg. Improvement/Round (pp)': data['avg_absolute_improvement'] * 100,  # as percentage points
            'Best Round Improvement (pp)': data['best_round_improvement'] * 100,  # as percentage points
            'Best Round': data['best_round_index'],
            'Total Rounds': data['total_rounds']
        }
        rows.append(row)
        
    df = pd.DataFrame(rows)
    
    # Sort by dataset, then by final accuracy
    df = df.sort_values(['Dataset', 'Final Accuracy (%)'], ascending=[True, False])
    
    return df

def export_convergence_data(convergence_data, output_file):
    """Export convergence data to JSON with NumPy type handling"""
    import json
    import numpy as np
    
    # Create a custom JSON encoder that handles NumPy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, np.bool)):
                return bool(obj)
            return super(NumpyEncoder, self).default(obj)
    
    # Create serializable version of the data
    serializable_data = {}
    
    for (dataset, method), data in convergence_data.items():
        key = f"{dataset}_{method}"
        serializable_data[key] = {
            'accuracy_history': data['accuracy_history'],
            'absolute_changes': data['absolute_changes'],
            'percentage_changes': [float(x) if not np.isinf(x) else 9999.0 for x in data['percentage_changes']],
            'avg_absolute_improvement': data['avg_absolute_improvement'],
            'final_accuracy': data['final_accuracy'],
            'total_rounds': data['total_rounds'],
            'best_round_improvement': data['best_round_improvement'],
            'best_round_index': data['best_round_index']
        }
            
    # Write to file using the custom encoder
    with open(output_file, 'w') as f:
        json.dump(serializable_data, f, indent=2, cls=NumpyEncoder)
        
    print(f"Exported convergence data to {output_file}")

def plot_convergence_rates(convergence_data, dataset_filter=None, save_path=None):
    """Create plots of convergence rates"""
    # Filter data if needed
    if dataset_filter:
        plot_data = {k: v for k, v in convergence_data.items() if k[0] == dataset_filter}
    else:
        plot_data = convergence_data
        
    # Group by dataset
    datasets = set(k[0] for k in plot_data.keys())
    
    # Set up seaborn style
    sns.set_style("whitegrid")
    
    # Create color map for methods
    methods = sorted(set(k[1] for k in plot_data.keys()))
    # Use plt.colormaps instead of get_cmap (updated for newer matplotlib)
    if hasattr(plt, 'colormaps'):
        # For newer matplotlib versions
        colors = {method: plt.colormaps['tab10'](i) for i, method in enumerate(methods)}
    else:
        # Fallback for older versions
        colors = {method: plt.cm.tab10(i) for i, method in enumerate(methods)}
    
    for dataset in datasets:
        # Filter data for this dataset
        dataset_data = {k: v for k, v in plot_data.items() if k[0] == dataset}
        
        # Plot 1: Accuracy over rounds
        plt.figure(figsize=(12, 6))
        
        for key, data in dataset_data.items():
            method = key[1]  # Extract method from the tuple key
            rounds = list(range(1, len(data['accuracy_history']) + 1))
            plt.plot(rounds, [acc * 100 for acc in data['accuracy_history']], 
                     marker='o', label=method, color=colors[method], linewidth=2)
            
        plt.title(f'Accuracy Over Rounds - {dataset}', fontsize=14)
        plt.xlabel('Communication Round', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, f'accuracy_{dataset}.png'), dpi=300)
            
        # Plot 2: Absolute improvement per round
        plt.figure(figsize=(12, 6))
        
        for key, data in dataset_data.items():
            method = key[1]  # Extract method from the tuple key
            if data['absolute_changes']:
                rounds = list(range(2, len(data['accuracy_history']) + 1))
                plt.plot(rounds, [change * 100 for change in data['absolute_changes']], 
                         marker='o', label=method, color=colors[method], linewidth=2)
                
        plt.title(f'Absolute Improvement Per Round - {dataset}', fontsize=14)
        plt.xlabel('Communication Round', fontsize=12)
        plt.ylabel('Accuracy Improvement (percentage points)', fontsize=12)
        plt.legend(fontsize=10)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, f'improvement_{dataset}.png'), dpi=300)
            
        # Plot 3: Percentage improvement per round
        plt.figure(figsize=(12, 6))
        
        for key, data in dataset_data.items():
            method = key[1]  # Extract method from the tuple key
            if data['percentage_changes']:
                rounds = list(range(2, len(data['accuracy_history']) + 1))
                
                # Filter out infinite values for display
                pct_changes = data['percentage_changes'].copy()
                for i, val in enumerate(pct_changes):
                    if val > 1000 or np.isinf(val):  # Cap extremely high values
                        pct_changes[i] = 1000
                        
                plt.plot(rounds, pct_changes, marker='o', label=method, color=colors[method], linewidth=2)
                
        plt.title(f'Percentage Improvement Per Round - {dataset}', fontsize=14)
        plt.xlabel('Communication Round', fontsize=12)
        plt.ylabel('Relative Improvement (%)', fontsize=12)
        plt.legend(fontsize=10)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, f'pct_improvement_{dataset}.png'), dpi=300)
            
        # Plot 4: Average convergence rate comparison
        plt.figure(figsize=(12, 6))
        
        method_list = []
        avg_improvements = []
        final_accuracies = []
        
        for key, data in sorted(dataset_data.items(), key=lambda x: x[1]['final_accuracy'], reverse=True):
            method = key[1]  # Extract method from the tuple key
            method_list.append(method)
            avg_improvements.append(data['avg_absolute_improvement'] * 100)  # as percentage points
            final_accuracies.append(data['final_accuracy'] * 100)  # as percentage
            
        # Create bar chart with two bars per method
        x = np.arange(len(method_list))
        width = 0.35
        
        plt.bar(x - width/2, avg_improvements, width, label='Avg. Improvement/Round (pp)', color='skyblue')
        plt.bar(x + width/2, final_accuracies, width, label='Final Accuracy (%)', color='lightcoral')
        
        plt.title(f'Convergence Rate vs Final Accuracy - {dataset}', fontsize=14)
        plt.xticks(x, method_list, rotation=45, ha='right')
        plt.ylabel('Percentage / Percentage Points', fontsize=12)
        plt.legend(fontsize=10)
        
        # Add value labels on top of bars
        for i, v in enumerate(avg_improvements):
            plt.text(i - width/2, v + 1, f'{v:.2f}', ha='center', fontsize=9)
            
        for i, v in enumerate(final_accuracies):
            plt.text(i + width/2, v + 1, f'{v:.2f}', ha='center', fontsize=9)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, f'convergence_rate_{dataset}.png'), dpi=300)
            
        # Plot 5: Heatmap of improvement across rounds and methods
        methods_with_data = []
        improvement_data = []
        
        # Skip the heatmap if we don't have enough data
        if not all('absolute_changes' in data for data in dataset_data.values()):
            continue
            
        try:
            # Calculate max rounds for padding
            max_rounds = max(len(data['absolute_changes']) for data in dataset_data.values())
            
            for key, data in dataset_data.items():
                method = key[1]  # Extract method from the tuple key
                if data['absolute_changes']:
                    methods_with_data.append(method)
                    improvements = [change * 100 for change in data['absolute_changes']]
                    # Pad with NaN if needed to make all lists the same length
                    padded_improvements = improvements + [np.nan] * (max_rounds - len(improvements))
                    improvement_data.append(padded_improvements)
            
            if improvement_data and len(improvement_data) > 1:  # Need at least 2 methods for heatmap
                plt.figure(figsize=(12, 8))
                
                # Convert to DataFrame for heatmap
                df = pd.DataFrame(improvement_data, index=methods_with_data)
                df.columns = [f"Round {i+2}" for i in range(df.shape[1])]
                
                # Create heatmap
                sns.heatmap(df, annot=True, cmap="RdYlGn", center=0, fmt=".1f", linewidths=.5)
                
                plt.title(f'Accuracy Improvement Heatmap - {dataset}', fontsize=14)
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(os.path.join(save_path, f'heatmap_{dataset}.png'), dpi=300)
        except Exception as e:
            print(f"Error creating heatmap for {dataset}: {e}")

def analyze_cflag_parameters(method_results, save_path=None):
    """Analyze the impact of different CFLAG-BD parameters"""
    # Group by parameter values
    cflag_results = {k: v for k, v in method_results.items() if k[1].lower() == 'cflag-bd'}
    
    if not cflag_results:
        print("No CFLAG-BD results to analyze")
        return
        
    parameter_data = []
    
    for (dataset, _), results_list in cflag_results.items():
        for result in results_list:
            if 'params' in result and 'final_accuracy' in result:
                params = result['params']
                data_entry = {
                    'dataset': dataset,
                    'final_accuracy': result['final_accuracy'] * 100,
                    'num_clusters': params.get('num_clusters', 3),
                    'dedup_threshold': params.get('dedup_threshold', 0.01),
                    'client_fraction': params.get('client_fraction', 0.2),
                    'storage_savings': result.get('storage_savings', 0)
                }
                parameter_data.append(data_entry)
    
    if not parameter_data:
        print("No parameter data available for CFLAG-BD")
        return
        
    # Create DataFrame
    param_df = pd.DataFrame(parameter_data)
    
    # Create plots for parameter impact
    for param in ['num_clusters', 'dedup_threshold', 'client_fraction']:
        plt.figure(figsize=(10, 6))
        
        # Group by parameter value
        param_groups = param_df.groupby(param)
        
        # Calculate mean accuracy for each parameter value
        mean_accuracies = param_groups['final_accuracy'].mean()
        sem_accuracies = param_groups['final_accuracy'].sem()
        
        # Create bar chart
        ax = mean_accuracies.plot(kind='bar', yerr=sem_accuracies, capsize=5, color='skyblue')
        
        plt.title(f'Impact of {param} on Final Accuracy', fontsize=14)
        plt.xlabel(param, fontsize=12)
        plt.ylabel('Final Accuracy (%)', fontsize=12)
        plt.ylim(0, 100)
        
        # Add value labels on top of bars
        for i, v in enumerate(mean_accuracies):
            ax.text(i, v + 1, f'{v:.2f}', ha='center', fontsize=9)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, f'param_impact_{param}.png'), dpi=300)
            
    # Plot relationship between storage savings and accuracy
    plt.figure(figsize=(10, 6))
    
    sns.scatterplot(data=param_df, x='storage_savings', y='final_accuracy', hue='num_clusters', 
                   palette='viridis', s=100, alpha=0.7)
    
    plt.title('Storage Savings vs. Final Accuracy', fontsize=14)
    plt.xlabel('Storage Savings (%)', fontsize=12)
    plt.ylabel('Final Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Num Clusters')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'storage_vs_accuracy.png'), dpi=300)

def visualize_client_clusters_tsne(self, perplexity=30, n_iter=1000):
    """
    Create t-SNE visualization of client clustering
    
    Args:
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations for t-SNE
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    # Extract features for all clients
    all_features, feature_names = self.extract_client_features()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    features_2d = tsne.fit_transform(all_features)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Color by cluster assignment
    for cluster_idx in range(self.num_clusters):
        cluster_mask = self.client_clusters == cluster_idx
        plt.scatter(
            features_2d[cluster_mask, 0],
            features_2d[cluster_mask, 1],
            label=f'Cluster {cluster_idx}',
            alpha=0.7
        )
    
    # Add labels and legend
    plt.title('t-SNE Visualization of Client Clusters')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(alpha=0.3)
    
    return plt.gcf()
def main():
    # Directory containing result files
    results_dir = './results'
    
    # Create output directory for plots
    output_dir = './convergence_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load result files
    method_results = load_json_files(results_dir, dataset_filter='fashion_mnist')
    
    # Calculate convergence rates
    convergence_data = calculate_convergence_rates(method_results)
    
    # Create comparison table
    comparison_df = create_comparison_table(convergence_data)
    print("Convergence Rate Comparison:")
    print(comparison_df)
    
    # Save comparison table
    comparison_df.to_csv(os.path.join(output_dir, 'convergence_comparison.csv'), index=False)
    
    # Export convergence data
    export_convergence_data(convergence_data, os.path.join(output_dir, 'convergence_data.json'))
    
    # Create plots
    plot_convergence_rates(convergence_data, save_path=output_dir)
    
    # Analyze CFLAG-BD parameters
    analyze_cflag_parameters(method_results, save_path=output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()