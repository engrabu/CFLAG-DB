import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class ConvergenceRateAnalyzer:
    """
    Tool for analyzing and visualizing convergence rates in federated learning methods
    """
    def __init__(self, results_dir='./results'):
        """
        Initialize analyzer
        
        Args:
            results_dir: Directory containing experiment result JSON files
        """
        self.results_dir = results_dir
        self.method_results = defaultdict(list)
        self.convergence_data = {}
        
    def load_results(self, dataset_filter=None):
        """
        Load and process all experiment results
        
        Args:
            dataset_filter: Optional filter to only process specific dataset
        """
        json_files = [f for f in os.listdir(self.results_dir) if f.endswith('.json')]
        
        for filename in json_files:
            if dataset_filter and not filename.startswith(dataset_filter):
                continue
                
            filepath = os.path.join(self.results_dir, filename)
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                # Only process if it has both dataset and method
                if 'dataset' in data and 'method' in data:
                    dataset = data['dataset']
                    method = data['method']
                    
                    # Add to method results
                    self.method_results[(dataset, method)].append(data)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                
        print(f"Loaded results from {len(self.method_results)} different dataset/method combinations")
                
    def calculate_convergence_rates(self):
        """
        Calculate convergence rates for all methods
        """
        self.convergence_data = {}
        
        for (dataset, method), results_list in self.method_results.items():
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
                self.convergence_data[(dataset, method)] = {
                    'accuracy_history': accuracy_history,
                    'absolute_changes': absolute_changes,
                    'percentage_changes': percentage_changes,
                    'avg_absolute_improvement': avg_improvement,
                    'final_accuracy': accuracy_history[-1] if accuracy_history else 0,
                    'total_rounds': len(accuracy_history),
                    'best_round_improvement': max(absolute_changes) if absolute_changes else 0,
                    'best_round_index': np.argmax(absolute_changes) + 1 if absolute_changes else 0
                }
    
    def export_convergence_data(self, output_file='convergence_data.json'):
        """
        Export convergence data to JSON
        
        Args:
            output_file: File to write JSON data to
        """
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
        
        for (dataset, method), data in self.convergence_data.items():
            key = f"{dataset}_{method}"
            
            # Convert NumPy types to Python native types
            serializable_data[key] = {
                'accuracy_history': [float(x) if isinstance(x, np.floating) else x for x in data['accuracy_history']],
                'absolute_changes': [float(x) if isinstance(x, np.floating) else x for x in data['absolute_changes']],
                'percentage_changes': [float(x) if not np.isinf(x) else 9999.0 for x in data['percentage_changes']],
                'avg_absolute_improvement': float(data['avg_absolute_improvement']),
                'final_accuracy': float(data['final_accuracy']),
                'total_rounds': int(data['total_rounds']),
                'best_round_improvement': float(data['best_round_improvement']),
                'best_round_index': int(data['best_round_index'])
            }
                
        # Write to file using the custom encoder
        with open(output_file, 'w') as f:
            json.dump(serializable_data, f, indent=2, cls=NumpyEncoder)
                
        print(f"Exported convergence data to {output_file}")
        
    def create_comparison_table(self):
        """
        Create comparison table of convergence metrics
        
        Returns:
            df: Pandas DataFrame with convergence comparison
        """
        rows = []
        
        for (dataset, method), data in self.convergence_data.items():
            row = {
                'Dataset': dataset,
                'Method': method,
                'Final Accuracy': data['final_accuracy'] * 100,  # as percentage
                'Avg. Improvement/Round': data['avg_absolute_improvement'] * 100,  # as percentage points
                'Best Round Improvement': data['best_round_improvement'] * 100,  # as percentage points
                'Best Round': data['best_round_index'],
                'Total Rounds': data['total_rounds']
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # Sort by dataset, then by final accuracy
        df = df.sort_values(['Dataset', 'Final Accuracy'], ascending=[True, False])
        
        return df
    
    def plot_convergence_rates(self, dataset_filter=None, save_path=None):
        """
        Create plots of convergence rates
        
        Args:
            dataset_filter: Optional filter to only plot specific dataset
            save_path: Optional path to save plots
            
        Returns:
            figs: Dictionary of figure objects
        """
        figs = {}
        
        # Filter data if needed
        if dataset_filter:
            plot_data = {k: v for k, v in self.convergence_data.items() if k[0] == dataset_filter}
        else:
            plot_data = self.convergence_data
            
        # Group by dataset
        datasets = set(k[0] for k in plot_data.keys())
        
        # Create color map for methods
        methods = sorted(set(k[1] for k in plot_data.keys()))
        cmap = plt.colormaps['tab10'].resampled(len(methods))
        colors = {method: cmap(i) for i, method in enumerate(methods)}
        
        for dataset in datasets:
            # Filter data for this dataset
            dataset_data = {k: v for k, v in plot_data.items() if k[0] == dataset}
            
            # Plot 1: Accuracy over rounds
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            
            for (_, method), data in dataset_data.items():
                rounds = list(range(1, len(data['accuracy_history']) + 1))
                ax1.plot(rounds, [acc * 100 for acc in data['accuracy_history']], 
                         marker='o', label=method, color=colors[method])
                
            ax1.set_title(f'Accuracy Over Rounds - {dataset}')
            ax1.set_xlabel('Communication Round')
            ax1.set_ylabel('Accuracy (%)')
            ax1.grid(True)
            ax1.legend()
            
            figs[f'accuracy_{dataset}'] = fig1
            
            if save_path:
                fig1.savefig(os.path.join(save_path, f'accuracy_{dataset}.png'))
                
            # Plot 2: Absolute improvement per round
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            
            for (_, method), data in dataset_data.items():
                if data['absolute_changes']:
                    rounds = list(range(2, len(data['accuracy_history']) + 1))
                    ax2.plot(rounds, [change * 100 for change in data['absolute_changes']], 
                             marker='o', label=method, color=colors[method])
                    
            ax2.set_title(f'Absolute Improvement Per Round - {dataset}')
            ax2.set_xlabel('Communication Round')
            ax2.set_ylabel('Accuracy Improvement (pp)')
            ax2.grid(True)
            ax2.legend()
            
            figs[f'improvement_{dataset}'] = fig2
            
            if save_path:
                fig2.savefig(os.path.join(save_path, f'improvement_{dataset}.png'))
                
            # Plot 3: Percentage improvement per round
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            
            for (_, method), data in dataset_data.items():
                if data['percentage_changes']:
                    rounds = list(range(2, len(data['accuracy_history']) + 1))
                    
                    # Filter out infinite values for display
                    pct_changes = data['percentage_changes'].copy()
                    for i, val in enumerate(pct_changes):
                        if val > 1000:  # Cap extremely high values
                            pct_changes[i] = 1000
                            
                    ax3.plot(rounds, pct_changes, marker='o', label=method, color=colors[method])
                    
            ax3.set_title(f'Percentage Improvement Per Round - {dataset}')
            ax3.set_xlabel('Communication Round')
            ax3.set_ylabel('Relative Improvement (%)')
            ax3.grid(True)
            ax3.legend()
            
            figs[f'pct_improvement_{dataset}'] = fig3
            
            if save_path:
                fig3.savefig(os.path.join(save_path, f'pct_improvement_{dataset}.png'))
                
            # Plot 4: Average convergence rate comparison
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            
            methods = []
            avg_improvements = []
            final_accuracies = []
            
            for (_, method), data in sorted(dataset_data.items(), key=lambda x: x[1]['avg_absolute_improvement'], reverse=True):
                methods.append(method)
                avg_improvements.append(data['avg_absolute_improvement'] * 100)  # as percentage points
                final_accuracies.append(data['final_accuracy'] * 100)  # as percentage
                
            # Create bar chart with two bars per method
            x = np.arange(len(methods))
            width = 0.35
            
            ax4.bar(x - width/2, avg_improvements, width, label='Avg. Improvement/Round (pp)', color='skyblue')
            ax4.bar(x + width/2, final_accuracies, width, label='Final Accuracy (%)', color='lightcoral')
            
            ax4.set_title(f'Convergence Rate vs Final Accuracy - {dataset}')
            ax4.set_xticks(x)
            ax4.set_xticklabels(methods)
            ax4.legend()
            ax4.grid(axis='y')
            
            # Add value labels on top of bars
            for i, v in enumerate(avg_improvements):
                ax4.text(i - width/2, v + 1, f'{v:.2f}', ha='center')
                
            for i, v in enumerate(final_accuracies):
                ax4.text(i + width/2, v + 1, f'{v:.2f}', ha='center')
                
            figs[f'convergence_rate_{dataset}'] = fig4
            
            if save_path:
                fig4.savefig(os.path.join(save_path, f'convergence_rate_{dataset}.png'))
                
        return figs

# Example usage
if __name__ == "__main__":
    analyzer = ConvergenceRateAnalyzer()
    analyzer.load_results(dataset_filter="cifar10")
    analyzer.calculate_convergence_rates()
    
    # Export data to JSON
    analyzer.export_convergence_data("cifar10_convergence_data.json")
    
    # Create comparison table
    comparison_df = analyzer.create_comparison_table()
    print("Convergence Rate Comparison:")
    print(comparison_df)
    comparison_df.to_csv("convergence_comparison.csv", index=False)
    
    # Create and save plots
    os.makedirs("convergence_plots", exist_ok=True)
    analyzer.plot_convergence_rates(save_path="convergence_plots")