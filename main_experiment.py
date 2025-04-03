import os
import argparse
import time
from experiment_runner import ExperimentRunner


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Federated Learning Experiments')
    
    # Basic parameters
    parser.add_argument('--results_dir', type=str, default='./results', 
                        help='Directory to save results')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save plots to results directory')
    
    # Experiment type
    parser.add_argument('--experiment', type=str, choices=[
        'single', 'comparative', 'ablation', 'scalability', 'all'
    ], default='comparative', help='Type of experiment to run')
    
    # Dataset and method selection
    parser.add_argument('--dataset', type=str, default='fashion_mnist',
                       help='Dataset for single experiment')
    parser.add_argument('--method', type=str, default='cflag-bd',
                       help='Method for single experiment')
    
    # Experiment parameters
    parser.add_argument('--num_clients', type=int, default=2000,
                       help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=20,
                       help='Number of communication rounds')
    parser.add_argument('--local_epochs', type=int, default=5,
                       help='Number of local training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--client_fraction', type=float, default=0.2,
                       help='Fraction of clients to select each round')
    parser.add_argument('--num_clusters', type=int, default=3,
                       help='Number of clusters for clustering methods')
    parser.add_argument('--dedup_threshold', type=float, default=0.01,
                       help='Threshold for deduplication in CFLAG-BD')
    parser.add_argument('--fedprox_mu', type=float, default=0.01,
                       help='Proximal term weight for FedProx')
    parser.add_argument('--quantization_bits', type=int, default=8,
                       help='Number of bits for quantization in CQFL')
    parser.add_argument('--non_iid', action='store_true',
                       help='Use non-IID data distribution')
    parser.add_argument('--use_blockchain', action='store_true',
                       help='Use blockchain in CFLAG-BD')
    parser.add_argument('--dynamic_lr', action='store_true',
                   help='Use dynamic learning rate for CFLAG-BD')
    parser.add_argument('--lr_schedule', type=str, default='adaptive',
                    choices=['step', 'exponential', 'cosine', 'adaptive', 'cyclic'],
                    help='Learning rate schedule type')

    return parser.parse_args()
def generate_clustering_visualizations(clustering_methods, save_dir):
    """
    Generate visualizations for clustering-based FL methods
    
    Args:
        clustering_methods: Dictionary mapping dataset to list of (method_name, method_instance) tuples
        save_dir: Directory to save visualizations
    """
    import os
    
    for dataset, methods in clustering_methods.items():
        print(f"Generating cluster visualizations for {dataset}...")
        
        # Create directory for this dataset
        dataset_dir = os.path.join(save_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Find CFLAG-BD instance if available
        cflag_bd_instance = None
        for method_name, instance in methods:
            if method_name.lower() == 'cflag-bd':
                cflag_bd_instance = instance
                break
        
        if cflag_bd_instance:
            # Generate CFLAG-BD visualizations
            
            # Cluster assignments heatmap
            if hasattr(cflag_bd_instance, 'visualize_cluster_assignments'):
                try:
                    heatmap_path = os.path.join(dataset_dir, 'cflag_cluster_assignments.png')
                    cflag_bd_instance.visualize_cluster_assignments(save_path=heatmap_path)
                    print(f"Created cluster assignment heatmap at {heatmap_path}")
                except Exception as e:
                    print(f"Error creating cluster assignment heatmap: {e}")
            
            # t-SNE visualization
            if hasattr(cflag_bd_instance, 'visualize_client_clusters_tsne'):
                try:
                    tsne_path = os.path.join(dataset_dir, 'cflag_tsne.png')
                    cflag_bd_instance.visualize_client_clusters_tsne(save_path=tsne_path)
                    print(f"Created t-SNE visualization at {tsne_path}")
                except Exception as e:
                    print(f"Error creating t-SNE visualization: {e}")
            
            # Compare with other clustering methods
            for other_name, other_instance in methods:
                if other_name.lower() != 'cflag-bd' and hasattr(other_instance, 'cluster_history'):
                    try:
                        compare_path = os.path.join(dataset_dir, f'compare_{other_name}_vs_cflag.png')
                        cflag_bd_instance.visualize_cluster_assignments(
                            save_path=compare_path,
                            compare_with=other_instance
                        )
                        print(f"Created comparison with {other_name} at {compare_path}")
                    except Exception as e:
                        print(f"Error creating comparison with {other_name}: {e}")
        else:
            print(f"No CFLAG-BD instance found for {dataset}, skipping visualizations")
def collect_clustering_methods(runner, datasets, methods):
    """
    Collect clustering method instances directly from experiment runner
    
    Args:
        runner: The ExperimentRunner instance
        datasets: List of datasets used
        methods: List of methods used
        
    Returns:
        clustering_methods: Dictionary of method instances by dataset
    """
    clustering_methods = {}
    
    # Access method instances directly from the runner
    for dataset in datasets:
        clustering_methods[dataset] = []
        
        for method in methods:
            if method.lower() in ['cflag-bd', 'ifca', 'layercfl', 'cqfl']:
                # Look for method instance in runner's namespace
                method_attr = None
                
                # Try different attribute naming patterns
                possible_attrs = [
                    f"{method.lower().replace('-', '_')}",
                    f"{dataset}_{method.lower().replace('-', '_')}",
                    f"{method.lower().replace('-', '_')}_{dataset}"
                ]
                
                for attr_name in possible_attrs:
                    if hasattr(runner, attr_name):
                        method_attr = getattr(runner, attr_name)
                        print(f"Found {method} instance for {dataset} as {attr_name}")
                        clustering_methods[dataset].append((method, method_attr))
                        break
                
                if not method_attr:
                    # Try searching all attributes for matching instance
                    for attr_name in dir(runner):
                        attr = getattr(runner, attr_name)
                        if (attr_name.startswith(method.lower().replace('-', '_')) or 
                            attr_name.endswith(method.lower().replace('-', '_'))) and \
                           hasattr(attr, 'cluster_history'):
                            print(f"Found {method} instance for {dataset} as {attr_name}")
                            clustering_methods[dataset].append((method, attr))
                            break
    
    return clustering_methods
def generate_debug_visualization(save_dir):
    """Generate debugging visualization with synthetic data"""
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    print("Generating synthetic cluster visualizations for debugging...")

    # Create directory
    os.makedirs(save_dir, exist_ok=True)

    # Create synthetic data - two types of clustering patterns
    n_clients = 50
    n_rounds = 20
    n_clusters = 3

    # Pattern 1: Random clustering that changes a lot (unstable)
    unstable_clusters = []
    for round_idx in range(n_rounds):
        # Random assignments with 50% probability of change
        if round_idx == 0:
            assignments = np.random.randint(0, n_clusters, size=n_clients)
        else:
            assignments = unstable_clusters[-1].copy()
            change_indices = np.random.choice(n_clients, size=int(0.5*n_clients), replace=False)
            for idx in change_indices:
                current = assignments[idx]
                choices = [c for c in range(n_clusters) if c != current]
                assignments[idx] = np.random.choice(choices)
        unstable_clusters.append(assignments)

    # Pattern 2: More stable clustering with only 10% changes
    stable_clusters = []
    for round_idx in range(n_rounds):
        if round_idx == 0:
            # Initial clustering with clear pattern - group by client ID
            assignments = np.zeros(n_clients, dtype=int)
            group_size = n_clients // n_clusters
            for i in range(n_clusters-1):
                assignments[i*group_size:(i+1)*group_size] = i
            assignments[(n_clusters-1)*group_size:] = n_clusters-1
        else:
            assignments = stable_clusters[-1].copy()
            # Only 10% of clients change clusters each round
            change_indices = np.random.choice(n_clients, size=int(0.1*n_clients), replace=False)
            for idx in change_indices:
                current = assignments[idx]
                choices = [c for c in range(n_clusters) if c != current]
                assignments[idx] = np.random.choice(choices)
        stable_clusters.append(assignments)

    # Create heatmap visualizations
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))

    # Plot unstable clustering
    unstable_matrix = np.array(unstable_clusters).T  # Transpose to get clients x rounds
    sns.heatmap(unstable_matrix, ax=axes[0], cmap='viridis', cbar_kws={'label': 'Cluster'})
    axes[0].set_title('Unstable Clustering: Client Assignments over Rounds')
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Client ID')

    # Plot stable clustering
    stable_matrix = np.array(stable_clusters).T  # Transpose to get clients x rounds
    sns.heatmap(stable_matrix, ax=axes[1], cmap='viridis', cbar_kws={'label': 'Cluster'})
    axes[1].set_title('Stable Clustering: Client Assignments over Rounds')
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Client ID')

    plt.tight_layout()

    # Save figure
    viz_path = os.path.join(save_dir, 'cluster_visualization_debug.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Debug visualization saved to {viz_path}")

    # Create t-SNE visualization
    from sklearn.manifold import TSNE

    # Generate synthetic feature data
    np.random.seed(42)
    features = np.random.rand(n_clients, 10)  # 10-dimensional features

    # Add some cluster structure to features
    for i in range(n_clients):
        cluster = stable_clusters[-1][i]  # Use final cluster assignment
        # Add cluster-specific bias to features
        features[i] += np.array([cluster * 0.5] * 10)

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=10, n_iter=1000, random_state=42)
    features_2d = tsne.fit_transform(features)

    # Create plot
    plt.figure(figsize=(10, 8))

    # Color by cluster assignment
    for cluster_idx in range(n_clusters):
        cluster_mask = stable_clusters[-1] == cluster_idx
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

    # Save the plot
    tsne_path = os.path.join(save_dir, 'tsne_visualization_debug.png')
    plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Debug t-SNE visualization saved to {tsne_path}")
              
def main():
    """Main function to run experiments"""
    args = parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    plots_dir = os.path.join(args.results_dir, 'plots')
    if args.save_plots:
        os.makedirs(plots_dir, exist_ok=True)
    
    # Initialize experiment runner
    runner = ExperimentRunner(results_dir=args.results_dir)
    
    # Set experiment parameters
    params = {
        'num_clients': args.num_clients,
        'num_rounds': args.num_rounds,
        'local_epochs': args.local_epochs,
        'batch_size': args.batch_size,
        'learning_rate': 0.001,  # Fixed learning rate
        'client_fraction': args.client_fraction,
        'num_clusters': args.num_clusters,
        'dedup_threshold': args.dedup_threshold,
        'fedprox_mu': args.fedprox_mu,
        'non_iid': args.non_iid,
        'use_blockchain': args.use_blockchain,
        'quantization_bits': args.quantization_bits,
        'dynamic_lr':args.dynamic_lr,
        'lr_schedule':args.lr_schedule
    }
    
    # Run specified experiment
    if args.experiment == 'single' or args.experiment == 'all':
        print(f"\n=== Running Single Experiment: {args.dataset} + {args.method} ===\n")
        single_results = runner.run_single_experiment(args.dataset, args.method, params)
        
    if args.experiment == 'comparative' or args.experiment == 'all':
        print("\n=== Running Comparative Experiments ===\n")
        
        # For faster experiments, limit to a subset of datasets and methods
        #datasets = ['fashion_mnist', 'cifar10']
        datasets = ['cifar10']
        methods = ['cflag-bd','fedprox', 'ifca', 'layercfl', 'cqfl']
        #methods = ['fedavg', 'cflag-bd','fedprox', 'ifca', 'layercfl']
        
        comparative_results = runner.run_comparative_experiments(datasets, methods, params)
        
        # Create comparison table
        print("\n=== Performance Comparison ===\n")
        comparison_df, improvement_df = runner.create_comparison_table(comparative_results)
        print("Performance Table:")
        print(comparison_df)
        print("\nImprovement Table:")
        print(improvement_df)
        
        # Save tables to CSV
        comparison_df.to_csv(os.path.join(args.results_dir, 'comparison_table.csv'), index=False)
        improvement_df.to_csv(os.path.join(args.results_dir, 'improvement_table.csv'), index=False)
        
        # Create comparative plots
        if args.save_plots:
            print("\nCreating comparative plots...")
            _ = runner.plot_comparative_results(comparative_results, plots_dir)
        
        # After running experiments and before creating plots
    if args.save_plots:
        print("\nGenerating clustering visualizations...")
        clustering_viz_dir = os.path.join(plots_dir, 'clustering')
        os.makedirs(clustering_viz_dir, exist_ok=True)
        
        # Direct access to method instances during execution
        # Store references to actual method instances
        clustering_method_instances = {}
        
        # When creating a method in run_comparative_experiments, store it
        for dataset in datasets:
            for method in methods:
                if method.lower() in ['cflag-bd', 'ifca', 'layercfl', 'cqfl']:
                    try:
                        # Instead of just running experiments, capture the method instances
                        print(f"Creating {method} for {dataset}...")
                        global_model = get_model_for_dataset(dataset, device)
                        clients = runner.create_clients(
                            client_datasets, device, batch_size=params['batch_size'], learning_rate=params['learning_rate']
                        )
                        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
                        
                        # Create the method instance
                        method_instance = runner.create_method(method, global_model, clients, test_loader, device, params)
                        
                        # Run a dummy round to initialize clustering
                        print(f"Running initialization round for {method}...")
                        method_instance.run_round(client_fraction=params['client_fraction'], epochs=1)
                        
                        # Store the instance
                        if dataset not in clustering_method_instances:
                            clustering_method_instances[dataset] = []
                        clustering_method_instances[dataset].append((method, method_instance))
                        print(f"Successfully captured {method} instance for {dataset}")
                    except Exception as e:
                        print(f"Error creating {method} for {dataset}: {e}")
        
        # Now generate visualizations using these instances
        for dataset, method_instances in clustering_method_instances.items():
            print(f"Generating visualizations for {dataset}...")
            dataset_dir = os.path.join(clustering_viz_dir, dataset)
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Find CFLAG-BD instance
            cflag_bd_instance = None
            for method_name, instance in method_instances:
                if method_name.lower() == 'cflag-bd':
                    cflag_bd_instance = instance
                    break
            
            if cflag_bd_instance:
                # Create standalone visualization for CFLAG-BD
                print(f"Creating visualizations for CFLAG-BD...")
                try:
                    # Cluster assignments heatmap
                    if hasattr(cflag_bd_instance, 'visualize_cluster_assignments'):
                        viz_path = os.path.join(dataset_dir, 'cflag_bd_clusters.png')
                        cflag_bd_instance.visualize_cluster_assignments(save_path=viz_path)
                        print(f"Created cluster heatmap at {viz_path}")
                    
                    # t-SNE visualization if available
                    if hasattr(cflag_bd_instance, 'visualize_client_clusters_tsne'):
                        tsne_path = os.path.join(dataset_dir, 'cflag_bd_tsne.png')
                        cflag_bd_instance.visualize_client_clusters_tsne(save_path=tsne_path)
                        print(f"Created t-SNE visualization at {tsne_path}")
                    
                    # Create comparisons with other methods
                    for other_method, other_instance in method_instances:
                        if other_method.lower() != 'cflag-bd':
                            compare_path = os.path.join(dataset_dir, f'compare_{other_method}_vs_cflagbd.png')
                            if hasattr(other_instance, 'cluster_history'):
                                cflag_bd_instance.visualize_cluster_assignments(
                                    save_path=compare_path, compare_with=other_instance
                                )
                                print(f"Created comparison with {other_method} at {compare_path}")
                except Exception as e:
                    print(f"Error creating CFLAG-BD visualizations: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"No CFLAG-BD instance found for {dataset}")
        
        # Add convergence rate analysis
        print("\n=== Analyzing Convergence Rates ===\n")
        convergence_df, convergence_data = runner.analyze_convergence_rates(comparative_results)
        
        print("Convergence Rate Comparison:")
        print(convergence_df)
        
        # Save convergence data and table
        convergence_df.to_csv(os.path.join(args.results_dir, 'convergence_comparison.csv'), index=False)
        runner.export_convergence_data(convergence_data, os.path.join(args.results_dir, 'convergence_data.json'))
        
        # Create convergence plots
        if args.save_plots:
            print("\nCreating convergence rate plots...")
            convergence_plots_dir = os.path.join(plots_dir, 'convergence')
            os.makedirs(convergence_plots_dir, exist_ok=True)
            _ = runner.plot_convergence_rates(convergence_data, save_path=convergence_plots_dir)
        
        if args.save_plots:
            print("\nGenerating debug visualizations...")
            generate_debug_visualization(clustering_viz_dir)
        
    if args.experiment == 'ablation' or args.experiment == 'all':
        print("\n=== Running Ablation Study ===\n")
        ablation_results = runner.run_ablation_study(params, args.dataset)
        
        if args.save_plots:
            print("\nCreating ablation plots...")
            _ = runner.plot_ablation_results(ablation_results, plots_dir)
        
    if args.experiment == 'scalability' or args.experiment == 'all':
        print("\n=== Running Scalability Study ===\n")
        scalability_results = runner.run_scalability_study('cflag-bd', args.dataset, params)
        
        if args.save_plots:
            print("\nCreating scalability plots...")
            _ = runner.plot_scalability_results(scalability_results, plots_dir)
    
    print(f"\nAll experiments completed. Results saved to {args.results_dir}")
    if args.save_plots:
        print(f"Plots saved to {plots_dir}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")