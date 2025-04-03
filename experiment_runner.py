import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import os
import json
from collections import defaultdict
from torch.utils.data import DataLoader
from scipy import stats
import copy

# Import our implementations
from model_architectures import get_model_for_dataset
from enhanced_client import EnhancedClient
from baseline_algorithms import FedAvg, FedProx, IFCA, LayerCFL, CQFL
from cflag_bd_algorithm import CFLAGBD
from multi_dataset_support import DatasetManager
from learning_rate_scheduler import integrate_dynamic_lr_with_cflagbd


class ExperimentRunner:
    """
    Comprehensive experiment runner for federated learning algorithms
    """

    def __init__(self, results_dir='./results'):
        """
        Initialize experiment runner

        Args:
            results_dir: Directory to save results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        # Available datasets
        self.available_datasets = [
            'fashion_mnist',
            'cifar10',
            'har',
            'iot',
            'chestxray'
        ]

        # Available methods
        self.available_methods = [
            'fedavg',
            'fedprox',
            'ifca',
            'layercfl',
            'cqfl',
            'cflag-bd'
        ]

        # Default hyperparameters
        self.default_params = {
            'num_clients': 100,
            'num_rounds': 50,
            'local_epochs': 5,
            'batch_size': 32,
            'learning_rate': 0.01,
            'client_fraction': 0.2,
            'num_clusters': 3,
            'dedup_threshold': 0.01,
            'fedprox_mu': 0.01,
            'non_iid': True,
            'use_blockchain': True,
            'quantization_bits': 8
        }

        # Track all experiment results
        self.experiment_results = {}

    def setup_data(self, dataset_name, num_clients=100, non_iid=True):
        """
        Setup dataset and clients

        Args:
            dataset_name: Name of dataset to use
            num_clients: Number of clients to create
            non_iid: Whether to distribute data in non-IID fashion

        Returns:
            client_datasets: List of client datasets
            test_loader: DataLoader for test dataset
        """
        print(
            f"Setting up {dataset_name} dataset for {num_clients} clients (non_iid={non_iid})...")

        # Initialize dataset manager
        data_manager = DatasetManager(
            data_root='../data', num_clients=num_clients, iid=not non_iid)

        # Load and partition dataset
        client_datasets, test_dataset = data_manager.create_client_datasets(
            dataset_name)

        # Create test loader
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        return client_datasets, test_loader

    def create_clients(self, client_datasets, device, batch_size=32, learning_rate=0.01):
        """
        Create client objects

        Args:
            client_datasets: List of client datasets
            device: Device to run on
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer

        Returns:
            clients: List of client objects
        """
        clients = []

        for i, dataset in enumerate(client_datasets):
            # Randomize processing power and reliability
            processing_power = np.random.uniform(0.5, 2.0)
            reliability = np.random.uniform(0.8, 1.0)

            client = EnhancedClient(
                client_id=i,
                dataset=dataset,
                device=device,
                batch_size=batch_size,
                learning_rate=learning_rate,
                processing_power=processing_power,
                reliability=reliability
            )

            clients.append(client)

        return clients

    def create_method(self, method_name, global_model, clients, test_loader, device, params):
        """Create federated learning method

                Args:
                        method_name: Name of method to create
                        global_model: Global model
                        clients: List of client objects
                        test_loader: DataLoader for test dataset
                        device: Device to run on
                        params: Dictionary of method parameters

                Returns:
                        method: Federated learning method object
                """
        if method_name.lower() == 'fedavg':
            return FedAvg(global_model, clients, test_loader, device)
        elif method_name.lower() == 'fedprox':
            return FedProx(global_model, clients, test_loader, device, mu=params.get('fedprox_mu', 0.01))
        elif method_name.lower() == 'ifca':
            return IFCA(global_model, clients, test_loader, device, num_clusters=params.get('num_clusters', 3))
        elif method_name.lower() == 'layercfl':
            return LayerCFL(global_model, clients, test_loader, device, num_clusters=params.get('num_clusters', 3))
        elif method_name.lower() == 'cqfl':
            return CQFL(global_model, clients, test_loader, device,
                        num_clusters=params.get('num_clusters', 3),
                        quantization_bits=params.get('quantization_bits', 8))
        elif method_name.lower() == 'cflag-bd':
            method = CFLAGBD(global_model, clients, test_loader, device,
                            num_clusters=params.get('num_clusters', 3),
                            dedup_threshold=params.get('dedup_threshold', 0.01),
                            use_blockchain=params.get('use_blockchain', True))
            
            # Add this: Integrate dynamic learning rate if parameter is set
            if params.get('dynamic_lr', False):
                method = integrate_dynamic_lr_with_cflagbd(
                    method, 
                    schedule_type=params.get('lr_schedule', 'adaptive')
                )
                
            return method
        else:
            raise ValueError(f"Unknown method: {method_name}")

    def run_single_experiment(self, dataset_name, method_name, params=None):
        """Run a single experiment with specified parameters
        Args:
            dataset_name: Name of dataset to use
            method_name: Name of method to use
            params: Dictionary of parameters (if None, use defaults)

        Returns:
            results: Dictionary of experiment results
        """
        if params is None:
            params = self.default_params.copy()

        print(f"Running experiment: {dataset_name} + {method_name}")
        print(f"Parameters: {params}")

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Setup data
        client_datasets, test_loader = self.setup_data(
            dataset_name,
            num_clients=params['num_clients'],
            non_iid=params['non_iid']
        )

        # Create clients
        clients = self.create_clients(
            client_datasets,
            device,
            batch_size=params['batch_size'],
            learning_rate=params['learning_rate']
        )

        # Create global model
        global_model = get_model_for_dataset(dataset_name, device)

        # Create method
        method = self.create_method(
            method_name,
            global_model,
            clients,
            test_loader,
            device,
            params
        )

        # Run simulation
        start_time = time.time()
        results = method.run_simulation(
            num_rounds=params['num_rounds'],
            client_fraction=params['client_fraction'],
            epochs=params['local_epochs']
        )
        total_time = time.time() - start_time

        # Add experiment metadata
        results['dataset'] = dataset_name
        results['method'] = method_name
        results['params'] = params
        results['total_wall_time'] = total_time

        # Serialize results safely
        def to_serializable(obj):
            if isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_serializable(v) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(to_serializable(v) for v in obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        json_results = to_serializable(results)

        # Save results
        experiment_id = f"{dataset_name}_{method_name}_{int(time.time())}"
        results_path = os.path.join(self.results_dir, f"{experiment_id}.json")

        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        # Store results in memory
        self.experiment_results[experiment_id] = results

        print(f"Experiment completed. Results saved to {results_path}")
        print(f"Final accuracy: {results['final_accuracy']:.4f}")

        return results

    def run_comparative_experiments(self, datasets=None, methods=None, params=None):
        """
        Run comparative experiments across datasets and methods

        Args:
            datasets: List of datasets to use (if None, use all)
            methods: List of methods to use (if None, use all)
            params: Dictionary of parameters (if None, use defaults)

        Returns:
            comparative_results: Dictionary mapping (dataset, method) to results
        """
        if datasets is None:
            datasets = self.available_datasets

        if methods is None:
            methods = self.available_methods

        if params is None:
            params = self.default_params.copy()

        print(f"Running comparative experiments:")
        print(f"Datasets: {datasets}")
        print(f"Methods: {methods}")
        print(f"Parameters: {params}")

        comparative_results = {}

        for dataset in datasets:
            for method in methods:
                try:
                    results = self.run_single_experiment(
                        dataset, method, params)
                    comparative_results[(dataset, method)] = results
                except Exception as e:
                    print(f"Error running {dataset} + {method}: {e}")

        return comparative_results

    def run_ablation_study(self, params=None, dataset='fashion_mnist'):
        """
        Run ablation study on CFLAG-BD parameters

        Args:
            params: Base parameters (if None, use defaults)
            dataset: Dataset to use for ablation

        Returns:
            ablation_results: Dictionary of ablation results
        """
        if params is None:
            params = self.default_params.copy()

        print(f"Running ablation study on {dataset}")

        # Parameters to ablate
        ablation_params = {
            'num_clusters': [2, 3, 5, 7],
            'dedup_threshold': [0.001, 0.01, 0.05, 0.1],
            'client_fraction': [0.1, 0.2, 0.5, 1.0],
            'local_epochs': [1, 3, 5, 10]
        }

        ablation_results = {}

        # Run base experiment with default parameters
        base_params = params.copy()
        base_results = self.run_single_experiment(
            dataset, 'cflag-bd', base_params)
        ablation_results['base'] = base_results

        # Run ablation for each parameter
        for param_name, param_values in ablation_params.items():
            param_results = {}

            for value in param_values:
                if value == params.get(param_name):
                    # Skip if this is the default value (already in base results)
                    param_results[value] = base_results
                    continue

                # Update parameter
                test_params = params.copy()
                test_params[param_name] = value

                # Run experiment
                results = self.run_single_experiment(
                    dataset, 'cflag-bd', test_params)
                param_results[value] = results

            ablation_results[param_name] = param_results

        return ablation_results

    def run_scalability_study(self, method='cflag-bd', dataset='fashion_mnist', params=None):
        """
        Run scalability study with increasing client numbers

        Args:
            method: Method to test scalability
            dataset: Dataset to use
            params: Base parameters (if None, use defaults)

        Returns:
            scalability_results: Dictionary of scalability results
        """
        if params is None:
            params = self.default_params.copy()

        print(f"Running scalability study for {method} on {dataset}")

        # Client counts to test
        client_counts = [10, 50, 100, 200, 500, 1000,5000,10000]

        scalability_results = {}

        for num_clients in client_counts:
            # Update client count
            test_params = params.copy()
            test_params['num_clients'] = num_clients

            # Adjust client fraction to keep absolute client count manageable
            if num_clients > 200:
                test_params['client_fraction'] = min(
                    100 / num_clients, test_params['client_fraction'])

            # Run experiment
            results = self.run_single_experiment(dataset, method, test_params)
            scalability_results[num_clients] = results

        return scalability_results

    def create_comparison_table(self, comparative_results=None):
        """
        Create comparison table from experiment results

        Args:
            comparative_results: Dictionary of comparative results (if None, use stored results)

        Returns:
            df: Pandas DataFrame with comparison results
        """
        if comparative_results is None:
            if not self.experiment_results:
                raise ValueError("No experiment results available")
            comparative_results = self.experiment_results

        # Create data for table
        rows = []

        for key, results in comparative_results.items():
            if isinstance(key, tuple) and len(key) == 2:
                dataset, method = key
            else:
                # Handle the case when the key is an experiment_id string
                dataset = results.get('dataset', 'unknown')
                method = results.get('method', 'unknown')

            try:
                row = {
                    'Dataset': dataset,
                    'Method': method,
                    'Final Accuracy (%)': results.get('final_accuracy', 0) * 100,
                    'Training Time (s)': results.get('training_time', 0),
                    'Communication Cost (MB)': results.get('communication_cost', 0) / (1024 * 1024),
                    'Total Wall Time (s)': results.get('total_wall_time', 0)
                }

                # Add storage savings for CFLAG-BD
                if method.lower() == 'cflag-bd':
                    row['Storage Savings (%)'] = results.get('storage_savings', 0)

                rows.append(row)
            except Exception as e:
                print(f"Warning: Could not process result for {dataset}/{method}: {e}")

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Check if DataFrame is empty
        if df.empty:
            print("Warning: No valid experiment results to compare.")
            return df, pd.DataFrame()  # Return empty DataFrames

        # Calculate improvements relative to FedAvg
        improvements = {}

        for dataset in df['Dataset'].unique():
            # Get FedAvg row for this dataset
            fedavg_row = df[(df['Dataset'] == dataset) & (
                df['Method'].str.lower() == 'fedavg')]

            if fedavg_row.empty:
                continue

            fedavg_accuracy = fedavg_row['Final Accuracy (%)'].values[0]
            fedavg_comm_cost = fedavg_row['Communication Cost (MB)'].values[0]
            fedavg_time = fedavg_row['Training Time (s)'].values[0]

            # Calculate improvements for other methods
            dataset_rows = df[df['Dataset'] == dataset]

            for _, row in dataset_rows.iterrows():
                method = row['Method']

                if method.lower() == 'fedavg':
                    continue

                accuracy_imp = row['Final Accuracy (%)'] - fedavg_accuracy
                comm_imp = (
                    1 - row['Communication Cost (MB)'] / fedavg_comm_cost) * 100 if fedavg_comm_cost > 0 else 0
                time_imp = (1 - row['Training Time (s)'] / fedavg_time) * 100 if fedavg_time > 0 else 0

                improvements[(dataset, method)] = {
                    'Accuracy Improvement (pp)': accuracy_imp,
                    'Communication Reduction (%)': comm_imp,
                    'Training Time Reduction (%)': time_imp
                }

        # Add improvements to DataFrame
        improvement_rows = []

        for key, imps in improvements.items():
            dataset, method = key
            row = {'Dataset': dataset, 'Method': method}
            row.update(imps)
            improvement_rows.append(row)

        improvement_df = pd.DataFrame(improvement_rows)

        return df, improvement_df
    def plot_comparative_results(self, comparative_results=None, save_path=None):
        """
        Create comparative plots from experiment results

        Args:
            comparative_results: Dictionary of comparative results (if None, use stored results)
            save_path: Path to save plots (if None, don't save)

        Returns:
            figs: Dictionary of figure objects
        """
        if comparative_results is None:
            if not self.experiment_results:
                raise ValueError("No experiment results available")
            comparative_results = self.experiment_results

        # Create figure dictionary
        figs = {}

        # Get unique datasets and methods
        datasets = set()
        methods = set()

        for key in comparative_results.keys():
            if isinstance(key, tuple) and len(key) == 2:
                dataset, method = key
                datasets.add(dataset)
                methods.add(method)

        # Group results by dataset
        dataset_results = defaultdict(dict)

        for key, results in comparative_results.items():
            if isinstance(key, tuple) and len(key) == 2:
                dataset, method = key
                dataset_results[dataset][method] = results
            else:
                # Handle the case when the key is an experiment_id string
                dataset = results['dataset']
                method = results['method']
                dataset_results[dataset][method] = results

        # Plot 1: Accuracy over rounds for each dataset
        for dataset, methods_dict in dataset_results.items():
            fig, ax = plt.subplots(figsize=(10, 6))

            for method, results in methods_dict.items():
                if 'accuracy_history' in results:
                    accuracy_history = results['accuracy_history']
                    rounds = list(range(1, len(accuracy_history) + 1))
                    ax.plot(
                        rounds, [acc * 100 for acc in accuracy_history], label=method)

            ax.set_title(f'Accuracy Over Rounds - {dataset}')
            ax.set_xlabel('Communication Round')
            ax.set_ylabel('Accuracy (%)')
            ax.grid(True)
            ax.legend()

            figs[f'accuracy_{dataset}'] = fig

            if save_path:
                fig.savefig(os.path.join(save_path, f'accuracy_{dataset}.png'))

        # Plot 2: Bar chart of final accuracies
        fig, ax = plt.subplots(figsize=(12, 8))

        # Prepare data for grouped bar chart
        bar_data = defaultdict(list)
        bar_methods = list(methods)
        bar_datasets = list(datasets)

        for dataset in bar_datasets:
            for method in bar_methods:
                if method in dataset_results[dataset]:
                    results = dataset_results[dataset][method]
                    bar_data[method].append(results['final_accuracy'] * 100)
                else:
                    bar_data[method].append(0)

        # Create bar positions
        bar_width = 0.8 / len(bar_methods)
        r = np.arange(len(bar_datasets))

        # Plot bars
        for i, method in enumerate(bar_methods):
            ax.bar(r + i * bar_width, bar_data[method], width=bar_width, label=method,
                   yerr=0.5)  # Add small error bars for visibility

        # Add labels and legend
        ax.set_title('Final Accuracy Comparison')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Accuracy (%)')
        ax.set_xticks(r + bar_width * (len(bar_methods) - 1) / 2)
        ax.set_xticklabels(bar_datasets)
        ax.legend()

        figs['final_accuracy'] = fig

        if save_path:
            fig.savefig(os.path.join(save_path, 'final_accuracy.png'))

        # Plot 3: Communication cost comparison
        fig, ax = plt.subplots(figsize=(12, 8))

        # Prepare data
        comm_data = defaultdict(list)

        for dataset in bar_datasets:
            for method in bar_methods:
                if method in dataset_results[dataset]:
                    results = dataset_results[dataset][method]
                    # Convert to MB
                    comm_data[method].append(
                        results['communication_cost'] / (1024 * 1024))
                else:
                    comm_data[method].append(0)

        # Plot bars
        for i, method in enumerate(bar_methods):
            ax.bar(r + i * bar_width,
                   comm_data[method], width=bar_width, label=method)

        # Add labels and legend
        ax.set_title('Communication Cost Comparison')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Communication Cost (MB)')
        ax.set_xticks(r + bar_width * (len(bar_methods) - 1) / 2)
        ax.set_xticklabels(bar_datasets)
        ax.legend()

        figs['communication_cost'] = fig

        if save_path:
            fig.savefig(os.path.join(save_path, 'communication_cost.png'))

        # Plot 4: Training time comparison
        fig, ax = plt.subplots(figsize=(12, 8))

        # Prepare data
        time_data = defaultdict(list)

        for dataset in bar_datasets:
            for method in bar_methods:
                if method in dataset_results[dataset]:
                    results = dataset_results[dataset][method]
                    time_data[method].append(results['training_time'])
                else:
                    time_data[method].append(0)

        # Plot bars
        for i, method in enumerate(bar_methods):
            ax.bar(r + i * bar_width,
                   time_data[method], width=bar_width, label=method)

        # Add labels and legend
        ax.set_title('Training Time Comparison')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Training Time (s)')
        ax.set_xticks(r + bar_width * (len(bar_methods) - 1) / 2)
        ax.set_xticklabels(bar_datasets)
        ax.legend()

        figs['training_time'] = fig

        if save_path:
            fig.savefig(os.path.join(save_path, 'training_time.png'))

        # Plot 5: Improvement over FedAvg
        if 'fedavg' in [m.lower() for m in methods]:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Prepare data
            methods_without_fedavg = [
                m for m in bar_methods if m.lower() != 'fedavg']
            datasets_with_fedavg = []
            improvements = []

            for dataset in bar_datasets:
                if 'fedavg' in dataset_results[dataset]:
                    datasets_with_fedavg.append(dataset)
                    fedavg_acc = dataset_results[dataset]['fedavg']['final_accuracy'] * 100
                    dataset_improvements = []

                    for method in methods_without_fedavg:
                        if method in dataset_results[dataset]:
                            acc = dataset_results[dataset][method]['final_accuracy'] * 100
                            dataset_improvements.append(acc - fedavg_acc)
                        else:
                            dataset_improvements.append(0)

                    improvements.append(dataset_improvements)

            # Convert to numpy array
            improvements = np.array(improvements)

            # Create heatmap
            im = ax.imshow(improvements, cmap='RdYlGn')

            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Accuracy Improvement (pp)',
                               rotation=-90, va="bottom")

            # Add labels
            ax.set_xticks(np.arange(len(methods_without_fedavg)))
            ax.set_yticks(np.arange(len(datasets_with_fedavg)))
            ax.set_xticklabels(methods_without_fedavg)
            ax.set_yticklabels(datasets_with_fedavg)

            # Rotate x tick labels
            plt.setp(ax.get_xticklabels(), rotation=45,
                     ha="right", rotation_mode="anchor")

            # Add title
            ax.set_title('Accuracy Improvement over FedAvg')

            # Add values to cells
            for i in range(len(datasets_with_fedavg)):
                for j in range(len(methods_without_fedavg)):
                    ax.text(j, i, f"{improvements[i, j]:.2f}",
                            ha="center", va="center", color="black" if improvements[i, j] < 5 else "white")

            figs['improvement_heatmap'] = fig

            if save_path:
                fig.savefig(os.path.join(save_path, 'improvement_heatmap.png'))

        # Plot 6: Statistical significance
        if 'cflag-bd' in [m.lower() for m in methods]:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Prepare data
            p_values = []
            effect_sizes = []
            comparison_methods = []

            for method in methods:
                if method.lower() != 'cflag-bd':
                    comparison_methods.append(method)
                    method_p_values = []
                    method_effect_sizes = []

                    for dataset in datasets:
                        if 'cflag-bd' in dataset_results[dataset] and method in dataset_results[dataset]:
                            cflag_results = dataset_results[dataset]['cflag-bd']
                            other_results = dataset_results[dataset][method]

                            # Get round-by-round accuracy
                            if 'accuracy_history' in cflag_results and 'accuracy_history' in other_results:
                                cflag_acc = cflag_results['accuracy_history']
                                other_acc = other_results['accuracy_history']

                                # Limit to minimum length of both
                                min_len = min(len(cflag_acc), len(other_acc))
                                cflag_acc = cflag_acc[:min_len]
                                other_acc = other_acc[:min_len]

                                # Perform paired t-test
                                t_stat, p_val = stats.ttest_rel(
                                    cflag_acc, other_acc)

                                # Calculate effect size (Cohen's d)
                                d = (np.mean(cflag_acc) - np.mean(other_acc)) / \
                                     np.std(np.array(cflag_acc) - np.array(other_acc))

                                method_p_values.append(p_val)
                                method_effect_sizes.append(abs(d))
                            else:
                                method_p_values.append(1.0)
                                method_effect_sizes.append(0.0)
                        else:
                            method_p_values.append(1.0)
                            method_effect_sizes.append(0.0)

                    # Average p-value and effect size across datasets
                    avg_p = np.mean(method_p_values)
                    avg_effect = np.mean(method_effect_sizes)

                    p_values.append(avg_p)
                    effect_sizes.append(avg_effect)

            # Create scatter plot
            scatter = ax.scatter(
                effect_sizes, [-np.log10(p) for p in p_values], s=100)

            # Add labels for each point
            for i, method in enumerate(comparison_methods):
                ax.annotate(method, (effect_sizes[i], -np.log10(p_values[i])),
                            xytext=(5, 5), textcoords='offset points')

            # Add reference lines
            ax.axhline(y=-np.log10(0.05), color='r',
                       linestyle='--', label='p=0.05')
            ax.axhline(y=-np.log10(0.01), color='g',
                       linestyle='--', label='p=0.01')
            ax.axvline(x=0.2, color='gray', linestyle='--',
                       label='Small Effect')
            ax.axvline(x=0.5, color='gray', linestyle='-',
                       label='Medium Effect')
            ax.axvline(x=0.8, color='gray', linestyle='-.',
                       label='Large Effect')

            # Add labels and legend
            ax.set_title('Statistical Significance of CFLAG-BD Improvements')
            ax.set_xlabel('Effect Size (abs Cohen\'s d)')
            ax.set_ylabel('-log10(p-value)')
            ax.legend()

            figs['statistical_significance'] = fig

            if save_path:
                fig.savefig(os.path.join(
                    save_path, 'statistical_significance.png'))

        return figs

    def plot_ablation_results(self, ablation_results, save_path=None):
        """
        Plot ablation study results

        Args:
            ablation_results: Dictionary of ablation results
            save_path: Path to save plots (if None, don't save)

        Returns:
            figs: Dictionary of figure objects
        """
        figs = {}

        # Plot for each ablated parameter
        for param_name, param_results in ablation_results.items():
            if param_name == 'base':
                continue

            fig, ax = plt.subplots(figsize=(10, 6))

            # Prepare data
            param_values = sorted(param_results.keys())
            accuracies = [param_results[v]['final_accuracy']
                * 100 for v in param_values]

            # Create line plot
            ax.plot(param_values, accuracies, 'o-')

            # Add labels
            ax.set_title(f'Ablation Study: {param_name}')
            ax.set_xlabel(param_name)
            ax.set_ylabel('Accuracy (%)')
            ax.grid(True)

            # Add value annotations
            for i, v in enumerate(param_values):
                ax.annotate(f"{accuracies[i]:.2f}%", (param_values[i], accuracies[i]),
                            xytext=(0, 10), textcoords='offset points', ha='center')

            figs[f'ablation_{param_name}'] = fig

            if save_path:
                fig.savefig(os.path.join(
                    save_path, f'ablation_{param_name}.png'))

        # Combined ablation plot
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        axs = axs.flatten()

        # Plot each parameter in a separate subplot
        for i, (param_name, param_results) in enumerate(ablation_results.items()):
            if param_name == 'base' or i >= 4:
                continue

            ax = axs[i]

            # Prepare data
            param_values = sorted(param_results.keys())
            accuracies = [param_results[v]['final_accuracy']
                * 100 for v in param_values]

            # Create line plot
            ax.plot(param_values, accuracies, 'o-')

            # Add labels
            ax.set_title(f'Ablation: {param_name}')
            ax.set_xlabel(param_name)
            ax.set_ylabel('Accuracy (%)')
            ax.grid(True)

            # Add value annotations
            for j, v in enumerate(param_values):
                ax.annotate(f"{accuracies[j]:.2f}%", (param_values[j], accuracies[j]),
                            xytext=(0, 10), textcoords='offset points', ha='center')

        plt.tight_layout()
        figs['ablation_combined'] = fig

        if save_path:
            fig.savefig(os.path.join(save_path, 'ablation_combined.png'))

        return figs

    def plot_scalability_results(self, scalability_results, save_path=None):
        """
        Plot scalability study results

        Args:
            scalability_results: Dictionary of scalability results
            save_path: Path to save plots (if None, don't save)

        Returns:
            figs: Dictionary of figure objects
        """
        figs = {}

        # Sort client counts
        client_counts = sorted(scalability_results.keys())

        # Prepare data
        accuracies = [scalability_results[n]
            ['final_accuracy'] * 100 for n in client_counts]
        training_times = [scalability_results[n]['training_time']
            for n in client_counts]
        communication_costs = [
            scalability_results[n]['communication_cost'] / (1024 * 1024) for n in client_counts]

        # Plot 1: Accuracy vs. Client Count
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(client_counts, accuracies, 'o-')
        ax.set_title('Scalability: Accuracy vs. Client Count')
        ax.set_xlabel('Number of Clients')
        ax.set_ylabel('Accuracy (%)')
        ax.set_xscale('log')
        ax.grid(True)

        # Add value annotations
        for i, n in enumerate(client_counts):
            ax.annotate(f"{accuracies[i]:.2f}%", (n, accuracies[i]),
                        xytext=(0, 10), textcoords='offset points', ha='center')

        figs['scalability_accuracy'] = fig

        if save_path:
            fig.savefig(os.path.join(save_path, 'scalability_accuracy.png'))

        # Plot 2: Training Time vs. Client Count
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(client_counts, training_times, 'o-')
        ax.set_title('Scalability: Training Time vs. Client Count')
        ax.set_xlabel('Number of Clients')
        ax.set_ylabel('Training Time (s)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)

        figs['scalability_time'] = fig

        if save_path:
            fig.savefig(os.path.join(save_path, 'scalability_time.png'))

        # Plot 3: Communication Cost vs. Client Count
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(client_counts, communication_costs, 'o-')
        ax.set_title('Scalability: Communication Cost vs. Client Count')
        ax.set_xlabel('Number of Clients')
        ax.set_ylabel('Communication Cost (MB)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)

        figs['scalability_communication'] = fig

        if save_path:
            fig.savefig(os.path.join(
                save_path, 'scalability_communication.png'))

        # Plot 4: Combined Scalability
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        # Accuracy
        axs[0].plot(client_counts, accuracies, 'o-', color='blue')
        axs[0].set_title('Accuracy vs. Client Count')
        axs[0].set_xlabel('Number of Clients')
        axs[0].set_ylabel('Accuracy (%)')
        axs[0].set_xscale('log')
        axs[0].grid(True)

        # Training Time
        axs[1].plot(client_counts, training_times, 'o-', color='orange')
        axs[1].set_title('Training Time vs. Client Count')
        axs[1].set_xlabel('Number of Clients')
        axs[1].set_ylabel('Training Time (s)')
        axs[1].set_xscale('log')
        axs[1].set_yscale('log')
        axs[1].grid(True)

        # Communication Cost
        axs[2].plot(client_counts, communication_costs, 'o-', color='green')
        axs[2].set_title('Communication Cost vs. Client Count')
        axs[2].set_xlabel('Number of Clients')
        axs[2].set_ylabel('Communication Cost (MB)')
        axs[2].set_xscale('log')
        axs[2].set_yscale('log')
        axs[2].grid(True)

        plt.tight_layout()
        figs['scalability_combined'] = fig

        if save_path:
            fig.savefig(os.path.join(save_path, 'scalability_combined.png'))

        return figs
    
    def analyze_convergence_rates(self, comparative_results=None):
        """
        Analyze convergence rates from experiment results
        
        Args:
            comparative_results: Optional dictionary of comparative results
            
        Returns:
            convergence_df: DataFrame with convergence metrics
        """
        if comparative_results is None:
            if not self.experiment_results:
                raise ValueError("No experiment results available")
            comparative_results = self.experiment_results
            
        # Prepare data for analysis
        convergence_data = {}
        
        for key, results in comparative_results.items():
            if isinstance(key, tuple) and len(key) == 2:
                dataset, method = key
            else:
                # Handle case when key is an experiment_id string
                dataset = results['dataset']
                method = results['method']
                
            if 'accuracy_history' in results and len(results['accuracy_history']) > 1:
                accuracy_history = results['accuracy_history']
                
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
        
        # Create DataFrame for comparison
        rows = []
        
        for (dataset, method), data in convergence_data.items():
            row = {
                'Dataset': dataset,
                'Method': method,
                'Final Accuracy (%)': data['final_accuracy'] * 100,
                'Avg. Improvement/Round (pp)': data['avg_absolute_improvement'] * 100,
                'Best Round Improvement (pp)': data['best_round_improvement'] * 100,
                'Best Round': data['best_round_index'],
                'Total Rounds': data['total_rounds']
            }
            rows.append(row)
            
        convergence_df = pd.DataFrame(rows)
        
        # Sort by dataset, then by final accuracy
        convergence_df = convergence_df.sort_values(['Dataset', 'Final Accuracy (%)'], ascending=[True, False])
        
        return convergence_df, convergence_data

    def export_convergence_data(self, convergence_data, output_file='convergence_data.json'):
        """
        Export convergence data to JSON
        
        Args:
            convergence_data: Dictionary of convergence data
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

    def plot_convergence_rates(self, convergence_data, dataset_filter=None, save_path=None):
        """
        Create plots of convergence rates
        
        Args:
            convergence_data: Dictionary of convergence data
            dataset_filter: Optional filter to only plot specific dataset
            save_path: Optional path to save plots
            
        Returns:
            figs: Dictionary of figure objects
        """
        figs = {}
        
        # Filter data if needed
        if dataset_filter:
            plot_data = {k: v for k, v in convergence_data.items() if k[0] == dataset_filter}
        else:
            plot_data = convergence_data
            
        # Group by dataset
        datasets = set(k[0] for k in plot_data.keys())
        
        # Create color map for methods
        methods = sorted(set(k[1] for k in plot_data.keys()))
        cmap = plt.cm.get_cmap('tab10', len(methods))
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
                
            # Plot 3: Average convergence rate comparison
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            
            methods_list = []
            avg_improvements = []
            
            for (_, method), data in sorted(dataset_data.items(), key=lambda x: x[1]['avg_absolute_improvement'], reverse=True):
                methods_list.append(method)
                avg_improvements.append(data['avg_absolute_improvement'] * 100)  # as percentage points
                
            # Create bar chart
            bars = ax3.bar(methods_list, avg_improvements, color='skyblue')
            
            ax3.set_title(f'Average Convergence Rate - {dataset}')
            ax3.set_ylabel('Average Improvement per Round (pp)')
            ax3.grid(axis='y')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax3.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
                
            figs[f'convergence_rate_{dataset}'] = fig3
            
            if save_path:
                fig3.savefig(os.path.join(save_path, f'convergence_rate_{dataset}.png'))
                
        return figs