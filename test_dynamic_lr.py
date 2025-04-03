import torch
import time
import os
from experiment_runner import ExperimentRunner
from learning_rate_scheduler import integrate_dynamic_lr_with_cflagbd
from model_architectures import get_model_for_dataset

# Make sure output directory exists
os.makedirs('./results', exist_ok=True)

# Initialize experiment runner
runner = ExperimentRunner(results_dir='./results')

# Set experiment parameters
params = {
    'num_clients': 10,
    'num_rounds': 10,
    'local_epochs': 5,
    'batch_size': 32,
    'learning_rate': 0.01,  # Starting with higher learning rate to better see the effects
    'client_fraction': 0.2,
    'num_clusters': 3,
    'dedup_threshold': 0.01,
    'non_iid': False,
    'use_blockchain': False
}

print("\n=== Testing Learning Rate Schedules with CFLAG-BD ===\n")

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Setup data - do this once to ensure fair comparison
print("\nSetting up dataset...")
client_datasets, test_loader = runner.setup_data('fashion_mnist', params['num_clients'], params['non_iid'])
clients = runner.create_clients(client_datasets, device, params['batch_size'], params['learning_rate'])

# Test different learning rate schedules
for schedule in ['step', 'exponential', 'cosine', 'adaptive', 'cyclic']:
    print(f"\n\n{'='*50}")
    print(f"=== Testing {schedule.upper()} Learning Rate Schedule ===")
    print(f"{'='*50}\n")
    
    # Create new model and method
    global_model = get_model_for_dataset('fashion_mnist', device)
    
    # Create new clients with fresh optimizers
    fresh_clients = runner.create_clients(client_datasets, device, params['batch_size'], params['learning_rate'])
    
    # Create new CFLAG-BD instance
    method = runner.create_method('cflag-bd', global_model, fresh_clients, test_loader, device, params)
    
    # Apply dynamic learning rate
    method = integrate_dynamic_lr_with_cflagbd(method, schedule_type=schedule, verbose=True)
    
    # Run simulation
    print(f"\nRunning simulation with {schedule} learning rate schedule...")
    start_time = time.time()
    results = method.run_simulation(
        num_rounds=params['num_rounds'],
        client_fraction=params['client_fraction'],
        epochs=params['local_epochs']
    )
    run_time = time.time() - start_time
    
    # Show results
    print(f"\nResults for {schedule.upper()} schedule:")
    print(f"Final accuracy: {results['final_accuracy']:.4f}")
    print(f"Training time: {results['training_time']:.2f}s")
    print(f"Wall clock time: {run_time:.2f}s")
    print(f"Accuracy history: {[f'{acc:.4f}' for acc in results['accuracy_history']]}")
    
    # Try to plot learning rate history
    try:
        plot_file = f"./results/lr_history_{schedule}.png"
        method.plot_learning_rate_history(save_path=plot_file)
    except Exception as e:
        print(f"Could not plot learning rate history: {e}")

print("\n=== All tests complete ===")