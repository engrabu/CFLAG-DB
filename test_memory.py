import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
import numpy as np
import os
import time
import gc
import copy

# Import memory-efficient implementations
from memory_efficient_cflagbd import MemoryEfficientCFLAGBD, memory_stats
from lazy_dataset import create_memory_efficient_clients, LazyDatasetPartition
from model_architectures import get_model_for_dataset
from enhanced_client import EnhancedClient

# Ensure output directories exist
os.makedirs('../data', exist_ok=True)
os.makedirs('./results', exist_ok=True)
os.makedirs('./client_storage', exist_ok=True)

# Configure experiment
NUM_CLIENTS = 10000
CLIENT_FRACTION = 0.1  # Only use 10% of clients per round to save memory
NUM_ROUNDS = 10
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
NUM_CLUSTERS = 5
OFFLOAD_TO_DISK = True
CLIENT_BATCH_SIZE = 50  # Process clients in batches of 50

print(f"Running large-scale CFLAG-BD with {NUM_CLIENTS} clients")
print("Initial memory usage:")
memory_stats()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load dataset once
print("Loading dataset...")
train_dataset = FashionMNIST(
    root='../data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = FashionMNIST(
    root='../data',
    train=False,
    download=True,
    transform=transformgit
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False
)

print("Creating clients...")
# Create memory-efficient clients with lazy dataset partitions
clients = create_memory_efficient_clients(
    base_dataset=train_dataset,
    num_clients=NUM_CLIENTS,
    device=device,
    client_class=EnhancedClient,
    batch_size=BATCH_SIZE,
    learning_rate=0.01,
    non_iid=True
)

print(f"Successfully created {len(clients)} clients")
print("Current memory usage:")
memory_stats()

# Create global model
global_model = get_model_for_dataset('fashion_mnist', device)

# Run memory-efficient CFLAG-BD
print("\nInitializing Memory-Efficient CFLAG-BD...")
cflag_bd = MemoryEfficientCFLAGBD(
    global_model=global_model,
    clients=clients,
    test_loader=test_loader,
    device=device,
    num_clusters=NUM_CLUSTERS,
    dedup_threshold=0.01,
    use_blockchain=False,  # Disable blockchain to save memory
    batch_size=CLIENT_BATCH_SIZE,
    offload_to_disk=OFFLOAD_TO_DISK,
    storage_path="./client_storage"
)

# Free some memory before starting simulation
clients = None
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("\nStarting federated learning simulation...")
start_time = time.time()
results = cflag_bd.run_simulation(
    num_rounds=NUM_ROUNDS,
    client_fraction=CLIENT_FRACTION,
    epochs=LOCAL_EPOCHS
)
total_time = time.time() - start_time

print("\nSimulation complete!")
print(f"Total time: {total_time:.2f} seconds")
print(f"Final accuracy: {results['final_accuracy']:.4f}")
print(f"Accuracy history: {[f'{acc:.4f}' for acc in results['accuracy_history']]}")
print(f"Total training time: {results['training_time']:.2f} seconds")
print(f"Total communication cost: {results['communication_cost'] / (1024 * 1024):.2f} MB")

# Save results
import json
with open('./results/large_scale_results.json', 'w') as f:
    # Convert numpy values to Python native types
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        else:
            return obj
            
    serializable_results = convert_to_serializable(results)
    json.dump(serializable_results, f, indent=2)

print("Results saved to ./results/large_scale_results.json")
print("Final memory usage:")
memory_stats()