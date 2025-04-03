import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torch.utils.data import DataLoader, Dataset, random_split, Subset, TensorDataset
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from collections import defaultdict
import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
import zipfile
import gzip
import shutil
import urllib.request
from io import BytesIO
import h5py
import copy
import time
import json
import hashlib  # Added for blockchain hashing
from hardware_profiler import HardwareProfiler, ArchitectureProfiler

class BlockchainSimulator:
    """
    Simulates a blockchain for federated learning
    """
    def __init__(self, block_size=1024*1024, consensus_delay=0.1):
        """
        Initialize blockchain simulator
        
        Args:
            block_size: Maximum block size in bytes
            consensus_delay: Simulated delay for consensus in seconds
        """
        self.chain = []
        self.pending_transactions = []
        self.block_size = block_size  # 1MB default
        self.consensus_delay = consensus_delay
        self.transaction_history = []
        
        # Create genesis block
        self.create_genesis_block()
        
    def create_genesis_block(self):
        """Create the first block in the chain (genesis block)"""
        genesis_block = {
            'index': 0,
            'timestamp': time.time(),
            'transactions': [],
            'previous_hash': "0" * 64,
            'nonce': 0,
            'hash': self.calculate_hash(0, time.time(), [], "0" * 64, 0)
        }
        
        self.chain.append(genesis_block)
        
    def calculate_hash(self, index, timestamp, transactions, previous_hash, nonce):
        """
        Calculate hash of block
        
        Args:
            index: Block index
            timestamp: Block creation timestamp
            transactions: List of transactions
            previous_hash: Hash of previous block
            nonce: Nonce for mining
            
        Returns:
            hash_value: SHA-256 hash of block
        """
        # Create string representation of block data
        block_string = f"{index}{timestamp}{json.dumps(transactions)}{previous_hash}{nonce}"
        
        # Calculate SHA-256 hash
        hash_object = hashlib.sha256(block_string.encode())
        hash_value = hash_object.hexdigest()
        
        return hash_value
        
    def add_transaction(self, sender, data_hash, metadata):
        """
        Add transaction to pending transactions
        
        Args:
            sender: Sender ID (client or server)
            data_hash: Hash of transaction data
            metadata: Additional transaction metadata
            
        Returns:
            success: Whether transaction was added successfully
        """
        transaction = {
            'sender': sender,
            'timestamp': time.time(),
            'data_hash': data_hash,
            'metadata': metadata
        }
        
        # Add to pending transactions
        self.pending_transactions.append(transaction)
        
        # Record in history
        self.transaction_history.append(transaction)
        
        return True
        
    def create_block(self):
        """
        Create new block with pending transactions
        
        Returns:
            new_block: New block added to chain
            tx_count: Number of transactions included
        """
        if not self.pending_transactions:
            return None, 0
            
        # Get transactions that fit in block
        transactions = []
        total_size = 0
        remaining_txs = []
        
        for tx in self.pending_transactions:
            tx_size = len(json.dumps(tx).encode())
            if total_size + tx_size <= self.block_size:
                transactions.append(tx)
                total_size += tx_size
            else:
                remaining_txs.append(tx)
                
        if not transactions:
            # If no transactions fit, include at least the first one
            transactions = [self.pending_transactions[0]]
            remaining_txs = self.pending_transactions[1:]
            
        # Get previous block
        previous_block = self.chain[-1]
        
        # Create new block
        new_block = {
            'index': len(self.chain),
            'timestamp': time.time(),
            'transactions': transactions,
            'previous_hash': previous_block['hash'],
            'nonce': 0
        }
        
        # Calculate hash
        new_block['hash'] = self.calculate_hash(
            new_block['index'],
            new_block['timestamp'],
            new_block['transactions'],
            new_block['previous_hash'],
            new_block['nonce']
        )
        
        # Simulate consensus delay
        time.sleep(self.consensus_delay)
        
        # Add block to chain
        self.chain.append(new_block)
        
        # Update pending transactions
        self.pending_transactions = remaining_txs
        
        return new_block, len(transactions)
        
    def verify_chain(self):
        """
        Verify integrity of blockchain
        
        Returns:
            valid: Whether blockchain is valid
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # Verify hash
            if current_block['hash'] != self.calculate_hash(
                current_block['index'],
                current_block['timestamp'],
                current_block['transactions'],
                current_block['previous_hash'],
                current_block['nonce']
            ):
                return False
                
            # Verify previous hash
            if current_block['previous_hash'] != previous_block['hash']:
                return False
                
        return True
        
    def get_chain_stats(self):
        """
        Get blockchain statistics
        
        Returns:
            stats: Dictionary of blockchain statistics
        """
        if not self.chain:
            return {
                'block_count': 0,
                'transaction_count': 0,
                'avg_transactions_per_block': 0,
                'avg_block_size': 0,
                'chain_valid': True
            }
            
        tx_counts = [len(block['transactions']) for block in self.chain]
        
        stats = {
            'block_count': len(self.chain),
            'transaction_count': sum(tx_counts),
            'avg_transactions_per_block': np.mean(tx_counts) if tx_counts else 0,
            'avg_block_size': np.mean([len(json.dumps(block).encode()) for block in self.chain]),
            'chain_valid': self.verify_chain()
        }
        
        return stats


class DedupStorage:
    """
    Storage manager with deduplication capabilities
    """
    def __init__(self, threshold=0.01):
        """
        Initialize deduplication storage
        
        Args:
            threshold: Threshold for considering parameter changes significant
        """
        self.previous_weights = {}  # Previous model weights by client/cluster
        self.dedup_threshold = threshold
        self.storage_savings = []
        self.block_savings = []
        
    def deduplicate_weights(self, client_id, current_weights):
        """
        Deduplicate model weights by comparing with previous weights
        
        Args:
            client_id: Client or cluster ID (must be hashable, like an integer)
            current_weights: Current model weights (OrderedDict from model.state_dict())
            
        Returns:
            deduped_weights: Deduplicated weights (only significant changes)
            compression_ratio: Ratio of size reduction
        """
        # Convert numpy integer types to Python int for hashability
        if hasattr(client_id, 'item') and callable(getattr(client_id, 'item')):
            # This handles numpy integer types like np.int32, np.int64, etc.
            client_id = int(client_id)
        
        # Add additional type checking to provide more helpful error messages
        if not isinstance(client_id, (int, str, tuple, bool)):
            # We have to use the client_id's value in the dictionary, so it must be hashable
            raise TypeError(f"client_id must be hashable (like int, str), got {type(client_id)}")
            
        # Make sure current_weights has all parameters
        if not current_weights:
            print("Warning: Empty current_weights dictionary passed to deduplicate_weights")
            return current_weights, 0.0
        
        # First submission or model structure change detection
        if client_id not in self.previous_weights:
            # First submission, store full weights
            self.previous_weights[client_id] = {k: v.clone() for k, v in current_weights.items()}
            return current_weights, 0.0
        
        # Check if model structure has changed (different keys)
        prev_keys = set(self.previous_weights[client_id].keys())
        curr_keys = set(current_weights.keys())
        
        if prev_keys != curr_keys:
            print(f"Warning: Model structure changed for client {client_id}")
            print(f"Missing keys: {prev_keys - curr_keys}")
            print(f"New keys: {curr_keys - prev_keys}")
            # Update stored weights and return full weights
            self.previous_weights[client_id] = {k: v.clone() for k, v in current_weights.items()}
            return current_weights, 0.0
                
        deduped_weights = {}
        param_count = 0
        included_count = 0
        
        # For each parameter in the model
        for key in current_weights.keys():
            # Get current and previous values
            current = current_weights[key]
            previous = self.previous_weights[client_id].get(key, torch.zeros_like(current))
            
            # Calculate absolute differences
            diff = torch.abs(current - previous)
            significant_changes = diff > self.dedup_threshold
            
            # Only include parameters with significant changes
            if significant_changes.any():
                # Create sparse representation with indices and values
                indices = torch.nonzero(significant_changes).cpu()
                values = current[significant_changes].cpu()
                
                deduped_weights[key] = {
                    'indices': indices,
                    'values': values,
                    'shape': current.shape
                }
                
                included_count += len(values)
            else:
                # Include a reference to indicate this key was processed but had no changes
                deduped_weights[key] = {
                    'indices': torch.tensor([]),
                    'values': torch.tensor([]),
                    'shape': current.shape,
                    'no_change': True
                }
            
            param_count += current.numel()
        
        # Update stored weights with current weights
        self.previous_weights[client_id] = {k: v.clone() for k, v in current_weights.items()}
        
        # Calculate compression ratio
        compression_ratio = 1.0 - (included_count / param_count) if param_count > 0 else 0.0
        
        self.storage_savings.append(compression_ratio * 100)
        
        return deduped_weights, compression_ratio
    def reconstruct_weights(self, client_id, deduped_weights):
        """
        Reconstruct full weights from deduplicated weights
        
        Args:
            client_id: Client or cluster ID (must be hashable)
            deduped_weights: Deduplicated weights
            
        Returns:
            reconstructed_weights: Full reconstructed weights
        """
        # Convert numpy integer types to Python int for hashability
        if hasattr(client_id, 'item') and callable(getattr(client_id, 'item')):
            # This handles numpy integer types like np.int32, np.int64, etc.
            client_id = int(client_id)
            
        # Ensure client_id is hashable
        if not isinstance(client_id, (int, str, tuple, bool)):
            raise TypeError(f"client_id must be hashable (like int, str), got {type(client_id)}")
            
        if client_id not in self.previous_weights:
            # If no previous weights are found, return the weights as is
            # assuming they are not actually deduplicated
            return deduped_weights
            
        # If deduped_weights is a regular state dict, it's not actually deduplicated
        if all(isinstance(v, torch.Tensor) for v in deduped_weights.values()):
            return deduped_weights
            
        reconstructed_weights = {}
        
        for key, sparse_data in deduped_weights.items():
            # Start with previous values
            if key in self.previous_weights[client_id]:
                reconstructed = self.previous_weights[client_id][key].clone()
            else:
                reconstructed = torch.zeros(sparse_data['shape'])
            
            # Update with new values at specified indices
            for idx, value in zip(sparse_data['indices'], sparse_data['values']):
                idx_tuple = tuple(idx.tolist())
                reconstructed[idx_tuple] = value
                
            reconstructed_weights[key] = reconstructed
            
        return reconstructed_weights
    
    def get_storage_stats(self):
        """
        Get storage statistics
        
        Returns:
            stats: Dictionary of storage statistics
        """
        stats = {
            'avg_compression_ratio': np.mean(self.storage_savings) if self.storage_savings else 0,
            'total_clients': len(self.previous_weights),
            'storage_trend': self.storage_savings
        }
        
        return stats


class CFLAGBD:
    """
    CFLAG-BD: Blockchain-based Clustered Federated Learning with Weight Deduplication
    """
    def __init__(self, global_model, clients, test_loader, device,
            num_clusters=3, dedup_threshold=0.01, use_blockchain=True):
        """
        Initialize CFLAG-BD algorithm with hardware awareness
        
        Args:
            global_model: Global model to be trained
            clients: List of client objects
            test_loader: DataLoader for test dataset
            device: Device to run computations on
            num_clusters: Number of clusters to form
            dedup_threshold: Threshold for deduplication
            use_blockchain: Whether to use blockchain
        """
        self.global_model = global_model
        self.clients = clients
        self.test_loader = test_loader
        self.device = device
        self.num_clusters = num_clusters
        self.dedup_threshold = dedup_threshold
        self.use_blockchain = use_blockchain
        
        # Initialize cluster models
        self.cluster_models = [copy.deepcopy(global_model) for _ in range(num_clusters)]
        
        # Initialize client cluster assignments
        self.client_clusters = np.random.randint(0, num_clusters, size=len(clients))
        
        # Initialize specialized model variants
        self.specialized_models = self.create_specialized_models(global_model)
        
        # Initialize blockchain if enabled
        if use_blockchain:
            self.blockchain = BlockchainSimulator()
        else:
            self.blockchain = None
            
        # Initialize deduplication storage
        self.dedup_storage = DedupStorage(threshold=dedup_threshold)
        
        # Track metrics
        self.metrics = {
            'global_accuracy': [],
            'cluster_accuracies': [],
            'client_accuracies': [],
            'training_time': [],
            'communication_cost': [],
            'storage_savings': [],
            'blockchain_stats': [],
            'clustering_quality': [],
            'cluster_sizes': [],
            'hardware_distribution': []  # New metric
        }
        
        # Track cluster history for stability analysis
        self.cluster_history = []
        
    def extract_client_features(self, client_indices=None):
        """
        Extract features from clients for clustering with backward compatibility
        
        Args:
            client_indices: Indices of clients to extract features from
            
        Returns:
            features: Array of client features
            feature_names: Names of extracted features
        """
        if client_indices is None:
            client_indices = list(range(len(self.clients)))
            
        features = []
        
        for idx in client_indices:
            client = self.clients[idx]
            
            # Extract client features
            client_features = client.extract_clustering_features()
            
            # Create basic feature vector (always available)
            feature_vector = [
                # Original features
                client_features.get('update_mean', 0.0),
                client_features.get('update_std', 0.0),
                client_features.get('processing_power', 1.0),
                client_features.get('reliability_score', 1.0),
                client_features.get('latest_accuracy', 0.0),
                client_features.get('latest_loss', 0.0),
                client_features.get('training_time', 0.0),
                client_features.get('convergence_rate', 0.0),
            ]
            
            # Add hardware-aware features only if available
            # This makes the method backward compatible
            if 'device_type' in client_features:
                feature_vector.extend([
                    client_features.get('device_type', 0.0),
                    client_features.get('compute_score', 1.0),
                    client_features.get('cpu_count', 1),
                    client_features.get('memory_gb', 1.0),
                ])
                
                if 'model_size' in client_features:
                    feature_vector.extend([
                        client_features.get('model_size', 0),
                        client_features.get('compute_intensity', 1.0),
                        client_features.get('conv_layers', 0),
                        client_features.get('fc_layers', 0)
                    ])
            
            features.append(feature_vector)
        
        # Basic feature names (always present)
        feature_names = [
            'update_mean',
            'update_std',
            'processing_power',
            'reliability_score',
            'latest_accuracy',
            'latest_loss',
            'training_time',
            'convergence_rate',
        ]
        
        # Add hardware feature names if those features are included
        if len(features[0]) > len(feature_names):
            feature_names.extend([
                'device_type',
                'compute_score',
                'cpu_count',
                'memory_gb',
            ])
            
            # Add architecture feature names if those features are included
            if len(features[0]) > len(feature_names):
                feature_names.extend([
                    'model_size',
                    'compute_intensity',
                    'conv_layers',
                    'fc_layers'
                ])
                
        return np.array(features), feature_names
    
    def create_specialized_models(self, base_model):
        """
        Create specialized model variants for different hardware profiles
        
        Args:
            base_model: Base model architecture
            
        Returns:
            model_variants: Dictionary of specialized models
        """
        model_variants = {
            'high_compute': self._optimize_for_high_compute(copy.deepcopy(base_model)),
            'low_compute': self._optimize_for_low_compute(copy.deepcopy(base_model)),
            'gpu': self._optimize_for_gpu(copy.deepcopy(base_model)),
            'cpu': self._optimize_for_cpu(copy.deepcopy(base_model))
        }
        
        return model_variants

    def _optimize_for_high_compute(self, model):
        """
        Optimize model for high compute devices
        This is a placeholder - implement actual optimizations based on your models
        """
        # You may want to increase model capacity for high-compute devices
        return model

    def _optimize_for_low_compute(self, model):
        """
        Optimize model for low compute devices
        This is a placeholder - implement actual optimizations based on your models
        """
        # You may want to reduce model capacity for low-compute devices
        # E.g., reduce number of filters in conv layers or neurons in FC layers
        return model
    def _optimize_for_gpu(self, model):
        """
        Optimize model for GPU execution
        This is a placeholder - implement actual GPU optimizations
        """
        # GPU optimizations might include using operations that are efficient on GPUs
        return model

    def _optimize_for_cpu(self, model):
        """
        Optimize model for CPU execution
        This is a placeholder - implement actual CPU optimizations
        """
        # CPU optimizations might include using operations that are efficient on CPUs
        return model

    def select_hardware_diverse_clients(self, client_fraction):
        """
        Select a diverse set of clients based on hardware capabilities
        
        Args:
            client_fraction: Fraction of clients to select
            
        Returns:
            selected_indices: Indices of selected clients
        """
        num_clients = max(1, int(client_fraction * len(self.clients)))
        
        # Group clients by device type
        gpu_clients = []
        cpu_clients = []
        
        for i, client in enumerate(self.clients):
            if hasattr(client, 'hardware_profile') and client.hardware_profile.get('gpu_available', False):
                gpu_clients.append(i)
            else:
                cpu_clients.append(i)
        
        # Calculate how many of each to select
        total_gpu = len(gpu_clients)
        total_cpu = len(cpu_clients)
        total = total_gpu + total_cpu
        
        # Select proportionally to maintain hardware distribution
        gpu_fraction = total_gpu / total if total > 0 else 0
        num_gpu = min(total_gpu, round(num_clients * gpu_fraction))
        num_cpu = min(total_cpu, num_clients - num_gpu)
        
        # If we can't get enough of one type, compensate with the other
        if num_gpu < round(num_clients * gpu_fraction) and num_cpu < total_cpu:
            num_cpu = min(total_cpu, num_clients - num_gpu)
        elif num_cpu < num_clients - num_gpu and num_gpu < total_gpu:
            num_gpu = min(total_gpu, num_clients - num_cpu)
        
        # Select the clients
        selected_gpu_clients = np.random.choice(gpu_clients, num_gpu, replace=False).tolist() if num_gpu > 0 else []
        selected_cpu_clients = np.random.choice(cpu_clients, num_cpu, replace=False).tolist() if num_cpu > 0 else []
        
        # Combine selections
        selected_indices = selected_gpu_clients + selected_cpu_clients
        
        return selected_indices

    def assign_specialized_models(self):
        """
        Assign specialized model variants to clusters based on hardware profiles
        """
        # Initialize specialized models if not already done
        if not hasattr(self, 'specialized_models'):
            self.specialized_models = self.create_specialized_models(self.global_model)
        
        # Calculate hardware profile for each cluster
        cluster_profiles = {}
        
        for cluster_idx in range(self.num_clusters):
            # Get clients in this cluster
            cluster_clients = [i for i, c in enumerate(self.client_clusters) if c == cluster_idx]
            
            if not cluster_clients:
                continue
                
            # Calculate average hardware metrics
            gpu_clients = 0
            total_compute = 0
            
            for client_idx in cluster_clients:
                client = self.clients[client_idx]
                if hasattr(client, 'hardware_profile'):
                    if client.hardware_profile.get('gpu_available', False):
                        gpu_clients += 1
                    total_compute += client.hardware_profile.get('compute_score', 1.0)
            
            # Calculate averages
            gpu_fraction = gpu_clients / len(cluster_clients)
            avg_compute = total_compute / len(cluster_clients)
            
            cluster_profiles[cluster_idx] = {
                'gpu_fraction': gpu_fraction,
                'avg_compute': avg_compute
            }
            
        # Assign specialized models based on profiles
        for cluster_idx, profile in cluster_profiles.items():
            if profile['gpu_fraction'] > 0.5:
                # Mostly GPU clients
                model_variant = self.specialized_models['gpu']
            else:
                # Mostly CPU clients
                model_variant = self.specialized_models['cpu']
                
            # Further specialize based on compute power
            if profile['avg_compute'] > 10:  # Threshold may need adjustment
                if profile['gpu_fraction'] <= 0.5:
                    model_variant = self.specialized_models['high_compute']
            else:
                if profile['gpu_fraction'] <= 0.5:
                    model_variant = self.specialized_models['low_compute']
                    
            # Update cluster model with specialized variant
            # Need to preserve state dict values but update architecture
            current_state = self.cluster_models[cluster_idx].state_dict()
            self.cluster_models[cluster_idx] = copy.deepcopy(model_variant)
            
            try:
                # Try to load previous state dict
                self.cluster_models[cluster_idx].load_state_dict(current_state)
            except Exception as e:
                print(f"Error loading state dict for specialized model: {e}")
                # If architecture changed too much, keep the specialized model as is

    def cluster_clients(self, features):
        """
        Cluster clients based on extracted features with improved handling of edge cases
        
        Args:
            features: Array of client features
            
        Returns:
            cluster_labels: Cluster assignments
            silhouette: Silhouette score for clustering quality
        """
        # Handle the case when we have too few samples
        if len(features) <= self.num_clusters:
            print(f"Warning: Number of samples ({len(features)}) <= number of clusters ({self.num_clusters})")
            print("Using simple assignment instead of clustering")
            # Just assign each sample to its own cluster, or distribute evenly
            if len(features) <= 1:
                cluster_labels = np.zeros(len(features), dtype=int)
            else:
                cluster_labels = np.arange(len(features)) % self.num_clusters
            return cluster_labels, 0.0  # Return zero silhouette score
            
        # Continue with normal clustering if we have enough samples
        # Normalize features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)
        
        # Determine appropriate number of clusters based on sample size
        actual_num_clusters = min(self.num_clusters, max(2, len(features) // 3))
        
        # Apply hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=actual_num_clusters,
            affinity='euclidean',
            linkage='ward'
        )
        
        cluster_labels = clustering.fit_predict(normalized_features)
        
        # Calculate silhouette score if possible
        silhouette = 0.0  # Default value
        try:
            # Count samples per cluster to check if silhouette score is calculable
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            
            if len(unique_labels) >= 2 and all(count >= 2 for count in counts):
                silhouette = silhouette_score(normalized_features, cluster_labels)
            else:
                print("Warning: Cannot calculate silhouette score - not enough samples per cluster")
        except Exception as e:
            print(f"Warning: Could not calculate silhouette score: {e}")
        
        # If we had to reduce the number of clusters, expand back to the desired number
        if actual_num_clusters < self.num_clusters:
            # Map the reduced clusters back to the original number by splitting the largest clusters
            for i in range(actual_num_clusters, self.num_clusters):
                # Find the cluster with the most samples
                unique_labels, counts = np.unique(cluster_labels, return_counts=True)
                largest_cluster = unique_labels[np.argmax(counts)]
                
                # Get indices of samples in the largest cluster
                largest_cluster_indices = np.where(cluster_labels == largest_cluster)[0]
                
                if len(largest_cluster_indices) >= 2:
                    # Split the cluster in half randomly
                    split_size = len(largest_cluster_indices) // 2
                    to_reassign = np.random.choice(largest_cluster_indices, split_size, replace=False)
                    
                    # Assign to new cluster
                    cluster_labels[to_reassign] = i
        
        return cluster_labels, silhouette
        
    def train_clients(self, client_indices=None, epochs=5):
        """
        Train selected clients
        
        Args:
            client_indices: Indices of clients to train
            epochs: Number of local training epochs
            
        Returns:
            client_updates: List of model updates
            training_stats: Dictionary of training statistics
        """
        if client_indices is None:
            client_indices = list(range(len(self.clients)))
            
        client_updates = []
        training_times = []
        client_accuracies = []
        
        # Distribute appropriate cluster models to clients
        for idx in client_indices:
            cluster_idx = self.client_clusters[idx]
            self.clients[idx].set_model(copy.deepcopy(self.cluster_models[cluster_idx]))
            
        # Train clients
        for idx in client_indices:
            start_time = time.time()
            update = self.clients[idx].train(epochs=epochs)
            training_time = time.time() - start_time
            
            client_updates.append(update)
            training_times.append(training_time)
            
            # Record client accuracy
            if self.clients[idx].round_accuracies:
                client_accuracies.append(self.clients[idx].round_accuracies[-1])
                
            # Update client's reliability score based on performance
            global_avg = np.mean(client_accuracies) if client_accuracies else None
            self.clients[idx].update_reliability_score(
                success=True,
                global_average=global_avg
            )
            
        # Compute statistics
        training_stats = {
            'mean_time': np.mean(training_times) if training_times else 0,
            'total_time': np.sum(training_times) if training_times else 0,
            'client_accuracies': client_accuracies
        }
        
        return client_updates, training_stats
        
    def process_and_store_updates(self, client_indices, client_updates):
        """
        Process updates from clients with improved error handling
        
        Args:
            client_indices: Indices of participating clients
            client_updates: List of client updates
            
        Returns:
            cluster_updates: Aggregated updates per cluster
            blockchain_stats: Dictionary of blockchain statistics
        """
        cluster_updates = {}
        blockchain_stats = {
            'reduced_cost': 0,
            'original_cost': 0,
            'compression_ratio': 0
        }
        
        # Validate client updates - check for None or empty updates
        valid_indices = []
        valid_updates = []
        for i, client_id in enumerate(client_indices):
            if i >= len(client_updates):
                print(f"Warning: Missing update for client {client_id}")
                continue
                
            update = client_updates[i]
            if update is None:
                print(f"Warning: Null update for client {client_id}")
                continue
                
            if not update:  # Empty dict
                print(f"Warning: Empty update for client {client_id}")
                continue
                
            # Check if update has the expected keys (basic validation)
            if not all(isinstance(v, torch.Tensor) for v in update.values()):
                print(f"Warning: Invalid update format for client {client_id}")
                continue
                
            valid_indices.append(client_id)
            valid_updates.append(update)
        
        # Check if we have any valid updates
        if not valid_updates:
            print("Error: No valid client updates to process")
            return {}, blockchain_stats
            
        # Apply deduplication to client updates
        reconstructed_updates = []
        total_compression = 0.0
        
        for i, client_id in enumerate(valid_indices):
            update = valid_updates[i]
            
            try:
                # Deduplicate weights
                deduped_update, compression_ratio = self.dedup_storage.deduplicate_weights(client_id, update)
                reconstructed = self.dedup_storage.reconstruct_weights(client_id, deduped_update)
                
                # Validate reconstruction success
                if reconstructed is None or not reconstructed:
                    print(f"Warning: Failed to reconstruct update for client {client_id}")
                    continue
                    
                reconstructed_updates.append(reconstructed)
                total_compression += compression_ratio
            except Exception as e:
                print(f"Error processing update for client {client_id}: {str(e)}")
                continue
                
        # If no updates reconstructed successfully, return empty results
        if not reconstructed_updates:
            print("Error: No client updates could be reconstructed")
            return {}, blockchain_stats
        
        # Group client updates by cluster index
        cluster_groups = defaultdict(list)
        cluster_weights = defaultdict(list)
        
        for i, client_id in enumerate(valid_indices):
            if i >= len(reconstructed_updates):
                continue  # Skip if this client's update wasn't reconstructed
                
            cluster_idx = self.client_clusters[client_id]
            cluster_groups[cluster_idx].append(reconstructed_updates[i])
            
            # Use default weight of 1.0 if client_weights not available
            client_weight = getattr(self.clients[client_id], 'reliability_score', 1.0)
            cluster_weights[cluster_idx].append(client_weight)
        
        # Aggregate updates for each cluster
        for cluster_idx, updates in cluster_groups.items():
            if not updates:
                continue
                
            weights = cluster_weights[cluster_idx]
            cluster_updates[cluster_idx] = self.aggregate_updates(updates, weights)
        
        # Calculate communication costs
        if valid_updates:
            # Estimate original size (in bytes)
            param_size_bytes = 0
            first_update = valid_updates[0]
            for param in first_update.values():
                param_size_bytes += param.numel() * 4  # 4 bytes per float32
            
            original_cost = param_size_bytes * len(valid_indices)
            
            # Only apply compression if we have at least one valid update
            if len(reconstructed_updates) > 0:
                compression_factor = total_compression / len(reconstructed_updates)
            else:
                compression_factor = 0
                
            reduced_cost = original_cost * (1 - compression_factor)
            
            blockchain_stats['original_cost'] = original_cost
            blockchain_stats['reduced_cost'] = reduced_cost
            blockchain_stats['compression_ratio'] = compression_factor
        
        # Blockchain operations (if enabled)
        if self.use_blockchain and self.blockchain and cluster_updates:
            try:
                # Add transactions to blockchain for each cluster update
                for cluster_idx, update in cluster_updates.items():
                    # Create a deterministic string representation of the update
                    update_str = str(sum(p.sum().item() for p in update.values()))
                    data_hash = str(hash(update_str))
                    
                    # Add to blockchain
                    self.blockchain.add_transaction(
                        sender=f"cluster_{cluster_idx}",
                        data_hash=data_hash,
                        metadata={"round": len(self.metrics['global_accuracy'])}
                    )
                
                # Create blocks with transactions
                self.blockchain.create_block()
            except Exception as e:
                print(f"Error in blockchain operations: {str(e)}")
    
        return cluster_updates, blockchain_stats

    def aggregate_updates(self, updates, weights=None):
        """
        Aggregate model updates with improved error handling
        
        Args:
            updates: List of model updates to aggregate
            weights: Weights for each update
            
        Returns:
            aggregated_update: Weighted average of updates
        """
        if not updates:
            return None
            
        # Use equal weights if not provided
        if weights is None:
            weights = [1.0 / len(updates)] * len(updates)
            
        # Ensure weights sum to 1
        weights = np.array(weights) / np.sum(weights)
        
        # Get all keys from all updates
        all_keys = set()
        for update in updates:
            all_keys.update(update.keys())
        
        # Initialize aggregated update
        aggregated_update = {}
        
        # For each parameter key, aggregate across all updates
        for key in all_keys:
            # Get all valid updates for this key
            valid_updates = []
            valid_weights = []
            
            for i, update in enumerate(updates):
                if key in update and update[key] is not None:
                    valid_updates.append(update[key])
                    valid_weights.append(weights[i])
                    
            if valid_updates:
                # Re-normalize weights for valid updates
                valid_weights = np.array(valid_weights) / np.sum(valid_weights)
                
                # First update sets the tensor shape/type
                aggregated_update[key] = torch.zeros_like(valid_updates[0])
                
                # Weighted average of parameters
                for i, update_val in enumerate(valid_updates):
                    aggregated_update[key] += update_val * valid_weights[i]
            else:
                # No valid updates for this key, use zeros
                # This shouldn't happen in normal operation, so print a warning
                print(f"Warning: No valid updates found for parameter {key}")
                
        # Verify the aggregated update has all required keys
        if not all(key in aggregated_update for key in all_keys):
            missing_keys = all_keys - set(aggregated_update.keys())
            print(f"Warning: Missing keys in aggregated update: {missing_keys}")
        
        return aggregated_update
        
    def update_cluster_models(self, cluster_updates):
        """
        Update cluster models with aggregated updates with improved error handling
        
        Args:
            cluster_updates: Dictionary mapping clusters to aggregated updates
        """
        for cluster_idx, update in cluster_updates.items():
            if update is not None:
                # Get the current model state
                current_state = self.cluster_models[cluster_idx].state_dict()
                
                # Check for missing keys in the update
                missing_keys = set(current_state.keys()) - set(update.keys())
                extra_keys = set(update.keys()) - set(current_state.keys())
                
                if missing_keys:
                    # Instead of just a warning, provide more detail for debugging
                    print(f"Warning: Cluster {cluster_idx} update missing keys: {missing_keys}")
                    print("Copying missing keys from current model state")
                    
                    # Create a comprehensive update dictionary with all keys
                    complete_update = {}
                    
                    # First add all available updates
                    for key in current_state.keys():
                        if key in update:
                            complete_update[key] = update[key]
                        else:
                            # For missing keys, copy from the current model
                            complete_update[key] = current_state[key].clone()
                    
                    # Now use this complete update
                    update = complete_update
                
                if extra_keys:
                    print(f"Warning: Cluster {cluster_idx} update contains extra keys: {extra_keys}")
                    # Remove extra keys that don't belong in the model
                    for key in extra_keys:
                        del update[key]
                
                # Verify parameter shapes match before updating
                shape_mismatch = False
                for key, param in update.items():
                    if key in current_state and param.shape != current_state[key].shape:
                        print(f"Error: Shape mismatch for parameter {key} in cluster {cluster_idx}")
                        print(f"Expected shape: {current_state[key].shape}, got: {param.shape}")
                        shape_mismatch = True
                
                # Only update if no shape mismatches were found
                if not shape_mismatch:
                    try:
                        self.cluster_models[cluster_idx].load_state_dict(update)
                    except Exception as e:
                        print(f"Error updating cluster {cluster_idx} model: {str(e)}")
                        print("Rolling back to previous model state")
                        # No need to do anything as we haven't modified the model yet
                else:
                    print(f"Skipping update for cluster {cluster_idx} due to parameter shape mismatches")
                
    def update_global_model(self):
        """
        Update global model by aggregating cluster models
        
        Returns:
            accuracy: Global model accuracy
        """
        # Evaluate each cluster model
        cluster_accuracies = []
        for model in self.cluster_models:
            self.global_model.load_state_dict(model.state_dict())
            accuracy = self.evaluate_global_model()
            cluster_accuracies.append(accuracy)
            
        # Find best performing cluster
        best_cluster = np.argmax(cluster_accuracies)
        
        # Use best cluster model as global model
        self.global_model.load_state_dict(self.cluster_models[best_cluster].state_dict())
        
        return cluster_accuracies[best_cluster], cluster_accuracies
        
    def evaluate_global_model(self):
        """
        Evaluate global model on test dataset
        
        Returns:
            accuracy: Test accuracy
        """
        self.global_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.global_model(data)
                
                # Compute accuracy based on output type
                if len(target.shape) > 1 and target.shape[1] > 1:
                    # Multi-label classification
                    predicted = (outputs > 0.5).float()
                    correct += (predicted == target).sum().item()
                    total += target.numel()
                else:
                    # Standard classification
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
        
        accuracy = correct / total if total > 0 else 0
        return accuracy
        
    def run_round(self, client_fraction=1.0, epochs=5):
        """
        Run one round of CFLAG-BD with backward compatibility
        
        Args:
            client_fraction: Fraction of clients to select for training
            epochs: Number of local training epochs
            
        Returns:
            round_stats: Dictionary of round statistics
        """
        round_start_time = time.time()
        
        # Check if hardware-aware selection is available
        has_hardware_awareness = hasattr(self, 'select_hardware_diverse_clients')
        
        # Select clients 
        if has_hardware_awareness:
            try:
                client_indices = self.select_hardware_diverse_clients(client_fraction)
            except Exception as e:
                print(f"Warning: Hardware-aware client selection failed: {e}")
                # Fall back to random selection
                has_hardware_awareness = False
                num_clients = max(1, int(client_fraction * len(self.clients)))
                client_indices = np.random.choice(len(self.clients), num_clients, replace=False)
        else:
            # Traditional random selection
            num_clients = max(1, int(client_fraction * len(self.clients)))
            client_indices = np.random.choice(len(self.clients), num_clients, replace=False)
        
        # Train selected clients
        client_updates, training_stats = self.train_clients(client_indices, epochs)
        
        # Extract features for clustering
        features, feature_names = self.extract_client_features(client_indices)
        
        # Cluster clients
        cluster_labels, silhouette = self.cluster_clients(features)
        self.metrics['clustering_quality'].append(silhouette)
        # Update client cluster assignments
        for i, idx in enumerate(client_indices):
            self.client_clusters[idx] = cluster_labels[i]
            
        # Track cluster history
        self.cluster_history.append(self.client_clusters.copy())
        
        # Assign specialized models if available
        if has_hardware_awareness and hasattr(self, 'assign_specialized_models'):
            try:
                self.assign_specialized_models()
            except Exception as e:
                print(f"Warning: Failed to assign specialized models: {e}")
        
        # Process client updates with deduplication and blockchain
        cluster_updates, blockchain_stats = self.process_and_store_updates(
            client_indices, client_updates
        )
        
        # Update cluster models
        self.update_cluster_models(cluster_updates)
        
        # Update global model
        global_accuracy, cluster_accuracies = self.update_global_model()
        
        # Track cluster sizes
        cluster_sizes = [np.sum(self.client_clusters == i) for i in range(self.num_clusters)]
        
        # Track hardware distribution if available
        hardware_distribution = {}
        if has_hardware_awareness:
            try:
                hardware_distribution = {
                    'gpu_clients': sum(1 for i in client_indices 
                                    if hasattr(self.clients[i], 'hardware_profile') 
                                    and self.clients[i].hardware_profile.get('gpu_available', False)),
                    'cpu_clients': sum(1 for i in client_indices 
                                    if not hasattr(self.clients[i], 'hardware_profile') 
                                    or not self.clients[i].hardware_profile.get('gpu_available', False))
                }
            except Exception as e:
                print(f"Warning: Failed to collect hardware distribution: {e}")
        
        # Track metrics
        self.metrics['global_accuracy'].append(global_accuracy)
        self.metrics['cluster_accuracies'].append(cluster_accuracies)
        self.metrics['client_accuracies'].append(training_stats['client_accuracies'])
        self.metrics['training_time'].append(training_stats['total_time'])
        self.metrics['communication_cost'].append(blockchain_stats['reduced_cost'])
        self.metrics['storage_savings'].append(blockchain_stats['compression_ratio'] * 100)
        self.metrics['blockchain_stats'].append(blockchain_stats)
        self.metrics['clustering_quality'].append(silhouette)
        self.metrics['cluster_sizes'].append(cluster_sizes)
        
        # Only add hardware_distribution if it exists
        if hardware_distribution and 'hardware_distribution' in self.metrics:
            self.metrics['hardware_distribution'].append(hardware_distribution)
        
        round_time = time.time() - round_start_time
        
        round_stats = {
            'accuracy': global_accuracy,
            'training_time': training_stats['total_time'],
            'communication_cost': blockchain_stats['reduced_cost'],
            'original_cost': blockchain_stats['original_cost'],
            'compression_ratio': blockchain_stats['compression_ratio'],
            'num_clients': len(client_indices),
            'cluster_sizes': cluster_sizes,
            'cluster_accuracies': cluster_accuracies,
            'silhouette_score': silhouette,
            'round_time': round_time
        }
        
        # Add hardware distribution if available
        if hardware_distribution:
            round_stats['hardware_distribution'] = hardware_distribution
        
        return round_stats
        
    def run_simulation(self, num_rounds=10, client_fraction=1.0, epochs=5):
        """
        Run federated learning simulation for multiple rounds
        
        Args:
            num_rounds: Number of communication rounds
            client_fraction: Fraction of clients to select each round
            epochs: Number of local training epochs
            
        Returns:
            results: Dictionary of simulation results
        """
        start_time = time.time()
        
        round_stats = []
        for round_idx in range(num_rounds):
            round_start = time.time()
            stats = self.run_round(client_fraction, epochs)
            round_time = time.time() - round_start
            
            stats['round'] = round_idx + 1
            stats['round_time'] = round_time
            round_stats.append(stats)
            
            print(f"Round {round_idx+1}/{num_rounds}, Accuracy: {stats['accuracy']:.4f}, Time: {round_time:.2f}s")
            
        total_time = time.time() - start_time
        
        # Calculate performance improvement
        if num_rounds > 1:
            accuracy_improvement = (self.metrics['global_accuracy'][-1] - self.metrics['global_accuracy'][0]) * 100
            storage_savings = np.mean(self.metrics['storage_savings'])
        else:
            accuracy_improvement = 0
            storage_savings = 0
            
        results = {
            'final_accuracy': self.metrics['global_accuracy'][-1],
            'accuracy_improvement': accuracy_improvement,
            'accuracy_history': self.metrics['global_accuracy'],
            'training_time': sum(self.metrics['training_time']),
            'communication_cost': sum(self.metrics['communication_cost']),
            'storage_savings': storage_savings,
            'total_time': total_time,
            'round_stats': round_stats,
            'blockchain_stats': self.metrics['blockchain_stats'][-1] if self.metrics['blockchain_stats'] else None
        }
        
        return results
        
    def analyze_clustering_stability(self):
        """
        Analyze stability of clustering across rounds
        
        Returns:
            stability_stats: Dictionary of clustering stability statistics
        """
        if len(self.cluster_history) < 2:
            return {'stability': 1.0, 'changes_per_round': 0}
            
        stability_scores = []
        changes_per_round = []
        
        for i in range(1, len(self.cluster_history)):
            previous = self.cluster_history[i-1]
            current = self.cluster_history[i]
            
            # Count changes in cluster assignments
            changes = sum(previous != current)
            stability = 1 - (changes / len(previous))
            
            stability_scores.append(stability)
            changes_per_round.append(changes)
            
        return {
            'stability': np.mean(stability_scores),
            'stability_trend': stability_scores,
            'changes_per_round': changes_per_round,
            'avg_changes_per_round': np.mean(changes_per_round)
        }
        
    def generate_performance_report(self, compare_with=None):
        """
        Generate comprehensive performance report
        
        Args:
            compare_with: Optional other method to compare with
            
        Returns:
            report: Dictionary of performance metrics
        """
        # Basic performance metrics
        report = {
            'accuracy': {
                'final': self.metrics['global_accuracy'][-1] if self.metrics['global_accuracy'] else 0,
                'trend': self.metrics['global_accuracy'],
                'improvement': (self.metrics['global_accuracy'][-1] - self.metrics['global_accuracy'][0]) * 100 
                              if len(self.metrics['global_accuracy']) > 1 else 0
            },
            'efficiency': {
                'communication_cost': sum(self.metrics['communication_cost']),
                'storage_savings': np.mean(self.metrics['storage_savings']) if self.metrics['storage_savings'] else 0,
                'training_time': sum(self.metrics['training_time'])
            },
            'clustering': {
                'quality': np.mean(self.metrics['clustering_quality']) if self.metrics['clustering_quality'] else 0,
                'stability': self.analyze_clustering_stability()['stability'],
                'final_sizes': self.metrics['cluster_sizes'][-1] if self.metrics['cluster_sizes'] else []
            }
        }
        
        # Add blockchain statistics if available
        if self.blockchain:
            blockchain_stats = self.blockchain.get_chain_stats()
            report['blockchain'] = {
                'blocks': blockchain_stats['block_count'],
                'transactions': blockchain_stats['transaction_count'],
                'avg_tx_per_block': blockchain_stats['avg_transactions_per_block'],
                'chain_valid': blockchain_stats['chain_valid']
            }
            
        # Add comparison if provided
        if compare_with:
            # Calculate improvements
            if compare_with.metrics['global_accuracy']:
                accuracy_diff = self.metrics['global_accuracy'][-1] - compare_with.metrics['global_accuracy'][-1]
                accuracy_improvement = accuracy_diff * 100  # as percentage
            else:
                accuracy_improvement = 0
                
            if compare_with.metrics['communication_cost']:
                cost_reduction = 1 - (sum(self.metrics['communication_cost']) / sum(compare_with.metrics['communication_cost']))
                cost_reduction_pct = cost_reduction * 100  # as percentage
            else:
                cost_reduction_pct = 0
                
            report['comparison'] = {
                'accuracy_improvement': accuracy_improvement,
                'communication_cost_reduction': cost_reduction_pct,
                'method_name': compare_with.__class__.__name__
            }
            
        return report
    
    def visualize_cluster_assignments(self, save_path=None, compare_with=None):
        """
        Create heatmaps visualizing client cluster assignments over rounds
        
        Args:
            save_path: Optional path to save the visualization
            compare_with: Optional other method to compare with (must have cluster_history)
            
        Returns:
            fig: Matplotlib figure object with the visualizations
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import os
        
        # Check if we have cluster history
        if not hasattr(self, 'cluster_history') or not self.cluster_history:
            print("No cluster history available for visualization")
            return None
        
        # Create a matrix of client assignments (rows=clients, columns=rounds)
        num_clients = len(self.cluster_history[0])
        num_rounds = len(self.cluster_history)
        
        # Select a subset of clients to visualize if there are too many
        max_clients_to_show = min(100, num_clients)  # Limit to 100 clients for visibility
        client_indices = np.linspace(0, num_clients-1, max_clients_to_show, dtype=int)
        
        # Create matrix of cluster assignments
        cflag_matrix = np.zeros((len(client_indices), num_rounds))
        for round_idx in range(num_rounds):
            for i, client_idx in enumerate(client_indices):
                cflag_matrix[i, round_idx] = self.cluster_history[round_idx][client_idx]
        
        # Calculate stability metrics
        stability_scores = []
        for i in range(1, len(self.cluster_history)):
            previous = self.cluster_history[i-1]
            current = self.cluster_history[i]
            
            # Count changes in cluster assignments
            changes = sum(previous[j] != current[j] for j in range(len(previous)))
            stability = 1 - (changes / len(previous))
            stability_scores.append(stability)
        
        avg_stability = np.mean(stability_scores) if stability_scores else 0.0
        
        # Create the visualization
        if compare_with is not None and hasattr(compare_with, 'cluster_history') and compare_with.cluster_history:
            # Set up comparative visualization
            fig, axes = plt.subplots(2, 1, figsize=(15, 12))
            
            # Create matrix for comparison method
            comp_matrix = np.zeros((len(client_indices), min(num_rounds, len(compare_with.cluster_history))))
            for round_idx in range(min(num_rounds, len(compare_with.cluster_history))):
                for i, client_idx in enumerate(client_indices):
                    if client_idx < len(compare_with.cluster_history[round_idx]):
                        comp_matrix[i, round_idx] = compare_with.cluster_history[round_idx][client_idx]
            
            # Calculate comparison stability
            comp_stability_scores = []
            for i in range(1, len(compare_with.cluster_history)):
                previous = compare_with.cluster_history[i-1]
                current = compare_with.cluster_history[i]
                
                # Count changes in cluster assignments
                changes = sum(previous[j] != current[j] for j in range(min(len(previous), len(current))))
                stability = 1 - (changes / min(len(previous), len(current)))
                comp_stability_scores.append(stability)
            
            comp_avg_stability = np.mean(comp_stability_scores) if comp_stability_scores else 0.0
            
            # Plot CFLAG-BD
            sns.heatmap(cflag_matrix, ax=axes[0], cmap='viridis', cbar_kws={'label': 'Cluster'})
            axes[0].set_title(f'CFLAG-BD: Client Assignments over Rounds (Stability: {avg_stability:.2f})')
            axes[0].set_xlabel('Round')
            axes[0].set_ylabel('Client ID')
            
            # Plot comparison method
            sns.heatmap(comp_matrix, ax=axes[1], cmap='viridis', cbar_kws={'label': 'Cluster'})
            axes[1].set_title(f'{compare_with.__class__.__name__}: Client Assignments over Rounds (Stability: {comp_avg_stability:.2f})')
            axes[1].set_xlabel('Round')
            axes[1].set_ylabel('Client ID')
            
        else:
            # Set up single visualization
            fig, ax = plt.subplots(figsize=(15, 10))
            
            # Plot CFLAG-BD
            sns.heatmap(cflag_matrix, ax=ax, cmap='viridis', cbar_kws={'label': 'Cluster'})
            ax.set_title(f'CFLAG-BD: Client Assignments over Rounds (Stability: {avg_stability:.2f})')
            ax.xlabel('Round')
            ax.ylabel('Client ID')
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        return fig

    def visualize_client_clusters_tsne(self, perplexity=30, n_iter=1000, save_path=None):
        """
        Create t-SNE visualization of client clustering
        
        Args:
            perplexity: t-SNE perplexity parameter
            n_iter: Number of iterations for t-SNE
            save_path: Optional path to save the visualization
            
        Returns:
            fig: Matplotlib figure with the t-SNE visualization
        """
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        import numpy as np
        import os
        
        # Get features for visualization
        try:
            # Extract features for all clients
            features, feature_names = self.extract_client_features()
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
            features_2d = tsne.fit_transform(features)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Color by cluster assignment
            for cluster_idx in range(self.num_clusters):
                cluster_mask = self.client_clusters == cluster_idx
                ax.scatter(
                    features_2d[cluster_mask, 0],
                    features_2d[cluster_mask, 1],
                    label=f'Cluster {cluster_idx}',
                    alpha=0.7
                )
            
            # Add labels and legend
            ax.set_title('t-SNE Visualization of Client Clusters')
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Save figure if requested
            if save_path:
                # Make sure the directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"t-SNE visualization saved to {save_path}")
            
            return fig
            
        except Exception as e:
            print(f"Error in t-SNE visualization: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_and_visualize_stability(self, compare_with=None, save_path=None):
        """
        Analyze and visualize the stability of clustering over rounds
        
        Args:
            compare_with: Optional other method to compare with
            save_path: Optional path to save the visualization
            
        Returns:
            fig: Matplotlib figure object with the stability visualization
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Calculate stability metrics
        stability_stats = self.analyze_clustering_stability()
        stability_trend = stability_stats.get('stability_trend', [])
        changes_per_round = stability_stats.get('changes_per_round', [])
        
        # Create visualization
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot stability trend
        rounds = list(range(1, len(stability_trend) + 1))
        axes[0].plot(rounds, stability_trend, 'o-', color='blue', label='CFLAG-BD')
        
        # Plot comparison method if provided
        if compare_with is not None and hasattr(compare_with, 'analyze_clustering_stability'):
            comp_stats = compare_with.analyze_clustering_stability()
            comp_trend = comp_stats.get('stability_trend', [])
            comp_rounds = list(range(1, len(comp_trend) + 1))
            
            if comp_trend:
                axes[0].plot(comp_rounds, comp_trend, 'o-', color='orange', 
                            label=compare_with.__class__.__name__)
        
        axes[0].set_title('Cluster Stability Over Rounds')
        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('Stability Score (0-1)')
        axes[0].set_ylim(0, 1.1)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot changes per round
        axes[1].bar(rounds, changes_per_round, color='blue', alpha=0.7, label='CFLAG-BD')
        
        # Plot comparison method if provided
        if compare_with is not None and hasattr(compare_with, 'analyze_clustering_stability'):
            comp_stats = compare_with.analyze_clustering_stability()
            comp_changes = comp_stats.get('changes_per_round', [])
            comp_rounds = list(range(1, len(comp_changes) + 1))
            
            if comp_changes:
                axes[1].bar(comp_rounds, comp_changes, color='orange', alpha=0.5,
                            label=compare_with.__class__.__name__)
        
        axes[1].set_title('Client Assignment Changes Per Round')
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('Number of Changes')
        axes[1].grid(True, axis='y', alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Stability analysis saved to {save_path}")
        
        return fig

    # Add method to compute cluster stability (as provided in the question)
    def compute_cluster_stability(self, clusters):
        """
        Compute stability of clustering between consecutive rounds
        
        Args:
            clusters: List of cluster assignment arrays for each round
            
        Returns:
            stability: Average stability score (0-1, higher is more stable)
        """
        if not clusters or len(clusters) < 2:
            return 0  # Not enough data to compute stability
        
        stability = []
        for i in range(1, len(clusters)):
            changes = sum(c1 != c2 for c1, c2 in zip(clusters[i], clusters[i-1]))
            stability.append(1 - changes / len(clusters[i]))
        
        return np.mean(stability)

    # To integrate with the experiment runner, add this method:
    def add_clustering_visualization(runner, comparative_results, save_path=None):
        """
        Add clustering visualization to experiment runner
        
        Args:
            runner: ExperimentRunner instance
            comparative_results: Results from comparative experiments
            save_path: Path to save visualizations
        """
        # Find methods that use clustering
        clustering_methods = []
        for key, results in comparative_results.items():
            if isinstance(key, tuple) and len(key) == 2:
                dataset, method = key
                # Check if this is a clustering-based method
                if method.lower() in ['cflag-bd', 'ifca', 'layercfl', 'cqfl']:
                    # Find the actual method instance
                    for experiment_id, exp_results in runner.experiment_results.items():
                        if exp_results.get('dataset') == dataset and exp_results.get('method') == method:
                            if hasattr(exp_results, 'cluster_history'):
                                clustering_methods.append((dataset, method, exp_results))
        
        # Create visualizations for each dataset
        for dataset in set(d for d, _, _ in clustering_methods):
            dataset_methods = [(m, r) for d, m, r in clustering_methods if d == dataset]
            
            if dataset_methods:
                # Use CFLAG-BD as primary if available
                cflag_bd = next((r for m, r in dataset_methods if m.lower() == 'cflag-bd'), None)
                
                if cflag_bd:
                    # Create visualization directory
                    viz_dir = os.path.join(save_path, 'clustering') if save_path else None
                    if viz_dir:
                        os.makedirs(viz_dir, exist_ok=True)
                    
                    # Create visualizations for CFLAG-BD
                    if hasattr(cflag_bd, 'visualize_cluster_assignments'):
                        for m, r in dataset_methods:
                            if m.lower() != 'cflag-bd' and hasattr(r, 'cluster_history'):
                                viz_path = os.path.join(viz_dir, f'{dataset}_cluster_comparison_{m}.png')
                                cflag_bd.visualize_cluster_assignments(save_path=viz_path, compare_with=r)
                    
                    # Create stability visualization
                    stability_path = os.path.join(viz_dir, f'{dataset}_stability_analysis.png')
                    if hasattr(cflag_bd, 'analyze_and_visualize_stability'):
                        for m, r in dataset_methods:
                            if m.lower() != 'cflag-bd':
                                cflag_bd.analyze_and_visualize_stability(compare_with=r, save_path=stability_path)