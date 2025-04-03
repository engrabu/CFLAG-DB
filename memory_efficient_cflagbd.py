import torch
import numpy as np
import gc
import psutil
import os
import time
from collections import defaultdict
import copy
from monitored_execution import MemoryMonitor, TimeoutError, ThreadingTimeout
import signal
import json

def memory_stats():
    """Print current memory usage statistics"""
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} memory: {torch.cuda.memory_allocated(i) / (1024 * 1024):.2f} MB allocated, "
                  f"{torch.cuda.memory_reserved(i) / (1024 * 1024):.2f} MB reserved")

class MemoryEfficientCFLAGBD:
    """
    Memory-efficient implementation of CFLAG-BD for large numbers of clients
    This is a wrapper/extension for the existing CFLAG-BD implementation
    """
    def __init__(self, global_model, clients, test_loader, device, 
                 num_clusters=3, dedup_threshold=0.01, use_blockchain=True,
                 batch_size=100, offload_to_disk=False, storage_path="./client_storage"):
        """
        Initialize memory-efficient CFLAG-BD
        
        Args:
            global_model: Global model to be trained
            clients: List of client objects (or dataset partitions to create clients from)
            test_loader: DataLoader for test dataset
            device: Device to run computations on
            num_clusters: Number of clusters to form
            dedup_threshold: Threshold for deduplication
            use_blockchain: Whether to use blockchain
            batch_size: Maximum number of clients to process at once
            offload_to_disk: Whether to offload inactive clients to disk
            storage_path: Path for storing offloaded clients
        """
        self.global_model = global_model
        self.full_client_list = clients  # Store reference but don't process yet
        self.test_loader = test_loader
        self.device = device
        self.num_clusters = num_clusters
        self.dedup_threshold = dedup_threshold
        self.use_blockchain = use_blockchain
        self.batch_size = batch_size
        self.offload_to_disk = offload_to_disk
        self.storage_path = storage_path
        
        if offload_to_disk:
            os.makedirs(storage_path, exist_ok=True)
        
        # Initialize minimal required structures
        self.active_clients = []
        self.client_clusters = np.random.randint(0, num_clusters, size=len(clients))
        self.cluster_models = [copy.deepcopy(global_model) for _ in range(num_clusters)]
        
        # Metrics tracking with minimal memory usage
        self.metrics = {
            'global_accuracy': [],
            'cluster_accuracies': [],
            'training_time': [],
            'communication_cost': [],
            'storage_savings': []
        }
        
        # Only initialize blockchain if enabled (it can consume memory)
        if use_blockchain:
            from cflag_bd_algorithm import BlockchainSimulator
            self.blockchain = BlockchainSimulator()
        else:
            self.blockchain = None
            
        # Initialize deduplication storage
        from cflag_bd_algorithm import DedupStorage
        self.dedup_storage = DedupStorage(threshold=dedup_threshold)
    
    def _load_client_batch(self, client_indices):
        """
        Load a batch of clients into memory with version compatibility
        
        Args:
            client_indices: Indices of clients to load
            
        Returns:
            active_clients: List of loaded client objects
        """
        if self.offload_to_disk:
            # Load clients from disk if they've been offloaded
            from enhanced_client import EnhancedClient
            
            # Check PyTorch version for compatibility
            import torch
            torch_version = torch.__version__.split('.')
            major_version = int(torch_version[0])
            minor_version = int(torch_version[1]) if len(torch_version) > 1 else 0
            supports_weights_only = (major_version > 2) or (major_version == 2 and minor_version >= 6)
            
            active_clients = []
            for idx in client_indices:
                client_path = os.path.join(self.storage_path, f"client_{idx}.pt")
                if os.path.exists(client_path):
                    try:
                        # Load based on PyTorch version
                        if supports_weights_only:
                            # PyTorch 2.6+: Try with weights_only=False first
                            try:
                                client_state = torch.load(client_path, weights_only=False)
                            except Exception as e:
                                print(f"Warning: Error loading client {idx} with weights_only=False: {e}")
                                # Try with explicit allowlist if first attempt fails
                                try:
                                    import numpy as np
                                    from torch.serialization import add_safe_globals
                                    
                                    # Add numpy scalar type to allowlist
                                    add_safe_globals(['numpy.core.multiarray.scalar'])
                                    
                                    # Try loading again with weights_only=True
                                    client_state = torch.load(client_path, weights_only=True)
                                except Exception as e2:
                                    print(f"Error loading client {idx} with both methods: {e2}")
                                    # Fall back to original client
                                    client = self.full_client_list[idx]
                                    active_clients.append(client)
                                    continue
                        else:
                            # PyTorch before 2.6: traditional loading
                            client_state = torch.load(client_path)
                    
                        # Recreate client object with stored state
                        client = EnhancedClient(
                            client_id=idx,
                            dataset=self.full_client_list[idx].dataset,
                            device=self.device
                        )
                        
                        # Load state into client
                        if hasattr(client, 'load_state_dict'):
                            client.load_state_dict(client_state)
                        else:
                            # Manual attribute setting if load_state_dict not available
                            for key, value in client_state.items():
                                if key == 'model_state' and hasattr(client, 'model'):
                                    client.model.load_state_dict(value)
                                elif key == 'optimizer_state' and hasattr(client, 'optimizer'):
                                    client.optimizer.load_state_dict(value)
                                else:
                                    setattr(client, key, value)
                                    
                    except Exception as e:
                        print(f"Warning: Failed to load client {idx} from disk: {e}")
                        print("Using original client instead.")
                        client = self.full_client_list[idx]
                else:
                    # If not previously saved, use original client
                    client = self.full_client_list[idx]
                
                active_clients.append(client)
        else:
            # Simply reference the original clients
            active_clients = [self.full_client_list[idx] for idx in client_indices]
            
        return active_clients
    
    def _offload_clients(self, clients, client_indices):
        """
        Offload clients to disk to free memory in a version-compatible way
        
        Args:
            clients: List of client objects to offload
            client_indices: Corresponding indices of these clients
        """
        if not self.offload_to_disk:
            return
        
        # Check PyTorch version for compatibility
        import torch
        torch_version = torch.__version__.split('.')
        major_version = int(torch_version[0])
        minor_version = int(torch_version[1]) if len(torch_version) > 1 else 0
        supports_weights_only = (major_version > 2) or (major_version == 2 and minor_version >= 6)
        
        print(f"PyTorch version: {torch.__version__}, supports weights_only: {supports_weights_only}")
                
        for i, client in enumerate(clients):
            idx = client_indices[i]
            client_path = os.path.join(self.storage_path, f"client_{idx}.pt")
            
            try:
                # Get serializable state dict
                if hasattr(client, 'get_state_dict'):
                    client_state = client.get_state_dict()
                else:
                    # If client doesn't have get_state_dict method, create a simple state dict
                    client_state = {
                        'client_id': getattr(client, 'client_id', idx),
                        'learning_rate': getattr(client, 'learning_rate', 0.01),
                        'batch_size': getattr(client, 'batch_size', 32),
                        'processing_power': getattr(client, 'processing_power', 1.0),
                        'reliability_score': getattr(client, 'reliability_score', 1.0),
                        'round_accuracies': getattr(client, 'round_accuracies', []),
                        'round_losses': getattr(client, 'round_losses', []),
                        'total_training_time': getattr(client, 'total_training_time', 0),
                    }
                    
                    # Add model state if available
                    if hasattr(client, 'model') and client.model is not None:
                        client_state['model_state'] = client.model.state_dict()
                        
                    # Add optimizer state if available    
                    if hasattr(client, 'optimizer') and client.optimizer is not None:
                        client_state['optimizer_state'] = client.optimizer.state_dict()
                
                # Use compatible save method based on PyTorch version
                if supports_weights_only:
                    # PyTorch 2.6+
                    torch.save(client_state, client_path, _use_new_zipfile_serialization=True, weights_only=False)
                else:
                    # PyTorch before 2.6
                    torch.save(client_state, client_path, _use_new_zipfile_serialization=True)
                
            except Exception as e:
                print(f"Warning: Could not offload client {idx}: {e}")
                print("This client will not be available for reload from disk.")
    
    # Update this method in memory_efficient_cflagbd.py

    def run_simulation(self, num_rounds=10, client_fraction=0.2, epochs=5, batch_timeout=300, round_timeout=1800):
        """
        Run federated learning simulation with robust monitoring and timeout protection
        Windows-compatible version that uses threading for timeouts
        
        Args:
            num_rounds: Number of communication rounds
            client_fraction: Fraction of clients to select each round
            epochs: Number of local training epochs
            batch_timeout: Timeout for each batch in seconds (default: 5 minutes)
            round_timeout: Timeout for each round in seconds (default: 30 minutes)
            
        Returns:
            results: Dictionary of simulation results
        """
        # Start memory monitoring
        memory_monitor = MemoryMonitor()
        memory_monitor.start()
        
        try:
            start_time = time.time()
            total_clients = len(self.full_client_list)
            
            print(f"Starting memory-efficient CFLAG-BD with {total_clients} clients")
            print(f"Initial memory usage:")
            memory_stats()
            
            # Import original CFLAG-BD to use its methods
            from cflag_bd_algorithm import CFLAGBD
            
            # Track progress
            round_stats = []
            completed_rounds = 0
            
            # Setup progress tracking
            progress_file = os.path.join(self.storage_path, "progress.json")
            
            # Create or load progress file
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r') as f:
                        progress_data = json.load(f)
                        completed_rounds = progress_data.get('completed_rounds', 0)
                        if 'metrics' in progress_data:
                            self.metrics = progress_data['metrics']
                        round_stats = progress_data.get('round_stats', [])
                        
                    print(f"Resuming from round {completed_rounds + 1} based on saved progress")
                except Exception as e:
                    print(f"Error loading progress file: {e}")
                    completed_rounds = 0
                    
            # Loop through rounds with timeout protection
            for round_idx in range(completed_rounds, num_rounds):
                round_start = time.time()
                print(f"\n{'='*50}")
                print(f"ROUND {round_idx+1}/{num_rounds} - STARTING")
                print(f"{'='*50}")
                
                try:
                    # Use a thread-based timeout for the round
                    round_timer = ThreadingTimeout(round_timeout)
                    round_timer.start()
                    
                    # Track activity timestamps for detecting hangs
                    last_activity = time.time()
                    
                    # Select clients for this round
                    clients_per_round = max(1, int(client_fraction * total_clients))
                    selected_indices = np.random.choice(total_clients, clients_per_round, replace=False)
                    
                    # Process clients in batches to limit memory usage
                    all_client_updates = []
                    total_training_time = 0
                    batch_stats = []
                    
                    print(f"Processing {clients_per_round} clients in batches of {self.batch_size}")
                    print(f"Total batches: {(len(selected_indices) + self.batch_size - 1) // self.batch_size}")
                    
                    batch_count = 0
                    for batch_start in range(0, len(selected_indices), self.batch_size):
                        # Check if round timer timed out
                        if round_timer.timed_out:
                            raise TimeoutError(f"Round exceeded time limit ({round_timeout}s)")
                            
                        batch_count += 1
                        batch_end = min(batch_start + self.batch_size, len(selected_indices))
                        batch_indices = selected_indices[batch_start:batch_end]
                        
                        print(f"\nBatch {batch_count}/{(len(selected_indices) + self.batch_size - 1) // self.batch_size}: "
                            f"clients {batch_start}-{batch_end-1}")
                        batch_start_time = time.time()
                        
                        # Use a thread-based timeout for each batch
                        batch_timer = ThreadingTimeout(batch_timeout)
                        batch_timer.start()
                        
                        try:
                            # Load clients for this batch
                            print("  Loading clients...")
                            self.active_clients = self._load_client_batch(batch_indices)
                            last_activity = time.time()
                            
                            # Check for batch timeout
                            if batch_timer.timed_out:
                                raise TimeoutError(f"Batch loading exceeded time limit ({batch_timeout}s)")
                            
                            # Verify client datasets
                            print("  Validating client datasets...")
                            valid_client_indices = []
                            valid_clients = []
                            
                            for i, client in enumerate(self.active_clients):
                                # Check if batch timer timed out
                                if batch_timer.timed_out:
                                    raise TimeoutError(f"Batch validation exceeded time limit ({batch_timeout}s)")
                                    
                                # Check if client has a valid dataset
                                dataset_valid = False
                                try:
                                    if hasattr(client, 'dataset') and client.dataset is not None:
                                        if len(client.dataset) > 0:
                                            dataset_valid = True
                                        else:
                                            print(f"  Warning: Client {batch_indices[i]} has empty dataset, skipping")
                                    else:
                                        print(f"  Warning: Client {batch_indices[i]} has no dataset, skipping")
                                except Exception as e:
                                    print(f"  Error checking dataset for client {batch_indices[i]}: {e}")
                                
                                if dataset_valid:
                                    valid_client_indices.append(batch_indices[i])
                                    valid_clients.append(client)
                            
                            # Skip batch if no valid clients
                            if not valid_clients:
                                print("  No valid clients in this batch, skipping")
                                batch_timer.cancel()
                                continue
                            
                            print(f"  Creating temporary CFLAG-BD with {len(valid_clients)} valid clients...")
                            # Create temporary CFLAG-BD instance for this batch of valid clients
                            temp_cflagbd = CFLAGBD(
                                global_model=self.global_model,
                                clients=valid_clients,
                                test_loader=self.test_loader,
                                device=self.device,
                                num_clusters=self.num_clusters,
                                dedup_threshold=self.dedup_threshold,
                                use_blockchain=self.use_blockchain
                            )
                            last_activity = time.time()
                            
                            # Check for batch timeout
                            if batch_timer.timed_out:
                                raise TimeoutError(f"Batch setup exceeded time limit ({batch_timeout}s)")
                            
                            # Copy over current cluster assignments
                            for i, idx in enumerate(valid_client_indices):
                                temp_cflagbd.client_clusters[i] = self.client_clusters[idx]
                            
                            temp_cflagbd.cluster_models = [copy.deepcopy(model) for model in self.cluster_models]
                            
                            # Train this batch of clients
                            print("  Training clients...")
                            temp_stats = temp_cflagbd.run_round(client_fraction=1.0, epochs=epochs)
                            last_activity = time.time()
                            
                            # Check for batch timeout
                            if batch_timer.timed_out:
                                raise TimeoutError(f"Batch training exceeded time limit ({batch_timeout}s)")
                            
                            print("  Updating cluster assignments and models...")
                            # Store updated cluster assignments
                            for i, idx in enumerate(valid_client_indices):
                                self.client_clusters[idx] = temp_cflagbd.client_clusters[i]
                            
                            # Collect client updates
                            if hasattr(temp_cflagbd, 'client_updates'):
                                all_client_updates.extend(temp_cflagbd.client_updates)
                            
                            # Update metrics
                            total_training_time += temp_stats['training_time']
                            
                            # Update cluster models if needed
                            self.cluster_models = temp_cflagbd.cluster_models
                            
                            batch_stats.append(temp_stats)
                            
                            # Offload clients to free memory
                            print("  Offloading clients to free memory...")
                            self._offload_clients(valid_clients, valid_client_indices)
                            
                            # Free memory
                            self.active_clients = []
                            del temp_cflagbd
                            gc.collect()
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                            
                            batch_time = time.time() - batch_start_time
                            print(f"  Batch complete. Time: {batch_time:.2f}s, "
                                f"Clients processed: {len(valid_clients)}")
                            print(f"  Current memory usage:")
                            memory_stats()
                            
                            # Cancel batch timer since we completed successfully
                            batch_timer.cancel()
                            
                        except TimeoutError as e:
                            # Make sure to cancel the timer
                            batch_timer.cancel()
                            
                            print(f"\n⚠️ TIMEOUT: Batch {batch_count} processing exceeded time limit ({batch_timeout}s)")
                            print(f"  Error: {e}")
                            print("  Cleaning up and continuing with next batch...")
                            # Free memory
                            self.active_clients = []
                            gc.collect()
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                            
                        except Exception as e:
                            # Make sure to cancel the timer
                            batch_timer.cancel()
                            
                            print(f"\n⚠️ ERROR processing batch {batch_count}: {e}")
                            print("  Traceback:")
                            import traceback
                            traceback.print_exc()
                            print("  Continuing with next batch...")
                            # Free memory
                            self.active_clients = []
                            gc.collect()
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                        # Save progress after each batch
                        if batch_stats:
                            try:
                                temp_progress = {
                                    'completed_rounds': round_idx,
                                    'completed_batches': batch_count,
                                    'metrics': self.metrics,
                                    'round_stats': round_stats
                                }
                                with open(progress_file, 'w') as f:
                                    json.dump(temp_progress, f)
                            except Exception as e:
                                print(f"Warning: Could not save progress: {e}")
                    
                    # Cancel round timer since we completed successfully
                    round_timer.cancel()
                    
                    # Skip round if no batches were processed successfully
                    if not batch_stats:
                        print(f"\n⚠️ Round {round_idx+1}/{num_rounds} failed: No batches processed successfully")
                        continue
                        
                    # Create unified stats for this round
                    stats = {
                        'round': round_idx + 1,
                        'accuracy': np.mean([s['accuracy'] for s in batch_stats]),
                        'training_time': total_training_time,
                        'communication_cost': sum(s.get('communication_cost', 0) for s in batch_stats),
                        'round_time': time.time() - round_start
                    }
                    
                    if all('cluster_accuracies' in s for s in batch_stats):
                        try:
                            stats['cluster_accuracies'] = [
                                np.mean([s['cluster_accuracies'][i] for s in batch_stats if i < len(s['cluster_accuracies'])])
                                for i in range(self.num_clusters)
                            ]
                        except Exception as e:
                            print(f"Error calculating cluster accuracies: {e}")
                            stats['cluster_accuracies'] = [0.0] * self.num_clusters
                            
                    if all('compression_ratio' in s for s in batch_stats):
                        stats['compression_ratio'] = np.mean([s.get('compression_ratio', 0) for s in batch_stats])
                        
                    round_stats.append(stats)
                    
                    # Update global metrics
                    self.metrics['global_accuracy'].append(stats['accuracy'])
                    if 'cluster_accuracies' in stats:
                        self.metrics['cluster_accuracies'].append(stats['cluster_accuracies'])
                    self.metrics['training_time'].append(stats['training_time'])
                    self.metrics['communication_cost'].append(stats['communication_cost'])
                    if 'compression_ratio' in stats:
                        self.metrics['storage_savings'].append(stats['compression_ratio'] * 100)
                    
                    # Track successful completion
                    completed_rounds = round_idx + 1
                    
                    # Save progress file
                    progress_data = {
                        'completed_rounds': completed_rounds,
                        'metrics': self.metrics,
                        'round_stats': round_stats
                    }
                    with open(progress_file, 'w') as f:
                        json.dump(progress_data, f)
                    
                    print(f"\n{'='*50}")
                    print(f"ROUND {round_idx+1}/{num_rounds} - COMPLETED")
                    print(f"Accuracy: {stats['accuracy']:.4f}, Time: {stats['round_time']:.2f}s")
                    print(f"{'='*50}")
                    
                except TimeoutError as e:
                    print(f"\n⚠️ TIMEOUT: Round {round_idx+1} exceeded time limit ({round_timeout}s)")
                    print(f"Error: {e}")
                    print("Saving progress and continuing with next round...")
                    
                    # Save partial progress
                    progress_data = {
                        'completed_rounds': round_idx,  # This round didn't complete
                        'metrics': self.metrics,
                        'round_stats': round_stats
                    }
                    with open(progress_file, 'w') as f:
                        json.dump(progress_data, f)
                    
                except Exception as e:
                    print(f"\n⚠️ ERROR in round {round_idx+1}: {e}")
                    print("Traceback:")
                    import traceback
                    traceback.print_exc()
                    print("Saving progress and continuing with next round...")
                    
                    # Save partial progress
                    progress_data = {
                        'completed_rounds': round_idx,  # This round didn't complete
                        'metrics': self.metrics,
                        'round_stats': round_stats
                    }
                    with open(progress_file, 'w') as f:
                        json.dump(progress_data, f)
            
            # Calculate final results
            total_time = time.time() - start_time
            
            # Handle the case where no rounds were completed successfully
            if not self.metrics['global_accuracy']:
                print("Warning: No rounds completed successfully")
                results = {
                    'final_accuracy': 0.0,
                    'accuracy_history': [],
                    'training_time': 0.0,
                    'communication_cost': 0.0,
                    'storage_savings': 0.0,
                    'total_time': total_time,
                    'round_stats': []
                }
            else:
                results = {
                    'final_accuracy': self.metrics['global_accuracy'][-1],
                    'accuracy_history': self.metrics['global_accuracy'],
                    'training_time': sum(self.metrics['training_time']),
                    'communication_cost': sum(self.metrics['communication_cost']),
                    'storage_savings': np.mean(self.metrics['storage_savings']) if self.metrics['storage_savings'] else 0,
                    'total_time': total_time,
                    'round_stats': round_stats,
                    'completed_rounds': completed_rounds
                }
            
            print(f"\n{'='*50}")
            print(f"TRAINING COMPLETE")
            print(f"Total time: {total_time:.2f}s")
            print(f"Rounds completed: {completed_rounds}/{num_rounds}")
            if self.metrics['global_accuracy']:
                print(f"Final accuracy: {self.metrics['global_accuracy'][-1]:.4f}")
            print(f"{'='*50}")
            
            print("Final memory usage:")
            memory_stats()
            
            return results
            
        except Exception as e:
            print(f"\n{'!'*50}")
            print(f"CRITICAL ERROR: {e}")
            print("Traceback:")
            import traceback
            traceback.print_exc()
            print(f"{'!'*50}")
            
            # Return partial results if available
            if self.metrics['global_accuracy']:
                return {
                    'final_accuracy': self.metrics['global_accuracy'][-1],
                    'accuracy_history': self.metrics['global_accuracy'],
                    'training_time': sum(self.metrics['training_time']),
                    'communication_cost': sum(self.metrics['communication_cost']),
                    'storage_savings': np.mean(self.metrics['storage_savings']) if self.metrics['storage_savings'] else 0,
                    'total_time': time.time() - start_time,
                    'round_stats': round_stats,
                    'error': str(e)
                }
            else:
                return {
                    'error': str(e),
                    'total_time': time.time() - start_time
                }
        
        finally:
            # Always stop memory monitor
            memory_monitor.stop()