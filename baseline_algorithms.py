import torch
import torch.nn as nn
import numpy as np
import copy
import time
from collections import defaultdict
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import torch.nn.functional as F

class FedAvg:
    """
    Implementation of FedAvg algorithm (McMahan et al., 2017)
    """
    def __init__(self, global_model, clients, test_loader, device):
        """
        Initialize FedAvg algorithm
        
        Args:
            global_model: Global model to be trained
            clients: List of client objects
            test_loader: DataLoader for test dataset
            device: Device to run computations on
        """
        self.global_model = global_model
        self.clients = clients
        self.test_loader = test_loader
        self.device = device
        self.metrics = {
            'global_accuracy': [],
            'client_accuracies': [],
            'training_time': [],
            'communication_cost': []
        }
        
    def train_clients(self, client_indices=None, epochs=5):
        """
        Train selected clients
        
        Args:
            client_indices: Indices of clients to train (if None, train all)
            epochs: Number of local training epochs
            
        Returns:
            client_updates: List of model updates from clients
            training_stats: Dictionary of training statistics
        """
        if client_indices is None:
            client_indices = list(range(len(self.clients)))
            
        client_updates = []
        training_times = []
        
        # Update clients with global model
        for idx in client_indices:
            self.clients[idx].set_model(copy.deepcopy(self.global_model))
        
        # Train selected clients
        for idx in client_indices:
            start_time = time.time()
            update = self.clients[idx].train(epochs=epochs)
            training_time = time.time() - start_time
            
            client_updates.append(update)
            training_times.append(training_time)
            
        # Compute statistics
        training_stats = {
            'mean_time': np.mean(training_times),
            'total_time': np.sum(training_times),
            'client_accuracies': [self.clients[idx].round_accuracies[-1] for idx in client_indices 
                                  if self.clients[idx].round_accuracies]
        }
        
        return client_updates, training_stats
        
    def aggregate_updates(self, client_updates, weights=None):
        """
        Aggregate client updates using weighted averaging
        
        Args:
            client_updates: List of client model updates
            weights: List of weights for each update (if None, use equal weights)
            
        Returns:
            global_update: Aggregated global model update
        """
        if not client_updates:
            return None
            
        # Use equal weights if not provided
        if weights is None:
            weights = [1.0 / len(client_updates)] * len(client_updates)
            
        # Ensure weights sum to 1
        weights = np.array(weights) / np.sum(weights)
        
        # Initialize global update with zeros like first client update
        global_update = {}
        for key in client_updates[0].keys():
            global_update[key] = torch.zeros_like(client_updates[0][key])
            
        # Weighted average of parameters
        for i, update in enumerate(client_updates):
            for key in update.keys():
                global_update[key] += update[key] * weights[i]
                
        return global_update
        
    def update_global_model(self, global_update):
        """
        Update global model with aggregated update
        
        Args:
            global_update: Aggregated model update
        """
        if global_update is not None:
            self.global_model.load_state_dict(global_update)
        
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
        Run one round of federated learning
        
        Args:
            client_fraction: Fraction of clients to select for training
            epochs: Number of local training epochs
            
        Returns:
            round_stats: Dictionary of round statistics
        """
        # Select clients
        num_clients = max(1, int(client_fraction * len(self.clients)))
        client_indices = np.random.choice(len(self.clients), num_clients, replace=False)
        
        # Train selected clients
        client_updates, training_stats = self.train_clients(client_indices, epochs)
        
        # Track communication cost (size of parameters)
        if client_updates:
            param_size = sum(p.numel() * 4 for p in client_updates[0].values())  # 4 bytes per float32
            communication_cost = param_size * len(client_indices)
        else:
            communication_cost = 0
        
        # Aggregate updates
        global_update = self.aggregate_updates(client_updates)
        
        # Update global model
        self.update_global_model(global_update)
        
        # Evaluate global model
        accuracy = self.evaluate_global_model()
        
        # Track metrics
        self.metrics['global_accuracy'].append(accuracy)
        self.metrics['client_accuracies'].append(training_stats['client_accuracies'])
        self.metrics['training_time'].append(training_stats['total_time'])
        self.metrics['communication_cost'].append(communication_cost)
        
        round_stats = {
            'accuracy': accuracy,
            'training_time': training_stats['total_time'],
            'communication_cost': communication_cost,
            'num_clients': len(client_indices)
        }
        
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
        
        results = {
            'final_accuracy': self.metrics['global_accuracy'][-1],
            'accuracy_history': self.metrics['global_accuracy'],
            'training_time': sum(self.metrics['training_time']),
            'communication_cost': sum(self.metrics['communication_cost']),
            'total_time': total_time,
            'round_stats': round_stats
        }
        
        return results


class FedProx(FedAvg):
    """
    Implementation of FedProx algorithm (Li et al., 2020)
    FedProx adds a proximal term to client optimization to limit client drift
    """
    def __init__(self, global_model, clients, test_loader, device, mu=0.01):
        """
        Initialize FedProx algorithm
        
        Args:
            global_model: Global model to be trained
            clients: List of client objects
            test_loader: DataLoader for test dataset
            device: Device to run computations on
            mu: Proximal term weight
        """
        super().__init__(global_model, clients, test_loader, device)
        self.mu = mu
        
    def train_clients(self, client_indices=None, epochs=5):
        """
        Train selected clients with proximal term
        
        Args:
            client_indices: Indices of clients to train (if None, train all)
            epochs: Number of local training epochs
            
        Returns:
            client_updates: List of model updates from clients
            training_stats: Dictionary of training statistics
        """
        if client_indices is None:
            client_indices = list(range(len(self.clients)))
            
        client_updates = []
        training_times = []
        
        # Keep copy of global model for proximal term
        global_params = copy.deepcopy(self.global_model.state_dict())
        
        # Update clients with global model
        for idx in client_indices:
            self.clients[idx].set_model(copy.deepcopy(self.global_model))
        
        # Train selected clients
        for idx in client_indices:
            client = self.clients[idx]
            
            start_time = time.time()
            
            # Custom training loop with proximal term
            train_loader = client.create_data_loader()
            client.model.train()
            
            epoch_losses = []
            for epoch in range(epochs):
                running_loss = 0.0
                samples = 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Zero gradients
                    client.optimizer.zero_grad()
                    
                    # Forward pass
                    output = client.model(data)
                    
                    # Calculate task loss
                    if isinstance(output, torch.Tensor) and output.shape[1] > 1 and len(target.shape) == 1:
                        task_loss = nn.CrossEntropyLoss()(output, target)
                    elif len(target.shape) > 1 and target.shape[1] > 1:
                        task_loss = nn.BCELoss()(output, target)
                    else:
                        task_loss = nn.NLLLoss()(output, target)
                    
                    # Calculate proximal term
                    proximal_term = 0
                    for w, w_t in zip(client.model.parameters(), self.global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)
                    
                    # Combined loss with proximal term
                    loss = task_loss + (self.mu / 2) * proximal_term
                    
                    # Backward pass and optimization
                    loss.backward()
                    client.optimizer.step()
                    
                    # Track statistics
                    running_loss += loss.item() * data.size(0)
                    samples += data.size(0)
                
                # Compute average loss for epoch
                epoch_loss = running_loss / samples
                epoch_losses.append(epoch_loss)
            
            # Record training time
            training_time = time.time() - start_time
            client.total_training_time += training_time
            
            # Calculate average loss for this round
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
            client.round_losses.append(avg_loss)
            
            # Evaluate client model
            accuracy = client.evaluate()
            
            # Get model update
            update = client.get_model_update()
            client_updates.append(update)
            training_times.append(training_time)
            
        # Compute statistics
        training_stats = {
            'mean_time': np.mean(training_times),
            'total_time': np.sum(training_times),
            'client_accuracies': [self.clients[idx].round_accuracies[-1] for idx in client_indices 
                                  if self.clients[idx].round_accuracies]
        }
        
        return client_updates, training_stats


class IFCA(FedAvg):
    """
    Implementation of IFCA algorithm (Ghosh et al., 2020)
    Iterative Federated Clustering Algorithm
    """
    def __init__(self, global_model, clients, test_loader, device, num_clusters=3):
        """
        Initialize IFCA algorithm
        
        Args:
            global_model: Global model architecture (will be replicated for clusters)
            clients: List of client objects
            test_loader: DataLoader for test dataset
            device: Device to run computations on
            num_clusters: Number of clusters to form
        """
        super().__init__(global_model, clients, test_loader, device)
        self.num_clusters = num_clusters
        
        # Create model for each cluster
        self.cluster_models = [copy.deepcopy(global_model) for _ in range(num_clusters)]
        
        # Initialize client cluster assignments randomly
        self.client_clusters = np.random.randint(0, num_clusters, size=len(clients))
        
        # Metrics specific to clustering
        self.metrics.update({
            'cluster_sizes': [],
            'cluster_accuracies': [],
            'clustering_quality': []
        })
        
    def assign_clients_to_clusters(self):
        """
        Assign each client to the best performing cluster model
        
        Returns:
            cluster_assignments: Array of cluster assignments for each client
        """
        cluster_assignments = np.zeros(len(self.clients), dtype=int)
        
        for i, client in enumerate(self.clients):
            best_loss = float('inf')
            best_cluster = 0
            
            # Evaluate client data on each cluster model
            for cluster_idx, model in enumerate(self.cluster_models):
                # Create temporary client with this cluster's model
                temp_client = copy.deepcopy(client)
                temp_client.set_model(copy.deepcopy(model))
                
                # Evaluate without training
                eval_loader = temp_client.create_data_loader()
                model.eval()
                
                # Compute loss on client data
                running_loss = 0.0
                samples = 0
                
                with torch.no_grad():
                    for data, target in eval_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = model(data)
                        
                        # Calculate loss based on output type
                        if isinstance(output, torch.Tensor) and output.shape[1] > 1 and len(target.shape) == 1:
                            loss = nn.CrossEntropyLoss()(output, target)
                        elif len(target.shape) > 1 and target.shape[1] > 1:
                            loss = nn.BCELoss()(output, target)
                        else:
                            loss = nn.NLLLoss()(output, target)
                            
                        running_loss += loss.item() * data.size(0)
                        samples += data.size(0)
                
                avg_loss = running_loss / samples if samples > 0 else float('inf')
                
                # Update best cluster if loss is lower
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_cluster = cluster_idx
            
            cluster_assignments[i] = best_cluster
            
        return cluster_assignments
        
    def run_round(self, client_fraction=1.0, epochs=5):
        """
        Run one round of federated learning
        
        Args:
            client_fraction: Fraction of clients to select for training
            epochs: Number of local training epochs
            
        Returns:
            round_stats: Dictionary of round statistics
        """
        # Select clients
        num_clients = max(1, int(client_fraction * len(self.clients)))
        client_indices = np.random.choice(len(self.clients), num_clients, replace=False)
        
        # Train selected clients
        client_updates, training_stats = self.train_clients(client_indices, epochs)
        
        # Track communication cost (size of parameters)
        if client_updates:
            param_size = sum(p.numel() * 4 for p in client_updates[0].values())  # 4 bytes per float32
            communication_cost = param_size * len(client_indices)
        else:
            communication_cost = 0
        
        # Aggregate updates
        global_update = self.aggregate_updates(client_updates)
        
        # Update global model
        self.update_global_model(global_update)
        
        # Evaluate global model
        accuracy = self.evaluate_global_model()
        
        # Track metrics
        self.metrics['global_accuracy'].append(accuracy)
        self.metrics['client_accuracies'].append(training_stats['client_accuracies'])
        self.metrics['training_time'].append(training_stats['total_time'])
        self.metrics['communication_cost'].append(communication_cost)
        
        round_stats = {
            'accuracy': accuracy,
            'training_time': training_stats['total_time'],
            'communication_cost': communication_cost,
            'num_clients': len(client_indices)
        }
        
        return round_stats

class LayerCFL(FedAvg):
    """
    Implementation of LayerCFL algorithm (Luo et al., 2021)
    Layer-wise Clustered Federated Learning
    """
    def __init__(self, global_model, clients, test_loader, device, num_clusters=3):
        """
        Initialize LayerCFL algorithm
        
        Args:
            global_model: Global model to be trained
            clients: List of client objects
            test_loader: DataLoader for test dataset
            device: Device to run computations on
            num_clusters: Number of clusters to form
        """
        super().__init__(global_model, clients, test_loader, device)
        self.num_clusters = num_clusters
        
        # Get layer names from model
        self.layer_names = list(global_model.state_dict().keys())
        
        # Initialize layer-wise clustering
        self.layer_clusters = {layer: np.random.randint(0, num_clusters, size=len(clients)) 
                               for layer in self.layer_names}
        
        # Create layer-wise model clusters
        self.layer_cluster_models = {}
        for layer in self.layer_names:
            self.layer_cluster_models[layer] = []
            for _ in range(num_clusters):
                # Create a copy of the layer parameters
                layer_params = copy.deepcopy(global_model.state_dict()[layer])
                self.layer_cluster_models[layer].append(layer_params)
        
        # Metrics specific to layer-wise clustering
        self.metrics.update({
            'layer_clustering_quality': [],
            'layer_cluster_sizes': []
        })
        
    def cosine_similarity_matrix(self, layer_updates):
        """
        Compute cosine similarity matrix between client updates for a layer
        
        Args:
            layer_updates: List of client updates for specific layer
            
        Returns:
            similarity_matrix: Matrix of cosine similarities
        """
        num_clients = len(layer_updates)
        similarity_matrix = torch.zeros((num_clients, num_clients))
        
        # Flatten updates for cosine similarity
        flattened_updates = []
        for update in layer_updates:
            flat_update = update.flatten()
            flattened_updates.append(flat_update)
        
        # Compute pairwise cosine similarities
        for i in range(num_clients):
            for j in range(i, num_clients):
                if i == j:
                    similarity_matrix[i, j] = 1.0  # Self-similarity
                else:
                    cos_sim = F.cosine_similarity(flattened_updates[i].unsqueeze(0), 
                                                 flattened_updates[j].unsqueeze(0))[0].item()
                    similarity_matrix[i, j] = cos_sim
                    similarity_matrix[j, i] = cos_sim  # Symmetric
        
        return similarity_matrix
        
    def cluster_layer_updates(self, layer_updates, client_indices):
        """
        Cluster clients based on their layer updates
        
        Args:
            layer_updates: List of client updates for specific layer
            client_indices: Indices of clients that provided updates
            
        Returns:
            layer_cluster_assignments: Cluster assignments for each client
        """
        # Compute similarity matrix
        similarity_matrix = self.cosine_similarity_matrix(layer_updates)
        
        # Convert to distance matrix (1 - similarity)
        distance_matrix = 1 - similarity_matrix
        
        # Apply hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=min(self.num_clusters, len(client_indices)),
            affinity='precomputed',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(distance_matrix.cpu().numpy())
        
        # Create full assignment array
        layer_cluster_assignments = np.zeros(len(self.clients), dtype=int)
        for i, idx in enumerate(client_indices):
            layer_cluster_assignments[idx] = cluster_labels[i]
            
        return layer_cluster_assignments
        
    def aggregate_layer_updates(self, layer_updates, cluster_assignments, client_indices):
        """
        Aggregate layer updates by cluster
        
        Args:
            layer_updates: List of client updates for specific layer
            cluster_assignments: Cluster assignments for each client
            client_indices: Indices of clients that provided updates
            
        Returns:
            cluster_aggregates: Aggregated updates for each cluster
        """
        # Group updates by cluster
        cluster_groups = defaultdict(list)
        for i, idx in enumerate(client_indices):
            cluster_idx = cluster_assignments[idx]
            cluster_groups[cluster_idx].append(layer_updates[i])
        
        # Aggregate updates for each cluster
        cluster_aggregates = {}
        for cluster_idx, updates in cluster_groups.items():
            if updates:
                # Average updates in this cluster
                cluster_agg = torch.stack(updates).mean(dim=0)
                cluster_aggregates[cluster_idx] = cluster_agg
                
        return cluster_aggregates
        
    def run_round(self, client_fraction=1.0, epochs=5):
        """
        Run one round of LayerCFL
        
        Args:
            client_fraction: Fraction of clients to select for training
            epochs: Number of local training epochs
            
        Returns:
            round_stats: Dictionary of round statistics
        """
        # Select clients
        num_clients = max(1, int(client_fraction * len(self.clients)))
        client_indices = np.random.choice(len(self.clients), num_clients, replace=False)
        
        # Train selected clients
        client_updates, training_stats = self.train_clients(client_indices, epochs)
        
        # Track communication cost (size of parameters)
        if client_updates:
            param_size = sum(p.numel() * 4 for p in client_updates[0].values())  # 4 bytes per float32
            communication_cost = param_size * len(client_indices)
        else:
            communication_cost = 0
            
        # Process each layer separately
        layer_cluster_assignments = {}
        layer_cluster_sizes = {}
        
        for layer_name in self.layer_names:
            # Extract layer updates from all clients
            layer_updates = [update[layer_name] for update in client_updates]
            
            # Cluster clients based on layer updates
            layer_cluster_assignments[layer_name] = self.cluster_layer_updates(
                layer_updates, client_indices
            )
            
            # Aggregate updates by cluster
            cluster_aggregates = self.aggregate_layer_updates(
                layer_updates, layer_cluster_assignments[layer_name], client_indices
            )
            
            # Update layer-wise cluster models
            for cluster_idx, aggregate in cluster_aggregates.items():
                self.layer_cluster_models[layer_name][cluster_idx] = aggregate
                
            # Track cluster sizes for this layer
            layer_cluster_sizes[layer_name] = [
                np.sum(layer_cluster_assignments[layer_name] == i) 
                for i in range(self.num_clusters)
            ]
        
        # Create global model by assigning each client to best layer-wise clusters
        global_update = self.global_model.state_dict()
        
        # For each layer, set parameters to the most common cluster's aggregate
        for layer_name in self.layer_names:
            # Find most populated cluster
            cluster_counts = np.bincount(
                layer_cluster_assignments[layer_name][client_indices], 
                minlength=self.num_clusters
            )
            most_common_cluster = np.argmax(cluster_counts)
            
            # Use most common cluster's parameters
            global_update[layer_name] = self.layer_cluster_models[layer_name][most_common_cluster]
            
        # Update global model
        self.global_model.load_state_dict(global_update)
        
        # Evaluate global model
        accuracy = self.evaluate_global_model()
        
        # Track metrics
        self.metrics['global_accuracy'].append(accuracy)
        self.metrics['client_accuracies'].append(training_stats['client_accuracies'])
        self.metrics['training_time'].append(training_stats['total_time'])
        self.metrics['communication_cost'].append(communication_cost)
        self.metrics['layer_cluster_sizes'].append(layer_cluster_sizes)
        
        round_stats = {
            'accuracy': accuracy,
            'training_time': training_stats['total_time'],
            'communication_cost': communication_cost,
            'num_clients': len(client_indices),
            'layer_cluster_sizes': layer_cluster_sizes
        }
        
        return round_stats
class CQFL(FedAvg):
    """
    Implementation of CQFL algorithm (Clustered Quantized Federated Learning)
    Vahidian et al., 2021
    """
    def __init__(self, global_model, clients, test_loader, device, 
                 num_clusters=3, quantization_bits=8):
        """
        Initialize CQFL algorithm
        
        Args:
            global_model: Global model to be trained
            clients: List of client objects
            test_loader: DataLoader for test dataset
            device: Device to run computations on
            num_clusters: Number of clusters to form
            quantization_bits: Number of bits for quantization
        """
        super().__init__(global_model, clients, test_loader, device)
        self.num_clusters = num_clusters
        self.quantization_bits = quantization_bits
        
        # Initialize cluster models
        self.cluster_models = [copy.deepcopy(global_model) for _ in range(num_clusters)]
        
        # Initialize client cluster assignments
        self.client_clusters = np.random.randint(0, num_clusters, size=len(clients))
        
        # Metrics specific to CQFL
        self.metrics.update({
            'cluster_sizes': [],
            'cluster_accuracies': [],
            'quantization_compression': []
        })
        
    def quantize_update(self, update, bits=8):
        """
        Quantize model update to reduce communication cost
        
        Args:
            update: Model update (state dict)
            bits: Number of bits for quantization
            
        Returns:
            quantized_update: Quantized model update
            compression_ratio: Ratio of size reduction
        """
        quantized_update = {}
        original_size = 0
        quantized_size = 0
        
        for key, param in update.items():
            # Flatten parameter
            flat_param = param.flatten()
            original_size += flat_param.numel() * 32  # Original bits (float32)
            
            # Find min and max values
            min_val = flat_param.min().item()
            max_val = flat_param.max().item()
            
            # Scale to [0, 2^bits - 1]
            scale = (max_val - min_val) / (2**bits - 1) if max_val > min_val else 1.0
            
            # Quantize values
            quantized_values = torch.round((flat_param - min_val) / scale).to(torch.int)
            
            # Store quantized parameters and metadata
            quantized_update[key] = {
                'values': quantized_values,
                'min_val': min_val,
                'scale': scale,
                'shape': param.shape
            }
            
            # Calculate quantized size
            quantized_size += quantized_values.numel() * bits  # Quantized values
            quantized_size += 64  # min_val and scale (float32 each)
            
        # Calculate compression ratio
        compression_ratio = 1.0 - (quantized_size / original_size)
        
        return quantized_update, compression_ratio
        
    def dequantize_update(self, quantized_update):
        """
        Dequantize model update
        
        Args:
            quantized_update: Quantized model update
            
        Returns:
            update: Dequantized model update
        """
        update = {}
        
        for key, q_param in quantized_update.items():
            # Dequantize values
            dequantized = q_param['values'].float() * q_param['scale'] + q_param['min_val']
            
            # Reshape to original shape
            update[key] = dequantized.reshape(q_param['shape'])
            
        return update
        
    def cluster_clients(self, client_models):
        """
        Cluster clients based on model similarity
        
        Args:
            client_models: List of client models
            
        Returns:
            cluster_assignments: Cluster assignments for each client
        """
        # Extract flattened parameters from each model
        flattened_params = []
        for model in client_models:
            params = []
            for param in model.parameters():
                params.append(param.detach().cpu().flatten())
            flattened_params.append(torch.cat(params))
            
        # Stack parameters for clustering
        param_matrix = torch.stack(flattened_params)
        
        # Compute pairwise cosine similarity
        similarity_matrix = torch.zeros((len(client_models), len(client_models)))
        for i in range(len(client_models)):
            for j in range(i, len(client_models)):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    cos_sim = F.cosine_similarity(
                        param_matrix[i].unsqueeze(0), 
                        param_matrix[j].unsqueeze(0)
                    )[0].item()
                    similarity_matrix[i, j] = cos_sim
                    similarity_matrix[j, i] = cos_sim
                    
        # Convert to distance matrix
        distance_matrix = 1 - similarity_matrix
        
        # Apply clustering
        clustering = AgglomerativeClustering(
            n_clusters=min(self.num_clusters, len(client_models)),
            affinity='precomputed',
            linkage='average'
        )
        
        cluster_assignments = clustering.fit_predict(distance_matrix.numpy())
        
        return cluster_assignments
        
    def run_round(self, client_fraction=1.0, epochs=5):
        """
        Run one round of CQFL
        
        Args:
            client_fraction: Fraction of clients to select for training
            epochs: Number of local training epochs
            
        Returns:
            round_stats: Dictionary of round statistics
        """
        # Select clients
        num_clients = max(1, int(client_fraction * len(self.clients)))
        client_indices = np.random.choice(len(self.clients), num_clients, replace=False)
        
        # Initialize clients with their cluster models
        for idx in client_indices:
            cluster_idx = self.client_clusters[idx]
            self.clients[idx].set_model(copy.deepcopy(self.cluster_models[cluster_idx]))
            
        # Train clients
        client_models = []
        training_times = []
        quantized_updates = []
        compression_ratios = []
        
        for idx in client_indices:
            # Train client
            start_time = time.time()
            self.clients[idx].train(epochs=epochs)
            training_time = time.time() - start_time
            training_times.append(training_time)
            
            # Store client model
            client_models.append(copy.deepcopy(self.clients[idx].model))
            
            # Quantize client update
            client_update = self.clients[idx].get_model_update()
            quantized_update, compression_ratio = self.quantize_update(
                client_update, self.quantization_bits
            )
            
            quantized_updates.append(quantized_update)
            compression_ratios.append(compression_ratio)
            
        # Re-cluster clients based on model similarity
        if len(client_models) >= self.num_clusters:
            new_clusters = self.cluster_clients(client_models)
            
            # Update cluster assignments for selected clients
            for i, idx in enumerate(client_indices):
                self.client_clusters[idx] = new_clusters[i]
                
        # Track cluster sizes
        cluster_sizes = [np.sum(self.client_clusters == i) for i in range(self.num_clusters)]
        self.metrics['cluster_sizes'].append(cluster_sizes)
        
        # Group updates by cluster
        cluster_updates = defaultdict(list)
        for i, idx in enumerate(client_indices):
            cluster_idx = self.client_clusters[idx]
            dequantized_update = self.dequantize_update(quantized_updates[i])
            cluster_updates[cluster_idx].append(dequantized_update)
            
        # Aggregate updates for each cluster
        for cluster_idx, updates in cluster_updates.items():
            if updates:
                cluster_update = self.aggregate_updates(updates)
                self.cluster_models[cluster_idx].load_state_dict(cluster_update)
                
        # Set global model to best performing cluster model
        cluster_accuracies = []
        for cluster_idx, model in enumerate(self.cluster_models):
            # Evaluate cluster model
            self.global_model = copy.deepcopy(model)
            accuracy = self.evaluate_global_model()
            cluster_accuracies.append(accuracy)
            
        # Use best performing cluster model as global model
        best_cluster = np.argmax(cluster_accuracies)
        self.global_model = copy.deepcopy(self.cluster_models[best_cluster])
        best_accuracy = cluster_accuracies[best_cluster]
        
        # Calculate communication cost with quantization
        original_model_size = sum(p.numel() * 32 for p in self.global_model.parameters())  # bits
        avg_compression = np.mean(compression_ratios)
        communication_cost = original_model_size * len(client_indices) * (1 - avg_compression) / 8  # bytes
        
        # Track metrics
        self.metrics['global_accuracy'].append(best_accuracy)
        self.metrics['cluster_accuracies'].append(cluster_accuracies)
        self.metrics['training_time'].append(sum(training_times))
        self.metrics['communication_cost'].append(communication_cost)
        self.metrics['quantization_compression'].append(avg_compression)
        
        round_stats = {
            'accuracy': best_accuracy,
            'training_time': sum(training_times),
            'communication_cost': communication_cost,
            'num_clients': len(client_indices),
            'cluster_sizes': cluster_sizes,
            'cluster_accuracies': cluster_accuracies,
            'best_cluster': best_cluster,
            'compression_ratio': avg_compression
        }
        
        return round_stats