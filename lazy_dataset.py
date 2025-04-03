from torch.utils.data import Dataset

class LazyDatasetPartition(Dataset):
    """
    Memory-efficient dataset partition that doesn't store all data in memory
    Only accesses the base dataset when an item is actually requested
    """
    def __init__(self, base_dataset, indices):
        """
        Initialize lazy dataset partition
        
        Args:
            base_dataset: Original dataset to partition
            indices: Indices to include in this partition
        """
        self.base_dataset = base_dataset
        self.indices = indices
        
    def __getitem__(self, idx):
        """Get item from base dataset using stored indices"""
        if idx >= len(self.indices):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.indices)}")
        return self.base_dataset[self.indices[idx]]
        
    def __len__(self):
        """Get length of this partition"""
        return len(self.indices)


def create_memory_efficient_clients(base_dataset, num_clients, device, client_class, 
                                   batch_size=32, learning_rate=0.01, non_iid=False):
    """
    Create memory-efficient clients with lazy dataset partitions
    
    Args:
        base_dataset: Original full dataset
        num_clients: Number of clients to create
        device: Device to use for computation
        client_class: Class to use for creating clients (e.g., EnhancedClient)
        batch_size: Batch size for training
        learning_rate: Learning rate for optimization
        non_iid: Whether to create non-IID data distribution
        
    Returns:
        clients: List of client objects
    """
    import numpy as np
    
    # Get all indices for the dataset
    all_indices = list(range(len(base_dataset)))
    
    if non_iid:
        # Group indices by label for non-IID distribution
        labels = []
        for i in all_indices:
            _, label = base_dataset[i]
            if hasattr(label, 'item'):
                label = label.item()
            labels.append(label)
            
        # Group indices by label
        label_indices = {}
        for idx, label in enumerate(labels):
            if label not in label_indices:
                label_indices[label] = []
            label_indices[label].append(idx)
            
        # Create non-IID partitions using Dirichlet distribution
        alpha = 0.5  # Lower alpha = more skew
        client_indices = [[] for _ in range(num_clients)]
        
        for label, indices in label_indices.items():
            # Generate proportion of each label for each client
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * len(indices) for p in proportions])
            proportions = proportions.astype(int)
            
            # Fix rounding errors
            proportions[-1] = len(indices) - np.sum(proportions[:-1])
            
            # Assign samples to clients
            start_idx = 0
            for i in range(num_clients):
                count = proportions[i]
                client_indices[i].extend(indices[start_idx:start_idx + count])
                start_idx += count
    else:
        # Create IID partitions
        np.random.shuffle(all_indices)
        items_per_client = len(all_indices) // num_clients
        client_indices = []
        
        for i in range(num_clients):
            start_idx = i * items_per_client
            end_idx = start_idx + items_per_client if i < num_clients - 1 else len(all_indices)
            client_indices.append(all_indices[start_idx:end_idx])
    
    # Create clients with lazy dataset partitions
    clients = []
    for i in range(num_clients):
        # Create lazy dataset partition
        client_dataset = LazyDatasetPartition(base_dataset, client_indices[i])
        
        # Create client with this dataset
        client = client_class(
            client_id=i,
            dataset=client_dataset,
            device=device,
            batch_size=batch_size,
            learning_rate=learning_rate,
            processing_power=np.random.uniform(0.5, 2.0),
            reliability=np.random.uniform(0.8, 1.0)
        )
        
        clients.append(client)
        
    return clients