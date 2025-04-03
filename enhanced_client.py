import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from hardware_profiler import HardwareProfiler, ArchitectureProfiler
import numpy as np
import time
import copy

class EnhancedClient:
    """
    Enhanced client implementation for federated learning with hardware and reliability tracking
    """
    def __init__(self, client_id, dataset, device, model=None, batch_size=32, 
                 learning_rate=0.01, processing_power=None, reliability=1.0):
        """
        Initialize client
        
        Args:
            client_id: Unique identifier for the client
            dataset: Client's training dataset
            device: Device to run computations on
            model: PyTorch model (if None, must be set later)
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            processing_power: Relative processing capability (simulated)
            reliability: Initial reliability score (0-1)
        """
        self.client_id = client_id
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.learning_rate = learning_rate
        
        # Track training metrics
        self.round_accuracies = []
        self.round_losses = []
        self.total_training_time = 0
        self.training_history = []
        
        # Initialize hardware profiler
        self.hardware_profiler = HardwareProfiler()
        self.hardware_profile = self.hardware_profiler.get_hardware_profile()
        
        # Architecture profiler will be initialized when model is set
        self.architecture_profiler = None
        self.architecture_profile = None
        
        # Use actual measured processing power if available, otherwise use provided or default
        if processing_power is None:
            self.processing_power = self.hardware_profile['compute_score']
        else:
            self.processing_power = processing_power
            
        self.reliability_score = reliability
        self.communication_history = []
        self.convergence_rate = []
        
        # Create optimizer if model is provided
        if model is not None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            self._initialize_architecture_profiler()
        
    def set_model(self, model):
        """Set client model and reinitialize optimizer"""
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def create_data_loader(self):
        """Create DataLoader for client dataset"""
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
    
    def train(self, epochs=5, evaluate=True):
        """
        Train client model for specified number of epochs
        
        Args:
            epochs: Number of training epochs
            evaluate: Whether to evaluate model after training
            
        Returns:
            model_update: Updated model parameters
        """
        if self.model is None:
            raise ValueError("Model not set for client")
            
        train_loader = self.create_data_loader()
        
        # Set model to training mode
        self.model.train()
        
        # Track metrics
        epoch_losses = []
        start_time = time.time()
        
        # Simulate different processing capabilities
        actual_epochs = max(1, int(epochs * self.processing_power))
        
        # Add random noise to simulate unreliable clients with some probability
        if np.random.random() > self.reliability_score:
            actual_epochs = max(1, actual_epochs // 2)  # Fewer epochs for unreliable clients
            
        # Training loop
        for epoch in range(actual_epochs):
            running_loss = 0.0
            samples = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                
                # Calculate loss based on output type
                if isinstance(output, torch.Tensor) and output.shape[1] > 1 and len(target.shape) == 1:
                    # Standard classification
                    loss = nn.CrossEntropyLoss()(output, target)
                elif len(target.shape) > 1 and target.shape[1] > 1:
                    # Multi-label classification (e.g., ChestXray)
                    loss = nn.BCELoss()(output, target)
                else:
                    # Default to NLL loss
                    loss = nn.NLLLoss()(output, target)
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                # Track statistics
                running_loss += loss.item() * data.size(0)
                samples += data.size(0)
            
            # Compute average loss for epoch
            epoch_loss = running_loss / samples
            epoch_losses.append(epoch_loss)
            
        # Record training time
        training_time = time.time() - start_time
        self.total_training_time += training_time
        
        # Calculate average loss for this round
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
        self.round_losses.append(avg_loss)
        
        # Evaluate if requested
        if evaluate:
            accuracy = self.evaluate()
        else:
            accuracy = None
            
        # Track training history
        self.training_history.append({
            'loss': avg_loss,
            'accuracy': accuracy,
            'time': training_time,
            'epochs': actual_epochs
        })
        
        # Update convergence rate
        if len(self.round_accuracies) >= 2:
            rate = self.round_accuracies[-1] - self.round_accuracies[-2]
            self.convergence_rate.append(rate)
            
        # Return model update
        return self.get_model_update()
    
    def evaluate(self):
        """
        Evaluate client model on client dataset
        
        Returns:
            accuracy: Evaluation accuracy
        """
        if self.model is None:
            raise ValueError("Model not set for client")
            
        # Create evaluation data loader
        eval_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        
        # Set model to evaluation mode
        self.model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in eval_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                
                # Compute accuracy based on task type
                if len(target.shape) > 1 and target.shape[1] > 1:
                    # Multi-label classification (e.g., ChestXray)
                    predicted = (outputs > 0.5).float()
                    correct += (predicted == target).sum().item()
                    total += target.numel()
                else:
                    # Standard classification
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
        
        # Calculate accuracy
        accuracy = correct / total if total > 0 else 0
        
        # Track accuracy
        self.round_accuracies.append(accuracy)
        
        return accuracy
    
    def get_model_update(self):
        """
        Get client model parameters, ensuring all parameters are included
        
        Returns:
            model_parameters: Dictionary of model parameters
        """
        if self.model is None:
            raise ValueError("Model not set for client")
            
        # Make sure we capture the complete state dict
        model_state = self.model.state_dict()
        update = {}
        
        # Ensure each parameter is properly copied
        for key, param in model_state.items():
            # Clone the parameter to avoid reference issues
            update[key] = param.clone().detach()
            
        # Verify all expected keys are present
        if len(update) != len(model_state):
            print(f"Warning: Client {self.client_id} update is missing keys.")
            print(f"Expected {len(model_state)} parameters, got {len(update)}")
            
        return update
    
    def measure_processing_power(self):
        """
        Measure client's processing capability using benchmarking
        
        Returns:
            processing_power: Relative processing capability (normalized)
        """
        # Simulate benchmarking by running a standard computation
        start_time = time.time()
        
        # Perform a standardized computation task
        test_size = 1000
        test_tensor = torch.randn(test_size, test_size).to(self.device)
        for _ in range(5):
            _ = torch.mm(test_tensor, test_tensor)
            
        compute_time = time.time() - start_time
        
        # Normalize between 0.5 and 2.0
        # Lower time = higher processing power
        norm_factor = 1.0  # Baseline reference
        power = norm_factor / max(compute_time, 1e-6)
        
        # Clip to desired range
        power = np.clip(power, 0.5, 2.0)
        
        return power
    
    def update_reliability_score(self, success=True, target_accuracy=None, global_average=None):
        """
        Update client reliability score based on training success and performance
        
        Args:
            success: Whether client completed training successfully
            target_accuracy: Target accuracy to compare against
            global_average: Global average accuracy for reference
            
        Returns:
            new_score: Updated reliability score
        """
        # Start with current score
        new_score = self.reliability_score
        
        # Factor 1: Training success (e.g., no crashes or timeouts)
        if not success:
            new_score *= 0.8  # Penalize failures
        
        # Factor 2: Accuracy compared to targets (if provided)
        if target_accuracy is not None and self.round_accuracies:
            client_accuracy = self.round_accuracies[-1]
            accuracy_ratio = min(client_accuracy / max(target_accuracy, 1e-6), 1.5)
            new_score = 0.7 * new_score + 0.3 * accuracy_ratio
            
        # Factor 3: Convergence behavior (using last 3 rounds if available)
        if len(self.round_accuracies) >= 3:
            # Positive trend in accuracy
            if self.round_accuracies[-1] > self.round_accuracies[-2] > self.round_accuracies[-3]:
                new_score *= 1.1  # Reward steady improvement
            # Oscillating behavior
            elif abs(self.round_accuracies[-1] - self.round_accuracies[-3]) < 0.01:
                new_score *= 0.95  # Slight penalty for stagnation
        
        # Factor 4: Contribution relative to global average (if provided)
        if global_average is not None and self.round_accuracies:
            client_accuracy = self.round_accuracies[-1]
            if client_accuracy > global_average:
                new_score *= 1.05  # Reward above-average contribution
            elif client_accuracy < 0.8 * global_average:
                new_score *= 0.9  # Penalize significantly below-average
        
        # Ensure score stays in valid range
        new_score = np.clip(new_score, 0.1, 1.0)
        
        # Update client's score
        self.reliability_score = new_score
        
        return new_score
    def _initialize_architecture_profiler(self):
        """Initialize architecture profiler with current model"""
        if self.model is not None:
            self.architecture_profiler = ArchitectureProfiler(self.model)
            self.architecture_profile = self.architecture_profiler.get_architecture_profile()
        
    def set_model(self, model):
        """Set client model and reinitialize optimizer"""
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Initialize architecture profiler with new model
        self._initialize_architecture_profiler()
    def extract_clustering_features(self):
        """
        Extract features for client clustering
        
        Returns:
            features: Dictionary of client features for clustering
        """
        if self.model is None:
            raise ValueError("Model not set for client")
            
        # Get model update for gradient-based features
        model_update = self.get_model_update()
        
        # Flatten parameters for analysis
        flattened_update = torch.cat([param.flatten() for param in model_update.values()])
        
        # Extract statistical features
        features = {
            # Model update features
            'update_mean': flattened_update.mean().item(),
            'update_std': flattened_update.std().item(),
            
            # Client characteristics
            'processing_power': self.processing_power,
            'reliability_score': self.reliability_score,
            
            # Training behavior
            'latest_accuracy': self.round_accuracies[-1] if self.round_accuracies else 0.0,
            'latest_loss': self.round_losses[-1] if self.round_losses else float('inf'),
            'training_time': self.total_training_time / max(1, len(self.round_accuracies)),
            
            # Convergence metrics
            'convergence_rate': np.mean(self.convergence_rate[-3:]) if len(self.convergence_rate) >= 3 else 0.0,
            
            # Model characteristics
            'model_size': sum(p.numel() for p in self.model.parameters())
        }
        
        return features
    
    def get_state_dict(self):
        """
        Get a serializable state dictionary for the client
        This allows saving and loading client state efficiently
        
        Returns:
            state_dict: Dictionary with essential client state
        """
        state_dict = {
            'client_id': self.client_id,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'processing_power': self.processing_power,
            'reliability_score': self.reliability_score,
            'round_accuracies': self.round_accuracies,
            'round_losses': self.round_losses,
            'total_training_time': self.total_training_time,
            'convergence_rate': self.convergence_rate
        }
        
        # Include model and optimizer state if available
        if hasattr(self, 'model') and self.model is not None:
            state_dict['model_state'] = self.model.state_dict()
            
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            state_dict['optimizer_state'] = self.optimizer.state_dict()
            
        # Include hardware profile if available
        if hasattr(self, 'hardware_profile') and self.hardware_profile is not None:
            state_dict['hardware_profile'] = self.hardware_profile
            
        # Include architecture profile if available
        if hasattr(self, 'architecture_profile') and self.architecture_profile is not None:
            state_dict['architecture_profile'] = self.architecture_profile
            
        return state_dict

    def load_state_dict(self, state_dict):
        """
        Load client state from a state dictionary
        
        Args:
            state_dict: Dictionary with client state
        """
        # Load basic attributes
        self.client_id = state_dict['client_id']
        self.learning_rate = state_dict['learning_rate']
        self.batch_size = state_dict['batch_size']
        self.processing_power = state_dict['processing_power']
        self.reliability_score = state_dict['reliability_score']
        self.round_accuracies = state_dict['round_accuracies']
        self.round_losses = state_dict['round_losses']
        self.total_training_time = state_dict['total_training_time']
        self.convergence_rate = state_dict['convergence_rate']
        
        # Load model state if available
        if 'model_state' in state_dict and hasattr(self, 'model') and self.model is not None:
            self.model.load_state_dict(state_dict['model_state'])
            
        # Load optimizer state if available
        if 'optimizer_state' in state_dict and hasattr(self, 'optimizer') and self.optimizer is not None:
            self.optimizer.load_state_dict(state_dict['optimizer_state'])
        elif 'model_state' in state_dict and hasattr(self, 'model') and self.model is not None:
            # If model is loaded but optimizer isn't, reinitialize optimizer
            import torch.optim as optim
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
        # Load hardware profile if available
        if 'hardware_profile' in state_dict:
            self.hardware_profile = state_dict['hardware_profile']
            
        # Load architecture profile if available
        if 'architecture_profile' in state_dict:
            self.architecture_profile = state_dict['architecture_profile']
    
    # Add to enhanced_client.py

def create_data_loader(self):
    """Create DataLoader for client dataset with empty dataset protection"""
    # Check if dataset exists and is not empty
    if not hasattr(self, 'dataset') or self.dataset is None:
        print(f"Warning: Client {self.client_id} has no dataset")
        # Create a dummy dataset with one sample to avoid errors
        dummy_data = torch.zeros(1, 28, 28)  # Assuming MNIST-like data
        dummy_label = torch.zeros(1, dtype=torch.long)
        from torch.utils.data import TensorDataset
        self.dataset = TensorDataset(dummy_data, dummy_label)
    
    # Check dataset length
    try:
        dataset_len = len(self.dataset)
        if dataset_len == 0:
            print(f"Warning: Client {self.client_id} has empty dataset")
            # Create a dummy dataset with one sample
            dummy_data = torch.zeros(1, 28, 28)  # Assuming MNIST-like data
            dummy_label = torch.zeros(1, dtype=torch.long)
            from torch.utils.data import TensorDataset
            self.dataset = TensorDataset(dummy_data, dummy_label)
    except Exception as e:
        print(f"Error checking dataset length for client {self.client_id}: {e}")
        # Create a dummy dataset
        dummy_data = torch.zeros(1, 28, 28)  # Assuming MNIST-like data
        dummy_label = torch.zeros(1, dtype=torch.long)
        from torch.utils.data import TensorDataset
        self.dataset = TensorDataset(dummy_data, dummy_label)
    
    return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def train(self, epochs=5, evaluate=True):
        """
        Train client model with empty dataset protection
        
        Args:
            epochs: Number of training epochs
            evaluate: Whether to evaluate model after training
            
        Returns:
            model_update: Updated model parameters
        """
        if self.model is None:
            raise ValueError("Model not set for client")
        
        try:
            train_loader = self.create_data_loader()
            
            # Set model to training mode
            self.model.train()
            
            # Track metrics
            epoch_losses = []
            start_time = time.time()
            
            # Simulate different processing capabilities
            actual_epochs = max(1, int(epochs * self.processing_power))
            
            # Add random noise to simulate unreliable clients with some probability
            if np.random.random() > self.reliability_score:
                actual_epochs = max(1, actual_epochs // 2)  # Fewer epochs for unreliable clients
                
            # Training loop
            for epoch in range(actual_epochs):
                running_loss = 0.0
                samples = 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    output = self.model(data)
                    
                    # Calculate loss based on output type
                    if isinstance(output, torch.Tensor) and output.shape[1] > 1 and len(target.shape) == 1:
                        # Standard classification
                        loss = nn.CrossEntropyLoss()(output, target)
                    elif len(target.shape) > 1 and target.shape[1] > 1:
                        # Multi-label classification (e.g., ChestXray)
                        loss = nn.BCELoss()(output, target)
                    else:
                        # Default to NLL loss
                        loss = nn.NLLLoss()(output, target)
                    
                    # Backward pass and optimization
                    loss.backward()
                    self.optimizer.step()
                    
                    # Track statistics
                    running_loss += loss.item() * data.size(0)
                    samples += data.size(0)
                
                # Compute average loss for epoch
                epoch_loss = running_loss / samples if samples > 0 else float('inf')
                epoch_losses.append(epoch_loss)
                
            # Record training time
            training_time = time.time() - start_time
            self.total_training_time += training_time
            
            # Calculate average loss for this round
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
            self.round_losses.append(avg_loss)
            
            # Evaluate if requested
            if evaluate:
                accuracy = self.evaluate()
            else:
                accuracy = None
                
            # Track training history
            self.training_history.append({
                'loss': avg_loss,
                'accuracy': accuracy,
                'time': training_time,
                'epochs': actual_epochs
            })
            
            # Update convergence rate
            if len(self.round_accuracies) >= 2:
                rate = self.round_accuracies[-1] - self.round_accuracies[-2]
                self.convergence_rate.append(rate)
                
            # Return model update
            return self.get_model_update()
        
        except Exception as e:
            print(f"Error during training for client {self.client_id}: {e}")
            # Return current model state without training
            # This avoids crashing the entire process when one client fails
            return self.get_model_update()
