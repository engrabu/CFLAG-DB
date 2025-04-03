import numpy as np
import torch
import types

class LearningRateScheduler:
    """
    Manages dynamic learning rate schedules for federated learning
    """
    def __init__(self, initial_lr=0.01, schedule_type='adaptive', 
                 decay_factor=0.5, step_size=10, min_lr=0.0001,
                 patience=3, cooldown=1):
        """
        Initialize learning rate scheduler
        
        Args:
            initial_lr: Starting learning rate
            schedule_type: Type of schedule ('step', 'exponential', 'cosine', 'adaptive', 'cyclic')
            decay_factor: Factor to multiply learning rate by (for step and exponential)
            step_size: Number of rounds between step decays
            min_lr: Minimum learning rate
            patience: Number of rounds with no improvement before reducing LR (for adaptive)
            cooldown: Number of rounds to wait after reducing LR (for adaptive)
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.schedule_type = schedule_type
        self.decay_factor = decay_factor
        self.step_size = step_size
        self.min_lr = min_lr
        self.round = 0
        self.performance_history = []
        self.patience = patience
        self.cooldown = cooldown
        self.wait_counter = 0
        self.cooldown_counter = 0
        self.best_performance = -float('inf')
        self.lr_history = [initial_lr]
        
    def step(self, round_performance=None):
        """
        Update learning rate based on the schedule and current round
        
        Args:
            round_performance: Optional performance metrics from the last round
            
        Returns:
            new_lr: Updated learning rate
        """
        self.round += 1
        
        if round_performance is not None:
            self.performance_history.append(round_performance)
            
        # Calculate new learning rate based on schedule type
        if self.schedule_type == 'step':
            # Step decay: reduce LR by decay factor every step_size rounds
            if self.round % self.step_size == 0:
                self.current_lr = max(self.current_lr * self.decay_factor, self.min_lr)
                
        elif self.schedule_type == 'exponential':
            # Exponential decay: continuously reduce LR
            self.current_lr = max(self.initial_lr * (self.decay_factor ** self.round), self.min_lr)
            
        elif self.schedule_type == 'cosine':
            # Cosine annealing: smooth decay following cosine curve
            max_rounds = 100  # Adjust as needed
            self.current_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
                             (1 + np.cos(np.pi * self.round / max_rounds))
                             
        elif self.schedule_type == 'adaptive':
            # Adaptive learning rate based on performance (similar to ReduceLROnPlateau)
            if round_performance is not None and len(self.performance_history) > 0:
                current_perf = round_performance.get('accuracy', 0)
                
                # If we're in cooldown period, don't adjust LR
                if self.cooldown_counter > 0:
                    self.cooldown_counter -= 1
                else:
                    # Check if performance improved
                    if current_perf > self.best_performance:
                        self.best_performance = current_perf
                        self.wait_counter = 0
                    else:
                        self.wait_counter += 1
                        
                        # If waited long enough with no improvement, reduce LR
                        if self.wait_counter >= self.patience:
                            self.current_lr = max(self.current_lr * self.decay_factor, self.min_lr)
                            self.wait_counter = 0
                            self.cooldown_counter = self.cooldown
                
        elif self.schedule_type == 'cyclic':
            # Cyclic learning rate: oscillate between min and max
            cycle_size = 2 * self.step_size
            cycle = np.floor(1 + self.round / cycle_size)
            x = np.abs(self.round / self.step_size - 2 * cycle + 1)
            self.current_lr = self.min_lr + (self.initial_lr - self.min_lr) * max(0, 1 - x)
        
        # Add to history
        self.lr_history.append(self.current_lr)
            
        return self.current_lr
        
    def get_lr(self):
        """Get current learning rate"""
        return self.current_lr
        
    def reset(self):
        """Reset scheduler state"""
        self.current_lr = self.initial_lr
        self.round = 0
        self.performance_history = []
        self.wait_counter = 0
        self.cooldown_counter = 0
        self.best_performance = -float('inf')


def integrate_dynamic_lr_with_cflagbd(cflagbd_instance, schedule_type='adaptive', verbose=True):
    """
    Integrate dynamic learning rate with an existing CFLAG-BD instance
    
    Args:
        cflagbd_instance: Existing CFLAG-BD algorithm instance
        schedule_type: Type of learning rate schedule
        verbose: Whether to display learning rate changes
        
    Returns:
        cflagbd_instance: Modified CFLAG-BD instance with dynamic LR
    """
    # Get initial learning rate from clients
    initial_lr = cflagbd_instance.clients[0].learning_rate if cflagbd_instance.clients else 0.01
    
    # Create global scheduler
    cflagbd_instance.global_lr_scheduler = LearningRateScheduler(
        initial_lr=initial_lr,
        schedule_type=schedule_type
    )
    
    # Create per-cluster schedulers
    cflagbd_instance.cluster_lr_schedulers = [
        LearningRateScheduler(
            initial_lr=initial_lr,
            schedule_type=schedule_type
        ) for _ in range(cflagbd_instance.num_clusters)
    ]
    
    # Track previous learning rates
    cflagbd_instance.previous_cluster_lrs = [initial_lr] * cflagbd_instance.num_clusters
    cflagbd_instance.previous_client_lrs = {i: initial_lr for i in range(len(cflagbd_instance.clients))}
    
    # Store the original run_round method
    original_run_round = cflagbd_instance.run_round
    
    # Define the enhanced run_round method with dynamic learning rate
    def enhanced_run_round(client_fraction=1.0, epochs=5):
        # Select clients
        client_indices = cflagbd_instance.select_hardware_diverse_clients(client_fraction) \
            if hasattr(cflagbd_instance, 'select_hardware_diverse_clients') \
            else np.random.choice(len(cflagbd_instance.clients), 
                                 max(1, int(client_fraction * len(cflagbd_instance.clients))), 
                                 replace=False).tolist()
        
        # Get cluster assignments for selected clients
        client_clusters = [cflagbd_instance.client_clusters[i] for i in client_indices]
        
        # Track learning rate changes
        lr_changes = []
        cluster_lr_changes = []
        
        # Update learning rates for each client based on its cluster
        for i, client_idx in enumerate(client_indices):
            cluster_idx = client_clusters[i]
            
            # Get learning rate from cluster scheduler
            new_lr = cflagbd_instance.cluster_lr_schedulers[cluster_idx].get_lr()
            
            # Track cluster learning rate changes
            prev_cluster_lr = cflagbd_instance.previous_cluster_lrs[cluster_idx]
            if abs(new_lr - prev_cluster_lr) > 1e-6:  # Small epsilon to account for floating point
                cluster_lr_changes.append((cluster_idx, prev_cluster_lr, new_lr))
                cflagbd_instance.previous_cluster_lrs[cluster_idx] = new_lr
            
            # Adjust for hardware if possible
            if hasattr(cflagbd_instance, 'adjust_learning_rate_for_hardware'):
                new_lr = cflagbd_instance.adjust_learning_rate_for_hardware(client_idx, new_lr)
                
            # Store previous learning rate
            prev_lr = cflagbd_instance.previous_client_lrs.get(client_idx, initial_lr)
            
            # Update client's learning rate
            if hasattr(cflagbd_instance.clients[client_idx], 'set_learning_rate'):
                cflagbd_instance.clients[client_idx].set_learning_rate(new_lr)
            else:
                # If no set_learning_rate method, directly update optimizer
                client = cflagbd_instance.clients[client_idx]
                client.learning_rate = new_lr
                if hasattr(client, 'optimizer') and client.optimizer is not None:
                    for param_group in client.optimizer.param_groups:
                        param_group['lr'] = new_lr
            
            # Check if learning rate changed and record the change
            if abs(new_lr - prev_lr) > 1e-6:  # Small epsilon to account for floating point
                lr_changes.append((client_idx, prev_lr, new_lr))
                cflagbd_instance.previous_client_lrs[client_idx] = new_lr
        
        # Display learning rate changes if verbose
        if verbose:
            if cluster_lr_changes:
                print("\n=== Cluster Learning Rate Changes ===")
                for cluster_idx, old_lr, new_lr in cluster_lr_changes:
                    change_pct = ((new_lr - old_lr) / old_lr) * 100
                    change_direction = "▲" if new_lr > old_lr else "▼"
                    print(f"Cluster {cluster_idx}: {old_lr:.6f} → {new_lr:.6f} {change_direction} ({change_pct:.2f}%)")
            
            if lr_changes:
                print("\n=== Client Learning Rate Changes ===")
                for client_idx, old_lr, new_lr in lr_changes:
                    change_pct = ((new_lr - old_lr) / old_lr) * 100
                    change_direction = "▲" if new_lr > old_lr else "▼"
                    cluster_idx = cflagbd_instance.client_clusters[client_idx]
                    print(f"Client {client_idx} (Cluster {cluster_idx}): {old_lr:.6f} → {new_lr:.6f} {change_direction} ({change_pct:.2f}%)")
        
        # Call the original run_round method
        round_stats = original_run_round(client_fraction, epochs)
        
        # Update learning rate schedulers based on performance
        global_lr_old = cflagbd_instance.global_lr_scheduler.get_lr()
        global_lr_new = cflagbd_instance.global_lr_scheduler.step(round_stats)
        
        if verbose and abs(global_lr_old - global_lr_new) > 1e-6:
            change_pct = ((global_lr_new - global_lr_old) / global_lr_old) * 100
            change_direction = "▲" if global_lr_new > global_lr_old else "▼"
            print(f"\n=== Global Learning Rate Changed ===")
            print(f"Global LR: {global_lr_old:.6f} → {global_lr_new:.6f} {change_direction} ({change_pct:.2f}%)")
        
        # Update cluster schedulers if cluster accuracies are available
        if 'cluster_accuracies' in round_stats:
            for cluster_idx, accuracy in enumerate(round_stats['cluster_accuracies']):
                cluster_lr_old = cflagbd_instance.cluster_lr_schedulers[cluster_idx].get_lr()
                cluster_lr_new = cflagbd_instance.cluster_lr_schedulers[cluster_idx].step({'accuracy': accuracy})
                
                if verbose and abs(cluster_lr_old - cluster_lr_new) > 1e-6:
                    change_pct = ((cluster_lr_new - cluster_lr_old) / cluster_lr_old) * 100
                    change_direction = "▲" if cluster_lr_new > cluster_lr_old else "▼"
                    print(f"\nCluster {cluster_idx} LR (post-round): {cluster_lr_old:.6f} → {cluster_lr_new:.6f} {change_direction} ({change_pct:.2f}%)")
        
        # Add learning rate info to round_stats
        round_stats['global_lr'] = cflagbd_instance.global_lr_scheduler.get_lr()
        round_stats['cluster_lrs'] = [scheduler.get_lr() for scheduler in cflagbd_instance.cluster_lr_schedulers]
        round_stats['lr_changes'] = len(lr_changes)
        
        return round_stats
    
    # Replace the run_round method
    cflagbd_instance.run_round = enhanced_run_round
    
    # Add method to adjust learning rate based on hardware
    def adjust_learning_rate_for_hardware(client_idx, base_lr):
        """
        Adjust learning rate based on client hardware profile
        
        Args:
            client_idx: Client index
            base_lr: Base learning rate
            
        Returns:
            adjusted_lr: Hardware-adjusted learning rate
        """
        client = cflagbd_instance.clients[client_idx]
        
        # Default adjustment factor
        adjustment = 1.0
        
        # Adjust based on hardware if available
        if hasattr(client, 'hardware_profile') and client.hardware_profile is not None:
            # Higher compute score = can handle higher learning rates
            compute_score = client.hardware_profile.get('compute_score', 1.0)
            compute_factor = min(compute_score / 10.0, 2.0)
            
            # GPU availability = can handle higher learning rates
            gpu_factor = 1.5 if client.hardware_profile.get('gpu_available', False) else 1.0
            
            # Combine factors
            adjustment = min(compute_factor * gpu_factor, 3.0)  # Cap at 3x
            
        # Adjust learning rate
        adjusted_lr = base_lr * adjustment
        
        return adjusted_lr
    
    cflagbd_instance.adjust_learning_rate_for_hardware = adjust_learning_rate_for_hardware
    
    # Add method to client class if needed
    if not hasattr(cflagbd_instance.clients[0], 'set_learning_rate'):
        for client in cflagbd_instance.clients:
            def set_learning_rate(self, new_lr):
                """Update learning rate for client optimizer"""
                self.learning_rate = new_lr
                if hasattr(self, 'optimizer') and self.optimizer is not None:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
            
            # Bind the method to the client instance
            client.set_learning_rate = types.MethodType(set_learning_rate, client)
    
    # Add method to plot learning rate history
    def plot_learning_rate_history(save_path=None):
        """
        Plot learning rate history for all schedulers
        
        Args:
            save_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        # Plot global LR
        plt.plot(cflagbd_instance.global_lr_scheduler.lr_history, 
                 label='Global', linewidth=2, color='black')
        
        # Plot cluster LRs
        for i, scheduler in enumerate(cflagbd_instance.cluster_lr_schedulers):
            plt.plot(scheduler.lr_history, 
                     label=f'Cluster {i}', 
                     linestyle='--')
            
        plt.xlabel('Training Round')
        plt.ylabel('Learning Rate')
        plt.title(f'Learning Rate History ({schedule_type.title()} Schedule)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale often better for learning rates
        
        if save_path:
            plt.savefig(save_path)
            print(f"Learning rate plot saved to {save_path}")
        
        plt.show()
        
    cflagbd_instance.plot_learning_rate_history = plot_learning_rate_history
    
    print(f"Dynamic learning rate ({schedule_type}) integrated with CFLAG-BD")
    
    return cflagbd_instance