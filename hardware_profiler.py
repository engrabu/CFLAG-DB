import torch
import time
import psutil
import platform
import numpy as np

class HardwareProfiler:
    """
    Measures and tracks hardware capabilities of clients
    """
    def __init__(self):
        """Initialize the hardware profiler"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.has_gpu = torch.cuda.is_available()
        self.gpu_name = torch.cuda.get_device_name(0) if self.has_gpu else None
        self.gpu_memory = torch.cuda.get_device_properties(0).total_memory if self.has_gpu else 0
        self.cpu_count = psutil.cpu_count(logical=True)
        self.system_memory = psutil.virtual_memory().total
        self.platform = platform.system()
        self.architecture = platform.machine()
        
    def benchmark_compute_power(self, test_size=1000, iterations=5):
        """
        Benchmark computation power using matrix multiplication
        
        Args:
            test_size: Size of test matrices
            iterations: Number of iterations for benchmark
            
        Returns:
            compute_score: Relative computation power score
        """
        # Create test tensors
        if self.has_gpu:
            a = torch.randn(test_size, test_size, device=self.device)
            b = torch.randn(test_size, test_size, device=self.device)
            
            # Warmup
            for _ in range(2):
                _ = torch.matmul(a, b)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(iterations):
                _ = torch.matmul(a, b)
                torch.cuda.synchronize()
            compute_time = time.time() - start_time
        else:
            a = torch.randn(test_size, test_size)
            b = torch.randn(test_size, test_size)
            
            # Warmup
            for _ in range(2):
                _ = torch.matmul(a, b)
            
            # Benchmark
            start_time = time.time()
            for _ in range(iterations):
                _ = torch.matmul(a, b)
            compute_time = time.time() - start_time
        
        # Calculate compute score (higher is better)
        compute_score = (test_size * test_size * test_size * iterations) / compute_time / 1e9
        
        return compute_score
        
    def get_hardware_profile(self):
        """
        Get comprehensive hardware profile
        
        Returns:
            profile: Dictionary of hardware capabilities
        """
        # Run benchmarks
        compute_score = self.benchmark_compute_power()
        
        # Create hardware profile
        profile = {
            'device_type': 'gpu' if self.has_gpu else 'cpu',
            'gpu_available': self.has_gpu,
            'gpu_name': self.gpu_name,
            'gpu_memory_gb': self.gpu_memory / 1e9 if self.has_gpu else 0,
            'cpu_count': self.cpu_count,
            'system_memory_gb': self.system_memory / 1e9,
            'compute_score': compute_score,
            'platform': self.platform,
            'architecture': self.architecture
        }
        
        return profile


class ArchitectureProfiler:
    """
    Analyzes neural network architecture characteristics
    """
    def __init__(self, model):
        """
        Initialize architecture profiler
        
        Args:
            model: PyTorch neural network model
        """
        self.model = model
        
    def get_layer_distribution(self):
        """
        Get distribution of layer types in the model
        
        Returns:
            layer_counts: Dictionary mapping layer types to counts
        """
        layer_counts = {}
        
        for name, module in self.model.named_modules():
            layer_type = module.__class__.__name__
            
            if layer_type in layer_counts:
                layer_counts[layer_type] += 1
            else:
                layer_counts[layer_type] = 1
                
        return layer_counts
        
    def calculate_compute_intensity(self):
        """
        Calculate compute-to-memory ratio of the model
        
        Returns:
            compute_intensity: Ratio of computations to memory accesses
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        total_macs = 0  # Multiply-accumulate operations
        
        # Estimate MACs for common layer types
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                # For each output pixel: kernel_size^2 * in_channels MACs
                if hasattr(module, 'kernel_size'):
                    if isinstance(module.kernel_size, tuple):
                        k_h, k_w = module.kernel_size
                    else:
                        k_h = k_w = module.kernel_size
                    
                    # Estimate output dimensions if not directly available
                    out_h = out_w = 32  # Default estimate
                    
                    try:
                        # MAC calculation - simplified estimate
                        macs_per_layer = k_h * k_w * module.in_channels * module.out_channels * out_h * out_w
                        total_macs += macs_per_layer
                    except AttributeError:
                        pass  # Skip if attributes not available
                
            elif isinstance(module, torch.nn.Linear):
                try:
                    # For linear: in_features * out_features MACs
                    macs_per_layer = module.in_features * module.out_features
                    total_macs += macs_per_layer
                except AttributeError:
                    pass  # Skip if attributes not available
        
        # If estimation is too low, use a rough heuristic based on parameter count
        if total_macs < total_params:
            total_macs = total_params * 10  # Rough estimate
            
        # Compute intensity is ratio of compute to memory
        compute_intensity = total_macs / (total_params * 4)  # 4 bytes per parameter
        
        return compute_intensity
        
    def get_architecture_profile(self):
        """
        Get comprehensive architecture profile
        
        Returns:
            profile: Dictionary of architecture characteristics
        """
        # Basic model size information
        total_params = sum(p.numel() for p in self.model.parameters())
        param_size_bytes = total_params * 4  # Assuming float32 parameters
        
        # Layer information
        layer_distribution = self.get_layer_distribution()
        
        # Get depth (number of layers with parameters)
        depth = len(list(self.model.parameters()))
        
        # Calculate compute intensity
        compute_intensity = self.calculate_compute_intensity()
        
        # Count different types of layers
        conv_layers = sum(1 for m in self.model.modules() if isinstance(m, torch.nn.Conv2d))
        fc_layers = sum(1 for m in self.model.modules() if isinstance(m, torch.nn.Linear))
        pooling_layers = sum(1 for m in self.model.modules() 
                            if isinstance(m, (torch.nn.MaxPool2d, torch.nn.AvgPool2d)))
        
        # Create architecture profile
        profile = {
            'total_parameters': total_params,
            'model_size_mb': param_size_bytes / 1e6,
            'layer_distribution': layer_distribution,
            'model_depth': depth,
            'compute_intensity': compute_intensity,
            'conv_layers': conv_layers,
            'fc_layers': fc_layers,
            'pooling_layers': pooling_layers
        }
        
        return profile