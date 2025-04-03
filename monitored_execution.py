import threading
import time
import psutil
import os
import gc
import torch
import numpy as np
from functools import wraps

class TimeoutError(Exception):
    """Custom exception for timeout errors"""
    pass

class ThreadingTimeout:
    """Cross-platform timeout implementation using threading"""
    def __init__(self, seconds):
        self.seconds = seconds
        self.timer = None
        self.timed_out = False

    def timeout_handler(self):
        self.timed_out = True
        # Getting the current thread's ID to raise exception in the right thread
        import threading
        import ctypes
        import inspect
        
        # Find the thread where the timer was started
        current_thread = threading.current_thread()
        
        # Get all active threads
        for thread_id, frame in sys._current_frames().items():
            try:
                # Get the thread object from its ID
                for thread in threading.enumerate():
                    if thread.ident == thread_id and thread is not current_thread:
                        # Raise exception in the thread
                        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                            ctypes.c_long(thread_id),
                            ctypes.py_object(TimeoutError("Operation timed out"))
                        )
                        if res > 1:
                            # If more than one thread was affected, undo
                            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                                ctypes.c_long(thread_id), 
                                None
                            )
            except Exception:
                pass

    def start(self):
        self.timer = threading.Timer(self.seconds, self.timeout_handler)
        self.timer.daemon = True
        self.timer.start()

    def cancel(self):
        if self.timer:
            self.timer.cancel()
        
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.cancel()
        return type is TimeoutError

def with_timeout(seconds):
    """
    Decorator to add timeout to functions (cross-platform)
    
    Args:
        seconds: Timeout in seconds
        
    Returns:
        Decorated function with timeout
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            timeout = ThreadingTimeout(seconds)
            try:
                with timeout:
                    return func(*args, **kwargs)
            except TimeoutError:
                raise
        return wrapper
    return decorator

class MemoryMonitor:
    """Monitor memory usage and raise exception if it exceeds limit"""
    def __init__(self, memory_limit_gb=None, check_interval=5):
        """
        Initialize memory monitor
        
        Args:
            memory_limit_gb: Memory limit in GB (default: 90% of system memory)
            check_interval: Interval between checks in seconds
        """
        self.process = psutil.Process(os.getpid())
        
        # Default to 90% of system memory if not specified
        if memory_limit_gb is None:
            system_memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            self.memory_limit_bytes = int(system_memory_gb * 0.9 * 1024 * 1024 * 1024)
        else:
            self.memory_limit_bytes = int(memory_limit_gb * 1024 * 1024 * 1024)
            
        self.check_interval = check_interval
        self.stop_flag = threading.Event()
        self.monitor_thread = None
        
    def memory_check(self):
        """Check memory usage periodically"""
        print(f"Memory monitor started. Limit: {self.memory_limit_bytes / (1024*1024*1024):.2f} GB")
        
        while not self.stop_flag.is_set():
            try:
                # Get current memory usage
                memory_info = self.process.memory_info()
                current_memory = memory_info.rss
                
                # Check if memory usage exceeds limit
                if current_memory > self.memory_limit_bytes:
                    print(f"\n⚠️ WARNING: Memory usage ({current_memory/(1024*1024*1024):.2f} GB) "
                          f"exceeded limit ({self.memory_limit_bytes/(1024*1024*1024):.2f} GB)")
                    print("Attempting emergency memory cleanup...")
                    
                    # Try to free some memory
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    # Check again after cleanup
                    current_memory = self.process.memory_info().rss
                    if current_memory > self.memory_limit_bytes:
                        print(f"Memory usage still high after cleanup: {current_memory/(1024*1024*1024):.2f} GB")
                        print("Raising MemoryError to prevent system crash")
                        self.stop()
                        os._exit(1)  # Force exit to prevent hanging
                
                # Wait for next check
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"Error in memory monitor: {e}")
                time.sleep(self.check_interval)
        
    def start(self):
        """Start memory monitoring in background thread"""
        self.stop_flag.clear()
        self.monitor_thread = threading.Thread(target=self.memory_check)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop(self):
        """Stop memory monitoring"""
        self.stop_flag.set()
        if self.monitor_thread is not None:
            self.monitor_thread.join(timeout=1.0)