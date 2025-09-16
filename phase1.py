"""
Phase 1: Environment Setup
==========================
Handles environment preparation and validation for GLM-4.5-Air quantization.

This module verifies system requirements, configures CUDA environment,
sets up memory settings, and prepares monitoring infrastructure.
"""

import os
import sys
import subprocess
import psutil
import logging
import shutil
import torch
import threading
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import importlib
import pkg_resources


@dataclass
class HardwareConfig:
    """Hardware configuration for quantization."""
    gpu_memory_gb: int
    cpu_memory_gb: int
    gpu_name: str
    cuda_version: str
    disk_space_gb: int
    offload_folder: Path
    

class EnvironmentSetup:
    """Handles environment preparation and validation."""
    
    def __init__(self, hardware_config: HardwareConfig):
        """
        Initialize environment setup with hardware configuration.
        
        Args:
            hardware_config: Hardware configuration dataclass
        """
        self.hardware = hardware_config
        self.logger = logging.getLogger(__name__)
        self.dep_manager = DependencyManager()  # Fix #6: Create once and reuse
        
    def verify_system_requirements(self) -> Dict[str, bool]:
        """
        Verify system meets minimum requirements.
        
        Checks:
        - GPU availability and memory
        - CPU memory
        - Disk space
        - CUDA installation
        - Python version
        
        Returns:
            Dict with requirement checks and their status
        """
        requirements = {}
        
        # Check Python version (3.8+ required for llmcompressor)
        python_version = sys.version_info
        requirements['python_version'] = python_version >= (3, 8)
        if not requirements['python_version']:
            self.logger.warning(f"Python {python_version.major}.{python_version.minor} detected. Python 3.8+ recommended.")
        
        # Check GPU availability
        try:
            requirements['cuda_available'] = torch.cuda.is_available()
            if requirements['cuda_available']:
                gpu_count = torch.cuda.device_count()
                requirements['gpu_found'] = gpu_count > 0
                
                if gpu_count > 0:
                    # Check GPU memory
                    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    requirements['gpu_memory_sufficient'] = gpu_mem_gb >= 20  # Minimum 20GB for GLM-4.5-Air
                    
                    # Check if it's the expected GPU
                    gpu_name = torch.cuda.get_device_name(0)
                    requirements['gpu_match'] = self.hardware.gpu_name.lower() in gpu_name.lower()
                    
                    self.logger.info(f"GPU detected: {gpu_name} with {gpu_mem_gb:.1f}GB memory")
                else:
                    requirements['gpu_found'] = False
                    requirements['gpu_memory_sufficient'] = False
                    requirements['gpu_match'] = False
            else:
                requirements['cuda_available'] = False
                requirements['gpu_found'] = False
                requirements['gpu_memory_sufficient'] = False
                self.logger.error("CUDA is not available. GPU acceleration will not work.")
                
        except Exception as e:
            self.logger.error(f"Error checking GPU: {e}")
            requirements['cuda_available'] = False
            requirements['gpu_found'] = False
            requirements['gpu_memory_sufficient'] = False
        
        # Check CPU memory
        cpu_mem = psutil.virtual_memory()
        cpu_mem_gb = cpu_mem.total / (1024**3)
        requirements['cpu_memory_sufficient'] = cpu_mem_gb >= 200  # Minimum 200GB for GLM-4.5-Air
        self.logger.info(f"System RAM: {cpu_mem_gb:.1f}GB (Available: {cpu_mem.available / (1024**3):.1f}GB)")
        
        # Check disk space
        requirements['disk_space_sufficient'] = self.check_disk_space(required_gb=300)
        
        # Check swap space
        requirements['swap_available'] = self.verify_swap_space(recommended_gb=100)
        
        # Check CUDA toolkit
        requirements['cuda_toolkit'] = self._check_cuda_toolkit()
        
        # Summary
        all_critical_met = all([
            requirements.get('cuda_available', False),
            requirements.get('gpu_found', False),
            requirements.get('gpu_memory_sufficient', False),
            requirements.get('cpu_memory_sufficient', False),
            requirements.get('disk_space_sufficient', False)
        ])
        
        requirements['all_critical_requirements_met'] = all_critical_met
        
        if all_critical_met:
            self.logger.info("✅ All critical system requirements met")
        else:
            self.logger.warning("⚠️ Some critical requirements not met. Review the requirements above.")
            
        return requirements
    
    def check_disk_space(self, required_gb: int = 300) -> bool:
        """
        Check if sufficient disk space is available.
        
        Args:
            required_gb: Required disk space in GB
            
        Returns:
            True if sufficient space available
        """
        try:
            # Check the disk where the offload folder will be
            offload_path = self.hardware.offload_folder
            if not offload_path.exists():
                offload_path = Path.cwd()  # Use current directory as fallback
            
            stat = shutil.disk_usage(str(offload_path))
            
            # Avoid division by zero
            if stat.total == 0:
                self.logger.error("Cannot determine disk space - total size is 0")
                return False
                
            available_gb = stat.free / (1024**3)
            total_gb = stat.total / (1024**3)
            used_percent = (stat.used / stat.total) * 100
            
            self.logger.info(f"Disk space - Total: {total_gb:.1f}GB, Available: {available_gb:.1f}GB, Used: {used_percent:.1f}%")
            
            if available_gb >= required_gb:
                self.logger.info(f"✅ Sufficient disk space: {available_gb:.1f}GB available (required: {required_gb}GB)")
                return True
            else:
                self.logger.warning(f"⚠️ Insufficient disk space: {available_gb:.1f}GB available (required: {required_gb}GB)")
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking disk space: {e}")
            return False
    
    def check_gpu_availability(self) -> Dict[str, Any]:
        """
        Check GPU availability and specifications.
        
        Returns:
            Dictionary with GPU information
        """
        gpu_info = {
            'available': False,
            'count': 0,
            'devices': []
        }
        
        try:
            if torch.cuda.is_available():
                gpu_info['available'] = True
                gpu_info['count'] = torch.cuda.device_count()
                gpu_info['cuda_version'] = torch.version.cuda
                
                # Store current device to restore later
                current_device = torch.cuda.current_device() if gpu_info['count'] > 0 else None
                
                for i in range(gpu_info['count']):
                    props = torch.cuda.get_device_properties(i)
                    device_info = {
                        'index': i,
                        'name': props.name,
                        'memory_gb': props.total_memory / (1024**3),
                        'capability': f"{props.major}.{props.minor}",
                        'multi_processor_count': props.multi_processor_count
                    }
                    
                    # Get current memory usage
                    torch.cuda.set_device(i)  # Fix #14: Removed redundant cuda.is_available() check
                    device_info['memory_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
                    device_info['memory_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
                    
                    gpu_info['devices'].append(device_info)
                    
                    self.logger.info(f"GPU {i}: {device_info['name']} - "
                                   f"Memory: {device_info['memory_gb']:.1f}GB, "
                                   f"Capability: {device_info['capability']}")
                
                # Restore original device
                if current_device is not None:
                    torch.cuda.set_device(current_device)
                    
            else:
                self.logger.warning("No CUDA-capable GPU detected")
                
        except Exception as e:
            self.logger.error(f"Error checking GPU availability: {e}")
            
        return gpu_info
    
    def setup_cuda_environment(self) -> bool:
        """
        Configure CUDA environment variables.
        
        Sets:
        - CUDA_VISIBLE_DEVICES
        - PYTORCH_CUDA_ALLOC_CONF
        - CUDA_LAUNCH_BLOCKING (for debugging)
        
        Returns:
            True if CUDA setup successful
        """
        try:
            # Set visible devices (using first GPU)
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            self.logger.info("Set CUDA_VISIBLE_DEVICES=0")
            
            # Configure PyTorch CUDA memory allocation
            # This helps with memory fragmentation issues
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            self.logger.info("Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")
            
            # Optional: Enable synchronous CUDA operations for better error messages
            # (disable in production for better performance)
            if self.logger.level == logging.DEBUG:
                os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
                self.logger.debug("Enabled CUDA_LAUNCH_BLOCKING for debugging")
            
            # Set memory fraction if needed (reserve some memory for system)
            if torch.cuda.is_available():
                # Use environment variable instead of non-existent function
                # This is handled by PYTORCH_CUDA_ALLOC_CONF already
                
                # Clear any existing cache
                torch.cuda.empty_cache()
                self.logger.info("Cleared CUDA cache")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up CUDA environment: {e}")
            return False
    
    def configure_memory_settings(self) -> Dict[str, Any]:
        """
        Configure memory settings for optimal performance.
        
        Configures:
        - PyTorch memory allocation
        - CPU memory limits
        - Swap space verification
        - Offloading directories
        
        Returns:
            Dictionary of configured memory settings
        """
        memory_config = {}
        
        try:
            # Get current memory status
            cpu_mem = psutil.virtual_memory()
            memory_config['cpu_total_gb'] = cpu_mem.total / (1024**3)
            memory_config['cpu_available_gb'] = cpu_mem.available / (1024**3)
            memory_config['cpu_percent_used'] = cpu_mem.percent
            
            # Configure PyTorch settings
            pytorch_settings = self.setup_pytorch_memory()
            memory_config['pytorch_configured'] = pytorch_settings
            
            # Check and configure swap
            swap = psutil.swap_memory()
            memory_config['swap_total_gb'] = swap.total / (1024**3)
            memory_config['swap_used_gb'] = swap.used / (1024**3)
            memory_config['swap_percent'] = swap.percent
            
            # Create and verify offload directories
            offload_dir = self.hardware.offload_folder
            offload_dir.mkdir(parents=True, exist_ok=True)
            memory_config['offload_dir'] = str(offload_dir)
            memory_config['offload_dir_ready'] = offload_dir.exists() and offload_dir.is_dir()
            
            # Set memory-related environment variables
            # Limit CPU memory usage to prevent system freeze
            max_cpu_mem = int(cpu_mem.total * 0.9)  # Use max 90% of system RAM
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:512,garbage_collection_threshold:0.6'
            
            memory_config['settings_applied'] = True
            
            self.logger.info(f"Memory configuration completed: "
                           f"CPU: {memory_config['cpu_available_gb']:.1f}/{memory_config['cpu_total_gb']:.1f}GB available, "
                           f"Swap: {memory_config['swap_total_gb']:.1f}GB")
            
        except Exception as e:
            self.logger.error(f"Error configuring memory settings: {e}")
            memory_config['settings_applied'] = False
            
        return memory_config
    
    def setup_pytorch_memory(self) -> bool:
        """
        Configure PyTorch-specific memory settings.
        
        Returns:
            True if configuration successful
        """
        try:
            # Set PyTorch multiprocessing method
            import torch.multiprocessing as mp
            mp.set_sharing_strategy('file_system')
            
            # Configure allocator settings for better memory management
            if torch.cuda.is_available():
                # Enable memory efficient attention if available
                if hasattr(torch.cuda, 'memory_efficient_attention'):
                    torch.cuda.memory_efficient_attention.enable()
                    self.logger.info("Enabled memory efficient attention")
                
                # Set growth increment for memory pool
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
                
                # Enable cudnn benchmarking for better performance
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                self.logger.info("Configured PyTorch CUDA settings")
            
            # Set number of threads for CPU operations
            num_cpus = psutil.cpu_count(logical=False)
            torch.set_num_threads(min(num_cpus, 16))  # Cap at 16 threads
            self.logger.info(f"Set PyTorch CPU threads to {min(num_cpus, 16)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up PyTorch memory: {e}")
            return False
    
    def verify_swap_space(self, recommended_gb: int = 100) -> bool:
        """
        Verify swap space configuration.
        
        Args:
            recommended_gb: Recommended swap space in GB
            
        Returns:
            True if swap space is adequate
        """
        try:
            swap = psutil.swap_memory()
            swap_total_gb = swap.total / (1024**3)
            swap_free_gb = swap.free / (1024**3)
            
            if swap_total_gb >= recommended_gb:
                self.logger.info(f"✅ Adequate swap space: {swap_total_gb:.1f}GB total, {swap_free_gb:.1f}GB free")
                return True
            elif swap_total_gb > 0:
                self.logger.warning(f"⚠️ Limited swap space: {swap_total_gb:.1f}GB (recommended: {recommended_gb}GB)")
                return True  # Some swap is better than none
            else:
                self.logger.warning("⚠️ No swap space configured. This may cause OOM issues during quantization.")
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking swap space: {e}")
            return False
    
    def verify_dependencies(self) -> Dict[str, str]:
        """
        Verify all required dependencies are installed.
        
        Returns:
            Dictionary of package names and versions
        """
        dependencies = {}
        required_packages = [
            'torch',
            'transformers',
            'accelerate',
            'llmcompressor',
            'safetensors',
            'pyyaml',
            'psutil',
            'numpy',
            'tqdm'
        ]
        
        # Fix #6: Use instance dep_manager instead of creating new one
        for package in required_packages:
            version = self.dep_manager.get_package_version(package)
            if version:
                dependencies[package] = version
                self.logger.info(f"✅ {package}: {version}")
            else:
                dependencies[package] = "NOT INSTALLED"
                self.logger.warning(f"⚠️ {package}: NOT INSTALLED")
        
        # Check for CUDA-enabled torch
        if 'torch' in dependencies and dependencies['torch'] != "NOT INSTALLED":
            dependencies['torch_cuda'] = torch.version.cuda if torch.cuda.is_available() else "CPU only"
            
        return dependencies
    
    def _check_cuda_toolkit(self) -> bool:
        """
        Check if CUDA toolkit is properly installed.
        
        Returns:
            True if CUDA toolkit is available
        """
        try:
            # Check if nvcc is available
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, 
                                  text=True,
                                  timeout=5,
                                  stdin=subprocess.DEVNULL)
            if result.returncode == 0:
                self.logger.info("CUDA toolkit detected via nvcc")
                return True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        # Check PyTorch CUDA
        if torch.cuda.is_available():
            self.logger.info(f"CUDA available through PyTorch (version: {torch.version.cuda})")
            return True
            
        self.logger.warning("CUDA toolkit not detected")
        return False
    
    def setup_monitoring(self) -> 'MonitoringService':
        """
        Set up system monitoring for the quantization process.
        
        Returns:
            Configured monitoring service instance
        """
        log_dir = self.hardware.offload_folder / "monitoring_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        monitoring = MonitoringService(log_dir)
        self.logger.info(f"Monitoring service initialized with log directory: {log_dir}")
        
        return monitoring
    
    def create_offload_directories(self) -> Dict[str, Path]:
        """
        Create necessary directories for offloading.
        
        Returns:
            Dictionary of created directory paths
        """
        directories = {}
        
        try:
            base_offload = self.hardware.offload_folder
            
            # Create main offload directory
            base_offload.mkdir(parents=True, exist_ok=True)
            directories['base'] = base_offload
            
            # Create subdirectories for different purposes
            subdirs = {
                'model_weights': base_offload / 'model_weights',
                'activations': base_offload / 'activations',
                'checkpoints': base_offload / 'checkpoints',
                'temp': base_offload / 'temp',
                'logs': base_offload / 'logs'
            }
            
            for name, path in subdirs.items():
                path.mkdir(parents=True, exist_ok=True)
                directories[name] = path
                self.logger.info(f"Created offload directory: {path}")
            
            # Verify write permissions
            test_file = base_offload / '.write_test'
            try:
                test_file.touch()
                test_file.unlink()
                directories['writable'] = True
            except Exception as e:
                self.logger.error(f"Cannot write to offload directory: {e}")
                directories['writable'] = False
                
        except Exception as e:
            self.logger.error(f"Error creating offload directories: {e}")
            
        return directories
    
    def optimize_system_settings(self) -> bool:
        """
        Optimize system settings for large model processing.
        
        Optimizations:
        - Disable unnecessary services
        - Set process priority
        - Configure file handles
        
        Returns:
            True if optimizations applied
        """
        optimizations_applied = []
        
        try:
            # Set process priority (nice value)
            # Lower nice value = higher priority
            if hasattr(os, 'nice'):
                try:
                    os.nice(-5)  # Slightly higher priority
                    optimizations_applied.append('process_priority')
                    self.logger.info("Set process priority to -5")
                except PermissionError:
                    self.logger.debug("Cannot change process priority (requires elevated permissions)")
            
            # Increase file descriptor limit (Unix-like systems)
            if sys.platform != 'win32':
                try:
                    import resource
                    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                    resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))
                    optimizations_applied.append('file_descriptors')
                    self.logger.info(f"Increased file descriptor limit to {min(4096, hard)}")
                except Exception as e:
                    self.logger.debug(f"Cannot modify file descriptor limit: {e}")
            
            # Disable Python garbage collection during critical operations
            # (Will be re-enabled selectively)
            import gc
            gc.set_threshold(700, 10, 10)  # Less aggressive GC
            optimizations_applied.append('gc_tuning')
            self.logger.info("Tuned garbage collection thresholds")
            
            # Set environment variables for better performance
            os.environ['OMP_NUM_THREADS'] = str(min(psutil.cpu_count(logical=False), 16))
            os.environ['MKL_NUM_THREADS'] = str(min(psutil.cpu_count(logical=False), 16))
            optimizations_applied.append('thread_settings')
            self.logger.info("Configured OpenMP and MKL thread settings")
            
            return len(optimizations_applied) > 0
            
        except Exception as e:
            self.logger.error(f"Error applying system optimizations: {e}")
            return False
    
    def generate_environment_report(self) -> str:
        """
        Generate comprehensive environment report.
        
        Returns:
            Formatted environment report string
        """
        report_lines = [
            "="*60,
            "ENVIRONMENT SETUP REPORT",
            "="*60,
            f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "HARDWARE CONFIGURATION:",
            "-"*30,
            f"GPU: {self.hardware.gpu_name}",
            f"GPU Memory: {self.hardware.gpu_memory_gb} GB",
            f"System RAM: {self.hardware.cpu_memory_gb} GB",
            f"CUDA Version: {self.hardware.cuda_version}",
            f"Disk Space Required: {self.hardware.disk_space_gb} GB",
            f"Offload Folder: {self.hardware.offload_folder}",
            "",
        ]
        
        # Add system check results
        requirements = self.verify_system_requirements()
        report_lines.extend([
            "SYSTEM REQUIREMENTS CHECK:",
            "-"*30,
        ])
        
        for check, status in requirements.items():
            status_icon = "✅" if status else "❌"
            report_lines.append(f"{status_icon} {check}: {status}")
        
        # Add GPU details
        gpu_info = self.check_gpu_availability()
        if gpu_info['available']:
            report_lines.extend([
                "",
                "GPU DETAILS:",
                "-"*30,
            ])
            for device in gpu_info['devices']:
                report_lines.extend([
                    f"Device {device['index']}: {device['name']}",
                    f"  Memory: {device['memory_gb']:.1f} GB",
                    f"  Compute Capability: {device['capability']}",
                ])
        
        # Add memory configuration
        memory_config = self.configure_memory_settings()
        report_lines.extend([
            "",
            "MEMORY CONFIGURATION:",
            "-"*30,
            f"CPU Available: {memory_config['cpu_available_gb']:.1f}/{memory_config['cpu_total_gb']:.1f} GB",
            f"Swap Total: {memory_config['swap_total_gb']:.1f} GB",
            f"Offload Directory: {memory_config['offload_dir']}",
        ])
        
        # Add dependency check
        dependencies = self.verify_dependencies()
        report_lines.extend([
            "",
            "DEPENDENCIES:",
            "-"*30,
        ])
        for package, version in dependencies.items():
            status_icon = "✅" if version != "NOT INSTALLED" else "❌"
            report_lines.append(f"{status_icon} {package}: {version}")
        
        report_lines.extend([
            "",
            "="*60
        ])
        
        return "\n".join(report_lines)


class MonitoringService:
    """Monitors system resources during quantization."""
    
    def __init__(self, log_dir: Path, interval_seconds: int = 10):
        """
        Initialize monitoring service.
        
        Args:
            log_dir: Directory for monitoring logs
            interval_seconds: Monitoring interval in seconds
        """
        self.log_dir = log_dir
        self.interval = interval_seconds
        self.is_monitoring = threading.Event()
        self.logger = logging.getLogger(__name__)
        self.monitoring_thread = None
        self.metrics_history = {
            'timestamps': [],
            'gpu_memory': [],
            'gpu_utilization': [],
            'cpu_percent': [],
            'ram_used': [],
            'disk_io': []
        }
        self.peak_metrics = {
            'gpu_memory_gb': 0,
            'gpu_utilization': 0,
            'cpu_percent': 0,
            'ram_used_gb': 0
        }
        
    def start_monitoring(self) -> None:
        """Start background system monitoring."""
        if self.is_monitoring.is_set():
            self.logger.warning("Monitoring already running")
            return
        
        self.is_monitoring.set()
        self.logger.info("Starting system monitoring")
        
        # Start monitoring in a separate thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        if not self.is_monitoring.is_set():
            return
        
        self.is_monitoring.clear()
        self.logger.info("Stopping system monitoring")
        
        # Wait for thread to finish
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
            
            # Fix #11: Check if thread is still alive after timeout
            if self.monitoring_thread.is_alive():
                self.logger.warning("Monitoring thread did not stop cleanly within timeout")
                # Give it a bit more time
                self.monitoring_thread.join(timeout=2)
                
                if self.monitoring_thread.is_alive():
                    self.logger.error("Monitoring thread still running - skipping final metrics export")
                    return  # Skip metrics export to avoid race condition
        
        # Safe to export metrics now
        self.export_metrics(self.log_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        while self.is_monitoring.is_set():
            try:
                metrics = self._collect_metrics()
                if metrics:  # Only update if metrics were collected
                    self._update_history(metrics)
                    self._update_peaks(metrics)
                time.sleep(self.interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                
    def _collect_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        metrics = {}
        
        # GPU metrics
        if torch.cuda.is_available():
            try:
                metrics['gpu_memory_gb'] = torch.cuda.memory_allocated() / (1024**3)
                metrics['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
                
                # Try to get utilization via nvidia-ml-py if available
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics['gpu_utilization'] = utilization.gpu
                    metrics['gpu_memory_utilization'] = utilization.memory
                    
                    # Temperature
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    metrics['gpu_temperature'] = temp
                    
                    # Power draw
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
                    metrics['gpu_power_watts'] = power
                    
                except ImportError:
                    pass  # nvidia-ml-py not available
                except Exception:
                    pass  # NVML call failed
                    
            except Exception as e:
                self.logger.debug(f"Error collecting GPU metrics: {e}")
        
        # CPU and memory metrics
        metrics['cpu_percent'] = psutil.cpu_percent(interval=1)
        metrics['cpu_per_core'] = psutil.cpu_percent(interval=1, percpu=True)
        
        mem = psutil.virtual_memory()
        metrics['ram_used_gb'] = mem.used / (1024**3)
        metrics['ram_available_gb'] = mem.available / (1024**3)
        metrics['ram_percent'] = mem.percent
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            metrics['disk_read_mb'] = disk_io.read_bytes / (1024**2)
            metrics['disk_write_mb'] = disk_io.write_bytes / (1024**2)
        
        # Network I/O (might be relevant for model downloading)
        net_io = psutil.net_io_counters()
        metrics['net_sent_mb'] = net_io.bytes_sent / (1024**2)
        metrics['net_recv_mb'] = net_io.bytes_recv / (1024**2)
        
        metrics['timestamp'] = datetime.now().isoformat()
        
        return metrics
    
    def _update_history(self, metrics: Dict[str, float]) -> None:
        """Update metrics history."""
        if not metrics:
            return
            
        self.metrics_history['timestamps'].append(metrics.get('timestamp'))
        
        # Keep only last 1000 samples to prevent memory growth
        max_samples = 1000
        if len(self.metrics_history['timestamps']) > max_samples:
            for key in self.metrics_history:
                if isinstance(self.metrics_history[key], list):
                    self.metrics_history[key] = self.metrics_history[key][-max_samples:]
        
        # Update specific metrics
        if 'gpu_memory_gb' in metrics:
            self.metrics_history['gpu_memory'].append(metrics['gpu_memory_gb'])
        if 'gpu_utilization' in metrics:
            self.metrics_history['gpu_utilization'].append(metrics['gpu_utilization'])
        if 'cpu_percent' in metrics:
            self.metrics_history['cpu_percent'].append(metrics['cpu_percent'])
        if 'ram_used_gb' in metrics:
            self.metrics_history['ram_used'].append(metrics['ram_used_gb'])
            
    def _update_peaks(self, metrics: Dict[str, float]) -> None:
        """Update peak metrics."""
        if not metrics:
            return
            
        for key in ['gpu_memory_gb', 'gpu_utilization', 'cpu_percent', 'ram_used_gb']:
            if key in metrics and key in self.peak_metrics:
                self.peak_metrics[key] = max(self.peak_metrics[key], metrics[key])
    
    def log_gpu_usage(self) -> Dict[str, float]:
        """
        Log current GPU usage.
        
        Returns:
            Dictionary with GPU metrics
        """
        gpu_metrics = {}
        
        if torch.cuda.is_available():
            gpu_metrics['memory_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
            gpu_metrics['memory_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
            gpu_metrics['memory_free_gb'] = (torch.cuda.get_device_properties(0).total_memory - 
                                            torch.cuda.memory_reserved()) / (1024**3)
            
            self.logger.info(f"GPU Memory - Allocated: {gpu_metrics['memory_allocated_gb']:.2f}GB, "
                           f"Reserved: {gpu_metrics['memory_reserved_gb']:.2f}GB, "
                           f"Free: {gpu_metrics['memory_free_gb']:.2f}GB")
        else:
            self.logger.warning("GPU not available for monitoring")
            
        return gpu_metrics
    
    def log_cpu_memory_usage(self) -> Dict[str, float]:
        """
        Log current CPU and memory usage.
        
        Returns:
            Dictionary with CPU/memory metrics
        """
        cpu_metrics = {}
        
        # CPU usage
        cpu_metrics['cpu_percent'] = psutil.cpu_percent(interval=1)
        cpu_metrics['cpu_per_core'] = psutil.cpu_percent(interval=1, percpu=True)
        
        # Memory usage
        mem = psutil.virtual_memory()
        cpu_metrics['ram_used_gb'] = mem.used / (1024**3)
        cpu_metrics['ram_available_gb'] = mem.available / (1024**3)
        cpu_metrics['ram_percent'] = mem.percent
        
        # Swap usage
        swap = psutil.swap_memory()
        cpu_metrics['swap_used_gb'] = swap.used / (1024**3)
        cpu_metrics['swap_percent'] = swap.percent
        
        self.logger.info(f"CPU: {cpu_metrics['cpu_percent']:.1f}%, "
                        f"RAM: {cpu_metrics['ram_used_gb']:.1f}/{cpu_metrics['ram_used_gb'] + cpu_metrics['ram_available_gb']:.1f}GB "
                        f"({cpu_metrics['ram_percent']:.1f}%), "
                        f"Swap: {cpu_metrics['swap_percent']:.1f}%")
        
        return cpu_metrics
    
    def log_disk_usage(self) -> Dict[str, float]:
        """
        Log disk usage and I/O statistics.
        
        Returns:
            Dictionary with disk metrics
        """
        disk_metrics = {}
        
        # Disk space
        disk = psutil.disk_usage('/')
        disk_metrics['disk_used_gb'] = disk.used / (1024**3)
        disk_metrics['disk_free_gb'] = disk.free / (1024**3)
        disk_metrics['disk_percent'] = disk.percent
        
        # Disk I/O
        io = psutil.disk_io_counters()
        if io:
            disk_metrics['read_mb'] = io.read_bytes / (1024**2)
            disk_metrics['write_mb'] = io.write_bytes / (1024**2)
            disk_metrics['read_count'] = io.read_count
            disk_metrics['write_count'] = io.write_count
        
        self.logger.info(f"Disk - Used: {disk_metrics['disk_used_gb']:.1f}GB, "
                        f"Free: {disk_metrics['disk_free_gb']:.1f}GB "
                        f"({disk_metrics['disk_percent']:.1f}%)")
        
        return disk_metrics
    
    def get_peak_usage(self) -> Dict[str, float]:
        """
        Get peak resource usage since monitoring started.
        
        Returns:
            Dictionary with peak usage metrics
        """
        return self.peak_metrics.copy()
    
    def generate_report(self) -> str:
        """
        Generate monitoring report.
        
        Returns:
            Formatted report string
        """
        report_lines = [
            "="*50,
            "MONITORING REPORT",
            "="*50,
            "",
            "PEAK USAGE:",
            "-"*30,
            f"GPU Memory: {self.peak_metrics['gpu_memory_gb']:.2f} GB",
            f"GPU Utilization: {self.peak_metrics['gpu_utilization']:.1f}%",
            f"CPU Usage: {self.peak_metrics['cpu_percent']:.1f}%",
            f"RAM Usage: {self.peak_metrics['ram_used_gb']:.1f} GB",
            "",
            "CURRENT STATUS:",
            "-"*30,
        ]
        
        # Add current metrics
        current_gpu = self.log_gpu_usage()
        current_cpu = self.log_cpu_memory_usage()
        
        if current_gpu:
            report_lines.append(f"GPU Memory Allocated: {current_gpu.get('memory_allocated_gb', 0):.2f} GB")
        report_lines.append(f"CPU Usage: {current_cpu['cpu_percent']:.1f}%")
        report_lines.append(f"RAM Usage: {current_cpu['ram_used_gb']:.1f} GB")
        
        if self.metrics_history['timestamps']:
            report_lines.extend([
                "",
                "MONITORING DURATION:",
                "-"*30,
                f"Start: {self.metrics_history['timestamps'][0]}",
                f"End: {self.metrics_history['timestamps'][-1]}",
                f"Samples Collected: {len(self.metrics_history['timestamps'])}",
            ])
        
        report_lines.append("="*50)
        
        return "\n".join(report_lines)
    
    def export_metrics(self, output_path: Path) -> bool:
        """
        Export collected metrics to file.
        
        Args:
            output_path: Path for metrics export
            
        Returns:
            True if export successful
        """
        try:
            # Fix #9: Better JSON serialization error handling
            export_data = {
                'peak_metrics': self.peak_metrics,
                'history': self.metrics_history,
                'summary': {
                    'monitoring_duration_seconds': len(self.metrics_history['timestamps']) * self.interval,
                    'samples_collected': len(self.metrics_history['timestamps']),
                    'interval_seconds': self.interval
                }
            }
            
            # Fix #5: Ensure directory exists before writing
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            except (TypeError, ValueError) as e:
                # Fix #9: Handle JSON serialization errors specifically
                self.logger.warning(f"Failed to serialize full metrics: {e}")
                
                # Try simplified export
                simplified_data = {
                    'peak_metrics': self.peak_metrics,
                    'summary': export_data['summary'],
                    'error': 'Full history too large or contains non-serializable objects'
                }
                
                with open(output_path, 'w') as f:
                    json.dump(simplified_data, f, indent=2, default=str)
                
                self.logger.info(f"Simplified metrics exported to {output_path}")
                return True
            
            self.logger.info(f"Metrics exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
            return False


class DependencyManager:
    """Manages Python package dependencies."""
    
    def __init__(self):
        """Initialize dependency manager."""
        self.logger = logging.getLogger(__name__)
        
    def check_package_installed(self, package_name: str) -> bool:
        """
        Check if a package is installed.
        
        Args:
            package_name: Name of the package
            
        Returns:
            True if package is installed
        """
        try:
            importlib.import_module(package_name)
            return True
        except ImportError:
            return False
    
    def get_package_version(self, package_name: str) -> Optional[str]:
        """
        Get version of installed package.
        
        Args:
            package_name: Name of the package
            
        Returns:
            Version string or None if not installed
        """
        try:
            # Try using pkg_resources first
            try:
                version = pkg_resources.get_distribution(package_name).version
                return version
            except:
                pass
            
            # Try importing and checking __version__
            module = importlib.import_module(package_name)
            if hasattr(module, '__version__'):
                return module.__version__
            
            # Special cases
            if package_name == 'torch':
                import torch
                return torch.__version__
            elif package_name == 'transformers':
                import transformers
                return transformers.__version__
            
            # If we can import but can't get version, return "installed"
            return "installed"
            
        except ImportError:
            return None
        except Exception as e:
            self.logger.debug(f"Error getting version for {package_name}: {e}")
            return None
    
    def install_package(self, package_spec: str) -> bool:
        """
        Install a Python package.
        
        Args:
            package_spec: Package specification (e.g., "torch>=2.0.0")
            
        Returns:
            True if installation successful
        """
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_spec],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                self.logger.info(f"Successfully installed {package_spec}")
                return True
            else:
                self.logger.error(f"Failed to install {package_spec}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Installation of {package_spec} timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error installing {package_spec}: {e}")
            return False
    
    def verify_cuda_toolkit(self) -> bool:
        """
        Verify CUDA toolkit installation.
        
        Returns:
            True if CUDA toolkit properly installed
        """
        cuda_available = False
        
        # Check via torch
        if self.check_package_installed('torch'):
            import torch
            if torch.cuda.is_available():
                cuda_available = True
                self.logger.info(f"CUDA available via PyTorch: {torch.version.cuda}")
        
        # Check nvcc
        try:
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, 
                                  text=True,
                                  timeout=5,
                                  stdin=subprocess.DEVNULL)
            if result.returncode == 0:
                cuda_available = True
                self.logger.info("CUDA toolkit (nvcc) found")
        except:
            pass
        
        # Check nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi'], 
                                  capture_output=True, 
                                  text=True,
                                  timeout=5,
                                  stdin=subprocess.DEVNULL)
            if result.returncode == 0:
                cuda_available = True
                self.logger.info("NVIDIA driver (nvidia-smi) found")
        except:
            pass
        
        return cuda_available
    
    def verify_torch_cuda(self) -> bool:
        """
        Verify PyTorch CUDA support.
        
        Returns:
            True if PyTorch has CUDA support
        """
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                self.logger.info(f"PyTorch CUDA support verified - "
                              f"PyTorch: {torch.__version__}, "
                              f"CUDA: {torch.version.cuda}")
                
                # Test CUDA operations
                try:
                    test_tensor = torch.randn(10, 10).cuda()
                    result = torch.sum(test_tensor)
                    self.logger.info("CUDA tensor operations working")
                    return True
                except Exception as e:
                    self.logger.error(f"CUDA tensor operations failed: {e}")
                    return False
            else:
                self.logger.warning("PyTorch does not have CUDA support")
                return False
                
        except ImportError:
            self.logger.error("PyTorch not installed")
            return False
        except Exception as e:
            self.logger.error(f"Error verifying PyTorch CUDA: {e}")
            return False


def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging for Phase 1.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"phase1_setup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('phase1_environment_setup')
    logger.info(f"Logging initialized - Log file: {log_file}")
    
    return logger


def run_phase1(hardware_config: HardwareConfig, 
               project_dir: Path) -> Dict[str, Any]:
    """
    Execute Phase 1: Environment Setup.
    
    Args:
        hardware_config: Hardware configuration
        project_dir: Project directory (can be string or Path)
        
    Returns:
        Dictionary with phase results and status
    """
    # Fix #17: Ensure project_dir is a Path object
    project_dir = Path(project_dir)
    
    # Set up logging
    log_dir = project_dir / "logs"
    logger = setup_logging(log_dir)
    
    logger.info("="*60)
    logger.info("PHASE 1: ENVIRONMENT SETUP STARTED")
    logger.info("="*60)
    
    results = {
        'success': False,
        'requirements_met': False,
        'cuda_ready': False,
        'monitoring_started': False,
        'directories_created': False,
        'report': None,
        'errors': []
    }
    
    try:
        # Initialize environment setup
        setup = EnvironmentSetup(hardware_config)
        
        # 1. Verify system requirements
        logger.info("Checking system requirements...")
        requirements = setup.verify_system_requirements()
        results['requirements'] = requirements
        results['requirements_met'] = requirements.get('all_critical_requirements_met', False)
        
        if not results['requirements_met']:
            logger.error("Critical system requirements not met")
            results['errors'].append("System requirements check failed")
            # Continue anyway to gather more information
        
        # 2. Setup CUDA environment
        logger.info("Setting up CUDA environment...")
        cuda_success = setup.setup_cuda_environment()
        results['cuda_ready'] = cuda_success
        
        if not cuda_success:
            logger.warning("CUDA setup encountered issues")
            results['errors'].append("CUDA setup incomplete")
        
        # 3. Configure memory settings
        logger.info("Configuring memory settings...")
        memory_config = setup.configure_memory_settings()
        results['memory_config'] = memory_config
        
        # 4. Create offload directories
        logger.info("Creating offload directories...")
        directories = setup.create_offload_directories()
        results['directories'] = directories
        results['directories_created'] = directories.get('writable', False)
        
        # 5. Apply system optimizations
        logger.info("Applying system optimizations...")
        optimizations = setup.optimize_system_settings()
        results['optimizations_applied'] = optimizations
        
        # 6. Start monitoring
        logger.info("Starting system monitoring...")
        monitoring_service = setup.setup_monitoring()
        monitoring_service.start_monitoring()
        results['monitoring_started'] = True
        results['monitoring_service'] = monitoring_service
        
        # 7. Generate report
        logger.info("Generating environment report...")
        report = setup.generate_environment_report()
        results['report'] = report
        
        # Fix #5: Ensure directory exists before saving report
        report_file = project_dir / "environment_report.txt"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Environment report saved to {report_file}")
        
        # Fix #20: Use .get() with defaults to prevent KeyError
        results['success'] = (
            results.get('requirements_met', False) and 
            results.get('cuda_ready', False) and 
            results.get('directories_created', False)
        )
        
        if results['success']:
            logger.info("✅ PHASE 1: ENVIRONMENT SETUP COMPLETED SUCCESSFULLY")
        else:
            logger.warning("⚠️ PHASE 1: ENVIRONMENT SETUP COMPLETED WITH WARNINGS")
            
    except Exception as e:
        logger.error(f"Fatal error in Phase 1: {e}", exc_info=True)
        results['errors'].append(str(e))
        results['success'] = False
    
    logger.info("="*60)
    
    return results
    
    def setup_monitoring(self) -> 'MonitoringService':
        """
        Set up system monitoring for the quantization process.
        
        Returns:
            Configured monitoring service instance
        """
        pass
    
    def create_offload_directories(self) -> Dict[str, Path]:
        """
        Create necessary directories for offloading.
        
        Returns:
            Dictionary of created directory paths
        """
        pass
    
    def optimize_system_settings(self) -> bool:
        """
        Optimize system settings for large model processing.
        
        Optimizations:
        - Disable unnecessary services
        - Set process priority
        - Configure file handles
        
        Returns:
            True if optimizations applied
        """
        pass
    
    def generate_environment_report(self) -> str:
        """
        Generate comprehensive environment report.
        
        Returns:
            Formatted environment report string
        """
        pass


class MonitoringService:
    """Monitors system resources during quantization."""
    
    def __init__(self, log_dir: Path, interval_seconds: int = 10):
        """
        Initialize monitoring service.
        
        Args:
            log_dir: Directory for monitoring logs
            interval_seconds: Monitoring interval in seconds
        """
        self.log_dir = log_dir
        self.interval = interval_seconds
        self.is_monitoring = False
        self.logger = logging.getLogger(__name__)
        
    def start_monitoring(self) -> None:
        """Start background system monitoring."""
        pass
    
    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        pass
    
    def log_gpu_usage(self) -> Dict[str, float]:
        """
        Log current GPU usage.
        
        Metrics:
        - Memory used/total
        - GPU utilization
        - Temperature
        - Power draw
        
        Returns:
            Dictionary with GPU metrics
        """
        pass
    
    def log_cpu_memory_usage(self) -> Dict[str, float]:
        """
        Log current CPU and memory usage.
        
        Metrics:
        - CPU utilization per core
        - RAM used/total
        - Swap used/total
        - Disk I/O
        
        Returns:
            Dictionary with CPU/memory metrics
        """
        pass
    
    def log_disk_usage(self) -> Dict[str, float]:
        """
        Log disk usage and I/O statistics.
        
        Returns:
            Dictionary with disk metrics
        """
        pass
    
    def get_peak_usage(self) -> Dict[str, float]:
        """
        Get peak resource usage since monitoring started.
        
        Returns:
            Dictionary with peak usage metrics
        """
        pass
    
    def generate_report(self) -> str:
        """
        Generate monitoring report.
        
        Returns:
            Formatted report string
        """
        pass
    
    def export_metrics(self, output_path: Path) -> bool:
        """
        Export collected metrics to file.
        
        Args:
            output_path: Path for metrics export
            
        Returns:
            True if export successful
        """
        pass


class DependencyManager:
    """Manages Python package dependencies."""
    
    def __init__(self):
        """Initialize dependency manager."""
        self.logger = logging.getLogger(__name__)
        
    def check_package_installed(self, package_name: str) -> bool:
        """
        Check if a package is installed.
        
        Args:
            package_name: Name of the package
            
        Returns:
            True if package is installed
        """
        pass
    
    def get_package_version(self, package_name: str) -> Optional[str]:
        """
        Get version of installed package.
        
        Args:
            package_name: Name of the package
            
        Returns:
            Version string or None if not installed
        """
        pass
    
    def install_package(self, package_spec: str) -> bool:
        """
        Install a Python package.
        
        Args:
            package_spec: Package specification (e.g., "torch>=2.0.0")
            
        Returns:
            True if installation successful
        """
        pass
    
    def verify_cuda_toolkit(self) -> bool:
        """
        Verify CUDA toolkit installation.
        
        Returns:
            True if CUDA toolkit properly installed
        """
        pass
    
    def verify_torch_cuda(self) -> bool:
        """
        Verify PyTorch CUDA support.
        
        Returns:
            True if PyTorch has CUDA support
        """
        pass


def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging for Phase 1.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    pass


def run_phase1(hardware_config: HardwareConfig, 
               project_dir: Path) -> Dict[str, Any]:
    """
    Execute Phase 1: Environment Setup.
    
    Args:
        hardware_config: Hardware configuration
        project_dir: Project directory
        
    Returns:
        Dictionary with phase results and status
    """
    pass


if __name__ == "__main__":
    # Example standalone execution
    config = HardwareConfig(
        gpu_memory_gb=24,
        cpu_memory_gb=256,
        gpu_name="RTX 3090",
        cuda_version="11.8",
        disk_space_gb=500,
        offload_folder=Path("./offload")
    )
    
    result = run_phase1(config, Path("./project"))
    print(f"Phase 1 completed: {result['success']}")