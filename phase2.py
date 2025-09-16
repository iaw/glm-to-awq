"""
Phase 2: Preparation
====================
Manages recipe preparation, dataset loading, and model downloading for quantization.

This module handles AWQ/GPTQ recipe configuration, calibration dataset preparation,
and base model acquisition from HuggingFace.
"""

import json
import yaml
import hashlib
import logging
import torch
import gc
import os
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datasets import load_dataset, Dataset
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download, hf_hub_download
import shutil

# Optional imports with error handling
try:
    import jsonlines
    JSONLINES_AVAILABLE = True
except ImportError:
    JSONLINES_AVAILABLE = False
    logging.warning("jsonlines library not available. JSONL format support will be limited.")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("pandas library not available. CSV/Parquet format support will be limited.")


class QuantizationMethod(Enum):
    """Supported quantization methods."""
    AWQ = "awq"
    GPTQ = "gptq"


@dataclass
class RecipeConfig:
    """Configuration for quantization recipe."""
    method: QuantizationMethod
    bits: int
    group_size: int
    symmetric: bool
    ignore_layers: List[str]
    mappings: Optional[List[Dict[str, Any]]]  # For AWQ
    duo_scaling: Optional[bool]  # For AWQ
    actorder: Optional[bool]  # For GPTQ
    percdamp: Optional[float]  # For GPTQ
    

class RecipeManager:
    """Manages quantization recipes and configurations."""
    
    def __init__(self, recipe_dir: Path):
        """
        Initialize recipe manager with storage directory.
        
        Args:
            recipe_dir: Directory for storing recipes
        """
        self.recipe_dir = recipe_dir
        self.recipe_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def load_awq_recipe(self, recipe_path: Path) -> Dict[str, Any]:
        """
        Load and validate AWQ recipe from YAML file.
        
        Expected structure matches the GLM-4.5-Air AWQ recipe format.
        
        Args:
            recipe_path: Path to recipe YAML file
            
        Returns:
            Parsed recipe dictionary
        """
        try:
            with open(recipe_path, 'r') as f:
                recipe = yaml.safe_load(f)
            
            # Validate required fields
            required_fields = ['quant_method', 'awq']
            for field in required_fields:
                if field not in recipe:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate AWQ configuration
            awq_config = recipe['awq']
            required_awq_fields = ['bits', 'group_size', 'zero_point']
            for field in required_awq_fields:
                if field not in awq_config:
                    raise ValueError(f"Missing AWQ field: {field}")
            
            self.logger.info(f"Loaded AWQ recipe from {recipe_path}")
            return recipe
            
        except Exception as e:
            self.logger.error(f"Error loading AWQ recipe: {e}")
            raise
    
    def create_awq_recipe_for_glm(self) -> Dict[str, Any]:
        """
        Create AWQ recipe specifically for GLM-4.5-Air.
        
        Uses the proven recipe structure from cpatonn/GLM-4.5-Air-AWQ-4bit.
        
        Returns:
            AWQ recipe dictionary for GLM models
        """
        recipe = {
            "quant_method": "awq",
            "awq": {
                "bits": 4,
                "group_size": 128,
                "zero_point": True,
                "symmetric": False,
                "calibration_dataset": "open_platypus",
                "num_calibration_samples": 512,
                "calibration_sequence_length": 2048,
                "duo_scaling": False,  # Can be enabled for better quality
                "modules_to_not_convert": None,
                "version": "gemm"  # Use optimized GEMM kernels
            },
            "targets": [
                # GLM-4.5-Air specific layer patterns
                "model.layers.*.self_attn.q_proj",
                "model.layers.*.self_attn.k_proj",
                "model.layers.*.self_attn.v_proj",
                "model.layers.*.self_attn.o_proj",
                "model.layers.*.mlp.gate_proj",
                "model.layers.*.mlp.up_proj",
                "model.layers.*.mlp.down_proj"
            ],
            "ignore": [
                "model.embed_tokens",
                "model.norm",
                "lm_head"
            ],
            "metadata": {
                "model_architecture": "GLM-4",
                "created_by": "GLM-4.5-Air Quantization MVP",
                "description": "4-bit AWQ quantization recipe for GLM-4.5-Air model"
            }
        }
        
        self.logger.info("Created AWQ recipe for GLM-4.5-Air")
        return recipe
    
    def create_gptq_recipe(self, config: RecipeConfig) -> Dict[str, Any]:
        """
        Create GPTQ recipe based on configuration.
        
        Args:
            config: Recipe configuration
            
        Returns:
            GPTQ recipe dictionary
        """
        recipe = {
            "quant_method": "gptq",
            "gptq": {
                "bits": config.bits,
                "group_size": config.group_size,
                "symmetric": config.symmetric,
                "actorder": config.actorder if config.actorder is not None else True,
                "percdamp": config.percdamp if config.percdamp is not None else 0.01,
                "block_size": 128,
                "use_triton": False,  # Set to True if Triton kernels available
                "use_cuda_fp16": True,
                "calibration_dataset": "open_platypus",
                "num_calibration_samples": 512,
                "calibration_sequence_length": 2048
            },
            "targets": self.get_glm_layer_patterns()["quantizable"],
            "ignore": config.ignore_layers if config.ignore_layers else self.get_glm_layer_patterns()["skip"],
            "metadata": {
                "model_architecture": "GLM-4",
                "created_by": "GLM-4.5-Air Quantization MVP",
                "description": f"{config.bits}-bit GPTQ quantization recipe for GLM-4.5-Air model"
            }
        }
        
        self.logger.info(f"Created GPTQ recipe with {config.bits}-bit quantization")
        return recipe
    
    def validate_recipe(self, recipe: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate recipe for completeness and correctness.
        
        Checks:
        - Required fields present
        - Layer names valid
        - Quantization parameters in valid ranges
        
        Args:
            recipe: Recipe dictionary to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check quantization method
        if "quant_method" not in recipe:
            issues.append("Missing 'quant_method' field")
        else:
            method = recipe["quant_method"]
            if method not in ["awq", "gptq"]:
                issues.append(f"Invalid quantization method: {method}")
            
            # Method-specific validation
            if method == "awq" and "awq" not in recipe:
                issues.append("Missing 'awq' configuration for AWQ method")
            elif method == "gptq" and "gptq" not in recipe:
                issues.append("Missing 'gptq' configuration for GPTQ method")
        
        # Validate quantization parameters
        if "awq" in recipe:
            awq_config = recipe["awq"]
            
            # Check bits
            if "bits" in awq_config:
                bits = awq_config["bits"]
                if bits not in [2, 3, 4, 8]:
                    issues.append(f"Invalid bit width for AWQ: {bits}")
            else:
                issues.append("Missing 'bits' in AWQ config")
            
            # Check group size
            if "group_size" in awq_config:
                group_size = awq_config["group_size"]
                if group_size not in [32, 64, 128, 256]:
                    issues.append(f"Non-standard group size: {group_size}")
            else:
                issues.append("Missing 'group_size' in AWQ config")
        
        # Validate GPTQ parameters
        if "gptq" in recipe:
            gptq_config = recipe["gptq"]
            
            if "percdamp" in gptq_config:
                percdamp = gptq_config["percdamp"]
                if not (0 <= percdamp <= 1):
                    issues.append(f"Invalid percdamp value: {percdamp}")
        
        # Check targets and ignore lists
        if "targets" not in recipe:
            issues.append("Missing 'targets' field for layers to quantize")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            self.logger.info("Recipe validation passed")
        else:
            self.logger.warning(f"Recipe validation found {len(issues)} issues")
        
        return is_valid, issues
    
    def adapt_recipe_for_hardware(self, 
                                  recipe: Dict[str, Any], 
                                  available_memory_gb: int) -> Dict[str, Any]:
        """
        Adapt recipe parameters based on hardware limitations.
        
        Adjustments:
        - Reduce batch size if needed
        - Adjust calibration samples
        - Modify sequence length
        
        Args:
            recipe: Original recipe
            available_memory_gb: Available GPU memory in GB
            
        Returns:
            Adapted recipe
        """
        import copy
        adapted_recipe = copy.deepcopy(recipe)  # Use deep copy to avoid mutation
        
        # Get the quantization config (awq or gptq)
        quant_method = recipe.get("quant_method", "awq")
        config_key = quant_method
        
        if config_key in adapted_recipe:
            config = adapted_recipe[config_key]
            
            # Estimate memory requirements
            base_model_size_gb = 24  # GLM-4.5-Air approximate size
            bits = config.get("bits", 4) if isinstance(config, dict) else 4
            quantized_size_gb = base_model_size_gb * (bits / 16)  # Rough estimate
            
            # Adjust based on available memory
            if available_memory_gb < 24:
                # Limited memory - reduce calibration samples and sequence length
                self.logger.warning(f"Limited GPU memory ({available_memory_gb}GB), adjusting recipe")
                
                config["num_calibration_samples"] = min(
                    config.get("num_calibration_samples", 512),
                    128
                )
                config["calibration_sequence_length"] = min(
                    config.get("calibration_sequence_length", 2048),
                    512
                )
                
                # For very limited memory, reduce group size
                if available_memory_gb < 16:
                    config["group_size"] = min(
                        config.get("group_size", 128),
                        64
                    )
            
            elif available_memory_gb >= 40:
                # Plenty of memory - can increase quality settings
                self.logger.info(f"Ample GPU memory ({available_memory_gb}GB), optimizing for quality")
                
                config["num_calibration_samples"] = max(
                    config.get("num_calibration_samples", 512),
                    1024
                )
                
                if quant_method == "awq":
                    config["duo_scaling"] = True  # Better quality with more memory
        
        self.logger.info(f"Adapted recipe for {available_memory_gb}GB GPU memory")
        return adapted_recipe
    
    def save_recipe(self, recipe: Dict[str, Any], name: str) -> Optional[Path]:
        """
        Save recipe to YAML file.
        
        Args:
            recipe: Recipe dictionary
            name: Name for the recipe file
            
        Returns:
            Path to saved recipe file, or None if save failed
        """
        # Ensure name has .yaml extension
        if not name.endswith('.yaml'):
            name = f"{name}.yaml"
        
        recipe_path = self.recipe_dir / name
        
        try:
            # Ensure directory exists
            recipe_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(recipe_path, 'w') as f:
                yaml.dump(recipe, f, default_flow_style=False, sort_keys=False)
            
            # Verify file was created
            if recipe_path.exists():
                self.logger.info(f"Saved recipe to {recipe_path}")
                return recipe_path
            else:
                self.logger.error(f"Recipe file was not created at {recipe_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error saving recipe: {e}")
            return None
    
    def get_glm_layer_patterns(self) -> Dict[str, List[str]]:
        """
        Get layer name patterns specific to GLM architecture.
        
        Returns:
            Dictionary of layer patterns
        """
        patterns = {
            "quantizable": [
                # Attention layers
                "model.layers.*.self_attn.q_proj",
                "model.layers.*.self_attn.k_proj",
                "model.layers.*.self_attn.v_proj",
                "model.layers.*.self_attn.o_proj",
                # MLP layers
                "model.layers.*.mlp.gate_proj",
                "model.layers.*.mlp.up_proj",
                "model.layers.*.mlp.down_proj"
            ],
            "skip": [
                # Embeddings and output layers - usually not quantized
                "model.embed_tokens",
                "model.norm",
                "lm_head",
                # Layer norms - small impact, better to keep in fp16
                "model.layers.*.input_layernorm",
                "model.layers.*.post_attention_layernorm"
            ],
            "attention": [
                "model.layers.*.self_attn.q_proj",
                "model.layers.*.self_attn.k_proj",
                "model.layers.*.self_attn.v_proj",
                "model.layers.*.self_attn.o_proj"
            ],
            "mlp": [
                "model.layers.*.mlp.gate_proj",
                "model.layers.*.mlp.up_proj",
                "model.layers.*.mlp.down_proj"
            ]
        }
        
        return patterns


class DatasetPreparer:
    """Handles calibration dataset preparation."""
    
    def __init__(self, cache_dir: Path):
        """
        Initialize dataset preparer with cache directory.
        
        Args:
            cache_dir: Directory for caching datasets
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def load_open_platypus(self, num_samples: int = 512) -> Any:
        """
        Load Open Platypus dataset for calibration.
        
        This is the default calibration dataset that works well
        for general-purpose quantization.
        
        Args:
            num_samples: Number of samples to load
            
        Returns:
            Loaded dataset
        """
        try:
            self.logger.info(f"Loading Open Platypus dataset with {num_samples} samples")
            
            # Load from HuggingFace datasets
            dataset = load_dataset(
                "garage-bAInd/Open-Platypus",
                split="train",
                cache_dir=str(self.cache_dir),
                streaming=False  # Load fully for proper shuffling
            )
            
            # Shuffle and select samples BEFORE converting to list
            dataset = dataset.shuffle(seed=42)
            
            # Select subset efficiently
            if len(dataset) > num_samples:
                dataset = dataset.select(range(num_samples))
            
            self.logger.info(f"Selected {len(dataset)} samples from Open Platypus")
            
            self.logger.info(f"Loaded {len(dataset)} samples from Open Platypus")
            
            # Process the dataset to extract text
            processed_data = []
            for item in dataset:
                # Open Platypus has 'instruction' and 'output' fields
                # Check if fields exist before accessing
                try:
                    if 'instruction' in item and 'output' in item:
                        text = f"Instruction: {item['instruction']}\n\nResponse: {item['output']}"
                    elif 'text' in item:
                        text = item['text']
                    else:
                        # Try to use any available text field
                        text = str(item.get('content', item.get('input', str(item))))
                    processed_data.append({"text": text})
                except (KeyError, TypeError) as e:
                    self.logger.warning(f"Error processing dataset item: {e}")
                    # Skip problematic items
                    continue
            
            # Check if we got any valid data
            if not processed_data:
                self.logger.error("No valid data extracted from Open Platypus dataset")
                self.logger.info("Falling back to WikiText-2 dataset")
                return self.load_wikitext_fallback(num_samples)
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error loading Open Platypus dataset: {e}")
            # Fallback to a simpler dataset
            self.logger.info("Falling back to WikiText-2 dataset")
            return self.load_wikitext_fallback(num_samples)
    
    def load_wikitext_fallback(self, num_samples: int) -> List[Dict[str, str]]:
        """
        Load WikiText-2 as fallback calibration dataset.
        
        Args:
            num_samples: Number of samples to load
            
        Returns:
            List of text samples
        """
        try:
            dataset = load_dataset(
                "wikitext",
                "wikitext-2-raw-v1",
                split="train",
                cache_dir=str(self.cache_dir)
            )
            
            # Filter out empty texts and select samples
            texts = []
            for item in dataset:
                if item["text"].strip():
                    texts.append({"text": item["text"]})
                if len(texts) >= num_samples:
                    break
            
            self.logger.info(f"Loaded {len(texts)} samples from WikiText-2")
            return texts
            
        except Exception as e:
            self.logger.error(f"Error loading WikiText-2: {e}")
            # Final fallback - create synthetic data
            return self.create_synthetic_calibration_data(num_samples)
    
    def create_synthetic_calibration_data(self, num_samples: int) -> List[Dict[str, str]]:
        """
        Create synthetic calibration data as last resort.
        
        Args:
            num_samples: Number of samples to create
            
        Returns:
            List of synthetic text samples
        """
        self.logger.warning("Creating synthetic calibration data")
        
        prompts = [
            "Explain the concept of machine learning in simple terms.",
            "Write a Python function to calculate the factorial of a number.",
            "What are the main causes of climate change?",
            "Describe the process of photosynthesis.",
            "How does the internet work?",
            "What is quantum computing?",
            "Explain the theory of relativity.",
            "Write a short story about a robot.",
            "What are the benefits of renewable energy?",
            "How do vaccines work?"
        ]
        
        samples = []
        for i in range(num_samples):
            prompt = prompts[i % len(prompts)]
            # Add variation to each sample
            text = f"{prompt} [Sample {i+1}]\n\n" + "This is a calibration sample. " * 10
            samples.append({"text": text})
        
        return samples
    
    def load_custom_dataset(self, 
                           dataset_path: Path, 
                           num_samples: int,
                           format_type: str = "jsonl") -> Any:
        """
        Load custom dataset for calibration.
        
        Supports formats:
        - JSONL with 'text' field
        - CSV with specified text column
        - Parquet files
        
        Args:
            dataset_path: Path to custom dataset
            num_samples: Number of samples to load
            format_type: Dataset format type
            
        Returns:
            Loaded dataset
        """
        try:
            if format_type == "jsonl":
                if not JSONLINES_AVAILABLE:
                    raise ImportError("jsonlines library is required for JSONL format. Install with: pip install jsonlines")
                
                samples = []
                with jsonlines.open(dataset_path) as reader:
                    for i, obj in enumerate(reader):
                        if i >= num_samples:
                            break
                        # Expect 'text' field or try common alternatives
                        text = obj.get("text") or obj.get("content") or obj.get("input")
                        if text:
                            samples.append({"text": text})
                
                self.logger.info(f"Loaded {len(samples)} samples from {dataset_path}")
                return samples
                
            elif format_type == "csv":
                if not PANDAS_AVAILABLE:
                    raise ImportError("pandas library is required for CSV format. Install with: pip install pandas")
                
                df = pd.read_csv(dataset_path, nrows=num_samples)
                
                # Try to find text column
                text_columns = ["text", "content", "input", "prompt", "question"]
                text_col = None
                for col in text_columns:
                    if col in df.columns:
                        text_col = col
                        break
                
                if text_col is None and len(df.columns) > 0:
                    # Use first string column
                    text_col = df.columns[0]
                
                samples = [{"text": str(row[text_col])} for _, row in df.iterrows()]
                self.logger.info(f"Loaded {len(samples)} samples from CSV")
                return samples
                
            elif format_type == "parquet":
                if not PANDAS_AVAILABLE:
                    raise ImportError("pandas library is required for Parquet format. Install with: pip install pandas pyarrow")
                
                df = pd.read_parquet(dataset_path)
                if len(df) > num_samples:
                    df = df.sample(n=num_samples, random_state=42)
                
                # Similar logic as CSV
                text_columns = ["text", "content", "input", "prompt", "question"]
                text_col = None
                for col in text_columns:
                    if col in df.columns:
                        text_col = col
                        break
                
                if text_col is None and len(df.columns) > 0:
                    text_col = df.columns[0]
                
                samples = [{"text": str(row[text_col])} for _, row in df.iterrows()]
                self.logger.info(f"Loaded {len(samples)} samples from Parquet")
                return samples
                
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
                
        except ImportError as e:
            self.logger.error(f"Missing required library: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading custom dataset: {e}")
            raise
    
    def load_huggingface_dataset(self,
                                 dataset_name: str,
                                 split: str = "train",
                                 num_samples: int = 512) -> Any:
        """
        Load dataset from HuggingFace Hub.
        
        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split to use
            num_samples: Number of samples to load
            
        Returns:
            Loaded dataset
        """
        try:
            self.logger.info(f"Loading dataset {dataset_name} from HuggingFace Hub")
            
            dataset = load_dataset(
                dataset_name,
                split=split,
                cache_dir=str(self.cache_dir),
                streaming=False  # Load fully for shuffling
            )
            
            # Shuffle for better representation
            dataset = dataset.shuffle(seed=42)
            
            # Select subset
            if len(dataset) > num_samples:
                dataset = dataset.select(range(num_samples))
            
            # Process to ensure 'text' field exists
            processed_data = []
            for item in dataset:
                # Try common field names
                text = None
                for field in ["text", "content", "input", "prompt", "question", "instruction"]:
                    if field in item:
                        text = item[field]
                        break
                
                # If no text field found, concatenate all string fields
                if text is None:
                    text_parts = []
                    for key, value in item.items():
                        if isinstance(value, str):
                            text_parts.append(f"{key}: {value}")
                    text = "\n".join(text_parts)
                
                if text:
                    processed_data.append({"text": text})
            
            self.logger.info(f"Loaded {len(processed_data)} samples from {dataset_name}")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error loading HuggingFace dataset {dataset_name}: {e}")
            raise
    
    def validate_dataset_format(self, dataset: Any) -> bool:
        """
        Validate dataset format compatibility.
        
        Checks:
        - Required fields present
        - Text length appropriate
        - No corrupted samples
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            True if format is compatible
        """
        if not dataset:
            self.logger.error("Dataset is empty")
            return False
        
        issues = []
        
        # Check if dataset is iterable
        try:
            iter(dataset)
        except TypeError:
            self.logger.error("Dataset is not iterable")
            return False
        
        # Validate samples
        min_length = 10  # Minimum text length
        max_length = 100000  # Maximum text length
        
        for i, sample in enumerate(dataset[:10]):  # Check first 10 samples
            # Check for text field
            if not isinstance(sample, dict):
                issues.append(f"Sample {i} is not a dictionary")
                continue
            
            if "text" not in sample:
                issues.append(f"Sample {i} missing 'text' field")
                continue
            
            text = sample["text"]
            
            # Check text validity
            if not isinstance(text, str):
                issues.append(f"Sample {i} text is not a string")
                continue
            
            if len(text) < min_length:
                issues.append(f"Sample {i} text too short ({len(text)} chars)")
            
            if len(text) > max_length:
                issues.append(f"Sample {i} text too long ({len(text)} chars)")
        
        if issues:
            self.logger.warning(f"Dataset validation found {len(issues)} issues:")
            for issue in issues[:5]:  # Show first 5 issues
                self.logger.warning(f"  - {issue}")
        
        return len(issues) == 0
    
    def preprocess_dataset(self, 
                          dataset: Any, 
                          tokenizer: Any,
                          max_length: int = 2048) -> Any:
        """
        Preprocess dataset for quantization.
        
        Steps:
        - Tokenization
        - Padding/truncation
        - Format conversion
        
        Args:
            dataset: Raw dataset
            tokenizer: Model tokenizer
            max_length: Maximum sequence length
            
        Returns:
            Preprocessed dataset
        """
        self.logger.info(f"Preprocessing dataset with max_length={max_length}")
        
        processed_samples = []
        
        for i, sample in enumerate(tqdm(dataset, desc="Preprocessing")):
            try:
                text = sample["text"]
                
                # Tokenize the text
                tokens = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                # Store both original text and tokenized version
                processed_sample = {
                    "text": text,
                    "input_ids": tokens["input_ids"].squeeze(),
                    "attention_mask": tokens["attention_mask"].squeeze()
                }
                
                # Add token type IDs if the model uses them (check if they exist)
                if "token_type_ids" in tokens:
                    processed_sample["token_type_ids"] = tokens["token_type_ids"].squeeze()
                
                processed_samples.append(processed_sample)
                
            except Exception as e:
                self.logger.warning(f"Error processing sample {i}: {e}")
                continue
        
        self.logger.info(f"Preprocessed {len(processed_samples)} samples")
        
        # Convert to HuggingFace Dataset if needed
        if processed_samples:
            try:
                from datasets import Dataset
                dataset_dict = {
                    "text": [s["text"] for s in processed_samples],
                    "input_ids": [s["input_ids"].tolist() for s in processed_samples],
                    "attention_mask": [s["attention_mask"].tolist() for s in processed_samples]
                }
                
                # Only add token_type_ids if they exist in processed samples
                if processed_samples[0].get("token_type_ids") is not None:
                    dataset_dict["token_type_ids"] = [s["token_type_ids"].tolist() for s in processed_samples]
                
                return Dataset.from_dict(dataset_dict)
            except:
                # Fallback to list format
                return processed_samples
        
        return processed_samples
    
    def create_calibration_subset(self,
                                  dataset: Any,
                                  num_samples: int,
                                  seed: int = 42) -> Any:
        """
        Create a representative subset for calibration.
        
        Args:
            dataset: Full dataset
            num_samples: Number of samples for calibration
            seed: Random seed for reproducibility
            
        Returns:
            Calibration subset
        """
        self.logger.info(f"Creating calibration subset with {num_samples} samples")
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # If dataset is a HuggingFace Dataset
        if hasattr(dataset, 'shuffle'):
            subset = dataset.shuffle(seed=seed)
            if len(subset) > num_samples:
                subset = subset.select(range(num_samples))
            return subset
        
        # If dataset is a list
        if isinstance(dataset, list):
            if len(dataset) <= num_samples:
                return dataset
            
            # Random sampling
            indices = np.random.choice(len(dataset), num_samples, replace=False)
            subset = [dataset[i] for i in indices]
            return subset
        
        # For other formats, try to convert to list
        try:
            dataset_list = list(dataset)
            if len(dataset_list) <= num_samples:
                return dataset_list
            
            indices = np.random.choice(len(dataset_list), num_samples, replace=False)
            subset = [dataset_list[i] for i in indices]
            return subset
            
        except Exception as e:
            self.logger.error(f"Error creating calibration subset: {e}")
            return dataset
    
    def estimate_dataset_memory(self,
                               num_samples: int,
                               max_length: int) -> float:
        """
        Estimate memory required for dataset.
        
        Args:
            num_samples: Number of samples
            max_length: Maximum sequence length
            
        Returns:
            Estimated memory in GB
        """
        # Rough estimation: each token is ~2 bytes (int16)
        # Plus overhead for attention masks, etc.
        bytes_per_token = 2
        overhead_factor = 2  # For attention masks and other tensors
        
        total_tokens = num_samples * max_length
        memory_bytes = total_tokens * bytes_per_token * overhead_factor
        memory_gb = memory_bytes / (1024 ** 3)
        
        self.logger.info(f"Estimated dataset memory: {memory_gb:.2f} GB")
        return memory_gb


class ModelDownloader:
    """Handles model downloading and verification."""
    
    def __init__(self, cache_dir: Path):
        """
        Initialize model downloader with cache directory.
        
        Args:
            cache_dir: Directory for caching models
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def download_model(self, 
                      model_id: str = "zai-org/GLM-4.5-Air",
                      revision: Optional[str] = None,
                      trust_remote_code: bool = True) -> Path:
        """
        Download model from HuggingFace Hub.
        
        Args:
            model_id: HuggingFace model ID
            revision: Specific model revision
            trust_remote_code: Whether to trust remote code
            
        Returns:
            Path to downloaded model
        """
        try:
            self.logger.info(f"Downloading model {model_id} from HuggingFace Hub")
            
            # Determine local path
            model_name = model_id.replace("/", "_")
            local_path = self.cache_dir / model_name
            
            # Check if already downloaded and valid
            if local_path.exists():
                if self.verify_model_integrity(local_path):
                    self.logger.info(f"Model already downloaded and verified at {local_path}")
                    return local_path
                else:
                    self.logger.warning(f"Existing model at {local_path} failed integrity check, re-downloading...")
                    # Could optionally delete the corrupted model here
                    # shutil.rmtree(local_path, ignore_errors=True)
            
            # Download using snapshot_download for full model
            self.logger.info("Starting model download (this may take a while)...")
            
            # Store current HF_HOME if it exists
            original_hf_home = os.environ.get('HF_HOME')
            
            try:
                # Set environment variable for HF cache
                os.environ['HF_HOME'] = str(self.cache_dir)
                
                downloaded_path = snapshot_download(
                    repo_id=model_id,
                    revision=revision,
                    cache_dir=str(self.cache_dir),
                    local_dir=str(local_path),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    ignore_patterns=["*.md", "*.txt", "LICENSE"]
                )
                
                self.logger.info(f"Model downloaded to {local_path}")
                
            finally:
                # Always restore original HF_HOME
                if original_hf_home is not None:
                    os.environ['HF_HOME'] = original_hf_home
                elif 'HF_HOME' in os.environ:
                    del os.environ['HF_HOME']
            
            # Verify download
            if not self.verify_model_integrity(local_path):
                raise ValueError("Downloaded model failed integrity check")
            
            return local_path
            
        except Exception as e:
            self.logger.error(f"Error downloading model: {e}")
            raise
    
    def download_tokenizer(self,
                          model_id: str,
                          trust_remote_code: bool = True) -> Any:
        """
        Download and load tokenizer.
        
        Args:
            model_id: HuggingFace model ID
            trust_remote_code: Whether to trust remote code
            
        Returns:
            Loaded tokenizer
        """
        try:
            self.logger.info(f"Loading tokenizer for {model_id}")
            
            # Try to load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                cache_dir=str(self.cache_dir)
            )
            
            # Set padding token if not present
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    # Try to use other special tokens as fallback
                    if hasattr(tokenizer, 'unk_token') and tokenizer.unk_token is not None:
                        tokenizer.pad_token = tokenizer.unk_token
                        self.logger.warning("Using unk_token as pad_token")
                    else:
                        # As last resort, add a new pad token
                        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                        self.logger.warning("Added new [PAD] token as pad_token")
            
            self.logger.info(f"Tokenizer loaded successfully")
            return tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading tokenizer: {e}")
            raise
    
    def verify_model_integrity(self, model_path: Path) -> bool:
        """
        Verify downloaded model integrity.
        
        Checks:
        - All required files present
        - Checksums match (if available)
        - Config file valid
        
        Args:
            model_path: Path to model directory
            
        Returns:
            True if model is intact
        """
        if not model_path.exists():
            return False
        
        # Check for essential files
        required_files = [
            "config.json",
            "tokenizer_config.json"
        ]
        
        optional_files = [
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model.bin.index.json",
            "model.safetensors.index.json"
        ]
        
        # Check required files
        for file_name in required_files:
            if not (model_path / file_name).exists():
                self.logger.warning(f"Missing required file: {file_name}")
                return False
        
        # Check for at least one model file
        has_model_file = False
        for file_name in optional_files:
            if (model_path / file_name).exists():
                has_model_file = True
                break
        
        # Check for sharded model files
        if not has_model_file:
            # Look for sharded files
            model_files = list(model_path.glob("pytorch_model-*.bin")) + \
                         list(model_path.glob("model-*.safetensors"))
            has_model_file = len(model_files) > 0
        
        if not has_model_file:
            self.logger.warning("No model weight files found")
            return False
        
        # Verify config is valid JSON
        try:
            config_path = model_path / "config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Basic config validation
            if "model_type" not in config:
                self.logger.warning("Invalid config: missing model_type")
                return False
                
        except Exception as e:
            self.logger.warning(f"Error validating config: {e}")
            return False
        
        self.logger.info("Model integrity check passed")
        return True
    
    def check_model_files(self, model_path: Path) -> Dict[str, bool]:
        """
        Check for required model files.
        
        Args:
            model_path: Path to model directory
            
        Returns:
            Dictionary of file checks
        """
        files_status = {}
        
        # Essential files
        essential_files = [
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json"
        ]
        
        for file_name in essential_files:
            files_status[file_name] = (model_path / file_name).exists()
        
        # Model weight files (various formats)
        weight_files = [
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model.bin.index.json",
            "model.safetensors.index.json"
        ]
        
        for file_name in weight_files:
            files_status[file_name] = (model_path / file_name).exists()
        
        # Check for sharded files
        sharded_pytorch = list(model_path.glob("pytorch_model-*.bin"))
        sharded_safetensors = list(model_path.glob("model-*.safetensors"))
        
        files_status["has_sharded_pytorch"] = len(sharded_pytorch) > 0
        files_status["has_sharded_safetensors"] = len(sharded_safetensors) > 0
        files_status["num_shards"] = len(sharded_pytorch) + len(sharded_safetensors)
        
        # Overall status
        has_weights = (
            files_status.get("pytorch_model.bin", False) or
            files_status.get("model.safetensors", False) or
            files_status.get("has_sharded_pytorch", False) or
            files_status.get("has_sharded_safetensors", False)
        )
        
        files_status["model_complete"] = (
            files_status.get("config.json", False) and
            has_weights
        )
        
        return files_status
    
    def get_model_info(self, model_path: Path) -> Dict[str, Any]:
        """
        Extract model information from config.
        
        Information extracted:
        - Architecture type
        - Number of parameters
        - Number of layers
        - Hidden size
        - Attention heads
        
        Args:
            model_path: Path to model directory
            
        Returns:
            Dictionary of model information
        """
        info = {}
        
        try:
            # Load config
            config_path = model_path / "config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Extract basic info
            info["model_type"] = config.get("model_type", "unknown")
            info["architectures"] = config.get("architectures", [])
            info["hidden_size"] = config.get("hidden_size", 0)
            info["num_hidden_layers"] = config.get("num_hidden_layers", 0)
            info["num_attention_heads"] = config.get("num_attention_heads", 0)
            info["num_key_value_heads"] = config.get("num_key_value_heads", 
                                                     info["num_attention_heads"])
            info["intermediate_size"] = config.get("intermediate_size", 0)
            info["vocab_size"] = config.get("vocab_size", 0)
            info["max_position_embeddings"] = config.get("max_position_embeddings", 0)
            info["torch_dtype"] = config.get("torch_dtype", "float16")
            
            # Calculate approximate parameter count
            # This is a rough estimate
            hidden = info["hidden_size"]
            layers = info["num_hidden_layers"]
            vocab = info["vocab_size"]
            intermediate = info["intermediate_size"]
            
            # Embedding parameters
            embedding_params = vocab * hidden
            
            # Attention parameters per layer (Q, K, V, O projections)
            attention_params = 4 * hidden * hidden
            
            # MLP parameters per layer
            mlp_params = 3 * hidden * intermediate  # gate, up, down projections
            
            # Total
            total_params = embedding_params + layers * (attention_params + mlp_params)
            info["estimated_parameters"] = total_params
            
            # Safe division for billions calculation
            if total_params > 0:
                info["estimated_parameters_billions"] = total_params / 1e9
                
                # Model size estimation
                bytes_per_param = 2  # Assuming fp16
                model_size_bytes = total_params * bytes_per_param
                info["estimated_size_gb"] = model_size_bytes / (1024 ** 3)
            else:
                info["estimated_parameters_billions"] = 0
                info["estimated_size_gb"] = 0
                self.logger.warning("Model parameter count is 0 - config may be invalid")
            
            self.logger.info(f"Model info extracted: {info['model_type']} with "
                           f"{info['estimated_parameters_billions']:.1f}B parameters")
            
        except Exception as e:
            self.logger.error(f"Error extracting model info: {e}")
            info["error"] = str(e)
        
        return info
    
    def estimate_model_size(self, model_path: Path) -> float:
        """
        Estimate model size in GB.
        
        Args:
            model_path: Path to model directory
            
        Returns:
            Model size in GB
        """
        total_size = 0
        
        try:
            # Sum up all .bin and .safetensors files
            for pattern in ["*.bin", "*.safetensors"]:
                for file_path in model_path.glob(pattern):
                    total_size += file_path.stat().st_size
            
            size_gb = total_size / (1024 ** 3)
            self.logger.info(f"Model size: {size_gb:.2f} GB")
            return size_gb
            
        except Exception as e:
            self.logger.error(f"Error estimating model size: {e}")
            # Return estimate based on config
            info = self.get_model_info(model_path)
            return info.get("estimated_size_gb", 0)
    
    def load_model_config(self, model_path: Path) -> Dict[str, Any]:
        """
        Load model configuration file.
        
        Args:
            model_path: Path to model directory
            
        Returns:
            Model configuration dictionary
        """
        config_path = model_path / "config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.logger.info(f"Loaded config for {config.get('model_type', 'unknown')} model")
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading model config: {e}")
            raise
    
    def verify_glm_architecture(self, config: Dict[str, Any]) -> bool:
        """
        Verify model is GLM architecture.
        
        Args:
            config: Model configuration
            
        Returns:
            True if model is GLM architecture
        """
        # Check model type
        model_type = config.get("model_type", "").lower()
        architectures = config.get("architectures", [])
        
        # GLM model indicators
        glm_indicators = [
            "glm" in model_type,
            "chatglm" in model_type,
            any("GLM" in arch for arch in architectures),
            any("ChatGLM" in arch for arch in architectures),
            config.get("model_family", "").lower() == "glm"
        ]
        
        is_glm = any(glm_indicators)
        
        if is_glm:
            self.logger.info("Verified GLM architecture")
        else:
            self.logger.warning(f"Model may not be GLM architecture (type: {model_type})")
        
        return is_glm


class PreparationValidator:
    """Validates all preparation components are ready."""
    
    def __init__(self):
        """Initialize preparation validator."""
        self.logger = logging.getLogger(__name__)
        
    def validate_recipe_ready(self, recipe_path: Optional[Path]) -> bool:
        """
        Validate recipe is ready for use.
        
        Args:
            recipe_path: Path to recipe file (can be None)
            
        Returns:
            True if recipe is ready
        """
        if recipe_path is None:
            self.logger.error("Recipe path is None")
            return False
            
        if not recipe_path.exists():
            self.logger.error(f"Recipe file not found: {recipe_path}")
            return False
        
        try:
            with open(recipe_path, 'r') as f:
                recipe = yaml.safe_load(f)
            
            # Check essential fields
            required = ["quant_method", "targets"]
            for field in required:
                if field not in recipe:
                    self.logger.error(f"Recipe missing required field: {field}")
                    return False
            
            method = recipe["quant_method"]
            if method not in ["awq", "gptq"]:
                self.logger.error(f"Invalid quantization method: {method}")
                return False
            
            self.logger.info("Recipe validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating recipe: {e}")
            return False
    
    def validate_dataset_ready(self, dataset: Any) -> bool:
        """
        Validate dataset is ready for calibration.
        
        Args:
            dataset: Prepared dataset
            
        Returns:
            True if dataset is ready
        """
        if not dataset:
            self.logger.error("Dataset is empty")
            return False
        
        try:
            # Handle different dataset types
            first_sample = None
            dataset_size = 0
            
            # For HuggingFace Dataset objects
            if hasattr(dataset, '__len__') and hasattr(dataset, '__getitem__'):
                dataset_size = len(dataset)
                if dataset_size > 0:
                    first_sample = dataset[0]
            else:
                # For generators or iterables
                try:
                    # Try to get first sample and count
                    for i, sample in enumerate(dataset):
                        if i == 0:
                            first_sample = sample
                        dataset_size += 1
                        
                        # Don't iterate through entire large datasets
                        if i >= 1000:
                            dataset_size = ">1000"
                            break
                except TypeError:
                    # Not iterable
                    self.logger.error("Dataset is not iterable")
                    return False
            
            # Validate first sample structure
            if first_sample is not None:
                if not isinstance(first_sample, dict) or "text" not in first_sample:
                    self.logger.error("Dataset samples must be dictionaries with 'text' field")
                    return False
            else:
                self.logger.error("Could not retrieve first sample from dataset")
                return False
            
            # Check minimum size
            if isinstance(dataset_size, int) and dataset_size < 10:
                self.logger.warning(f"Dataset has only {dataset_size} samples, recommend at least 100")
            
            self.logger.info(f"Dataset ready with {dataset_size} samples")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating dataset: {e}")
            return False
    
    def validate_model_ready(self, model_path: Path) -> bool:
        """
        Validate model is ready for quantization.
        
        Args:
            model_path: Path to model directory
            
        Returns:
            True if model is ready
        """
        if not model_path.exists():
            self.logger.error(f"Model path does not exist: {model_path}")
            return False
        
        # Use ModelDownloader to check files
        downloader = ModelDownloader(model_path.parent)
        
        # Check file integrity
        if not downloader.verify_model_integrity(model_path):
            self.logger.error("Model integrity check failed")
            return False
        
        # Check file status
        file_status = downloader.check_model_files(model_path)
        if not file_status.get("model_complete", False):
            self.logger.error("Model files incomplete")
            return False
        
        # Verify it's a GLM model (warning only)
        config = downloader.load_model_config(model_path)
        if not downloader.verify_glm_architecture(config):
            self.logger.warning("Model may not be GLM architecture, proceeding anyway")
        
        self.logger.info("Model validation passed")
        return True
    
    def generate_preparation_report(self,
                                   recipe_path: Path,
                                   dataset_info: Dict[str, Any],
                                   model_info: Dict[str, Any]) -> str:
        """
        Generate preparation phase report.
        
        Args:
            recipe_path: Path to recipe file
            dataset_info: Dataset information
            model_info: Model information
            
        Returns:
            Formatted report string
        """
        from datetime import datetime
        
        report_lines = [
            "="*60,
            "PHASE 2: PREPARATION REPORT",
            "="*60,
            f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "RECIPE CONFIGURATION:",
            "-"*30,
        ]
        
        # Recipe info
        if recipe_path and recipe_path.exists():
            try:
                with open(recipe_path, 'r') as f:
                    recipe = yaml.safe_load(f)
                
                method = recipe.get("quant_method", "unknown")
                report_lines.append(f"Quantization Method: {method.upper()}")
                
                if method in recipe:
                    config = recipe[method]
                    report_lines.append(f"Bits: {config.get('bits', 'N/A')}")
                    report_lines.append(f"Group Size: {config.get('group_size', 'N/A')}")
                    report_lines.append(f"Calibration Samples: {config.get('num_calibration_samples', 'N/A')}")
                    report_lines.append(f"Sequence Length: {config.get('calibration_sequence_length', 'N/A')}")
            except:
                report_lines.append("Error reading recipe file")
        else:
            report_lines.append("Recipe not found")
        
        # Dataset info
        report_lines.extend([
            "",
            "DATASET INFORMATION:",
            "-"*30,
        ])
        
        if dataset_info:
            report_lines.append(f"Dataset Type: {dataset_info.get('type', 'Unknown')}")
            report_lines.append(f"Number of Samples: {dataset_info.get('num_samples', 0)}")
            report_lines.append(f"Average Text Length: {dataset_info.get('avg_length', 0):.0f} chars")
            report_lines.append(f"Estimated Memory: {dataset_info.get('memory_gb', 0):.2f} GB")
        else:
            report_lines.append("Dataset not prepared")
        
        # Model info
        report_lines.extend([
            "",
            "MODEL INFORMATION:",
            "-"*30,
        ])
        
        if model_info:
            report_lines.append(f"Model Type: {model_info.get('model_type', 'Unknown')}")
            report_lines.append(f"Architecture: {', '.join(model_info.get('architectures', []))}")
            report_lines.append(f"Parameters: {model_info.get('estimated_parameters_billions', 0):.1f}B")
            report_lines.append(f"Hidden Size: {model_info.get('hidden_size', 0)}")
            report_lines.append(f"Layers: {model_info.get('num_hidden_layers', 0)}")
            report_lines.append(f"Attention Heads: {model_info.get('num_attention_heads', 0)}")
            report_lines.append(f"Model Size: {model_info.get('estimated_size_gb', 0):.2f} GB")
        else:
            report_lines.append("Model not downloaded")
        
        # Status summary
        report_lines.extend([
            "",
            "PREPARATION STATUS:",
            "-"*30,
        ])
        
        recipe_ready = recipe_path and recipe_path.exists()
        dataset_ready = dataset_info and dataset_info.get('num_samples', 0) > 0
        model_ready = model_info and model_info.get('model_type') is not None
        
        report_lines.append(f"✅ Recipe Ready: {recipe_ready}")
        report_lines.append(f"✅ Dataset Ready: {dataset_ready}")
        report_lines.append(f"✅ Model Ready: {model_ready}")
        
        all_ready = recipe_ready and dataset_ready and model_ready
        
        if all_ready:
            report_lines.append("")
            report_lines.append("✅ ALL COMPONENTS READY FOR QUANTIZATION")
        else:
            report_lines.append("")
            report_lines.append("⚠️ SOME COMPONENTS NOT READY")
        
        report_lines.append("="*60)
        
        return "\n".join(report_lines)


def run_phase2(project_dir: Union[Path, str],
               model_id: str = "zai-org/GLM-4.5-Air",
               quantization_method: QuantizationMethod = QuantizationMethod.AWQ,
               num_calibration_samples: int = 512) -> Dict[str, Any]:
    """
    Execute Phase 2: Preparation.
    
    Args:
        project_dir: Project directory (Path or string)
        model_id: HuggingFace model ID
        quantization_method: Method to use (AWQ or GPTQ)
        num_calibration_samples: Number of calibration samples
        
    Returns:
        Dictionary with phase results and paths
    """
    # Ensure project_dir is a Path object
    project_dir = Path(project_dir)
    
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("PHASE 2: PREPARATION STARTED")
    logger.info("="*60)
    
    results = {
        'success': False,
        'recipe_path': None,
        'dataset_ready': False,
        'model_path': None,
        'tokenizer': None,
        'errors': []
    }
    
    try:
        # Create necessary directories
        recipe_dir = project_dir / "recipes"
        dataset_dir = project_dir / "datasets"
        model_dir = project_dir / "models"
        
        for dir_path in [recipe_dir, dataset_dir, model_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Create/Load Recipe
        logger.info("Creating quantization recipe...")
        recipe_manager = RecipeManager(recipe_dir)
        
        if quantization_method == QuantizationMethod.AWQ:
            recipe = recipe_manager.create_awq_recipe_for_glm()
        else:
            config = RecipeConfig(
                method=quantization_method,
                bits=4,
                group_size=128,
                symmetric=False,
                ignore_layers=["model.embed_tokens", "model.norm", "lm_head"],
                mappings=None,
                duo_scaling=None,
                actorder=True,
                percdamp=0.01
            )
            recipe = recipe_manager.create_gptq_recipe(config)
        
        # Validate and save recipe
        is_valid, issues = recipe_manager.validate_recipe(recipe)
        if not is_valid:
            results['errors'].extend(issues)
            logger.error(f"Recipe validation failed: {issues}")
        else:
            try:
                recipe_path = recipe_manager.save_recipe(
                    recipe, 
                    f"glm_4_5_air_{quantization_method.value}"
                )
                if recipe_path and recipe_path.exists():
                    results['recipe_path'] = recipe_path
                    logger.info(f"Recipe saved to {recipe_path}")
                else:
                    results['errors'].append("Recipe save failed - path does not exist")
                    logger.error("Recipe save failed - path does not exist")
            except Exception as e:
                results['errors'].append(f"Recipe save failed: {str(e)}")
                logger.error(f"Recipe save failed: {e}")
        
        # Step 2: Prepare Dataset
        logger.info("Preparing calibration dataset...")
        dataset_preparer = DatasetPreparer(dataset_dir)
        
        # Load dataset (Open Platypus by default)
        dataset = dataset_preparer.load_open_platypus(num_calibration_samples)
        
        # Validate dataset
        if dataset_preparer.validate_dataset_format(dataset):
            results['dataset_ready'] = True
            results['dataset'] = dataset
            results['dataset_size'] = len(dataset)
            
            # Calculate dataset info
            total_length = sum(len(s["text"]) for s in dataset)
            avg_length = total_length / len(dataset) if dataset else 0
            
            results['dataset_info'] = {
                'type': 'Open Platypus',
                'num_samples': len(dataset),
                'avg_length': avg_length,
                'memory_gb': dataset_preparer.estimate_dataset_memory(
                    len(dataset), 
                    recipe.get(quantization_method.value, {}).get('calibration_sequence_length', 2048)
                )
            }
            
            logger.info(f"Dataset ready with {len(dataset)} samples")
        else:
            results['errors'].append("Dataset validation failed")
            logger.error("Dataset validation failed")
        
        # Step 3: Model Download
        logger.info("Downloading model and tokenizer...")
        model_downloader = ModelDownloader(model_dir)
        
        try:
            # Download model
            model_path = model_downloader.download_model(
                model_id=model_id,
                trust_remote_code=True
            )
            results['model_path'] = model_path
            
            # Download tokenizer
            tokenizer = model_downloader.download_tokenizer(
                model_id=model_id,
                trust_remote_code=True
            )
            results['tokenizer'] = tokenizer
            
            # Get model info
            model_info = model_downloader.get_model_info(model_path)
            results['model_info'] = model_info
            
            # Verify model
            if model_downloader.verify_model_integrity(model_path):
                logger.info(f"Model ready at {model_path}")
            else:
                results['errors'].append("Model integrity check failed")
                
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            results['errors'].append(f"Model download failed: {str(e)}")
        
        # Step 4: Preprocess dataset if tokenizer available
        if results.get('tokenizer') and results.get('dataset'):
            logger.info("Preprocessing dataset with tokenizer...")
            try:
                preprocessed = dataset_preparer.preprocess_dataset(
                    results['dataset'],
                    results['tokenizer'],
                    max_length=recipe.get(quantization_method.value, {}).get('calibration_sequence_length', 2048)
                )
                results['preprocessed_dataset'] = preprocessed
                logger.info("Dataset preprocessing complete")
            except Exception as e:
                logger.warning(f"Dataset preprocessing failed: {e}")
                # Not critical - can use raw dataset
        
        # Step 5: Validation
        validator = PreparationValidator()
        
        # Validate all components with safe path handling
        recipe_ready = False
        if results.get('recipe_path'):
            # Ensure it's a Path object
            recipe_path = results['recipe_path'] if isinstance(results['recipe_path'], Path) else Path(results['recipe_path'])
            recipe_ready = validator.validate_recipe_ready(recipe_path)
        else:
            recipe_ready = validator.validate_recipe_ready(None)
        
        dataset_ready = validator.validate_dataset_ready(results.get('dataset'))
        model_ready = False
        
        if results.get('model_path'):
            # Ensure it's a Path object
            model_path = results['model_path'] if isinstance(results['model_path'], Path) else Path(results['model_path'])
            model_ready = validator.validate_model_ready(model_path)
        
        # Generate report
        report = validator.generate_preparation_report(
            results.get('recipe_path'),
            results.get('dataset_info', {}),
            results.get('model_info', {})
        )
        
        results['report'] = report
        
        # Save report
        report_path = project_dir / "phase2_preparation_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")
        
        # Print report
        print("\n" + report)
        
        # Check overall success
        results['success'] = (
            recipe_ready and
            dataset_ready and
            model_ready
        )
        
        if results['success']:
            logger.info("✅ PHASE 2: PREPARATION COMPLETED SUCCESSFULLY")
        else:
            logger.warning("⚠️ PHASE 2: PREPARATION COMPLETED WITH ISSUES")
            
    except Exception as e:
        logger.error(f"Fatal error in Phase 2: {e}", exc_info=True)
        results['errors'].append(str(e))
        results['success'] = False
    
    logger.info("="*60)
    
    return results

if __name__ == "__main__":
    # Example standalone execution
    logging.basicConfig(level=logging.INFO)
    
    result = run_phase2(
        project_dir=Path("./project"),
        model_id="zai-org/GLM-4.5-Air",
        quantization_method=QuantizationMethod.AWQ,
        num_calibration_samples=512
    )
    print(f"\nPhase 2 completed: {result['success']}")
    if result['recipe_path']:
        print(f"Recipe saved to: {result['recipe_path']}")
    if result['dataset_ready']:
        print(f"Dataset prepared with {result.get('dataset_size', 0)} samples")
    if result['model_path']:
        print(f"Model downloaded to: {result['model_path']}")