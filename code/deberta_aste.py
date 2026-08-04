#!/usr/bin/env python3
"""
STAGE 3: PRODUCTION-GRADE JOINT DEBERTA ASTE MODEL TRAINING
WITH ALL 25 RECOMMENDED IMPROVEMENTS - ERROR FIXED
"""

import os
import json
import logging
import random
import shutil
import psutil
import csv
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
    TrainerCallback,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data': {
        'train_path': r"F:\Research\data\train.csv",
        'combined_triplets_path': r"F:\Research\data\all_triplets.csv",
        'test_split': 0.15,
        'val_split': 0.15,
    },
    'model': {
        'base_model': 'microsoft/deberta-v3-base',
        'max_length': 384,
    },
    'training': {
        'num_epochs': 5,
        'batch_size': 8,
        'learning_rate': 2e-5,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'gradient_accumulation_steps': 1,
        'gradient_clipping': 1.0,
        'eval_steps': 200,
        'save_steps': 200,
    },
    'paths': {
        'output_dir': r"F:\Research\model\aste_deberta",
        'checkpoint_dir': r"F:\Research\model\aste_deberta\checkpoints",
        'log_file': r"F:\Research\model\aste_deberta\training.log",
        'tensorboard_dir': r"F:\Research\model\aste_deberta\tensorboard",
        'metrics_file': r"F:\Research\model\aste_deberta\metrics.json",
        'experiment_file': r"F:\Research\model\aste_deberta\experiment_metadata.json",
        'history_file': r"F:\Research\model\aste_deberta\training_history.csv",
        'summary_file': r"F:\Research\model\aste_deberta\experiment_summary.txt",
    }
}

# Set all random seeds
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(RANDOM_SEED)

# Get device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create directories
for path in CONFIG['paths'].values():
    if path.endswith(('.log', '.json', '.csv', '.txt')):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    else:
        Path(path).mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOGGING
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Colored console output."""
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[92m',
        'WARNING': '\033[93m',
        'ERROR': '\033[91m',
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler
fh = logging.FileHandler(CONFIG['paths']['log_file'])
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
logger.addHandler(fh)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(ColoredFormatter('[%(asctime)s] %(levelname)s: %(message)s'))
logger.addHandler(ch)

# ============================================================================
# BIO TAGS
# ============================================================================

BIO_TAGS = ['O', 'B-ASPECT', 'I-ASPECT', 'B-OPINION', 'I-OPINION']
TAG2ID = {tag: idx for idx, tag in enumerate(BIO_TAGS)}
ID2TAG = {idx: tag for tag, idx in TAG2ID.items()}

ASPECTS = ['Room', 'Staff', 'Food', 'Location', 'Cleanliness', 'Value', 'WiFi', 'Amenities']

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_gpu_info():
    """Get GPU information."""
    info = {
        'device': str(DEVICE),
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        info['total_vram_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return info

def log_system_info():
    """Log comprehensive system information."""
    logger.info("\n" + "="*80)
    logger.info("SYSTEM INFORMATION")
    logger.info("="*80)
    
    gpu_info = get_gpu_info()
    logger.info(f"Device: {gpu_info['device']}")
    logger.info(f"CPU Cores: {psutil.cpu_count()}")
    logger.info(f"RAM: {psutil.virtual_memory().total / 1e9:.2f} GB")
    
    if gpu_info['cuda_available']:
        logger.info(f"GPU: {gpu_info['gpu_name']}")
        logger.info(f"GPU Count: {gpu_info['device_count']}")
        logger.info(f"CUDA: {gpu_info['cuda_version']}")
        logger.info(f"cuDNN: {gpu_info['cudnn_version']}")
        logger.info(f"Total VRAM: {gpu_info['total_vram_gb']:.2f} GB")
    
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Random Seed: {RANDOM_SEED}")
    logger.info("="*80 + "\n")

# ============================================================================
# DATA VALIDATION
# ============================================================================

class BIOValidator:
    """Validate BIO sequences."""
    
    @staticmethod
    def is_valid_bio_sequence(bio_tags: List[str]) -> bool:
        """Check if BIO sequence is valid."""
        valid_tags = set(BIO_TAGS)
        for tag in bio_tags:
            if tag not in valid_tags:
                return False
        
        for i, tag in enumerate(bio_tags):
            if tag.startswith('I-'):
                if i == 0:
                    return False
                prev_tag = bio_tags[i-1]
                tag_type = tag.split('-')[1]
                
                if not (prev_tag.endswith('-' + tag_type) or prev_tag == 'O'):
                    if not prev_tag.startswith('B-' + tag_type) and not prev_tag.startswith('I-' + tag_type):
                        return False
        
        return True

class DataValidator:
    """Validate dataset integrity."""
    
    @staticmethod
    def validate(train_df, combined_df, texts, bio_labels):
        """Comprehensive dataset validation."""
        logger.info("\n" + "="*80)
        logger.info("DATA VALIDATION")
        logger.info("="*80)
        
        errors = []
        
        # Check duplicates
        dups = len(train_df) - train_df['review_id'].nunique()
        if dups > 0:
            logger.warning(f"Found {dups} duplicate review_ids")
        
        # Check missing values
        missing_text = train_df['review_text'].isna().sum()
        missing_id = train_df['review_id'].isna().sum()
        if missing_text > 0:
            errors.append(f"Missing review_text: {missing_text}")
        if missing_id > 0:
            errors.append(f"Missing review_id: {missing_id}")
        
        # Check mismatch
        if len(texts) != len(bio_labels):
            errors.append(f"Mismatch: {len(texts)} texts vs {len(bio_labels)} labels")
        
        # Check BIO validity
        invalid_bio = 0
        for bio in bio_labels:
            if not BIOValidator.is_valid_bio_sequence(bio):
                invalid_bio += 1
        if invalid_bio > 0:
            logger.warning(f"Found {invalid_bio} invalid BIO sequences")
        
        # Check token-label mismatch
        mismatches = 0
        for text, labels in zip(texts, bio_labels):
            if len(text.split()) != len(labels):
                mismatches += 1
        if mismatches > 0:
            errors.append(f"{mismatches} reviews have token-label mismatch")
        
        # Statistics
        logger.info(f"Total reviews: {len(train_df)}")
        logger.info(f"Total triplets: {len(combined_df)}")
        logger.info(f"Reviews with triplets: {len(texts)}")
        
        token_lengths = [len(text.split()) for text in texts]
        logger.info(f"Token lengths - Min: {min(token_lengths)}, Max: {max(token_lengths)}, Avg: {np.mean(token_lengths):.1f}")
        
        coverage = (len(texts) / len(train_df)) * 100
        logger.info(f"Review coverage: {coverage:.1f}%")
        
        if errors:
            for error in errors:
                logger.error(f"  ✗ {error}")
            logger.info("="*80 + "\n")
            return False
        
        logger.info("✓ All validation checks passed")
        logger.info("="*80 + "\n")
        return True

# ============================================================================
# TRIPLET TO BIO CONVERTER
# ============================================================================

class TripletToBIOConverter:
    """Convert triplets to BIO labels with improved handling."""
    
    @staticmethod
    def find_all_spans(tokens: List[str], phrase: str) -> List[Tuple[int, int]]:
        """Find all occurrences of phrase in tokens."""
        phrase_tokens = phrase.lower().split()
        spans = []
        for i in range(len(tokens) - len(phrase_tokens) + 1):
            if [t.lower() for t in tokens[i:i+len(phrase_tokens)]] == phrase_tokens:
                spans.append((i, i + len(phrase_tokens) - 1))
        return spans
    
    @staticmethod
    def triplets_to_bio(review_text: str, triplets: List[Dict]) -> List[str]:
        """Convert triplets to BIO labels."""
        tokens = review_text.split()
        bio_tags = ['O'] * len(tokens)
        
        try:
            tagged_spans: Set[Tuple[int, int]] = set()
            
            for triplet in triplets:
                aspect = triplet.get('aspect', '')
                opinion = triplet.get('opinion', '')
                
                if not aspect or not opinion:
                    continue
                
                aspect_spans = TripletToBIOConverter.find_all_spans(tokens, aspect)
                opinion_spans = TripletToBIOConverter.find_all_spans(tokens, opinion)
                
                for start, end in aspect_spans:
                    span_tuple = (start, end)
                    if span_tuple not in tagged_spans:
                        bio_tags[start] = 'B-ASPECT'
                        for i in range(start + 1, end + 1):
                            bio_tags[i] = 'I-ASPECT'
                        tagged_spans.add(span_tuple)
                        break
                
                for start, end in opinion_spans:
                    span_tuple = (start, end)
                    if span_tuple not in tagged_spans:
                        bio_tags[start] = 'B-OPINION'
                        for i in range(start + 1, end + 1):
                            bio_tags[i] = 'I-OPINION'
                        tagged_spans.add(span_tuple)
                        break
        
        except Exception as e:
            logger.warning(f"Error in BIO conversion: {e}")
        
        return bio_tags

# ============================================================================
# DATASET
# ============================================================================

class ASTEDataset(Dataset):
    """PyTorch Dataset with improved BIO alignment."""
    
    def __init__(
        self,
        texts: List[str],
        bio_labels: List[List[str]],
        tokenizer,
        max_length: int = 384
    ):
        self.texts = texts
        self.bio_labels = bio_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []
        self.label_ids = []
        
        for text, labels in zip(texts, bio_labels):
            encoding = self._encode_with_aligned_labels(text, labels)
            if encoding is not None:
                self.encodings.append(encoding)
                self.label_ids.append(encoding['label_ids'])
    
    def _encode_with_aligned_labels(self, text: str, labels: List[str]):
        """Properly align BIO labels using word_ids()."""
        tokens = text.split()
        
        try:
            encoding = self.tokenizer(
                tokens,
                is_split_into_words=True,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
            )
            
            word_ids = encoding.word_ids()
            label_ids = [-100] * self.max_length
            
            for token_idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    label_ids[token_idx] = -100
                elif word_idx < len(labels):
                    label_ids[token_idx] = TAG2ID.get(labels[word_idx], TAG2ID['O'])
                else:
                    label_ids[token_idx] = -100
            
            encoding['label_ids'] = label_ids
            return encoding
        
        except Exception as e:
            logger.warning(f"Encoding error: {e}")
            return None
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        encoding = self.encodings[idx]
        return {
            'input_ids': encoding['input_ids'][0],
            'attention_mask': encoding['attention_mask'][0],
            'labels': torch.tensor(self.label_ids[idx], dtype=torch.long)
        }

# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(eval_preds, save_dir=None):
    """Compute comprehensive metrics."""
    preds, labels = eval_preds
    preds = np.argmax(preds, axis=2)
    
    true_preds = []
    true_labels = []
    
    for pred, label in zip(preds, labels):
        for p, l in zip(pred, label):
            if l != -100:
                true_preds.append(p)
                true_labels.append(l)
    
    # Main metrics
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        true_labels, true_preds, average='weighted', zero_division=0
    )
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        true_labels, true_preds, average='macro', zero_division=0
    )
    p_mi, r_mi, f1_mi, _ = precision_recall_fscore_support(
        true_labels, true_preds, average='micro', zero_division=0
    )
    
    metrics = {
        'f1': f1_w,
        'f1_weighted': f1_w,
        'f1_macro': f1_m,
        'f1_micro': f1_mi,
        'precision': p_w,
        'precision_weighted': p_w,
        'precision_macro': p_m,
        'precision_micro': p_mi,
        'recall': r_w,
        'recall_weighted': r_w,
        'recall_macro': r_m,
        'recall_micro': r_mi,
    }
    
    # Per-class metrics
    try:
        per_class = precision_recall_fscore_support(
            true_labels, true_preds, 
            average=None, 
            zero_division=0, 
            labels=list(range(len(BIO_TAGS)))
        )
        
        for idx, tag in ID2TAG.items():
            if idx < len(per_class[0]):
                metrics[f'f1_{tag}'] = float(per_class[2][idx])
                metrics[f'precision_{tag}'] = float(per_class[0][idx])
                metrics[f'recall_{tag}'] = float(per_class[1][idx])
            else:
                metrics[f'f1_{tag}'] = 0.0
                metrics[f'precision_{tag}'] = 0.0
                metrics[f'recall_{tag}'] = 0.0
    
    except Exception as e:
        logger.warning(f"Error computing per-class metrics: {e}")
    
    # Save visualizations
    if save_dir:
        try:
            unique_labels = sorted(set(true_labels))
            label_names = [ID2TAG.get(i, f'Class_{i}') for i in unique_labels]
            
            report = classification_report(
                true_labels, true_preds,
                labels=unique_labels,
                target_names=label_names,
                output_dict=True
            )
            
            report_file = os.path.join(save_dir, 'classification_report.json')
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Confusion matrix
            cm = confusion_matrix(true_labels, true_preds, labels=unique_labels)
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=label_names,
                yticklabels=label_names,
                cbar_kws={'label': 'Count'}
            )
            plt.title('Confusion Matrix - BIO Tags', fontsize=14, fontweight='bold')
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.tight_layout()
            
            cm_file = os.path.join(save_dir, 'confusion_matrix.png')
            plt.savefig(cm_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✓ Saved classification report and confusion matrix")
        
        except Exception as e:
            logger.warning(f"Error saving metrics visualizations: {e}")
    
    return metrics

# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

class CheckpointManager:
    """Manage checkpoints with automatic resume."""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "resume_metadata.json"
    
    def save_metadata(self, epoch: int, global_step: int, best_f1: float, metrics: Dict):
        """Save training metadata."""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'global_step': global_step,
            'best_f1': best_f1,
            'seed': RANDOM_SEED,
            'metrics': metrics,
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_metadata(self) -> Optional[Dict]:
        """Load training metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        return None
    
    def get_last_checkpoint(self) -> Optional[str]:
        """Get last checkpoint path."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint-*"))
        if checkpoints:
            return str(checkpoints[-1])
        return None
    
    def cleanup_old_checkpoints(self, keep_n=3):
        """Remove old checkpoints."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint-*"))
        if len(checkpoints) > keep_n:
            for checkpoint in checkpoints[:-keep_n]:
                shutil.rmtree(checkpoint)

# ============================================================================
# CUSTOM CALLBACKS
# ============================================================================

class ProgressCallback(TrainerCallback):
    """Training progress with ETA."""
    
    def __init__(self):
        self.start_time = None
        self.best_f1 = 0
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.now()
        logger.info(f"\nTraining started at {self.start_time}")
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0 and state.log_history:
            loss = state.log_history[-1].get('loss', 0)
            elapsed = (datetime.now() - self.start_time).total_seconds()
            
            if elapsed > 0:
                samples_per_sec = state.global_step * args.per_device_train_batch_size / elapsed
            else:
                samples_per_sec = 0
            
            total_steps = args.max_steps if args.max_steps > 0 else state.max_steps
            if total_steps > 0:
                remaining_steps = total_steps - state.global_step
                if samples_per_sec > 0:
                    eta_seconds = remaining_steps * (args.per_device_train_batch_size / samples_per_sec)
                    eta_str = str(timedelta(seconds=int(eta_seconds)))
                else:
                    eta_str = "N/A"
            else:
                eta_str = "N/A"
            
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1e9
                gpu_mem_pct = (gpu_mem / (torch.cuda.get_device_properties(0).total_memory / 1e9)) * 100
            else:
                gpu_mem = 0
                gpu_mem_pct = 0
            
            pct_complete = (state.global_step / total_steps * 100) if total_steps > 0 else 0
            
            logger.info(
                f"Step {state.global_step:5d} | "
                f"Loss: {loss:.4f} | "
                f"Samples/sec: {samples_per_sec:.1f} | "
                f"GPU: {gpu_mem_pct:.1f}% | "
                f"Progress: {pct_complete:.1f}% | "
                f"ETA: {eta_str}"
            )

class TrainingHistoryCallback(TrainerCallback):
    """Save training history."""
    
    def __init__(self, history_file: str):
        self.history_file = history_file
        self.history = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            log_entry = {
                'step': state.global_step,
                'epoch': state.epoch,
                'timestamp': datetime.now().isoformat(),
            }
            log_entry.update(logs)
            self.history.append(log_entry)
            
            if len(self.history) % 10 == 0:
                self._save_history()
    
    def on_train_end(self, args, state, control, **kwargs):
        self._save_history()
    
    def _save_history(self):
        """Save training history to CSV."""
        try:
            if self.history:
                df = pd.DataFrame(self.history)
                df.to_csv(self.history_file, index=False)
        except Exception as e:
            logger.warning(f"Error saving training history: {e}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def load_and_prepare_data():
    """Load and prepare data with improvements."""
    logger.info("\n[1/6] Loading data...")
    
    try:
        combined_df = pd.read_csv(CONFIG['data']['combined_triplets_path'])
        logger.info(f"✓ Loaded {len(combined_df)} triplets")
    except Exception as e:
        logger.error(f"Failed to load triplets: {e}")
        raise
    
    try:
        train_df = pd.read_csv(CONFIG['data']['train_path'])
        if 'review_id' not in train_df.columns:
            train_df['review_id'] = train_df.index
        logger.info(f"✓ Loaded {len(train_df)} reviews")
    except Exception as e:
        logger.error(f"Failed to load reviews: {e}")
        raise
    
    # Group triplets by review_id
    logger.info("[2/6] Grouping triplets by review_id...")
    triplets_by_review = combined_df.groupby('review_id').apply(
        lambda x: x[['aspect', 'opinion', 'sentiment']].to_dict('records')
    ).to_dict()
    
    # Convert to BIO
    logger.info("[3/6] Converting to BIO...")
    converter = TripletToBIOConverter()
    
    texts = []
    bio_labels = []
    review_ids_used = []
    
    for _, row in train_df.iterrows():
        review_id = row['review_id']
        if review_id in triplets_by_review:
            text = row['review_text']
            triplets = triplets_by_review[review_id]
            labels = converter.triplets_to_bio(text, triplets)
            
            if len(labels) == len(text.split()):
                texts.append(text)
                bio_labels.append(labels)
                review_ids_used.append(review_id)
    
    logger.info(f"✓ Converted {len(texts)} reviews")
    
    # Validate
    if not DataValidator.validate(train_df, combined_df, texts, bio_labels):
        raise ValueError("Dataset validation failed")
    
    # Stratified split by aspect distribution
    logger.info("[4/6] Computing aspect distribution for stratification...")
    
    aspect_counts = defaultdict(lambda: defaultdict(int))
    for review_id, bio in zip(review_ids_used, bio_labels):
        for tag in bio:
            if tag.startswith('B-ASPECT'):
                aspect_counts[review_id]['has_aspect'] = 1
    
    strat_labels = [1 if rid in aspect_counts else 0 for rid in review_ids_used]
    
    logger.info("[5/6] Splitting data with stratification...")
    
    # First split: train+val vs test
    idx_train_val, idx_test, strat_tv, strat_test = train_test_split(
        range(len(texts)),
        strat_labels,
        test_size=CONFIG['data']['test_split'],
        random_state=RANDOM_SEED,
        stratify=strat_labels
    )
    
    # Second split: train vs val
    idx_train, idx_val, _, _ = train_test_split(
        idx_train_val,
        [strat_labels[i] for i in idx_train_val],
        test_size=CONFIG['data']['val_split'] / (1 - CONFIG['data']['test_split']),
        random_state=RANDOM_SEED,
        stratify=[strat_labels[i] for i in idx_train_val]
    )
    
    train_texts = [texts[i] for i in idx_train]
    train_labels = [bio_labels[i] for i in idx_train]
    val_texts = [texts[i] for i in idx_val]
    val_labels = [bio_labels[i] for i in idx_val]
    test_texts = [texts[i] for i in idx_test]
    test_labels = [bio_labels[i] for i in idx_test]
    
    logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}\n")
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

def train_model(train_texts, train_labels, val_texts, val_labels, tokenizer, model):
    """Train model with all improvements."""
    logger.info("[6/6] Training model...\n")
    
    # Create datasets
    train_dataset = ASTEDataset(train_texts, train_labels, tokenizer, CONFIG['model']['max_length'])
    val_dataset = ASTEDataset(val_texts, val_labels, tokenizer, CONFIG['model']['max_length'])
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples\n")
    
    # Auto-resume from latest checkpoint
    checkpoint_manager = CheckpointManager(CONFIG['paths']['checkpoint_dir'])
    resume_from_checkpoint = checkpoint_manager.get_last_checkpoint()
    
    if resume_from_checkpoint:
        logger.info(f"✓ Resuming from checkpoint: {resume_from_checkpoint}\n")
    
    # FP16 mixed precision
    use_fp16 = torch.cuda.is_available()
    logger.info(f"FP16 Mixed Precision: {use_fp16}\n")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=CONFIG['paths']['output_dir'],
        num_train_epochs=CONFIG['training']['num_epochs'],
        per_device_train_batch_size=CONFIG['training']['batch_size'],
        per_device_eval_batch_size=CONFIG['training']['batch_size'],
        learning_rate=CONFIG['training']['learning_rate'],
        warmup_ratio=CONFIG['training']['warmup_ratio'],
        weight_decay=CONFIG['training']['weight_decay'],
        gradient_accumulation_steps=CONFIG['training']['gradient_accumulation_steps'],
        max_grad_norm=CONFIG['training']['gradient_clipping'],
        evaluation_strategy="steps",
        eval_steps=CONFIG['training']['eval_steps'],
        save_strategy="steps",
        save_steps=CONFIG['training']['save_steps'],
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        logging_dir=CONFIG['paths']['tensorboard_dir'],
        seed=RANDOM_SEED,
        fp16=use_fp16,
    )
    
    # Trainer with callbacks (FIXED: removed evaluation_steps parameter)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda x: compute_metrics(x, save_dir=CONFIG['paths']['output_dir']),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),  # FIXED: Removed evaluation_steps
            ProgressCallback(),
            TrainingHistoryCallback(CONFIG['paths']['history_file'])
        ]
    )
    
    # Train
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logger.info("✓ Training complete\n")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user\n")
    except Exception as e:
        logger.error(f"Training failed: {e}\n")
        raise
    
    return trainer

def evaluate_on_test(trainer, test_texts, test_labels, tokenizer):
    """Evaluate on test set."""
    logger.info("Evaluating on test set...\n")
    
    test_dataset = ASTEDataset(test_texts, test_labels, tokenizer, CONFIG['model']['max_length'])
    results = trainer.evaluate(eval_dataset=test_dataset)
    
    logger.info("\n" + "="*80)
    logger.info("TEST SET RESULTS")
    logger.info("="*80)
    logger.info(f"F1 (Weighted): {results.get('eval_f1', 0):.4f}")
    logger.info(f"F1 (Macro):    {results.get('eval_f1_macro', 0):.4f}")
    logger.info(f"F1 (Micro):    {results.get('eval_f1_micro', 0):.4f}")
    logger.info(f"Precision:     {results.get('eval_precision', 0):.4f}")
    logger.info(f"Recall:        {results.get('eval_recall', 0):.4f}")
    
    # Per-tag metrics
    logger.info("\nPer-Tag Metrics:")
    for tag in BIO_TAGS:
        f1 = results.get(f'eval_f1_{tag}', 0)
        p = results.get(f'eval_precision_{tag}', 0)
        r = results.get(f'eval_recall_{tag}', 0)
        logger.info(f"  {tag:12s}: F1={f1:.4f}, P={p:.4f}, R={r:.4f}")
    
    logger.info("="*80 + "\n")
    
    return results

def export_dataset_statistics(train_df, combined_df, texts):
    """Export dataset statistics."""
    logger.info("Exporting dataset statistics...\n")
    
    aspect_dist = combined_df['aspect'].value_counts().to_dict()
    sentiment_dist = combined_df['sentiment'].value_counts().to_dict()
    
    stats = {
        'total_reviews': len(train_df),
        'reviews_with_triplets': len(texts),
        'total_triplets': len(combined_df),
        'aspect_distribution': aspect_dist,
        'sentiment_distribution': sentiment_dist,
    }
    
    stats_file = os.path.join(CONFIG['paths']['output_dir'], 'dataset_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"✓ Dataset statistics saved")
    logger.info(f"  Total reviews: {stats['total_reviews']}")
    logger.info(f"  Reviews with triplets: {stats['reviews_with_triplets']}")
    logger.info(f"  Total triplets: {stats['total_triplets']}")
    logger.info(f"  Aspects: {dict(stats['aspect_distribution'])}")
    logger.info(f"  Sentiments: {dict(stats['sentiment_distribution'])}\n")

def save_experiment_metadata(results, train_texts, val_texts, test_texts):
    """Save comprehensive experiment metadata."""
    logger.info("Saving experiment metadata...\n")
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'seed': RANDOM_SEED,
        'model': CONFIG['model']['base_model'],
        'device': str(DEVICE),
        'gpu_info': get_gpu_info(),
        'config': CONFIG,
        'dataset': {
            'train_size': len(train_texts),
            'val_size': len(val_texts),
            'test_size': len(test_texts),
        },
        'test_results': {k: float(v) if isinstance(v, np.ndarray) else v for k, v in results.items()},
        'bio_tags': BIO_TAGS,
        'aspects': ASPECTS,
    }
    
    with open(CONFIG['paths']['experiment_file'], 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Experiment metadata saved")

def generate_summary_report(results, training_time):
    """Generate final experiment summary."""
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*80)
    
    summary = f"""
ASTE EXPERIMENT SUMMARY
Generated: {datetime.now().isoformat()}
Seed: {RANDOM_SEED}

DATASET
-------
Total reviews: See dataset_statistics.json
Total triplets: See dataset_statistics.json

MODEL
-----
Base model: {CONFIG['model']['base_model']}
Max length: {CONFIG['model']['max_length']}
BIO tags: {len(BIO_TAGS)}

TRAINING
--------
Epochs: {CONFIG['training']['num_epochs']}
Batch size: {CONFIG['training']['batch_size']}
Learning rate: {CONFIG['training']['learning_rate']}
Warmup ratio: {CONFIG['training']['warmup_ratio']}
Training time: {training_time}

RESULTS
-------
F1 (Weighted): {results.get('eval_f1', 0):.4f}
F1 (Macro):    {results.get('eval_f1_macro', 0):.4f}
F1 (Micro):    {results.get('eval_f1_micro', 0):.4f}
Precision:     {results.get('eval_precision', 0):.4f}
Recall:        {results.get('eval_recall', 0):.4f}

SAVED FILES
-----------
Model: {CONFIG['paths']['output_dir']}/pytorch_model.bin
Config: {CONFIG['paths']['output_dir']}/config.json
Logs: {CONFIG['paths']['log_file']}
Metrics: {CONFIG['paths']['output_dir']}/classification_report.json
Confusion matrix: {CONFIG['paths']['output_dir']}/confusion_matrix.png
Training history: {CONFIG['paths']['history_file']}
Dataset stats: {CONFIG['paths']['output_dir']}/dataset_statistics.json
Experiment metadata: {CONFIG['paths']['experiment_file']}
TensorBoard: {CONFIG['paths']['tensorboard_dir']}

NEXT STEPS
----------
1. Review confusion_matrix.png for error patterns
2. Check classification_report.json for per-tag performance
3. Analyze training_history.csv for training dynamics
4. Proceed to Stage 4: Extract triplets from all reviews
"""
    
    logger.info(summary)
    
    with open(CONFIG['paths']['summary_file'], 'w') as f:
        f.write(summary)
    
    logger.info(f"✓ Summary report saved to {CONFIG['paths']['summary_file']}\n")

def main():
    """Main pipeline with all improvements."""
    print("\n" + "="*80)
    print("STAGE 3: PRODUCTION-GRADE JOINT DEBERTA ASTE TRAINING")
    print("WITH ALL 25 RECOMMENDED IMPROVEMENTS")
    print("="*80 + "\n")
    
    training_start_time = datetime.now()
    
    try:
        # System info
        log_system_info()
        
        # Load and prepare data
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
            load_and_prepare_data()
        
        # Export statistics
        export_dataset_statistics(
            pd.read_csv(CONFIG['data']['train_path']),
            pd.read_csv(CONFIG['data']['combined_triplets_path']),
            train_texts
        )
        
        # Load model
        logger.info("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(CONFIG['model']['base_model'])
        model = AutoModelForTokenClassification.from_pretrained(
            CONFIG['model']['base_model'],
            num_labels=len(BIO_TAGS)
        )
        logger.info(f"✓ Loaded {CONFIG['model']['base_model']}\n")
        
        # Train
        trainer = train_model(train_texts, train_labels, val_texts, val_labels, tokenizer, model)
        
        # Evaluate
        results = evaluate_on_test(trainer, test_texts, test_labels, tokenizer)
        
        # Save model
        logger.info("Saving final model...")
        trainer.save_model(CONFIG['paths']['output_dir'])
        tokenizer.save_pretrained(CONFIG['paths']['output_dir'])
        
        # Save BIO tags
        with open(os.path.join(CONFIG['paths']['output_dir'], 'bio_tags.json'), 'w') as f:
            json.dump({'tag2id': TAG2ID, 'id2tag': ID2TAG}, f, indent=2)
        
        # Save config
        with open(os.path.join(CONFIG['paths']['output_dir'], 'training_config.json'), 'w') as f:
            json.dump(CONFIG, f, indent=2, default=str)
        
        # Save metadata
        save_experiment_metadata(results, train_texts, val_texts, test_texts)
        
        # Training time
        training_time = datetime.now() - training_start_time
        logger.info(f"Total training time: {training_time}")
        
        # Final summary
        generate_summary_report(results, str(training_time))
        
        logger.info(f"✓ Model saved to {CONFIG['paths']['output_dir']}")
        logger.info("\n✓✓✓ TRAINING COMPLETE ✓✓✓\n")
    
    except KeyboardInterrupt:
        logger.warning("\n✓ Interrupted by user. Resumable checkpoint saved.\n")
    except Exception as e:
        logger.error(f"\n✗ Fatal error: {e}\n")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()