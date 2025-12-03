"""
Training Script for IndraQuantum
Implements Knowledge Distillation (KD) from a teacher model to the quantum student model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
import argparse
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from typing import Optional, Dict, Any
import logging
from tqdm import tqdm

from indra.models.quantum_core import IndraQuantum


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Simple text dataset for language modeling"""
    
    def __init__(self, file_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load text data
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize
        self.encodings = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_overflowing_tokens=True,
            stride=max_length // 2
        )
        
        logger.info(f"Loaded {len(self.encodings['input_ids'])} sequences from {file_path}")
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx])
        }


class KnowledgeDistillationTrainer:
    """
    Trainer for Knowledge Distillation from a teacher model to IndraQuantum student
    """
    
    def __init__(
        self,
        student_model: IndraQuantum,
        teacher_model: nn.Module,
        tokenizer,
        config: Dict[str, Any],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs'],
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # KD hyperparameters
        self.temperature = config.get('temperature', 2.0)
        self.alpha = config.get('alpha', 0.5)  # Weight for distillation loss
        
        logger.info(f"Trainer initialized on device: {device}")
        logger.info(f"Student parameters: {self.student.get_num_params():,}")
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined distillation and task loss.
        
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            labels: Ground truth labels
            attention_mask: Optional attention mask
        
        Returns:
            Dictionary with loss components
        """
        # Shift logits and labels for language modeling
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
        
        # Task loss (cross-entropy with ground truth)
        task_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        # Distillation loss (KL divergence with teacher)
        student_probs = F.log_softmax(shift_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(shift_teacher_logits / self.temperature, dim=-1)
        
        distill_loss = F.kl_div(
            student_probs.view(-1, student_probs.size(-1)),
            teacher_probs.view(-1, teacher_probs.size(-1)),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * task_loss
        
        return {
            'total_loss': total_loss,
            'task_loss': task_loss,
            'distill_loss': distill_loss
        }
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.student.train()
        total_losses = {'total_loss': 0, 'task_loss': 0, 'distill_loss': 0}
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass through teacher
            with torch.no_grad():
                teacher_outputs = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits
            
            # Forward pass through student
            student_logits = self.student(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Compute loss
            losses = self.compute_distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=input_ids,
                attention_mask=attention_mask
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.student.parameters(),
                self.config.get('max_grad_norm', 1.0)
            )
            
            self.optimizer.step()
            
            # Accumulate losses
            for key in total_losses:
                total_losses[key] += losses[key].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total_loss'].item(),
                'task': losses['task_loss'].item(),
                'distill': losses['distill_loss'].item()
            })
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def train(self, dataloader: DataLoader, num_epochs: int, save_dir: str):
        """Full training loop"""
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{num_epochs}")
            logger.info(f"{'='*50}")
            
            # Train epoch
            metrics = self.train_epoch(dataloader, epoch)
            
            # Log metrics
            logger.info(f"Epoch {epoch} completed:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.4f}")
            
            # Update scheduler
            self.scheduler.step()
            
            # Save checkpoint
            if epoch % self.config.get('save_every', 1) == 0:
                checkpoint_path = os.path.join(
                    save_dir,
                    f"checkpoint_epoch_{epoch}.pt"
                )
                self.save_checkpoint(checkpoint_path, epoch, metrics)
        
        # Save final model
        final_path = os.path.join(save_dir, "final_model.pt")
        self.save_checkpoint(final_path, num_epochs, metrics)
        logger.info(f"\nTraining completed! Final model saved to {final_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train IndraQuantum with Knowledge Distillation')
    parser.add_argument('--config', type=str, default='configs/quantum_6gb.yaml',
                        help='Path to config file')
    parser.add_argument('--teacher', type=str, default='gpt2',
                        help='Teacher model name or path')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--log_file', type=str, default='training.log',
                        help='Path to log file')
    
    args = parser.parse_args()
    
    # Setup file logging
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Load teacher model and tokenizer
    logger.info(f"Loading teacher model: {args.teacher}")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher)
    
    # Create student model
    logger.info("Creating IndraQuantum student model")
    student_model = IndraQuantum(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        max_seq_length=config['max_seq_length']
    )
    
    # Create dataset and dataloader
    logger.info(f"Loading dataset from {args.dataset}")
    dataset = TextDataset(
        args.dataset,
        tokenizer,
        max_length=config['max_seq_length']
    )
    
    # DEBUG: Check max token ID
    all_ids = torch.tensor(dataset.encodings['input_ids']).flatten()
    max_id = all_ids.max().item()
    logger.info(f"DEBUG: Max Token ID in dataset: {max_id}")
    logger.info(f"DEBUG: Config Vocab Size: {config['vocab_size']}")
    
    if max_id >= config['vocab_size']:
        logger.error(f"CRITICAL: Max token ID {max_id} exceeds vocab size {config['vocab_size']}!")
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 0)
    )
    
    # Create trainer
    trainer = KnowledgeDistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        config=config,
        device=args.device
    )
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=args.device)
        trainer.student.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {checkpoint['epoch']}")

    # Train
    logger.info("Starting training...")
    # Modified train loop to support start_epoch
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(start_epoch, config['num_epochs'] + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{config['num_epochs']}")
        logger.info(f"{'='*50}")
        
        # Train epoch
        metrics = trainer.train_epoch(dataloader, epoch)
        
        # Log metrics
        logger.info(f"Epoch {epoch} completed:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        current_lr = trainer.scheduler.get_last_lr()[0]
        logger.info(f"  Learning Rate: {current_lr:.6e}")
        
        # Update scheduler
        trainer.scheduler.step()
        
        # Save checkpoint
        if epoch % config.get('save_every', 1) == 0:
            checkpoint_path = os.path.join(
                args.output_dir,
                f"checkpoint_epoch_{epoch}.pt"
            )
            trainer.save_checkpoint(checkpoint_path, epoch, metrics)
    
    # Save final model
    final_path = os.path.join(args.output_dir, "final_model.pt")
    trainer.save_checkpoint(final_path, config['num_epochs'], metrics)
    logger.info(f"\nTraining completed! Final model saved to {final_path}")


if __name__ == '__main__':
    main()
