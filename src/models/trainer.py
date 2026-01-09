"""
Training utilities and trainers for ML models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import mlflow
import mlflow.pytorch


class StressDataset(Dataset):
    """Dataset for stress prediction"""
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sequence_length: int = 50
    ):
        """
        Initialize dataset
        
        Args:
            features: Feature array [num_samples, num_features]
            labels: Label array [num_samples]
            sequence_length: Length of sequences for LSTM
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.sequence_length = sequence_length
        
        # Create sequences
        self.sequences, self.seq_labels = self._create_sequences()
    
    def _create_sequences(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create sequences from data"""
        sequences = []
        seq_labels = []
        
        for i in range(len(self.features) - self.sequence_length):
            seq = self.features[i:i + self.sequence_length]
            label = self.labels[i + self.sequence_length]
            
            sequences.append(seq)
            seq_labels.append(label)
        
        return torch.stack(sequences), torch.stack(seq_labels)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.seq_labels[idx]


class LSTMTrainer:
    """Trainer for LSTM stress prediction model"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001
    ):
        """
        Initialize trainer
        
        Args:
            model: LSTM model
            device: Device (CPU/GPU)
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        logger.info(f"Initialized LSTMTrainer on {device}")
    
    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (features, labels) in enumerate(progress_bar):
            features = features.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions, _ = self.model(features)
            
            # Calculate loss
            loss = self.criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        
        return {
            'train_loss': avg_loss
        }
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                # Forward pass
                predictions, _ = self.model(features)
                
                # Calculate loss
                loss = self.criterion(predictions, labels)
                total_loss += loss.item()
                
                # Store predictions and labels
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        mae = np.mean(np.abs(all_predictions - all_labels))
        rmse = np.sqrt(np.mean((all_predictions - all_labels) ** 2))
        
        # R² score
        ss_res = np.sum((all_labels - all_predictions) ** 2)
        ss_tot = np.sum((all_labels - np.mean(all_labels)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        return {
            'val_loss': avg_loss,
            'val_mae': mae,
            'val_rmse': rmse,
            'val_r2': r2
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        early_stopping_patience: int = 15,
        checkpoint_dir: Optional[Path] = None,
        mlflow_tracking: bool = True
    ) -> Dict[str, Any]:
        """
        Train model with early stopping
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
            mlflow_tracking: Enable MLflow tracking
            
        Returns:
            Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse': [],
            'val_r2': []
        }
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update history
            history['train_loss'].append(train_metrics['train_loss'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_mae'].append(val_metrics['val_mae'])
            history['val_rmse'].append(val_metrics['val_rmse'])
            history['val_r2'].append(val_metrics['val_r2'])
            
            # Log metrics
            logger.info(
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val MAE: {val_metrics['val_mae']:.4f} | "
                f"Val RMSE: {val_metrics['val_rmse']:.4f} | "
                f"Val R²: {val_metrics['val_r2']:.4f}"
            )
            
            # MLflow logging
            if mlflow_tracking:
                mlflow.log_metrics({
                    'train_loss': train_metrics['train_loss'],
                    'val_loss': val_metrics['val_loss'],
                    'val_mae': val_metrics['val_mae'],
                    'val_rmse': val_metrics['val_rmse'],
                    'val_r2': val_metrics['val_r2']
                }, step=epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['val_loss'])
            
            # Early stopping and checkpointing
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                
                # Save best model
                if checkpoint_dir:
                    checkpoint_path = checkpoint_dir / 'best_model.pth'
                    self.save_checkpoint(checkpoint_path, epoch, val_metrics)
                    logger.info(f"Saved best model to {checkpoint_path}")
            else:
                patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        logger.info("Training completed!")
        return history
    
    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }, path)
    
    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint


class VAETrainer:
    """Trainer for VAE model"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.0005
    ):
        """
        Initialize VAE trainer
        
        Args:
            model: VAE model
            device: Device (CPU/GPU)
            learning_rate: Learning rate
        """
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        
        logger.info(f"Initialized VAETrainer on {device}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training VAE")
        
        for batch_data in progress_bar:
            if isinstance(batch_data, (tuple, list)):
                features = batch_data[0]
                if len(batch_data) > 1:
                    conditions = batch_data[1]
                else:
                    conditions = None
            else:
                features = batch_data
                conditions = None
            
            features = features.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if conditions is not None and hasattr(self.model, 'forward'):
                # Conditional VAE
                conditions = conditions.to(self.device)
                reconstructed, mu, logvar = self.model(features, conditions)
            else:
                # Standard VAE
                reconstructed, mu, logvar = self.model(features)
            
            # Calculate loss
            total, recon, kl = self.model.loss_function(
                reconstructed, features, mu, logvar
            )
            
            # Backward pass
            total.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += total.item()
            total_recon_loss += recon.item()
            total_kl_loss += kl.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total.item(),
                'recon': recon.item(),
                'kl': kl.item()
            })
        
        return {
            'train_loss': total_loss / num_batches,
            'train_recon_loss': total_recon_loss / num_batches,
            'train_kl_loss': total_kl_loss / num_batches
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate VAE model"""
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                if isinstance(batch_data, (tuple, list)):
                    features = batch_data[0]
                    conditions = batch_data[1] if len(batch_data) > 1 else None
                else:
                    features = batch_data
                    conditions = None
                
                features = features.to(self.device)
                
                # Forward pass
                if conditions is not None:
                    conditions = conditions.to(self.device)
                    reconstructed, mu, logvar = self.model(features, conditions)
                else:
                    reconstructed, mu, logvar = self.model(features)
                
                # Calculate loss
                total, recon, kl = self.model.loss_function(
                    reconstructed, features, mu, logvar
                )
                
                total_loss += total.item()
                total_recon_loss += recon.item()
                total_kl_loss += kl.item()
        
        num_batches = len(val_loader)
        return {
            'val_loss': total_loss / num_batches,
            'val_recon_loss': total_recon_loss / num_batches,
            'val_kl_loss': total_kl_loss / num_batches
        }
