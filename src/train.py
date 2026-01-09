"""
End-to-end training pipeline for stress prediction model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
import mlflow
import mlflow.pytorch
from datetime import datetime
import yaml

from models import (
    create_stress_lstm_model,
    create_multi_horizon_model,
    LSTMTrainer,
    StressDataset
)
from utils.config import Config


class TrainingPipeline:
    """End-to-end training pipeline"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize training pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path) if config_path else Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.config.get("mlops.mlflow.tracking_uri", "http://localhost:5000"))
        mlflow.set_experiment(self.config.get("mlops.mlflow.experiment_name", "digital-bird-stress-twin"))
        
        logger.info(f"Initialized training pipeline on {self.device}")
    
    def load_data(self, data_path: Path) -> Dict[str, np.ndarray]:
        """
        Load training data
        
        Args:
            data_path: Path to data file
            
        Returns:
            Dictionary with features and labels
        """
        logger.info(f"Loading data from {data_path}")
        
        # Load data (assuming CSV or numpy format)
        if data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
            
            # Separate features and labels
            label_column = 'stress_level'  # Adjust as needed
            
            features = df.drop(columns=[label_column]).values
            labels = df[label_column].values
            
        elif data_path.suffix == '.npz':
            data = np.load(data_path)
            features = data['features']
            labels = data['labels']
        else:
            raise ValueError(f"Unsupported data format: {data_path.suffix}")
        
        logger.info(f"Loaded data: features shape={features.shape}, labels shape={labels.shape}")
        
        return {
            'features': features,
            'labels': labels
        }
    
    def create_data_loaders(
        self,
        data: Dict[str, np.ndarray],
        batch_size: int = 32,
        sequence_length: int = 50
    ) -> Dict[str, DataLoader]:
        """
        Create train/val/test data loaders
        
        Args:
            data: Dictionary with features and labels
            batch_size: Batch size
            sequence_length: Sequence length for LSTM
            
        Returns:
            Dictionary of data loaders
        """
        # Create dataset
        dataset = StressDataset(
            features=data['features'],
            labels=data['labels'],
            sequence_length=sequence_length
        )
        
        # Split into train/val/test
        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(
            f"Created data loaders: "
            f"train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}"
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
    
    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        input_size: int,
        run_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train model with MLflow tracking
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            input_size: Input feature dimension
            run_name: MLflow run name
            
        Returns:
            Training results dictionary
        """
        # Start MLflow run
        with mlflow.start_run(run_name=run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Get model configuration
            model_config = self.config.get('models.lstm.architecture', {})
            training_config = self.config.get('models.lstm.training', {})
            
            # Log parameters
            mlflow.log_params({
                'input_size': input_size,
                'hidden_size': model_config.get('hidden_size', 256),
                'num_layers': model_config.get('num_layers', 3),
                'dropout': model_config.get('dropout', 0.3),
                'bidirectional': model_config.get('bidirectional', True),
                'attention': model_config.get('attention', True),
                'learning_rate': training_config.get('learning_rate', 0.001),
                'batch_size': training_config.get('batch_size', 32),
                'epochs': training_config.get('epochs', 100)
            })
            
            # Create model
            model = create_stress_lstm_model(input_size, model_config)
            logger.info(f"Created model: {model.__class__.__name__}")
            
            # Create trainer
            trainer = LSTMTrainer(
                model=model,
                device=self.device,
                learning_rate=training_config.get('learning_rate', 0.001),
                weight_decay=training_config.get('weight_decay', 0.0001)
            )
            
            # Train
            checkpoint_dir = Path(self.config.get('paths.models.checkpoints', './models/checkpoints'))
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=training_config.get('epochs', 100),
                early_stopping_patience=training_config.get('early_stopping_patience', 15),
                checkpoint_dir=checkpoint_dir,
                mlflow_tracking=True
            )
            
            # Log final metrics
            best_val_loss = min(history['val_loss'])
            best_val_r2 = max(history['val_r2'])
            
            mlflow.log_metrics({
                'best_val_loss': best_val_loss,
                'best_val_r2': best_val_r2,
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1]
            })
            
            # Log model
            mlflow.pytorch.log_model(model, "model")
            
            # Save to registry if performance meets threshold
            if best_val_r2 >= self.config.get('mlops.model_registry.staging_threshold', 0.85):
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
                model_details = mlflow.register_model(
                    model_uri,
                    self.config.get('mlops.model_registry.model_name', 'bird-stress-predictor')
                )
                logger.info(f"Model registered: {model_details.name} version {model_details.version}")
            
            logger.info("Training completed successfully!")
            
            return {
                'history': history,
                'best_val_loss': best_val_loss,
                'best_val_r2': best_val_r2,
                'run_id': mlflow.active_run().info.run_id
            }
    
    def run_full_pipeline(
        self,
        data_path: Path,
        run_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run complete training pipeline
        
        Args:
            data_path: Path to training data
            run_name: MLflow run name
            
        Returns:
            Pipeline results
        """
        logger.info("="*80)
        logger.info("Starting Full Training Pipeline")
        logger.info("="*80)
        
        # Load data
        data = self.load_data(data_path)
        
        # Create data loaders
        training_config = self.config.get('models.lstm.training', {})
        loaders = self.create_data_loaders(
            data=data,
            batch_size=training_config.get('batch_size', 32),
            sequence_length=self.config.get('models.lstm.sequence_length', 50)
        )
        
        # Train model
        input_size = data['features'].shape[1]
        results = self.train_model(
            train_loader=loaders['train'],
            val_loader=loaders['val'],
            input_size=input_size,
            run_name=run_name
        )
        
        logger.info("="*80)
        logger.info("Pipeline Completed Successfully!")
        logger.info(f"Best Validation Loss: {results['best_val_loss']:.4f}")
        logger.info(f"Best Validation R²: {results['best_val_r2']:.4f}")
        logger.info(f"MLflow Run ID: {results['run_id']}")
        logger.info("="*80)
        
        return results


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Digital Bird Stress Twin model")
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to training data file'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='MLflow run name'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    config_path = Path(args.config) if args.config else None
    pipeline = TrainingPipeline(config_path)
    
    # Run training
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    results = pipeline.run_full_pipeline(
        data_path=data_path,
        run_name=args.run_name
    )


if __name__ == "__main__":
    main()
