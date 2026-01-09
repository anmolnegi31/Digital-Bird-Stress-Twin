"""
LSTM-based temporal stress prediction model with attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from loguru import logger


class AttentionLayer(nn.Module):
    """Attention mechanism for LSTM outputs"""
    
    def __init__(self, hidden_size: int):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention mechanism
        
        Args:
            lstm_outputs: LSTM outputs [batch, seq_len, hidden_size]
            
        Returns:
            context_vector: Weighted sum of LSTM outputs [batch, hidden_size]
            attention_weights: Attention weights [batch, seq_len]
        """
        # Calculate attention scores
        scores = self.attention_weights(lstm_outputs)  # [batch, seq_len, 1]
        scores = scores.squeeze(-1)  # [batch, seq_len]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1)  # [batch, seq_len]
        
        # Calculate context vector
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch, 1, seq_len]
            lstm_outputs  # [batch, seq_len, hidden_size]
        ).squeeze(1)  # [batch, hidden_size]
        
        return context_vector, attention_weights


class StressPredictionLSTM(nn.Module):
    """
    LSTM-based model for temporal stress prediction
    
    Architecture:
    - Bidirectional LSTM layers
    - Attention mechanism
    - Dropout regularization
    - Dense output layers
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True
    ):
        """
        Initialize LSTM model
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
            use_attention: Use attention mechanism
        """
        super(StressPredictionLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # Bidirectional multiplier
        self.num_directions = 2 if bidirectional else 1
        self.lstm_output_size = hidden_size * self.num_directions
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention layer
        if use_attention:
            self.attention = AttentionLayer(self.lstm_output_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.lstm_output_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        
        logger.info(
            f"Initialized StressPredictionLSTM: input_size={input_size}, "
            f"hidden_size={hidden_size}, num_layers={num_layers}, "
            f"bidirectional={bidirectional}, attention={use_attention}"
        )
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, seq_len, input_size]
            hidden: Hidden state tuple (h_0, c_0)
            
        Returns:
            output: Stress predictions [batch, 1]
            attention_weights: Attention weights if used [batch, seq_len]
        """
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)  # [batch, seq_len, lstm_output_size]
        
        # Apply attention or use last output
        attention_weights = None
        if self.use_attention:
            context_vector, attention_weights = self.attention(lstm_out)
        else:
            context_vector = lstm_out[:, -1, :]  # Take last time step
        
        # Fully connected layers
        out = F.relu(self.fc1(context_vector))
        out = self.bn1(out)
        out = self.dropout1(out)
        
        out = F.relu(self.fc2(out))
        out = self.bn2(out)
        out = self.dropout2(out)
        
        # Output layer with sigmoid activation (stress in [0, 1])
        out = torch.sigmoid(self.fc3(out))
        
        return out, attention_weights
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state
        
        Args:
            batch_size: Batch size
            device: Device (CPU/GPU)
            
        Returns:
            Tuple of (h_0, c_0) hidden states
        """
        h_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )
        c_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )
        return (h_0, c_0)
    
    def predict_sequence(
        self,
        x: torch.Tensor,
        future_steps: int = 1
    ) -> torch.Tensor:
        """
        Predict future stress values
        
        Args:
            x: Input sequence [batch, seq_len, input_size]
            future_steps: Number of future steps to predict
            
        Returns:
            Predictions [batch, future_steps]
        """
        self.eval()
        with torch.no_grad():
            predictions = []
            
            # Initial prediction
            current_input = x
            
            for _ in range(future_steps):
                pred, _ = self.forward(current_input)
                predictions.append(pred)
                
                # For next prediction, use the predicted value
                # (In production, you'd combine with future environmental features)
                # Here we just append the prediction to the sequence
                if future_steps > 1:
                    # Shift the sequence and add prediction
                    current_input = torch.cat([
                        current_input[:, 1:, :],
                        pred.unsqueeze(1).expand(-1, -1, x.size(-1))
                    ], dim=1)
            
            predictions = torch.cat(predictions, dim=1)  # [batch, future_steps]
            
        return predictions


class MultiHorizonLSTM(nn.Module):
    """
    LSTM model with multiple prediction horizons (24h, 48h, 72h)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        horizons: list = [24, 48, 72]
    ):
        """
        Initialize multi-horizon LSTM
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            horizons: List of prediction horizons (hours)
        """
        super(MultiHorizonLSTM, self).__init__()
        
        self.horizons = horizons
        
        # Shared LSTM backbone
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        lstm_output_size = hidden_size * 2  # Bidirectional
        
        # Attention
        self.attention = AttentionLayer(lstm_output_size)
        
        # Separate heads for each horizon
        self.horizon_heads = nn.ModuleDict()
        for h in horizons:
            self.horizon_heads[f'h{h}'] = nn.Sequential(
                nn.Linear(lstm_output_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
        
        logger.info(f"Initialized MultiHorizonLSTM with horizons: {horizons}")
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass for all horizons
        
        Args:
            x: Input tensor [batch, seq_len, input_size]
            
        Returns:
            Dictionary of predictions for each horizon
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        context_vector, _ = self.attention(lstm_out)
        
        # Predict for each horizon
        predictions = {}
        for h in self.horizons:
            pred = self.horizon_heads[f'h{h}'](context_vector)
            predictions[f'{h}h'] = pred
        
        return predictions


def create_stress_lstm_model(
    input_size: int,
    config: dict
) -> StressPredictionLSTM:
    """
    Factory function to create LSTM model from config
    
    Args:
        input_size: Number of input features
        config: Model configuration dictionary
        
    Returns:
        Initialized LSTM model
    """
    model = StressPredictionLSTM(
        input_size=input_size,
        hidden_size=config.get('hidden_size', 256),
        num_layers=config.get('num_layers', 3),
        dropout=config.get('dropout', 0.3),
        bidirectional=config.get('bidirectional', True),
        use_attention=config.get('attention', True)
    )
    
    return model


def create_multi_horizon_model(
    input_size: int,
    config: dict
) -> MultiHorizonLSTM:
    """
    Factory function to create multi-horizon LSTM model
    
    Args:
        input_size: Number of input features
        config: Model configuration dictionary
        
    Returns:
        Initialized multi-horizon LSTM model
    """
    model = MultiHorizonLSTM(
        input_size=input_size,
        hidden_size=config.get('hidden_size', 256),
        num_layers=config.get('num_layers', 3),
        dropout=config.get('dropout', 0.3),
        horizons=config.get('prediction_horizons', [24, 48, 72])
    )
    
    return model
