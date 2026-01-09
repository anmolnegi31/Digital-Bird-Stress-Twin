"""
Variational Autoencoder (VAE) for acoustic behavior simulation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from loguru import logger


class Encoder(nn.Module):
    """Encoder network for VAE"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        latent_dim: int,
        activation: str = 'relu'
    ):
        """
        Initialize encoder
        
        Args:
            input_dim: Input feature dimension (e.g., 40 for MFCCs)
            hidden_dims: List of hidden layer dimensions
            latent_dim: Latent space dimension
            activation: Activation function
        """
        super(Encoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Latent space layers
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        logger.debug(f"Initialized Encoder: input_dim={input_dim}, latent_dim={latent_dim}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, input_dim]
            
        Returns:
            mu: Mean of latent distribution [batch, latent_dim]
            logvar: Log variance of latent distribution [batch, latent_dim]
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class Decoder(nn.Module):
    """Decoder network for VAE"""
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list,
        output_dim: int,
        activation: str = 'relu'
    ):
        """
        Initialize decoder
        
        Args:
            latent_dim: Latent space dimension
            hidden_dims: List of hidden layer dimensions (reversed)
            output_dim: Output feature dimension
            activation: Activation function
        """
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
        
        # Build decoder layers
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*layers)
        
        # Output layer
        self.fc_out = nn.Linear(prev_dim, output_dim)
        
        logger.debug(f"Initialized Decoder: latent_dim={latent_dim}, output_dim={output_dim}")
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            z: Latent vector [batch, latent_dim]
            
        Returns:
            Reconstructed output [batch, output_dim]
        """
        h = self.decoder(z)
        out = self.fc_out(h)
        
        return out


class VAE(nn.Module):
    """
    Variational Autoencoder for acoustic behavior simulation
    
    This model learns the latent representation of bird vocalizations
    and can generate synthetic acoustic patterns conditioned on stress levels.
    """
    
    def __init__(
        self,
        input_dim: int = 40,
        latent_dim: int = 32,
        encoder_hidden_dims: list = None,
        decoder_hidden_dims: list = None,
        activation: str = 'relu',
        beta: float = 1.0
    ):
        """
        Initialize VAE
        
        Args:
            input_dim: Input feature dimension (MFCC dimension)
            latent_dim: Latent space dimension
            encoder_hidden_dims: Encoder hidden layer dimensions
            decoder_hidden_dims: Decoder hidden layer dimensions
            activation: Activation function
            beta: Beta coefficient for KL divergence (β-VAE)
        """
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Default architecture
        if encoder_hidden_dims is None:
            encoder_hidden_dims = [128, 64]
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [64, 128]
        
        # Encoder and Decoder
        self.encoder = Encoder(input_dim, encoder_hidden_dims, latent_dim, activation)
        self.decoder = Decoder(latent_dim, decoder_hidden_dims, input_dim, activation)
        
        logger.info(
            f"Initialized VAE: input_dim={input_dim}, latent_dim={latent_dim}, "
            f"beta={beta}"
        )
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, input_dim]
            
        Returns:
            reconstructed: Reconstructed output [batch, input_dim]
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructed = self.decoder(z)
        
        return reconstructed, mu, logvar
    
    def loss_function(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VAE loss function = Reconstruction Loss + β * KL Divergence
        
        Args:
            reconstructed: Reconstructed output
            original: Original input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            total_loss: Total VAE loss
            recon_loss: Reconstruction loss
            kl_loss: KL divergence loss
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed, original, reduction='mean')
        
        # KL divergence loss
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space
        
        Args:
            x: Input tensor [batch, input_dim]
            
        Returns:
            Latent representation [batch, latent_dim]
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to output
        
        Args:
            z: Latent vector [batch, latent_dim]
            
        Returns:
            Decoded output [batch, input_dim]
        """
        return self.decoder(z)
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample from latent space and generate outputs
        
        Args:
            num_samples: Number of samples to generate
            device: Device (CPU/GPU)
            
        Returns:
            Generated samples [num_samples, input_dim]
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z)
        return samples


class ConditionalVAE(nn.Module):
    """
    Conditional VAE for stress-conditioned acoustic generation
    
    This allows generating bird vocalizations conditioned on specific stress levels.
    """
    
    def __init__(
        self,
        input_dim: int = 40,
        condition_dim: int = 1,  # Stress level
        latent_dim: int = 32,
        encoder_hidden_dims: list = None,
        decoder_hidden_dims: list = None,
        activation: str = 'relu',
        beta: float = 1.0
    ):
        """
        Initialize Conditional VAE
        
        Args:
            input_dim: Input feature dimension
            condition_dim: Condition dimension (stress level)
            latent_dim: Latent space dimension
            encoder_hidden_dims: Encoder hidden dimensions
            decoder_hidden_dims: Decoder hidden dimensions
            activation: Activation function
            beta: Beta coefficient for KL divergence
        """
        super(ConditionalVAE, self).__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Default architecture
        if encoder_hidden_dims is None:
            encoder_hidden_dims = [128, 64]
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [64, 128]
        
        # Encoder with condition
        self.encoder = Encoder(
            input_dim + condition_dim,
            encoder_hidden_dims,
            latent_dim,
            activation
        )
        
        # Decoder with condition
        self.decoder = Decoder(
            latent_dim + condition_dim,
            decoder_hidden_dims,
            input_dim,
            activation
        )
        
        logger.info(
            f"Initialized ConditionalVAE: input_dim={input_dim}, "
            f"condition_dim={condition_dim}, latent_dim={latent_dim}"
        )
    
    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, input_dim]
            condition: Condition tensor [batch, condition_dim]
            
        Returns:
            reconstructed: Reconstructed output
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        # Concatenate input with condition
        x_cond = torch.cat([x, condition], dim=1)
        
        # Encode
        mu, logvar = self.encoder(x_cond)
        
        # Reparameterize
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Concatenate latent with condition
        z_cond = torch.cat([z, condition], dim=1)
        
        # Decode
        reconstructed = self.decoder(z_cond)
        
        return reconstructed, mu, logvar
    
    def generate_conditioned(
        self,
        stress_level: float,
        num_samples: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate acoustic patterns for a specific stress level
        
        Args:
            stress_level: Stress level [0, 1]
            num_samples: Number of samples to generate
            device: Device (CPU/GPU)
            
        Returns:
            Generated acoustic features [num_samples, input_dim]
        """
        # Sample from latent space
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        # Create condition tensor
        condition = torch.full((num_samples, self.condition_dim), stress_level, device=device)
        
        # Concatenate and decode
        z_cond = torch.cat([z, condition], dim=1)
        samples = self.decoder(z_cond)
        
        return samples
    
    def loss_function(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """VAE loss function"""
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, original, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


def create_vae_model(input_dim: int, config: dict) -> VAE:
    """
    Factory function to create VAE model from config
    
    Args:
        input_dim: Input feature dimension
        config: Model configuration dictionary
        
    Returns:
        Initialized VAE model
    """
    model = VAE(
        input_dim=input_dim,
        latent_dim=config.get('latent_dim', 32),
        encoder_hidden_dims=config.get('encoder_layers', [128, 64]),
        decoder_hidden_dims=config.get('decoder_layers', [64, 128]),
        activation=config.get('activation', 'relu'),
        beta=config.get('beta', 1.0)
    )
    
    return model


def create_conditional_vae_model(input_dim: int, config: dict) -> ConditionalVAE:
    """
    Factory function to create Conditional VAE model
    
    Args:
        input_dim: Input feature dimension
        config: Model configuration dictionary
        
    Returns:
        Initialized Conditional VAE model
    """
    model = ConditionalVAE(
        input_dim=input_dim,
        condition_dim=1,  # Stress level
        latent_dim=config.get('latent_dim', 32),
        encoder_hidden_dims=config.get('encoder_layers', [128, 64]),
        decoder_hidden_dims=config.get('decoder_layers', [64, 128]),
        activation=config.get('activation', 'relu'),
        beta=config.get('beta', 1.0)
    )
    
    return model
