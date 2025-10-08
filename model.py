"""
Stock Market Prediction Model Inspired by DeepSeek V3.2 Architecture
Implements: Sparse Attention, Mixture of Experts, Long Context Processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
import math


class SparseAttention(nn.Module):
    """
    Inspired by DeepSeek Sparse Attention (DSA)
    Uses learned indexing to select relevant time steps for attention
    """
    def __init__(self, d_model: int, n_heads: int, sparsity_ratio: float = 0.3):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.sparsity_ratio = sparsity_ratio
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # Indexer network for sparse selection (DSA-inspired)
        self.indexer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute importance scores for sparse selection
        importance = self.indexer(x).squeeze(-1)  # [batch, seq_len]
        
        # Select top-k based on sparsity ratio
        k = max(1, int(seq_len * self.sparsity_ratio))
        _, top_indices = torch.topk(importance, k, dim=-1)
        
        # Create sparse attention mask
        sparse_mask = torch.zeros(batch_size, seq_len, device=x.device)
        sparse_mask.scatter_(1, top_indices, 1.0)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply sparse mask
        sparse_mask = sparse_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
        attn_scores = attn_scores.masked_fill(sparse_mask == 0, float('-inf'))
        
        # Apply causal mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(attn_output)


class ExpertNetwork(nn.Module):
    """Single expert in the MoE system"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts inspired by DeepSeekMoE
    Different experts specialize in different market conditions
    """
    def __init__(self, d_model: int, d_ff: int, n_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        
        # Expert networks
        self.experts = nn.ModuleList([
            ExpertNetwork(d_model, d_ff) for _ in range(n_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(d_model, n_experts)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Compute gating scores
        gate_logits = self.gate(x)  # [batch, seq_len, n_experts]
        gate_scores = F.softmax(gate_logits, dim=-1)
        
        # Select top-k experts
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)
        
        # Process through selected experts
        output = torch.zeros_like(x)
        
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, :, i]
            expert_weight = top_k_scores[:, :, i].unsqueeze(-1)
            
            for expert_id in range(self.n_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += expert_weight[mask] * expert_output
        
        return output


class TransformerBlock(nn.Module):
    """Transformer block with Sparse Attention and MoE"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_experts: int, 
                 sparsity_ratio: float = 0.3, dropout: float = 0.1):
        super().__init__()
        
        self.sparse_attn = SparseAttention(d_model, n_heads, sparsity_ratio)
        self.moe = MixtureOfExperts(d_model, d_ff, n_experts)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Sparse attention with residual
        attn_out = self.sparse_attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        # MoE with residual
        moe_out = self.moe(self.norm2(x))
        x = x + self.dropout(moe_out)
        
        return x


class StockPredictionModel(nn.Module):
    """
    Complete stock prediction model inspired by DeepSeek V3.2
    - Handles long sequences (up to 5000 time steps)
    - Uses sparse attention for efficiency
    - MoE for different market regimes
    """
    def __init__(
        self,
        n_features: int,          # Number of stock features (OHLCV, indicators, etc.)
        d_model: int = 256,        # Model dimension (reduced from 671B!)
        n_heads: int = 8,
        d_ff: int = 1024,
        n_layers: int = 6,
        n_experts: int = 8,
        sparsity_ratio: float = 0.3,
        max_seq_len: int = 5000,   # Long context support
        dropout: float = 0.1,
        prediction_horizon: int = 1  # Steps ahead to predict
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.prediction_horizon = prediction_horizon
        
        # Input embedding
        self.input_proj = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, n_experts, sparsity_ratio, dropout)
            for _ in range(n_layers)
        ])
        
        # Output heads
        self.norm = nn.LayerNorm(d_model)
        
        # Multi-target prediction
        self.price_head = nn.Linear(d_model, prediction_horizon)  # Future prices
        self.direction_head = nn.Linear(d_model, prediction_horizon * 3)  # Up/Down/Neutral
        self.volatility_head = nn.Linear(d_model, prediction_horizon)  # Volatility forecast
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, n_features] - stock data
        Returns:
            prices: [batch, prediction_horizon] - predicted prices
            directions: [batch, prediction_horizon, 3] - up/down/neutral probs
            volatility: [batch, prediction_horizon] - volatility predictions
        """
        batch_size, seq_len, _ = x.shape
        
        # Embed input
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(x.device)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.norm(x)
        
        # Use last token for prediction
        last_hidden = x[:, -1, :]
        
        # Multi-target predictions
        prices = self.price_head(last_hidden)
        directions = self.direction_head(last_hidden).view(batch_size, self.prediction_horizon, 3)
        volatility = self.volatility_head(last_hidden)
        
        return prices, F.softmax(directions, dim=-1), F.relu(volatility)


class StockDataProcessor:
    """Data preprocessing for stock market data"""
    
    @staticmethod
    def create_features(df: pd.DataFrame) -> np.ndarray:
        """
        Create technical indicators and features from OHLCV data
        """
        features = []
        
        # Basic OHLCV
        features.extend([
            df['Open'].values,
            df['High'].values,
            df['Low'].values,
            df['Close'].values,
            df['Volume'].values
        ])
        
        # Returns
        returns = df['Close'].pct_change().fillna(0).values
        features.append(returns)
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            ma = df['Close'].rolling(window=window).mean().fillna(method='bfill').values
            features.append(ma)
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).fillna(50).values
        features.append(rsi)
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = (exp1 - exp2).fillna(0).values
        features.append(macd)
        
        # Bollinger Bands
        sma = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        upper_band = (sma + (std * 2)).fillna(method='bfill').values
        lower_band = (sma - (std * 2)).fillna(method='bfill').values
        features.extend([upper_band, lower_band])
        
        return np.stack(features, axis=1)
    
    @staticmethod
    def normalize_features(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize features using rolling statistics"""
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + 1e-8
        normalized = (features - mean) / std
        return normalized, mean, std


# Training example
def train_example():
    """Example training loop"""
    
    # Model configuration
    model = StockPredictionModel(
        n_features=15,  # OHLCV + indicators
        d_model=256,
        n_heads=8,
        d_ff=1024,
        n_layers=6,
        n_experts=8,
        sparsity_ratio=0.3,
        max_seq_len=5000,
        prediction_horizon=5
    )
    
    # Synthetic data example (replace with real data)
    batch_size = 16
    seq_len = 1000
    n_features = 15
    
    # Create synthetic stock data
    x = torch.randn(batch_size, seq_len, n_features)
    
    # Forward pass
    prices, directions, volatility = model(x)
    
    print(f"Price predictions shape: {prices.shape}")  # [16, 5]
    print(f"Direction predictions shape: {directions.shape}")  # [16, 5, 3]
    print(f"Volatility predictions shape: {volatility.shape}")  # [16, 5]
    
    # Loss calculation (example)
    target_prices = torch.randn(batch_size, 5)
    target_directions = torch.randint(0, 3, (batch_size, 5))
    target_volatility = torch.rand(batch_size, 5)
    
    price_loss = F.mse_loss(prices, target_prices)
    direction_loss = F.cross_entropy(
        directions.view(-1, 3), 
        target_directions.view(-1)
    )
    volatility_loss = F.mse_loss(volatility, target_volatility)
    
    total_loss = price_loss + direction_loss + 0.5 * volatility_loss
    
    print(f"\nTotal loss: {total_loss.item():.4f}")
    
    return model


if __name__ == "__main__":
    print("Stock Prediction Model - DeepSeek Inspired Architecture")
    print("=" * 60)
    model = train_example()
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"Total: {total_params:,}")
    print(f"Trainable: {trainable_params:,}")
    print(f"\nThis is {total_params / 671_000_000_000 * 100:.6f}% the size of DeepSeek V3.2!")
