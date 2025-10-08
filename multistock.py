"""
Multi-Stock Training System
Learns patterns across multiple stocks simultaneously with cross-stock attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import yfinance as yf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class StockEmbedding(nn.Module):
    """Learnable embeddings for different stocks"""
    
    def __init__(self, n_stocks: int, d_model: int):
        super().__init__()
        self.stock_embeddings = nn.Embedding(n_stocks, d_model)
        self.sector_embeddings = nn.Embedding(20, d_model)  # Different sectors
        
    def forward(self, stock_ids: torch.Tensor, sector_ids: torch.Tensor) -> torch.Tensor:
        return self.stock_embeddings(stock_ids) + self.sector_embeddings(sector_ids)


class CrossStockAttention(nn.Module):
    """
    Cross-stock attention mechanism
    Allows the model to learn from patterns across different stocks
    """
    
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
    def forward(self, query_stock: torch.Tensor, all_stocks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_stock: [batch, seq_len, d_model] - features of target stock
            all_stocks: [batch, n_stocks, seq_len, d_model] - features of all stocks
        """
        batch_size, seq_len, _ = query_stock.shape
        n_stocks = all_stocks.shape[1]
        
        # Project query from target stock
        Q = self.q_proj(query_stock).view(batch_size, seq_len, self.n_heads, self.head_dim)
        Q = Q.permute(0, 2, 1, 3)  # [batch, n_heads, seq_len, head_dim]
        
        # Project keys and values from all stocks
        all_stocks_flat = all_stocks.view(batch_size, n_stocks * seq_len, self.d_model)
        K = self.k_proj(all_stocks_flat).view(batch_size, n_stocks * seq_len, self.n_heads, self.head_dim)
        V = self.v_proj(all_stocks_flat).view(batch_size, n_stocks * seq_len, self.n_heads, self.head_dim)
        
        K = K.permute(0, 2, 1, 3)  # [batch, n_heads, n_stocks*seq_len, head_dim]
        V = V.permute(0, 2, 1, 3)
        
        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        return self.o_proj(attn_output)


class MultiStockTransformerBlock(nn.Module):
    """Transformer block with both self-attention and cross-stock attention"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Self attention (within stock)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Cross-stock attention
        self.cross_stock_attn = CrossStockAttention(d_model, n_heads)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        all_stocks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self attention
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # Cross-stock attention (if other stocks provided)
        if all_stocks is not None:
            cross_attn_out = self.cross_stock_attn(x, all_stocks)
            x = x + self.dropout(cross_attn_out)
            x = self.norm2(x)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm3(x)
        
        return x


class MultiStockPredictionModel(nn.Module):
    """
    Multi-stock prediction model
    Learns shared patterns across stocks while maintaining stock-specific parameters
    """
    
    def __init__(
        self,
        n_stocks: int,
        n_features: int,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        n_layers: int = 6,
        max_seq_len: int = 500,
        prediction_horizon: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_stocks = n_stocks
        self.d_model = d_model
        self.prediction_horizon = prediction_horizon
        
        # Stock embeddings
        self.stock_embedding = StockEmbedding(n_stocks, d_model)
        
        # Feature projection
        self.feature_proj = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_len, d_model)
        
        # Shared transformer layers
        self.shared_layers = nn.ModuleList([
            MultiStockTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers // 2)
        ])
        
        # Stock-specific layers
        self.stock_specific_layers = nn.ModuleList([
            MultiStockTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers // 2)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Prediction heads
        self.price_head = nn.Linear(d_model, prediction_horizon)
        self.direction_head = nn.Linear(d_model, prediction_horizon * 3)
        self.volatility_head = nn.Linear(d_model, prediction_horizon)
        
        # Correlation prediction (for portfolio optimization)
        self.correlation_head = nn.Linear(d_model, n_stocks)
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(
        self,
        features: torch.Tensor,
        stock_ids: torch.Tensor,
        sector_ids: torch.Tensor,
        all_stocks_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [batch, seq_len, n_features]
            stock_ids: [batch]
            sector_ids: [batch]
            all_stocks_features: [batch, n_stocks, seq_len, n_features] (optional)
        """
        batch_size, seq_len, _ = features.shape
        
        # Project features
        x = self.feature_proj(features)
        
        # Add stock embedding
        stock_emb = self.stock_embedding(stock_ids, sector_ids)  # [batch, d_model]
        stock_emb = stock_emb.unsqueeze(1).expand(-1, seq_len, -1)
        x = x + stock_emb
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        
        # Process all stocks features if provided
        all_stocks_encoded = None
        if all_stocks_features is not None:
            b, n_stocks, s, f = all_stocks_features.shape
            all_stocks_flat = all_stocks_features.view(b * n_stocks, s, f)
            all_stocks_encoded = self.feature_proj(all_stocks_flat)
            all_stocks_encoded = all_stocks_encoded.view(b, n_stocks, s, self.d_model)
        
        # Shared layers (learn common patterns across all stocks)
        for layer in self.shared_layers:
            x = layer(x, all_stocks_encoded)
        
        # Stock-specific layers
        for layer in self.stock_specific_layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # Use last hidden state for predictions
        last_hidden = x[:, -1, :]
        
        # Predictions
        prices = self.price_head(last_hidden)
        directions = self.direction_head(last_hidden).view(batch_size, self.prediction_horizon, 3)
        volatility = self.volatility_head(last_hidden)
        correlations = self.correlation_head(last_hidden)
        
        return (
            prices,
            F.softmax(directions, dim=-1),
            F.relu(volatility),
            torch.tanh(correlations)  # Correlation range [-1, 1]
        )


class MultiStockDataset(Dataset):
    """Dataset for multiple stocks"""
    
    def __init__(
        self,
        stock_data: Dict[str, pd.DataFrame],
        stock_to_id: Dict[str, int],
        stock_to_sector: Dict[str, int],
        seq_len: int = 500,
        prediction_horizon: int = 5
    ):
        self.stock_data = stock_data
        self.stock_to_id = stock_to_id
        self.stock_to_sector = stock_to_sector
        self.seq_len = seq_len
        self.prediction_horizon = prediction_horizon
        
        # Create index of all valid samples
        self.samples = []
        for symbol, df in stock_data.items():
            n_samples = len(df) - seq_len - prediction_horizon
            for i in range(n_samples):
                self.samples.append({
                    'symbol': symbol,
                    'idx': i,
                    'stock_id': stock_to_id[symbol],
                    'sector_id': stock_to_sector[symbol]
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        symbol = sample['symbol']
        i = sample['idx']
        
        df = self.stock_data[symbol]
        
        # Extract features
        feature_cols = [col for col in df.columns if col not in ['Date', 'Close', 'symbol']]
        features = df.iloc[i:i + self.seq_len][feature_cols].values
        
        # Extract targets
        target_prices = df.iloc[i + self.seq_len:i + self.seq_len + self.prediction_horizon]['Close'].values
        
        # Direction labels
        current_price = df.iloc[i + self.seq_len - 1]['Close']
        directions = []
        for price in target_prices:
            change = (price - current_price) / current_price
            if change < -0.001:
                directions.append(0)
            elif change > 0.001:
                directions.append(2)
            else:
                directions.append(1)
        
        return {
            'features': torch.FloatTensor(features),
            'target_prices': torch.FloatTensor(target_prices),
            'target_directions': torch.LongTensor(directions),
            'stock_id': torch.LongTensor([sample['stock_id']]),
            'sector_id': torch.LongTensor([sample['sector_id']]),
            'symbol': symbol
        }


class MultiStockDataLoader:
    """Load and prepare data for multiple stocks"""
    
    # Sector mapping
    SECTORS = {
        'Technology': 0, 'Financial': 1, 'Healthcare': 2, 'Consumer': 3,
        'Energy': 4, 'Industrial': 5, 'Materials': 6, 'Utilities': 7,
        'Real Estate': 8, 'Communication': 9, 'Other': 10
    }
    
    @staticmethod
    def download_stocks(
        symbols: List[str],
        start_date: str = '2020-01-01',
        end_date: str = '2024-12-31'
    ) -> Dict[str, pd.DataFrame]:
        """Download stock data from Yahoo Finance"""
        stock_data = {}
        
        print(f"Downloading data for {len(symbols)} stocks...")
        for symbol in tqdm(symbols):
            try:
                df = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if len(df) > 500:  # Ensure sufficient data
                    df = df.reset_index()
                    df['symbol'] = symbol
                    stock_data[symbol] = df
            except Exception as e:
                print(f"Failed to download {symbol}: {e}")
        
        print(f"Successfully downloaded {len(stock_data)} stocks")
        return stock_data
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe"""
        df = df.copy()
        
        # Basic returns
        df['returns'] = df['Close'].pct_change()
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        sma20 = df['Close'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        df['BB_upper'] = sma20 + (std20 * 2)
        df['BB_lower'] = sma20 - (std20 * 2)
        
        # Volatility
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Fill NaN
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    @staticmethod
    def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using rolling statistics"""
        df = df.copy()
        
        feature_cols = [col for col in df.columns 
                       if col not in ['Date', 'Close', 'symbol']]
        
        window = 252  # 1 year
        for col in feature_cols:
            rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
            rolling_std = df[col].rolling(window=window, min_periods=1).std()
            df[col] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
        
        return df
    
    @staticmethod
    def prepare_multistock_data(
        symbols: List[str],
        sectors: Dict[str, str],
        start_date: str = '2020-01-01',
        end_date: str = '2024-12-31'
    ) -> Tuple[Dict, Dict, Dict]:
        """Complete data preparation pipeline"""
        
        # Download data
        stock_data = MultiStockDataLoader.download_stocks(symbols, start_date, end_date)
        
        # Process each stock
        processed_data = {}
        for symbol, df in stock_data.items():
            df = MultiStockDataLoader.add_technical_indicators(df)
            df = MultiStockDataLoader.normalize_features(df)
            processed_data[symbol] = df
        
        # Create mappings
        stock_to_id = {symbol: i for i, symbol in enumerate(stock_data.keys())}
        stock_to_sector = {
            symbol: MultiStockDataLoader.SECTORS.get(sectors.get(symbol, 'Other'), 10)
            for symbol in stock_data.keys()
        }
        
        return processed_data, stock_to_id, stock_to_sector


def train_multistock_example():
    """Example of multi-stock training"""
    
    # Define portfolio of stocks
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'BAC', 'XOM', 'CVX', 'JNJ']
    
    # Sector mapping
    sectors = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
        'AMZN': 'Consumer', 'TSLA': 'Consumer',
        'JPM': 'Financial', 'BAC': 'Financial',
        'XOM': 'Energy', 'CVX': 'Energy',
        'JNJ': 'Healthcare'
    }
    
    print("Multi-Stock Training Pipeline")
    print("=" * 60)
    print(f"Stocks: {', '.join(symbols)}")
    
    # Prepare data
    print("\nPreparing data...")
    stock_data, stock_to_id, stock_to_sector = MultiStockDataLoader.prepare_multistock_data(
        symbols, sectors
    )
    
    # Create dataset
    dataset = MultiStockDataset(
        stock_data, 
        stock_to_id, 
        stock_to_sector,
        seq_len=500,
        prediction_horizon=5
    )
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Get feature dimension from first sample
    sample = dataset[0]
    n_features = sample['features'].shape[1]
    
    # Initialize model
    model = MultiStockPredictionModel(
        n_stocks=len(stock_to_id),
        n_features=n_features,
        d_model=256,
        n_heads=8,
        d_ff=1024,
        n_layers=6,
        prediction_horizon=5
    )
    
    print(f"\nModel initialized:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Stocks: {len(stock_to_id)}")
    print(f"Features: {n_features}")
    
    return model, train_loader, val_loader, stock_to_id


if __name__ == "__main__":
    model, train_loader, val_loader, stock_to_id = train_multistock_example()
    print("\n✓ Multi-stock training system ready!")
