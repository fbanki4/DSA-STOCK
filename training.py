"""
Complete Training Pipeline for Stock Market Prediction
Includes data loading, preprocessing, training, and evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class StockDataset(Dataset):
    """Dataset for stock time series"""
    
    def __init__(
        self, 
        data: np.ndarray, 
        targets: np.ndarray,
        seq_len: int = 500,
        prediction_horizon: int = 5
    ):
        self.data = data
        self.targets = targets
        self.seq_len = seq_len
        self.prediction_horizon = prediction_horizon
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.prediction_horizon
    
    def __getitem__(self, idx):
        # Input sequence
        x = self.data[idx:idx + self.seq_len]
        
        # Target: future prices and directions
        future_prices = self.targets[idx + self.seq_len:idx + self.seq_len + self.prediction_horizon]
        
        # Direction labels (0: down, 1: neutral, 2: up)
        current_price = self.targets[idx + self.seq_len - 1]
        directions = []
        for price in future_prices:
            change = (price - current_price) / current_price
            if change < -0.001:
                directions.append(0)  # Down
            elif change > 0.001:
                directions.append(2)  # Up
            else:
                directions.append(1)  # Neutral
        
        return {
            'features': torch.FloatTensor(x),
            'target_prices': torch.FloatTensor(future_prices),
            'target_directions': torch.LongTensor(directions)
        }


class StockDataLoader:
    """Load and preprocess stock market data"""
    
    @staticmethod
    def load_csv(filepath: str) -> pd.DataFrame:
        """Load stock data from CSV"""
        df = pd.read_csv(filepath)
        
        # Ensure required columns
        required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required):
            raise ValueError(f"CSV must contain columns: {required}")
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df
    
    @staticmethod
    def create_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        df = df.copy()
        
        # Returns
        df['returns'] = df['Close'].pct_change()
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
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
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / sma20
        
        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['ATR'] = ranges.max(axis=1).rolling(14).mean()
        
        # Volume indicators
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        # Price momentum
        df['momentum_1'] = df['Close'] / df['Close'].shift(1) - 1
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    @staticmethod
    def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix and target prices"""
        
        # Feature columns (exclude Date and Close which is target)
        feature_cols = [col for col in df.columns if col not in ['Date', 'Close']]
        
        features = df[feature_cols].values
        targets = df['Close'].values
        
        return features, targets
    
    @staticmethod
    def normalize_data(features: np.ndarray, method: str = 'rolling') -> Tuple[np.ndarray, Dict]:
        """Normalize features"""
        
        if method == 'rolling':
            # Use rolling statistics (more realistic for trading)
            window = 252  # 1 year of trading days
            normalized = np.zeros_like(features)
            
            for i in range(len(features)):
                start_idx = max(0, i - window)
                window_data = features[start_idx:i+1]
                
                mean = np.mean(window_data, axis=0)
                std = np.std(window_data, axis=0) + 1e-8
                
                normalized[i] = (features[i] - mean) / std
            
            stats = {'method': 'rolling', 'window': window}
        
        else:  # global normalization
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0) + 1e-8
            normalized = (features - mean) / std
            stats = {'method': 'global', 'mean': mean.tolist(), 'std': std.tolist()}
        
        return normalized, stats


class Trainer:
    """Training manager for stock prediction model"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        lr: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            features = batch['features'].to(self.device)
            target_prices = batch['target_prices'].to(self.device)
            target_directions = batch['target_directions'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            pred_prices, pred_directions, pred_volatility = self.model(features)
            
            # Calculate losses
            price_loss = nn.MSELoss()(pred_prices, target_prices)
            
            direction_loss = nn.CrossEntropyLoss()(
                pred_directions.view(-1, 3),
                target_directions.view(-1)
            )
            
            # Combined loss
            loss = price_loss + 0.5 * direction_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> Tuple[float, Dict]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        metrics = {
            'price_mae': 0,
            'direction_acc': 0,
            'total_samples': 0
        }
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                features = batch['features'].to(self.device)
                target_prices = batch['target_prices'].to(self.device)
                target_directions = batch['target_directions'].to(self.device)
                
                pred_prices, pred_directions, _ = self.model(features)
                
                # Losses
                price_loss = nn.MSELoss()(pred_prices, target_prices)
                direction_loss = nn.CrossEntropyLoss()(
                    pred_directions.view(-1, 3),
                    target_directions.view(-1)
                )
                
                loss = price_loss + 0.5 * direction_loss
                total_loss += loss.item()
                
                # Metrics
                metrics['price_mae'] += nn.L1Loss()(pred_prices, target_prices).item()
                
                pred_dirs = pred_directions.argmax(dim=-1)
                correct = (pred_dirs == target_directions).float().sum().item()
                metrics['direction_acc'] += correct
                metrics['total_samples'] += target_directions.numel()
        
        avg_loss = total_loss / len(self.val_loader)
        metrics['price_mae'] /= len(self.val_loader)
        metrics['direction_acc'] /= metrics['total_samples']
        
        return avg_loss, metrics
    
    def train(self, epochs: int, save_dir: str = './checkpoints'):
        """Full training loop"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss, metrics = self.validate()
            self.history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Price MAE: {metrics['price_mae']:.4f}")
            print(f"Direction Acc: {metrics['direction_acc']:.2%}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                    'metrics': metrics
                }, f'{save_dir}/best_model.pt')
                print("✓ Saved best model")
        
        # Save training history
        with open(f'{save_dir}/history.json', 'w') as f:
            json.dump(self.history, f)


# Example usage
def main():
    """Main training pipeline"""
    
    print("Stock Market Prediction - Training Pipeline")
    print("=" * 60)
    
    # 1. Load data (example with synthetic data - replace with real CSV)
    # df = StockDataLoader.load_csv('stock_data.csv')
    
    # Generate synthetic data for demonstration
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    np.random.seed(42)
    
    price = 100
    prices = [price]
    for _ in range(len(dates) - 1):
        price *= (1 + np.random.randn() * 0.02)
        prices.append(price)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * (1 + abs(np.random.randn() * 0.01)) for p in prices],
        'Low': [p * (1 - abs(np.random.randn() * 0.01)) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    # 2. Create technical indicators
    df = StockDataLoader.create_technical_indicators(df)
    
    # 3. Prepare features
    features, targets = StockDataLoader.prepare_features(df)
    
    # 4. Normalize
    features, _ = StockDataLoader.normalize_data(features, method='rolling')
    
    # 5. Train/val split
    split_idx = int(len(features) * 0.8)
    
    train_features = features[:split_idx]
    train_targets = targets[:split_idx]
    val_features = features[split_idx:]
    val_targets = targets[split_idx:]
    
    # 6. Create datasets
    train_dataset = StockDataset(train_features, train_targets, seq_len=500, prediction_horizon=5)
    val_dataset = StockDataset(val_features, val_targets, seq_len=500, prediction_horizon=5)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"\nDataset sizes:")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Features: {features.shape[1]}")
    
    # 7. Initialize model (import from previous artifact)
    # Assuming StockPredictionModel is imported
    from stock_deepseek_inspired import StockPredictionModel
    
    model = StockPredictionModel(
        n_features=features.shape[1],
        d_model=256,
        n_heads=8,
        d_ff=1024,
        n_layers=6,
        n_experts=8,
        sparsity_ratio=0.3,
        max_seq_len=5000,
        prediction_horizon=5
    )
    
    # 8. Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=1e-4
    )
    
    trainer.train(epochs=50, save_dir='./checkpoints')
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
