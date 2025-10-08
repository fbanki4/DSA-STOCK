"""
Complete End-to-End Pipeline
Multi-stock training + Backtesting + Portfolio Analysis
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class MultiStockTrainer:
    """Trainer for multi-stock model with advanced features"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer with different learning rates for different parts
        param_groups = [
            {'params': model.stock_embedding.parameters(), 'lr': 5e-5},
            {'params': model.shared_layers.parameters(), 'lr': 1e-4},
            {'params': model.stock_specific_layers.parameters(), 'lr': 1e-4},
            {'params': [p for n, p in model.named_parameters() 
                       if 'head' in n], 'lr': 2e-4}
        ]
        
        self.optimizer = optim.AdamW(param_groups, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_direction_acc': [], 'val_direction_acc': []
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            features = batch['features'].to(self.device)
            target_prices = batch['target_prices'].to(self.device)
            target_directions = batch['target_directions'].to(self.device)
            stock_ids = batch['stock_id'].squeeze(-1).to(self.device)
            sector_ids = batch['sector_id'].squeeze(-1).to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            pred_prices, pred_directions, pred_volatility, _ = self.model(
                features, stock_ids, sector_ids
            )
            
            # Multi-task loss
            price_loss = nn.MSELoss()(pred_prices, target_prices)
            
            direction_loss = nn.CrossEntropyLoss()(
                pred_directions.view(-1, 3),
                target_directions.view(-1)
            )
            
            # Volatility loss (optional, if we have volatility targets)
            # For now, we regularize it to be positive and reasonable
            vol_reg = torch.mean((pred_volatility - 0.02).pow(2))
            
            # Combined loss
            loss = price_loss + 0.5 * direction_loss + 0.01 * vol_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            pred_dirs = pred_directions.argmax(dim=-1)
            total_correct += (pred_dirs == target_directions).sum().item()
            total_samples += target_directions.numel()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{total_correct/total_samples:.3f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float, Dict]:
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        stock_performance = {}  # Track per-stock performance
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                features = batch['features'].to(self.device)
                target_prices = batch['target_prices'].to(self.device)
                target_directions = batch['target_directions'].to(self.device)
                stock_ids = batch['stock_id'].squeeze(-1).to(self.device)
                sector_ids = batch['sector_id'].squeeze(-1).to(self.device)
                
                pred_prices, pred_directions, _, _ = self.model(
                    features, stock_ids, sector_ids
                )
                
                # Losses
                price_loss = nn.MSELoss()(pred_prices, target_prices)
                direction_loss = nn.CrossEntropyLoss()(
                    pred_directions.view(-1, 3),
                    target_directions.view(-1)
                )
                
                loss = price_loss + 0.5 * direction_loss
                total_loss += loss.item()
                
                # Accuracy
                pred_dirs = pred_directions.argmax(dim=-1)
                correct = (pred_dirs == target_directions).float()
                total_correct += correct.sum().item()
                total_samples += target_directions.numel()
                
                # Per-stock tracking
                for i, symbol in enumerate(batch['symbol']):
                    if symbol not in stock_performance:
                        stock_performance[symbol] = {'correct': 0, 'total': 0}
                    
                    stock_performance[symbol]['correct'] += correct[i].sum().item()
                    stock_performance[symbol]['total'] += target_directions.shape[1]
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / total_samples
        
        # Calculate per-stock accuracy
        for symbol in stock_performance:
            perf = stock_performance[symbol]
            perf['accuracy'] = perf['correct'] / perf['total'] if perf['total'] > 0 else 0
        
        return avg_loss, accuracy, stock_performance
    
    def train(self, epochs: int, save_dir: str = './checkpoints'):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print('='*60)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_direction_acc'].append(train_acc)
            
            # Validate
            val_loss, val_acc, stock_perf = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_direction_acc'].append(val_acc)
            
            # Scheduler step
            self.scheduler.step()
            
            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2%}")
            
            # Show top/bottom performing stocks
            sorted_stocks = sorted(
                stock_perf.items(), 
                key=lambda x: x[1]['accuracy'], 
                reverse=True
            )
            
            print(f"\nTop 3 stocks:")
            for symbol, perf in sorted_stocks[:3]:
                print(f"  {symbol}: {perf['accuracy']:.2%}")
            
            print(f"\nBottom 3 stocks:")
            for symbol, perf in sorted_stocks[-3:]:
                print(f"  {symbol}: {perf['accuracy']:.2%}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'stock_performance': stock_perf
                }, f'{save_dir}/best_model.pt')
                print("\n✓ Saved best model")
        
        # Save history
        with open(f'{save_dir}/training_history.json', 'w') as f:
            json.dump(self.history, f)
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print('='*60)


class PortfolioBacktester:
    """Backtest on portfolio of stocks"""
    
    def __init__(
        self,
        model: nn.Module,
        stock_data: Dict[str, pd.DataFrame],
        stock_to_id: Dict[str, int],
        stock_to_sector: Dict[str, int],
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.model.eval()
        self.stock_data = stock_data
        self.stock_to_id = stock_to_id
        self.stock_to_sector = stock_to_sector
        self.device = device
    
    def backtest_portfolio(
        self,
        initial_capital: float = 100000.0,
        lookback: int = 500,
        rebalance_freq: int = 5  # Rebalance every 5 days
    ) -> Dict:
        """Backtest portfolio with periodic rebalancing"""
        
        # Portfolio state
        cash = initial_capital
        positions = {}  # {symbol: shares}
        equity_curve = [initial_capital]
        dates = []
        
        # Get common date range
        all_dates = set()
        for df in self.stock_data.values():
            all_dates.update(df['Date'].values)
        
        common_dates = sorted(all_dates)
        start_idx = lookback
        
        trades_log = []
        
        for i, date in enumerate(tqdm(common_dates[start_idx:], desc='Backtesting')):
            current_idx = start_idx + i
            
            # Get current prices
            current_prices = {}
            for symbol, df in self.stock_data.items():
                date_mask = df['Date'] == date
                if date_mask.any():
                    current_prices[symbol] = df[date_mask]['Close'].values[0]
            
            # Rebalancing logic
            if i % rebalance_freq == 0:
                # Generate predictions for all stocks
                predictions = {}
                
                for symbol in self.stock_data.keys():
                    if symbol not in current_prices:
                        continue
                    
                    df = self.stock_data[symbol]
                    date_idx = df[df['Date'] == date].index[0]
                    
                    if date_idx < lookback:
                        continue
                    
                    # Get features
                    feature_cols = [col for col in df.columns 
                                   if col not in ['Date', 'Close', 'symbol']]
                    features = df.iloc[date_idx - lookback:date_idx][feature_cols].values
                    
                    # Predict
                    with torch.no_grad():
                        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                        stock_id = torch.LongTensor([self.stock_to_id[symbol]]).to(self.device)
                        sector_id = torch.LongTensor([self.stock_to_sector[symbol]]).to(self.device)
                        
                        prices, directions, volatility, _ = self.model(
                            features_tensor, stock_id, sector_id
                        )
                        
                        # Get next-step prediction
                        pred_direction = directions[0, 0].cpu().numpy()  # [down, neutral, up]
                        pred_vol = volatility[0, 0].cpu().item()
                        
                        predictions[symbol] = {
                            'up_prob': pred_direction[2],
                            'down_prob': pred_direction[0],
                            'volatility': pred_vol,
                            'confidence': max(pred_direction[2], pred_direction[0])
                        }
                
                # Portfolio optimization: Select top N stocks
                n_positions = 5
                sorted_stocks = sorted(
                    predictions.items(),
                    key=lambda x: x[1]['up_prob'],
                    reverse=True
                )
                
                selected_stocks = [s[0] for s in sorted_stocks[:n_positions] 
                                  if s[1]['up_prob'] > 0.5]  # Only bullish stocks
                
                # Rebalance portfolio
                target_allocation = 1.0 / len(selected_stocks) if selected_stocks else 0
                portfolio_value = cash + sum(
                    positions.get(s, 0) * current_prices.get(s, 0)
                    for s in positions.keys()
                )
                
                # Sell positions not in selected stocks
                for symbol in list(positions.keys()):
                    if symbol not in selected_stocks:
                        shares = positions[symbol]
                        if symbol in current_prices:
                            cash += shares * current_prices[symbol] * 0.999  # 0.1% commission
                            trades_log.append({
                                'date': date,
                                'action': 'sell',
                                'symbol': symbol,
                                'shares': shares,
                                'price': current_prices[symbol]
                            })
                        del positions[symbol]
                
                # Buy/adjust positions in selected stocks
                for symbol in selected_stocks:
                    if symbol not in current_prices:
                        continue
                    
                    target_value = portfolio_value * target_allocation
                    target_shares = int(target_value / current_prices[symbol])
                    current_shares = positions.get(symbol, 0)
                    
                    shares_to_trade = target_shares - current_shares
                    
                    if shares_to_trade > 0:  # Buy
                        cost = shares_to_trade * current_prices[symbol] * 1.001
                        if cost <= cash:
                            cash -= cost
                            positions[symbol] = positions.get(symbol, 0) + shares_to_trade
                            trades_log.append({
                                'date': date,
                                'action': 'buy',
                                'symbol': symbol,
                                'shares': shares_to_trade,
                                'price': current_prices[symbol]
                            })
                    
                    elif shares_to_trade < 0:  # Sell
                        proceeds = abs(shares_to_trade) * current_prices[symbol] * 0.999
                        cash += proceeds
                        positions[symbol] += shares_to_trade
                        trades_log.append({
                            'date': date,
                            'action': 'sell',
                            'symbol': symbol,
                            'shares': abs(shares_to_trade),
                            'price': current_prices[symbol]
                        })
            
            # Update equity curve
            portfolio_value = cash + sum(
                positions.get(s, 0) * current_prices.get(s, 0)
                for s in positions.keys()
            )
            equity_curve.append(portfolio_value)
            dates.append(date)
        
        # Calculate metrics
        returns = np.diff(equity_curve) / equity_curve[:-1]
        total_return = (equity_curve[-1] - initial_capital) / initial_capital
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Max drawdown
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax
        max_drawdown = np.min(drawdown)
        
        return {
            'equity_curve': equity_curve,
            'dates': dates,
            'trades': trades_log,
            'metrics': {
                'total_return': total_return * 100,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown * 100,
                'num_trades': len(trades_log),
                'final_value': equity_curve[-1]
            }
        }


def main_pipeline():
    """Complete end-to-end pipeline"""
    
    print("="*80)
    print(" COMPLETE MULTI-STOCK PREDICTION & BACKTESTING PIPELINE")
    print("="*80)
    
    # 1. Setup
    from multistock_training import (
        MultiStockDataLoader, 
        MultiStockDataset, 
        MultiStockPredictionModel
    )
    
    # Define portfolio
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # Tech
        'JPM', 'BAC', 'WFC',  # Finance
        'XOM', 'CVX',  # Energy
        'JNJ', 'PFE',  # Healthcare
        'WMT', 'HD'   # Consumer
    ]
    
    sectors = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
        'AMZN': 'Consumer', 'TSLA': 'Technology',
        'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial',
        'XOM': 'Energy', 'CVX': 'Energy',
        'JNJ': 'Healthcare', 'PFE': 'Healthcare',
        'WMT': 'Consumer', 'HD': 'Consumer'
    }
    
    print(f"\n📊 Portfolio: {len(symbols)} stocks across {len(set(sectors.values()))} sectors")
    
    # 2. Data preparation
    print("\n" + "="*80)
    print("STEP 1: DATA PREPARATION")
    print("="*80)
    
    stock_data, stock_to_id, stock_to_sector = MultiStockDataLoader.prepare_multistock_data(
        symbols, sectors, start_date='2020-01-01', end_date='2024-12-31'
    )
    
    # 3. Create datasets
    print("\n" + "="*80)
    print("STEP 2: DATASET CREATION")
    print("="*80)
    
    dataset = MultiStockDataset(stock_data, stock_to_id, stock_to_sector, seq_len=500)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    
    # 4. Initialize model
    print("\n" + "="*80)
    print("STEP 3: MODEL INITIALIZATION")
    print("="*80)
    
    n_features = dataset[0]['features'].shape[1]
    
    model = MultiStockPredictionModel(
        n_stocks=len(stock_to_id),
        n_features=n_features,
        d_model=256,
        n_heads=8,
        d_ff=1024,
        n_layers=6,
        prediction_horizon=5
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # 5. Training
    print("\n" + "="*80)
    print("STEP 4: TRAINING")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    trainer = MultiStockTrainer(model, train_loader, val_loader, device)
    trainer.train(epochs=30, save_dir='./multistock_checkpoints')
    
    # 6. Backtesting
    print("\n" + "="*80)
    print("STEP 5: PORTFOLIO BACKTESTING")
    print("="*80)
    
    backtester = PortfolioBacktester(
        model, stock_data, stock_to_id, stock_to_sector, device
    )
    
    results = backtester.backtest_portfolio(
        initial_capital=100000,
        lookback=500,
        rebalance_freq=5
    )
    
    # 7. Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    metrics = results['metrics']
    print(f"\n📈 Portfolio Performance:")
    print(f"  Total Return:    {metrics['total_return']:>8.2f}%")
    print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:>8.2f}")
    print(f"  Max Drawdown:    {metrics['max_drawdown']:>8.2f}%")
    print(f"  Total Trades:    {metrics['num_trades']:>8,}")
    print(f"  Final Value:     ${metrics['final_value']:>,.2f}")
    
    print("\n✅ Pipeline complete!")
    
    return model, results


if __name__ == "__main__":
    model, results = main_pipeline()
