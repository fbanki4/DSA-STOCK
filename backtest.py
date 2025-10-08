"""
Comprehensive Backtesting Framework for Stock Market Predictions
Includes walk-forward validation, performance metrics, and trade simulation
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


@dataclass
class Trade:
    """Individual trade record"""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    position: str  # 'long' or 'short'
    shares: int
    pnl: float
    return_pct: float
    holding_period: int


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 100000.0
    position_size: float = 0.1  # 10% of capital per trade
    commission: float = 0.001  # 0.1% commission
    slippage: float = 0.0005  # 0.05% slippage
    max_positions: int = 5  # Maximum concurrent positions
    stop_loss: float = 0.05  # 5% stop loss
    take_profit: float = 0.15  # 15% take profit
    min_prediction_confidence: float = 0.6  # Minimum directional confidence


class Portfolio:
    """Portfolio manager for backtesting"""
    
    def __init__(self, initial_capital: float, config: BacktestConfig):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.config = config
        
        self.positions: Dict[str, Dict] = {}  # Active positions
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        self.dates: List[str] = []
        
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(
            pos['shares'] * current_prices.get(symbol, pos['entry_price'])
            for symbol, pos in self.positions.items()
        )
        return self.cash + positions_value
    
    def can_open_position(self) -> bool:
        """Check if we can open a new position"""
        return len(self.positions) < self.config.max_positions
    
    def open_position(
        self, 
        symbol: str, 
        date: str, 
        price: float, 
        direction: str,
        confidence: float
    ) -> bool:
        """Open a new position"""
        if not self.can_open_position() or symbol in self.positions:
            return False
        
        # Calculate position size
        position_value = self.cash * self.config.position_size
        
        # Apply slippage
        execution_price = price * (1 + self.config.slippage if direction == 'long' else 1 - self.config.slippage)
        
        # Calculate shares
        shares = int(position_value / execution_price)
        
        if shares == 0:
            return False
        
        # Calculate total cost with commission
        cost = shares * execution_price
        commission = cost * self.config.commission
        total_cost = cost + commission
        
        if total_cost > self.cash:
            return False
        
        # Update cash
        self.cash -= total_cost
        
        # Create position
        self.positions[symbol] = {
            'entry_date': date,
            'entry_price': execution_price,
            'shares': shares,
            'direction': direction,
            'confidence': confidence,
            'stop_loss': execution_price * (1 - self.config.stop_loss) if direction == 'long' 
                        else execution_price * (1 + self.config.stop_loss),
            'take_profit': execution_price * (1 + self.config.take_profit) if direction == 'long'
                          else execution_price * (1 - self.config.take_profit)
        }
        
        return True
    
    def close_position(self, symbol: str, date: str, price: float, reason: str = 'signal') -> Optional[Trade]:
        """Close an existing position"""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        
        # Apply slippage
        execution_price = price * (1 - self.config.slippage if pos['direction'] == 'long' else 1 + self.config.slippage)
        
        # Calculate P&L
        if pos['direction'] == 'long':
            pnl_per_share = execution_price - pos['entry_price']
        else:
            pnl_per_share = pos['entry_price'] - execution_price
        
        gross_pnl = pnl_per_share * pos['shares']
        commission = execution_price * pos['shares'] * self.config.commission
        net_pnl = gross_pnl - commission
        
        # Update cash
        proceeds = execution_price * pos['shares'] - commission
        self.cash += proceeds
        
        # Calculate return
        cost_basis = pos['entry_price'] * pos['shares']
        return_pct = (net_pnl / cost_basis) * 100
        
        # Create trade record
        holding_period = (pd.to_datetime(date) - pd.to_datetime(pos['entry_date'])).days
        
        trade = Trade(
            entry_date=pos['entry_date'],
            exit_date=date,
            entry_price=pos['entry_price'],
            exit_price=execution_price,
            position=pos['direction'],
            shares=pos['shares'],
            pnl=net_pnl,
            return_pct=return_pct,
            holding_period=holding_period
        )
        
        self.trades.append(trade)
        del self.positions[symbol]
        
        return trade
    
    def check_stops(self, symbol: str, date: str, current_price: float) -> Optional[Trade]:
        """Check if stop loss or take profit triggered"""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        
        if pos['direction'] == 'long':
            if current_price <= pos['stop_loss']:
                return self.close_position(symbol, date, current_price, 'stop_loss')
            elif current_price >= pos['take_profit']:
                return self.close_position(symbol, date, current_price, 'take_profit')
        else:  # short
            if current_price >= pos['stop_loss']:
                return self.close_position(symbol, date, current_price, 'stop_loss')
            elif current_price <= pos['take_profit']:
                return self.close_position(symbol, date, current_price, 'take_profit')
        
        return None
    
    def update_equity_curve(self, date: str, current_prices: Dict[str, float]):
        """Update equity curve"""
        portfolio_value = self.get_portfolio_value(current_prices)
        self.equity_curve.append(portfolio_value)
        self.dates.append(date)


class Backtester:
    """Main backtesting engine"""
    
    def __init__(self, model: torch.nn.Module, config: BacktestConfig, device: str = 'cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device
        
    def generate_signals(
        self, 
        features: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate predictions from model"""
        with torch.no_grad():
            features = features.to(self.device)
            prices, directions, volatility = self.model(features.unsqueeze(0))
            
            # Get first step predictions
            pred_price = prices[0, 0].cpu().numpy()
            pred_direction = directions[0, 0].cpu().numpy()  # [down, neutral, up] probabilities
            pred_volatility = volatility[0, 0].cpu().numpy()
            
        return pred_price, pred_direction, pred_volatility
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        features: np.ndarray,
        symbol: str = 'STOCK',
        lookback: int = 500
    ) -> Dict:
        """Run backtest on historical data"""
        
        portfolio = Portfolio(self.config.initial_capital, self.config)
        signals_log = []
        
        # Walk-forward testing
        for i in range(lookback, len(data) - 1):
            current_date = data.iloc[i]['Date']
            current_price = data.iloc[i]['Close']
            next_price = data.iloc[i + 1]['Close']
            
            # Get features for prediction
            feature_window = features[i - lookback:i]
            feature_tensor = torch.FloatTensor(feature_window)
            
            # Generate signal
            pred_price, pred_direction, pred_volatility = self.generate_signals(feature_tensor)
            
            # Determine action
            up_prob = pred_direction[2]
            down_prob = pred_direction[0]
            confidence = max(up_prob, down_prob)
            
            # Check stops for existing positions
            portfolio.check_stops(symbol, str(current_date), current_price)
            
            # Trading logic
            if symbol in portfolio.positions:
                # Check if we should close
                pos = portfolio.positions[symbol]
                
                if pos['direction'] == 'long' and down_prob > self.config.min_prediction_confidence:
                    trade = portfolio.close_position(symbol, str(current_date), current_price, 'signal_reversal')
                    if trade:
                        signals_log.append({
                            'date': current_date,
                            'action': 'close_long',
                            'price': current_price,
                            'pnl': trade.pnl
                        })
                
                elif pos['direction'] == 'short' and up_prob > self.config.min_prediction_confidence:
                    trade = portfolio.close_position(symbol, str(current_date), current_price, 'signal_reversal')
                    if trade:
                        signals_log.append({
                            'date': current_date,
                            'action': 'close_short',
                            'price': current_price,
                            'pnl': trade.pnl
                        })
            
            else:
                # Open new position if signal is strong enough
                if up_prob > self.config.min_prediction_confidence:
                    if portfolio.open_position(symbol, str(current_date), current_price, 'long', up_prob):
                        signals_log.append({
                            'date': current_date,
                            'action': 'open_long',
                            'price': current_price,
                            'confidence': up_prob
                        })
                
                elif down_prob > self.config.min_prediction_confidence:
                    if portfolio.open_position(symbol, str(current_date), current_price, 'short', down_prob):
                        signals_log.append({
                            'date': current_date,
                            'action': 'open_short',
                            'price': current_price,
                            'confidence': down_prob
                        })
            
            # Update equity curve
            portfolio.update_equity_curve(str(current_date), {symbol: current_price})
        
        # Close all remaining positions
        final_date = str(data.iloc[-1]['Date'])
        final_price = data.iloc[-1]['Close']
        for symbol in list(portfolio.positions.keys()):
            portfolio.close_position(symbol, final_date, final_price, 'end_of_backtest')
        
        # Calculate performance metrics
        metrics = self.calculate_metrics(portfolio, data)
        
        return {
            'portfolio': portfolio,
            'metrics': metrics,
            'signals_log': signals_log
        }
    
    def calculate_metrics(self, portfolio: Portfolio, data: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        if not portfolio.trades:
            return {'error': 'No trades executed'}
        
        # Basic metrics
        total_trades = len(portfolio.trades)
        winning_trades = [t for t in portfolio.trades if t.pnl > 0]
        losing_trades = [t for t in portfolio.trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t.pnl for t in portfolio.trades)
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Returns analysis
        returns = np.diff(portfolio.equity_curve) / portfolio.equity_curve[:-1]
        
        total_return = (portfolio.equity_curve[-1] - portfolio.initial_capital) / portfolio.initial_capital
        
        # Sharpe ratio (annualized)
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Maximum drawdown
        equity_curve = np.array(portfolio.equity_curve)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        calmar_ratio = (total_return * 100) / abs(max_drawdown * 100) if max_drawdown != 0 else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0
        
        # Average holding period
        avg_holding_period = np.mean([t.holding_period for t in portfolio.trades])
        
        metrics = {
            'total_return_pct': total_return * 100,
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown * 100,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate * 100,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_period': avg_holding_period,
            'final_equity': portfolio.equity_curve[-1]
        }
        
        return metrics
    
    def plot_results(self, results: Dict, save_path: Optional[str] = None):
        """Plot backtest results"""
        portfolio = results['portfolio']
        metrics = results['metrics']
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        
        # 1. Equity curve
        ax = axes[0, 0]
        dates = pd.to_datetime(portfolio.dates)
        ax.plot(dates, portfolio.equity_curve, linewidth=2)
        ax.axhline(y=portfolio.initial_capital, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax = axes[0, 1]
        equity = np.array(portfolio.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        ax.fill_between(dates, drawdown, 0, alpha=0.3, color='red')
        ax.plot(dates, drawdown, color='darkred', linewidth=1)
        ax.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        
        # 3. Trade P&L distribution
        ax = axes[1, 0]
        pnls = [t.pnl for t in portfolio.trades]
        ax.hist(pnls, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.set_title('Trade P&L Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('P&L ($)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # 4. Cumulative P&L
        ax = axes[1, 1]
        cumulative_pnl = np.cumsum(pnls)
        ax.plot(cumulative_pnl, linewidth=2, color='green')
        ax.set_title('Cumulative P&L', fontsize=14, fontweight='bold')
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Cumulative P&L ($)')
        ax.grid(True, alpha=0.3)
        
        # 5. Win/Loss analysis
        ax = axes[2, 0]
        categories = ['Winning\nTrades', 'Losing\nTrades']
        values = [metrics['winning_trades'], metrics['losing_trades']]
        colors = ['green', 'red']
        ax.bar(categories, values, color=colors, alpha=0.7)
        ax.set_title('Win/Loss Distribution', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Trades')
        
        # 6. Performance metrics table
        ax = axes[2, 1]
        ax.axis('off')
        
        metrics_text = f"""
        PERFORMANCE METRICS
        {'='*40}
        Total Return:        {metrics['total_return_pct']:.2f}%
        Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}
        Max Drawdown:        {metrics['max_drawdown_pct']:.2f}%
        Calmar Ratio:        {metrics['calmar_ratio']:.2f}
        
        Win Rate:            {metrics['win_rate']:.2f}%
        Profit Factor:       {metrics['profit_factor']:.2f}
        Total Trades:        {metrics['total_trades']}
        
        Avg Win:             ${metrics['avg_win']:.2f}
        Avg Loss:            ${metrics['avg_loss']:.2f}
        Avg Hold Period:     {metrics['avg_holding_period']:.1f} days
        
        Final Equity:        ${metrics['final_equity']:,.2f}
        """
        
        ax.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


def run_backtest_example():
    """Example of running backtest"""
    
    # Load model (assuming it's already trained)
    from stock_deepseek_inspired import StockPredictionModel
    
    model = StockPredictionModel(
        n_features=15,
        d_model=256,
        n_heads=8,
        d_ff=1024,
        n_layers=6,
        n_experts=8,
        sparsity_ratio=0.3,
        max_seq_len=5000,
        prediction_horizon=5
    )
    
    # Load checkpoint if available
    # checkpoint = torch.load('checkpoints/best_model.pt')
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    # Backtesting configuration
    config = BacktestConfig(
        initial_capital=100000.0,
        position_size=0.2,  # 20% per position
        commission=0.001,
        slippage=0.0005,
        max_positions=3,
        stop_loss=0.05,
        take_profit=0.15,
        min_prediction_confidence=0.65
    )
    
    # Initialize backtester
    backtester = Backtester(model, config, device='cpu')
    
    print("Backtesting framework ready!")
    print(f"Initial Capital: ${config.initial_capital:,.2f}")
    print(f"Max Positions: {config.max_positions}")
    print(f"Position Size: {config.position_size*100:.1f}%")
    
    return backtester, config


if __name__ == "__main__":
    backtester, config = run_backtest_example()
    print("\n✓ Backtesting framework initialized successfully!")
