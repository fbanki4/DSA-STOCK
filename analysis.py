"""
Visualization and Analysis Tools
For multi-stock predictions and portfolio performance
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PortfolioVisualizer:
    """Comprehensive visualization for portfolio analysis"""
    
    @staticmethod
    def plot_training_history(history: Dict, save_path: str = None):
        """Plot training metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax = axes[0]
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
        ax.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Accuracy
        ax = axes[1]
        ax.plot(epochs, history['train_direction_acc'], label='Train Accuracy', linewidth=2)
        ax.plot(epochs, history['val_direction_acc'], label='Val Accuracy', linewidth=2)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random (50%)')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Direction Accuracy', fontsize=12)
        ax.set_title('Direction Prediction Accuracy', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_portfolio_performance(results: Dict, save_path: str = None):
        """Comprehensive portfolio performance visualization"""
        
        equity_curve = results['equity_curve']
        dates = pd.to_datetime(results['dates'])
        trades = results['trades']
        metrics = results['metrics']
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Equity Curve
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(dates, equity_curve, linewidth=2, color='#2E86AB', label='Portfolio Value')
        ax1.fill_between(dates, equity_curve, alpha=0.3, color='#2E86AB')
        ax1.axhline(y=equity_curve[0], color='r', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.set_title('Portfolio Equity Curve', fontsize=16, fontweight='bold', pad=10)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Returns Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        returns = np.diff(equity_curve) / equity_curve[:-1] * 100
        ax2.hist(returns, bins=50, alpha=0.7, color='#A23B72', edgecolor='black')
        ax2.axvline(x=np.mean(returns), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(returns):.3f}%')
        ax2.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Return (%)', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Drawdown
        ax3 = fig.add_subplot(gs[1, 1])
        equity = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        ax3.fill_between(dates, drawdown, 0, alpha=0.5, color='red')
        ax3.plot(dates, drawdown, color='darkred', linewidth=1)
        ax3.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Date', fontsize=11)
        ax3.set_ylabel('Drawdown (%)', fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # 4. Trade Activity
        ax4 = fig.add_subplot(gs[1, 2])
        trades_df = pd.DataFrame(trades)
        if len(trades_df) > 0:
            trades_df['date'] = pd.to_datetime(trades_df['date'])
            monthly_trades = trades_df.groupby(trades_df['date'].dt.to_period('M')).size()
            monthly_trades.plot(kind='bar', ax=ax4, color='#F18F01', alpha=0.7)
            ax4.set_title('Monthly Trade Activity', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Month', fontsize=11)
            ax4.set_ylabel('Number of Trades', fontsize=11)
            ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # 5. Rolling Sharpe Ratio
        ax5 = fig.add_subplot(gs[2, 0])
        window = 60  # 60 days
        rolling_sharpe = []
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            sharpe = (np.mean(window_returns) / np.std(window_returns)) * np.sqrt(252)
            rolling_sharpe.append(sharpe)
        
        if rolling_sharpe:
            ax5.plot(dates[window:], rolling_sharpe, linewidth=2, color='#6A994E')
            ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax5.axhline(y=1, color='orange', linestyle='--', alpha=0.5, label='Good (>1)')
            ax5.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='Excellent (>2)')
            ax5.set_title('Rolling 60-Day Sharpe Ratio', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Date', fontsize=11)
            ax5.set_ylabel('Sharpe Ratio', fontsize=11)
            ax5.legend(fontsize=9)
            ax5.grid(True, alpha=0.3)
        
        # 6. Stock Allocation Over Time
        ax6 = fig.add_subplot(gs[2, 1:])
        if len(trades_df) > 0:
            # Track positions over time
            position_history = {}
            current_positions = {}
            
            for _, trade in trades_df.iterrows():
                date = trade['date']
                symbol = trade['symbol']
                
                if trade['action'] == 'buy':
                    current_positions[symbol] = current_positions.get(symbol, 0) + trade['shares']
                else:
                    current_positions[symbol] = current_positions.get(symbol, 0) - trade['shares']
                
                position_history[date] = current_positions.copy()
            
            # Plot as stacked area
            if position_history:
                all_symbols = set()
                for positions in position_history.values():
                    all_symbols.update(positions.keys())
                
                dates_list = sorted(position_history.keys())
                for symbol in all_symbols:
                    values = [position_history[d].get(symbol, 0) for d in dates_list]
                    ax6.plot(dates_list, values, label=symbol, linewidth=2, marker='o', markersize=3)
                
                ax6.set_title('Stock Positions Over Time', fontsize=14, fontweight='bold')
                ax6.set_xlabel('Date', fontsize=11)
                ax6.set_ylabel('Shares', fontsize=11)
                ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
                ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'Portfolio Analysis - Total Return: {metrics["total_return"]:.2f}%', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def create_interactive_dashboard(results: Dict):
        """Create interactive Plotly dashboard"""
        
        equity_curve = results['equity_curve']
        dates = pd.to_datetime(results['dates'])
        trades = results['trades']
        metrics = results['metrics']
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Portfolio Equity Curve', 'Drawdown',
                'Returns Distribution', 'Cumulative Returns',
                'Trade Activity', 'Performance Metrics'
            ),
            specs=[
                [{"colspan": 2}, None],
                [{"type": "histogram"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "table"}]
            ],
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # 1. Equity Curve
        fig.add_trace(
            go.Scatter(
                x=dates, y=equity_curve, 
                mode='lines', name='Portfolio Value',
                line=dict(color='#2E86AB', width=2),
                fill='tozeroy', fillcolor='rgba(46, 134, 171, 0.2)'
            ),
            row=1, col=1
        )
        
        # 2. Drawdown
        equity = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        
        fig.add_trace(
            go.Scatter(
                x=dates, y=drawdown,
                mode='lines', name='Drawdown',
                line=dict(color='red', width=1),
                fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.2)'
            ),
            row=2, col=2
        )
        
        # 3. Returns Distribution
        returns = np.diff(equity_curve) / equity_curve[:-1] * 100
        
        fig.add_trace(
            go.Histogram(
                x=returns, name='Returns',
                marker_color='#A23B72', opacity=0.7,
                nbinsx=50
            ),
            row=2, col=1
        )
        
        # 4. Trade Activity
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            trades_df['date'] = pd.to_datetime(trades_df['date'])
            monthly_trades = trades_df.groupby(trades_df['date'].dt.to_period('M')).size()
            
            fig.add_trace(
                go.Bar(
                    x=monthly_trades.index.astype(str),
                    y=monthly_trades.values,
                    name='Trades',
                    marker_color='#F18F01'
                ),
                row=3, col=1
            )
        
        # 5. Performance Metrics Table
        metrics_data = [
            ['Total Return', f"{metrics['total_return']:.2f}%"],
            ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.2f}"],
            ['Max Drawdown', f"{metrics['max_drawdown']:.2f}%"],
            ['Total Trades', f"{metrics['num_trades']:,}"],
            ['Final Value', f"${metrics['final_value']:,.2f}"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Metric</b>', '<b>Value</b>'],
                    fill_color='#2E86AB',
                    font=dict(color='white', size=12),
                    align='left'
                ),
                cells=dict(
                    values=list(zip(*metrics_data)),
                    fill_color='lavender',
                    align='left',
                    font=dict(size=11)
                )
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="📊 Portfolio Performance Dashboard",
            title_font_size=20,
            showlegend=True,
            height=1000,
            hovermode='x unified'
        )
        
        fig.show()
    
    @staticmethod
    def plot_stock_predictions(
        model, 
        stock_data: pd.DataFrame, 
        features: np.ndarray,
        stock_id: int,
        sector_id: int,
        lookback: int = 500,
        device: str = 'cpu'
    ):
        """Visualize predictions for a single stock"""
        
        model.eval()
        
        predictions = []
        actual_prices = []
        dates = []
        
        with torch.no_grad():
            for i in range(lookback, len(features) - 5):
                # Get features
                feature_window = features[i - lookback:i]
                feature_tensor = torch.FloatTensor(feature_window).unsqueeze(0).to(device)
                stock_id_tensor = torch.LongTensor([stock_id]).to(device)
                sector_id_tensor = torch.LongTensor([sector_id]).to(device)
                
                # Predict
                prices, _, _, _ = model(feature_tensor, stock_id_tensor, sector_id_tensor)
                pred_price = prices[0, 0].cpu().numpy()
                
                predictions.append(pred_price)
                actual_prices.append(stock_data.iloc[i]['Close'])
                dates.append(stock_data.iloc[i]['Date'])
        
        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Predictions vs Actual
        ax = axes[0]
        ax.plot(dates, actual_prices, label='Actual', linewidth=2, alpha=0.7)
        ax.plot(dates, predictions, label='Predicted', linewidth=2, alpha=0.7)
        ax.set_title('Stock Price Predictions vs Actual', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Prediction Error
        ax = axes[1]
        errors = np.array(predictions) - np.array(actual_prices)
        percentage_errors = (errors / np.array(actual_prices)) * 100
        
        ax.plot(dates, percentage_errors, color='red', linewidth=1)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.fill_between(dates, percentage_errors, 0, alpha=0.3, color='red')
        ax.set_title('Prediction Error', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Error (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        mae = np.mean(np.abs(percentage_errors))
        ax.text(0.02, 0.98, f'MAE: {mae:.2f}%', 
               transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()


def generate_report(results: Dict, save_path: str = 'portfolio_report.txt'):
    """Generate text report of portfolio performance"""
    
    metrics = results['metrics']
    trades = results['trades']
    
    report = f"""
{'='*80}
                    PORTFOLIO PERFORMANCE REPORT
{'='*80}

SUMMARY METRICS
{'-'*80}
Total Return:                {metrics['total_return']:>10.2f}%
Sharpe Ratio:                {metrics['sharpe_ratio']:>10.2f}
Maximum Drawdown:            {metrics['max_drawdown']:>10.2f}%
Total Number of Trades:      {metrics['num_trades']:>10,}
Final Portfolio Value:       ${metrics['final_value']:>10,.2f}

TRADING ACTIVITY
{'-'*80}
"""
    
    if trades:
        trades_df = pd.DataFrame(trades)
        
        # Trade statistics
        buy_trades = trades_df[trades_df['action'] == 'buy']
        sell_trades = trades_df[trades_df['action'] == 'sell']
        
        report += f"""
Total Buy Orders:            {len(buy_trades):>10,}
Total Sell Orders:           {len(sell_trades):>10,}
Average Trade Size:          {trades_df['shares'].mean():>10,.0f} shares
        
Most Traded Stocks:
"""
        stock_trade_counts = trades_df['symbol'].value_counts().head(5)
        for symbol, count in stock_trade_counts.items():
            report += f"  {symbol:6s} {count:>5} trades\n"
    
    report += f"\n{'='*80}\n"
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    print(report)
    return report


if __name__ == "__main__":
    print("Visualization tools loaded successfully!")
    print("\nAvailable functions:")
    print("  - PortfolioVisualizer.plot_training_history()")
    print("  - PortfolioVisualizer.plot_portfolio_performance()")
    print("  - PortfolioVisualizer.create_interactive_dashboard()")
    print("  - PortfolioVisualizer.plot_stock_predictions()")
    print("  - generate_report()")
