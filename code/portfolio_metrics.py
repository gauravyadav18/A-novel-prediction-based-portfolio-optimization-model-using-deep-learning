# portfolio_metrics.py
import numpy as np
import pandas as pd

def compute_metrics(portfolio_returns, benchmark_returns, risk_free_rate=0.025):
    """
    Compute portfolio evaluation metrics, including Sharpe Ratio and others.
    
    Args:
        portfolio_returns (np.ndarray): Daily portfolio returns.
        benchmark_returns (np.ndarray): Daily benchmark returns (proxy for market returns).
        risk_free_rate (float): Annualized risk-free rate (default 3%).
    
    Returns:
        dict: Dictionary of metrics.
    """
    # Annualization factor (252 trading days per year)
    trading_days = 252
    
    # Daily risk-free rate
    daily_risk_free = (1 + risk_free_rate) ** (1 / trading_days) - 1
    
    # Portfolio and benchmark annualized returns
    portfolio_return = np.mean(portfolio_returns) * trading_days
    benchmark_return = np.mean(benchmark_returns) * trading_days
    
    # Excess return (already annualized in original code)
    excess_returns = portfolio_returns - benchmark_returns
    er = np.mean(excess_returns) * trading_days
    
    # Standard deviation (annualized)
    sd = np.std(portfolio_returns) * np.sqrt(trading_days)
    
    # Information ratio
    ir = er / sd if sd != 0 else 0
    
    # Total return
    tor = (np.prod(1 + portfolio_returns) - 1) * 100
    
    # Maximum drawdown
    cum_returns = np.cumprod(1 + portfolio_returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdowns = (peak - cum_returns) / peak
    md = np.max(drawdowns) * 100
    
    # Turnover rate (simplified, assuming daily rebalancing)
    tur = np.mean(np.abs(np.diff(portfolio_returns))) * trading_days * 100
    
    # Sharpe Ratio
    excess_return = portfolio_return - risk_free_rate
    sharpe = excess_return / sd if sd != 0 else 0
    
    # Sortino Ratio (downside risk)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_sd = np.std(downside_returns) * np.sqrt(trading_days) if len(downside_returns) > 0 else 0
    sortino = excess_return / downside_sd if downside_sd != 0 else 0
    
    # Beta
    cov_matrix = np.cov(portfolio_returns, benchmark_returns)
    covariance = cov_matrix[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
    
    # Treynor Ratio
    treynor = excess_return / beta if beta != 0 else 0
    
    # Alpha
    alpha = portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
    
    # Profit Factor (treating daily returns as trades)
    profits = portfolio_returns[portfolio_returns > 0]
    losses = -portfolio_returns[portfolio_returns < 0]  # Convert to positive losses
    gross_profit = np.sum(profits) if len(profits) > 0 else 0
    gross_loss = np.sum(losses) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    return {
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        # 'Treynor Ratio': treynor,
        'Alpha (%)': alpha * 100,
         'Beta': beta,
        'Excess Return (%)': er * 100,
        'Standard Deviation': sd,
        # 'Information Ratio': ir,
        'Total Return (%)': tor,
        'Maximum Drawdown (%)': md,
        # 'Turnover Rate (%)': tur,
        # 'Profit Factor': profit_factor
    }