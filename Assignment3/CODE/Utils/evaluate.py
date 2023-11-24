import numpy as np


def calculate_sharpe_ratio(returns, risk_free_rate=0.005):
    """
    Calculate the Sharpe Ratio for a series of returns.
    :param returns: A list or array of return values.
    :param risk_free_rate: The risk-free rate of return.
    :return: The Sharpe Ratio.
    """
    returns = np.array(returns) if type(returns) == list else returns
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)


def calculate_max_drawdown(cumulative_returns):
    """
    Calculate the maximum drawdown for a series of cumulative returns.
    :param cumulative_returns: A list or array of cumulative return values.
    :return: The maximum drawdown.
    """
    peak = cumulative_returns[0]
    max_drawdown = 0
    for x in cumulative_returns:
        if x > peak:
            peak = x
        drawdown = (peak - x) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown


def calculate_annualized_return(cumulative_returns, num_days):
    """
    Calculate the annualized return from cumulative returns.
    :param cumulative_returns: A list or array of cumulative return values.
    :param num_days: The number of days over which the returns are calculated.
    :return: The annualized return.
    """
    if np.any(cumulative_returns):
        overall_return = cumulative_returns[-1]
        return (1 + overall_return) ** (365 / num_days) - 1
    return 0
