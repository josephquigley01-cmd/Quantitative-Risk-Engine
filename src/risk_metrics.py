import numpy as np

class PortfolioRiskAnalyzer:
    """
    Calculates institutional risk metrics based on simulated portfolio paths.
    """
    def __init__(self, simulated_prices, initial_prices, initial_portfolio_value, weights):
        self.simulated_prices = simulated_prices
        self.initial_prices = np.array(initial_prices)
        self.initial_portfolio_value = initial_portfolio_value
        self.weights = np.array(weights)
        
        # Calculate shares held per asset
        self.shares_held = (self.initial_portfolio_value * self.weights) / self.initial_prices
        
        # Calculate aggregate portfolio value at each time step
        self.portfolio_values_over_time = np.sum(self.simulated_prices * self.shares_held, axis=2)
        
        # Calculate final returns
        self.final_portfolio_values = self.portfolio_values_over_time[:, -1]
        self.portfolio_returns = (self.final_portfolio_values - self.initial_portfolio_value) / self.initial_portfolio_value

    def calculate_var(self, confidence_level=95):
        """Calculate Value at Risk (VaR) at a given confidence level."""
        percentile = 100 - confidence_level
        return np.percentile(self.portfolio_returns, percentile)

    def calculate_cvar(self, confidence_level=95):
        """Calculate Conditional Value at Risk (CVaR) - the average of losses exceeding VaR."""
        var = self.calculate_var(confidence_level)
        return np.mean(self.portfolio_returns[self.portfolio_returns <= var])