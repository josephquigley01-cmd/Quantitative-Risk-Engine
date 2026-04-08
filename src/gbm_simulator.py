import numpy as np
import scipy.linalg as la

class GBMSimulator:
    """
    Simulates Geometric Brownian Motion (GBM) for a multi-asset portfolio
    using Cholesky decomposition to map correlated real-world assets.
    """
    def __init__(self, initial_prices, expected_returns, volatilities, correlation_matrix, 
                 num_simulations=10000, time_horizon=1.0, num_steps=252, seed=42):
        self.initial_prices = np.array(initial_prices)
        self.expected_returns = np.array(expected_returns)
        self.volatilities = np.array(volatilities)
        self.correlation_matrix = np.array(correlation_matrix)
        self.num_simulations = num_simulations
        self.time_horizon = time_horizon
        self.num_steps = num_steps
        self.dt = time_horizon / num_steps
        self.num_assets = len(initial_prices)
        self.seed = seed

    def simulate(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            
        # Utilize SciPy Cholesky decomposition for mathematical heavy lifting
        # This maps real-world correlations into our stochastic process
        L = la.cholesky(self.correlation_matrix, lower=True)
        
        # Initialize 3D array: (simulations x steps x assets)
        prices = np.zeros((self.num_simulations, self.num_steps + 1, self.num_assets))
        prices[:, 0, :] = self.initial_prices
        
        # Pre-calculate deterministic drift
        drift = (self.expected_returns - 0.5 * self.volatilities**2) * self.dt
        
        for t in range(1, self.num_steps + 1):
            # Generate uncorrelated standard normal random shocks
            Z_uncorrelated = np.random.standard_normal((self.num_simulations, self.num_assets))
            
            # Induce correlation using the lower triangular matrix
            Z_correlated = Z_uncorrelated @ L.T
            
            # Calculate stochastic diffusion
            diffusion = self.volatilities * np.sqrt(self.dt) * Z_correlated
            
            # Update discrete prices: S(t+dt) = S(t) * exp(drift + diffusion)
            prices[:, t, :] = prices[:, t-1, :] * np.exp(drift + diffusion)
            
        return prices