import os
import matplotlib.pyplot as plt
from src.gbm_simulator import GBMSimulator
from src.risk_metrics import PortfolioRiskAnalyzer

def main():
    print("Initializing Quantitative Risk Engine...")

    # 1. Parameter Setup
    initial_prices = [150.0, 80.0, 120.0]
    expected_returns = [0.08, 0.10, 0.04]
    volatilities = [0.20, 0.25, 0.05]
    correlation_matrix = [
        [1.0, 0.6, -0.2],
        [0.6, 1.0, -0.1],
        [-0.2, -0.1, 1.0]
    ]
    initial_portfolio_value = 100000.0
    weights = [1/3, 1/3, 1/3]

    # 2. Simulation Execution
    print("Running 10,000 Monte Carlo simulations using SciPy Cholesky decomposition...")
    simulator = GBMSimulator(initial_prices, expected_returns, volatilities, correlation_matrix)
    simulated_prices = simulator.simulate()

    # 3. Calculate Risk Metrics
    print("Calculating Value at Risk (VaR) and Conditional VaR (CVaR)...")
    analyzer = PortfolioRiskAnalyzer(simulated_prices, initial_prices, initial_portfolio_value, weights)
    
    var_95 = analyzer.calculate_var(95)
    cvar_95 = analyzer.calculate_cvar(95)

    print("\n=========================================")
    print("      MONTE CARLO RISK ENGINE RESULTS    ")
    print("=========================================")
    print(f"95% Value at Risk (VaR):       {abs(var_95)*100:.2f}% (${abs(var_95)*initial_portfolio_value:,.2f})")
    print(f"95% Conditional VaR (CVaR):    {abs(cvar_95)*100:.2f}% (${abs(cvar_95)*initial_portfolio_value:,.2f})")
    print("=========================================\n")

    # 4. Generate Visualizations
    print("Generating visualizations...")
    os.makedirs("output", exist_ok=True)

    plt.style.use('ggplot')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot A: Spaghetti Plot
    for i in range(100):
        axes[0].plot(analyzer.portfolio_values_over_time[i, :], lw=1, alpha=0.5)
    axes[0].axhline(y=initial_portfolio_value, color='black', linestyle='--', label='Initial Investment')
    axes[0].set_title("Monte Carlo Simulation: Portfolio Paths (First 100)")
    axes[0].set_xlabel("Trading Days")
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].legend()

    # Plot B: Returns Histogram
    axes[1].hist(analyzer.portfolio_returns, bins=50, alpha=0.75, color='#4A90E2', edgecolor='black')
    axes[1].axvline(var_95, color='red', linestyle='dashed', linewidth=2, label=f'95% VaR: {var_95*100:.2f}%')
    axes[1].axvline(cvar_95, color='darkred', linestyle='dotted', linewidth=2, label=f'95% CVaR: {cvar_95*100:.2f}%')
    axes[1].set_title("Distribution of Final Portfolio Returns")
    axes[1].set_xlabel("Return")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("output/risk_dashboard.png", dpi=300)
    print("Visualizations successfully saved to 'output/risk_dashboard.png'.")

if __name__ == "__main__":
    main()