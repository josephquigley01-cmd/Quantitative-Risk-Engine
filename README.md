# Quantitative Risk Engine: Monte Carlo Portfolio Simulation

This repository contains a high-performance Python risk engine that evaluates the financial risk of a multi-asset portfolio using Monte Carlo simulations. The project maps correlated financial assets into a stochastic process to calculate institutional risk metrics, including **Value at Risk (VaR)** and **Conditional Value at Risk (CVaR)**.

## Mathematical Foundation
The simulation models asset paths using **Geometric Brownian Motion (GBM)**. The continuous-time stochastic differential equation (SDE) is defined as:

$$ dS_t = \mu S_t dt + \sigma S_t dW_t $$

Where:
* $S_t$ = Asset price at time $t$
* $\mu$ = Expected drift (annualized return)
* $\sigma$ = Asset volatility
* $dW_t$ = A Wiener process (Standard Brownian Motion)

### Modeling Correlated Assets (SciPy Matrix Decompositions)
In real-world markets, asset movements are not independent. To accurately simulate this, the engine relies on the **SciPy** library (`scipy.linalg.cholesky`) to perform a **Cholesky Decomposition** on the portfolio's correlation matrix $C$. 

By calculating the lower triangular matrix $L$ (where $C = L L^T$), we successfully transform uncorrelated standard normal random shocks into correlated shocks, thereby mapping real-world macro-market relationships into the stochastic diffusion component.

## Software Architecture
To prioritize execution speed and scalability, the simulation circumvents the need for slower, lower-level languages (like C++) by utilizing **NumPy's vectorized operations**. The Monte Carlo engine runs 10,000 parallel portfolio simulations spanning 252 trading steps in fractions of a second.

* `src/gbm_simulator.py`: Object-Oriented simulation engine housing the linear algebra routines and SDE calculations.
* `src/risk_metrics.py`: Isolated logic for calculating statistical percentiles and expected shortfalls.
* `main.py`: Entry point for configuring the mock portfolio and generating `matplotlib` visualizations.

## Getting Started

**1. Clone the repository.**
```bash
git clone [https://github.com/josephquigley01-cmd/Quantitative-Risk-Engine.git](https://github.com/josephquigley01-cmd/Quantitative-Risk-Engine.git)
cd Time-Series-Anomaly-Detection-for-Predictive-Maintenance
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run main.py**
```bash
python main.py
   
