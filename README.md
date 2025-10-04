# Stock Price Prediction with Transformer Architecture

> A deep learning framework comparing Transformer-based models with traditional RNNs for financial time series forecasting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Quick Start

```bash
git clone https://github.com/suphyusinhtet/stock-prediction-transformer.git
cd stock-prediction-transformer
pip install -r requirements.txt
python train_all_models.py --stock AAPL
```

## Project Overview

This research introduces **SwiftFormer**, a lightweight Transformer encoder designed for stock price forecasting, and compares it against LSTM, GRU, CNN-LSTM, and XGBoost using 20 years of data from five tech giants.

**Key Finding:** GRU consistently outperformed all models including the Transformer, achieving the lowest prediction errors across all stocks.

### Dataset
- **Companies:** Amazon, Apple, NVIDIA, Microsoft, Google
- **Period:** July 2005 - July 2025 (20 years)
- **Source:** Yahoo Finance API
- **Data Points:** 5,030 daily observations per stock

## Results Summary

**Best Model: GRU**

| Stock | GRU MAE | SwiftFormer MAE | LSTM MAE |
|-------|---------|-----------------|----------|
| AMZN  | **0.037** | 0.095 | 0.061 |
| AAPL  | **0.051** | 0.118 | 0.065 |
| NVDA  | **0.592** | 0.996 | 1.293 |
| MSFT  | **0.064** | 0.134 | 0.135 |
| GOOGL | **0.054** | 0.128 | 0.098 |

*Lower is better. Full results in [results/metrics](results/metrics)*

## Installation

**Requirements:**
- Python 3.8+
- PyTorch 1.10+
- CUDA (optional, for GPU acceleration)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Train Individual Model

```python
from models.swiftformer import SwiftFormer
from utils.data_preprocessing import load_and_preprocess

# Load data
train_data, test_data = load_and_preprocess('AAPL')

# Train SwiftFormer
model = SwiftFormer(d_model=128, n_heads=8, n_layers=4)
model.train(train_data, epochs=100)

# Predict
predictions = model.predict(test_data)
```

### Compare All Models

```bash
# Single stock
python train_all_models.py --stock AAPL --sequence_length 30

# All stocks
python train_all_models.py --all_stocks
```

### Regime-Aware Evaluation

```python
from utils.regime_detection import detect_regimes
from utils.evaluation import regime_aware_metrics

regimes = detect_regimes(stock_data)
metrics = regime_aware_metrics(predictions, actual, regimes)
```

## Model Architectures

### SwiftFormer (Proposed)
```
Input → Linear Projection → Positional Encoding 
  → 4x Transformer Encoder Layers → Output Linear
  
Parameters: 793,473
Embedding Dim: 128
Attention Heads: 8
Feed-forward Dim: 512
```

### Baseline Models
- **LSTM:** 64 units (17,217 params)
- **GRU:** 64 units (12,929 params)
- **CNN-LSTM:** 1D CNN + 64 LSTM units (21,121 params)
- **XGBoost:** 300 trees, depth 6

## Project Structure

```
├── data/                   # Stock price datasets
├── models/                 # Model implementations
│   ├── swiftformer.py
│   ├── lstm.py
│   ├── gru.py
│   ├── cnn_lstm.py
│   └── xgboost_model.py
├── utils/                  # Helper functions
│   ├── data_preprocessing.py
│   ├── evaluation.py
│   └── regime_detection.py
├── notebooks/              # Jupyter notebooks
├── results/                # Outputs and visualizations
└── train_all_models.py     # Main training script
```

## Methodology

**Data Preprocessing:**
- MinMax normalization to [-1, 1]
- 30-day sliding window sequences
- 90-10 train-test split

**Training:**
- Optimizer: AdamW (lr=0.001)
- Loss: MSE
- Scheduler: OneCycleLR
- Early stopping: 5 epochs patience

**Evaluation:**
- Metrics: MAE, RMSE, MSE
- Regime-aware analysis (stable/normal/volatile)
- Multi-step forecasting (1-10 steps)

## Key Findings

1. **GRU outperforms Transformer** - Simpler recurrent models better capture short-term dependencies in volatile financial data

2. **Regime matters** - Model performance varies significantly across market volatility conditions

3. **XGBoost struggles** - Tree-based methods poorly suited for temporal sequences without extensive feature engineering

4. **Multi-step degrades** - SwiftFormer accuracy drops 21.6% from 1-step to 10-step predictions

## Limitations

- Univariate analysis only (closing prices)
- No transaction costs considered
- Limited to tech sector stocks
- 30-day window may miss longer cycles

## Future Work

- [ ] Multivariate inputs (volume, sentiment, indicators)
- [ ] Ensemble methods (GRU + SwiftFormer)
- [ ] Real-time trading integration
- [ ] Cross-sector validation
- [ ] Attention mechanism interpretability

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

Data provided by Yahoo Finance API via `yfinance` library.

## Disclaimer

**For research and educational purposes only.** Not financial advice. Past performance doesn't guarantee future results. Consult professionals before trading.

## Contact

suphyusinhtet@gmail.com

Project Link: [https://github.com/suphyusinhtet/stockprice_prediction](https://github.com/suphyusinhtet/stockprice_prediction)
