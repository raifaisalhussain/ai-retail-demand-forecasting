# AI-Enabled Demand Forecasting for Efficient Inventory

MSc Computer Science Dissertation — University of Lincoln, 2025  
Supervisor: Dr. Khaled Bachour | Grade: Distinction

---

## Overview

This project compares three forecasting models — LSTM, ARIMA, and Decision Tree Regression — for predicting retail sales demand using the Walmart Store Sales dataset. Rather than stopping at model comparison, the project operationalises results into an interactive decision-support dashboard built with Flask and Dash, bridging the gap between academic research and practical retail inventory management.

---

## Key Results

| Model         | MAE    | RMSE   | Best For                                |
|---------------|--------|--------|-----------------------------------------|
| Decision Tree | 0.0497 | 0.0687 | Operational accuracy + interpretability |
| LSTM          | 0.0858 | 0.1125 | Non-linear seasonality + promo spikes   |
| ARIMA         | 0.1552 | 0.2041 | Long-term trend forecasting             |

- Decision Tree achieved the lowest error rates, outperforming both LSTM and ARIMA on the Walmart test set
- Promotional analysis revealed that marketing uplift is strongest in November–December, with promotions nearly doubling average sales volumes during peak holiday periods
- Feature importance analysis identified `product_id`, `month`, and `promotion` as the dominant demand drivers

---

## Features

- **Three-model comparative framework** — ARIMA (statistical), Decision Tree (ML), LSTM (deep learning) evaluated on the same dataset and metrics
- **Interactive Dash dashboard** with five tabs: Actual vs Predicted, Promotional Impact, Feature Importance, Model Comparison, and Results Summary
- **Promotional impact analysis** quantifying marketing uplift overall and broken down by month
- **End-to-end pipeline** from raw CSV ingestion through preprocessing, model training, evaluation, and live dashboard deployment via ngrok

---

## Tech Stack

| Layer         | Technology                          |
|---------------|-------------------------------------|
| Language      | Python 3.11                         |
| Deep Learning | TensorFlow 2.19 / Keras             |
| ML & Stats    | scikit-learn, statsmodels, pmdarima |
| Dashboard     | Plotly, Dash, Flask                 |
| Data          | pandas, NumPy                       |
| Deployment    | Google Colab, ngrok                 |

---

## Dataset

**Walmart Store Sales Forecasting** — publicly available on Kaggle.  
Download from: [kaggle.com/c/walmart-recruiting-store-sales-forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)  
Once downloaded, place `Walmart.csv` in the root directory before running.

Key variables used:

| Variable      | Role            | Notes                           |
|---------------|-----------------|---------------------------------|
| `sales`       | Target variable | MinMax scaled to [0,1]          |
| `promotion`   | Exogenous input | Binary flag from `Holiday_Flag` |
| `month`       | Seasonality     | Engineered from `date`          |
| `day_of_week` | Weekly pattern  | Engineered from `date`          |
| `product_id`  | Store control   | Label-encoded from `Store`      |

---

## How to Run

### Option 1: Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `Forecasting_Dashboard.ipynb`
3. Download `Walmart.csv` from Kaggle and upload to `/content/Walmart.csv`
4. Add your ngrok token (see below)
5. Run all cells — the public dashboard URL will print in the console

### Option 2: Local Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python forecasting_dashboard.py
```

The dashboard will be available via an ngrok public link printed in the terminal.

---

## ngrok Setup

To expose the dashboard publicly from Colab:

1. Get a free token at [ngrok.com](https://ngrok.com)
2. Insert it into `forecasting_dashboard.py`:
```python
NGROK_AUTH_TOKEN = "your_token_here"
```

---

## Project Structure
```
├── Forecasting_Dashboard.ipynb   # Main Colab notebook
├── forecasting_dashboard.py      # Standalone Python script
├── requirements.txt              # Python dependencies
├── .gitignore                    # Excludes dataset and cache files
└── README.md
```

---

## Dashboard Tabs

| Tab                 | Purpose                                                                |
|---------------------|------------------------------------------------------------------------|
| Actual vs Predicted | Visual comparison of each model's forecasts vs held-out actuals        |
| Promotional Impact  | Quantifies sales uplift from promotions, overall and by month          |
| Feature Importance  | Combined Decision Tree + LSTM surrogate feature importance scores      |
| Model Comparison    | MAE and RMSE bar charts across all three models                        |
| Results Summary     | Auto-generated narrative interpretation for non-technical stakeholders |

---

## Methodology

- **Data split:** 70% train / 15% validation / 15% test (chronological, no leakage)
- **ARIMA config:** (p=5, d=1, q=0) — linear trend baseline
- **Decision Tree:** `max_depth=5` — balances interpretability and accuracy
- **LSTM architecture:** 64 units → Dropout(0.2) → Dense(32) → output; Adam optimiser, early stopping (patience=5), 50 epochs, batch size 32
- **Evaluation metrics:** MAE and RMSE on held-out test set

---

## License

This project is open-source under the MIT License. The Walmart dataset is subject to Kaggle's terms of use and is not redistributed in this repository.

---

## Author

**Faisal Hussain**  
MSc Computer Science (Distinction) — University of Lincoln  
[linkedin.com/in/raifaisalhussain](https://linkedin.com/in/raifaisalhussain) | [github.com/raifaisalhussain](https://github.com/raifaisalhussain)
