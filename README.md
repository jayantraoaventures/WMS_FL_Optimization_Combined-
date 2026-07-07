# WMS_FL_Optimization_Combined-
Federated Learning system for optimizing Warehouse Management System (WMS) performance across 3 warehouses (Pharma/Electronics, Grocery/Apparel, Automotive) using SSD Regressor, LSTM &amp; Random Forest. Aggregates local models via FedAvg to reduce scan lag, API delay &amp; DB latency without sharing raw data.
# WMS Performance Optimization via Federated Learning

Privacy-preserving optimization of Warehouse Management System (WMS) latency using **Federated Learning (FL)** across three independently modeled warehouses. Each warehouse trains its own local model on its own data — raw data never leaves the node. Only model parameters are shared with a central FL server, which aggregates them (FedAvg) and broadcasts an optimization signal back to every warehouse.

## 🏗️ Project Overview

| Warehouse | Domain | Model | Data Source |
|---|---|---|---|
| **WH-A** | Pharma + Electronics (Zones A & B) | SSD Multi-Scale Regressor (3 SVR branches + GB meta-learner) | `logistics_dataset.csv` |
| **WH-B** | Groceries + Apparel | LSTM (NumPy, single-layer, seq-len=12) | `logistics_dataset.csv` + `Food Report - Oct. 2022.xlsx` (241 items × 31 days) |
| **WH-C** | Automotive (all zones) | Random Forest Regressor (150 trees) | `logistics_dataset.csv` + `Consumables Report - Oct. 2022.xlsx` (142 SKUs) |

Each local model predicts warehouse latency (scan lag, API delay, DB latency) from operational features. The FL server then:
1. Aggregates client updates via weighted **FedAvg** (weight ∝ number of samples)
2. Selects the best-performing model by R²
3. Derives a global **optimization signal** (latency reduction targets + system config recommendations)
4. Broadcasts it back so each warehouse can simulate its post-optimization latency

## 📁 Repository Structure

```
├── Code/
│   ├── main_real.py              # Master pipeline runner (4-step orchestration)
│   ├── data_loader.py             # Dataset loading & WMS feature engineering
│   ├── warehouse_models.py        # SSDRegressor, NumpyLSTM, WMSRandomForest
│   ├── federated_learning.py      # FL server, FL client, FedAvg aggregation
│   └── visualizations.py          # All 9 output figures
├── Dataset/
│   ├── Logistics Dataset/
│   │   └── logistics_dataset.csv              # 3,204 rows, 23 logistics KPI columns
│   └── Inventory Dataset/
│       ├── Food Report - Oct. 2022.xlsx        # 241 food items, daily usage
│       └── Consumables Report - Oct. 2022.xlsx  # 142 SKUs, spike patterns
├── Results/                       # Generated figures (fig0–fig8)
└── Report.docx                    # Full project write-up
```

## ⚙️ Pipeline

**Step 1 — Load & Engineer Data**
Loads and feature-engineers the logistics, food, and consumables datasets per warehouse.

**Step 2 — Local Model Training** (no data leaves the node)
- WH-A → SSD Multi-Scale Regressor
- WH-B → LSTM (NumPy, hidden=32, truncated BPTT)
- WH-C → Random Forest (150 trees, max_depth=12)

**Step 3 — Federated Learning Aggregation**
- Each client uploads model parameters (never raw data)
- FL server runs FedAvg and picks the best model by R²
- Global optimization signal (scan lag / API delay / DB latency reduction %, queue size, timeout, pool size, cache TTL, batch window) is computed and broadcast back

**Step 4 — Visualization**
Generates 9 figures: executive dashboard, before/after Gantt charts, per-metric comparisons, latency time-series, model performance (R²/RMSE/MAE), FL improvement heatmap, latency distribution shifts, feature importances, and a dual radar chart.

## 🚀 Usage

```bash
cd Code
python main_real.py
```

Make sure the dataset files are in the same directory as the scripts (or update `DATA_PATHS` in `main_real.py`):
```
logistics_dataset.csv
Food Report - Oct. 2022.xlsx
Consumables Report - Oct. 2022.xlsx
```

All outputs (engineered CSVs + 9 figures) are saved to `wms_fl_output/`.

## 📊 Sample Outputs

- `fig0_dashboard.png` — Executive KPI dashboard
- `fig1_gantt_before_after.png` — Latency before vs after FL optimization
- `fig4_model_performance.png` — R², RMSE, MAE across the three models
- `fig5_fl_improvement.png` — FL improvement bars + reduction heatmap
- `fig8_radar_performance.png` — Model quality vs optimization impact

## 🔒 Privacy Design

This project follows the core FL principle: **data stays local, only model parameters travel**. Each warehouse independently trains on its own data; the central server never sees raw records — only serialized parameters, sample counts, and R² scores.

## 🛠️ Tech Stack

- Python (NumPy, Pandas)
- Custom SSD Regressor, NumPy LSTM, Random Forest implementations
- Matplotlib for visualization
- Federated Learning (FedAvg)

## 👤 Author

3rd Year BTech CSE, Delhi Technological University (DTU)

## 📄 License

Feel free to add a license of your choice (MIT is a common default for academic/portfolio projects).
