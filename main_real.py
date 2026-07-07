"""
================================================================================
main_real.py
WMS FL Optimization — Real Dataset Project  ·  Master Runner
================================================================================
Orchestrates the complete 4-step pipeline using real inventory & logistics data:

  STEP 1 — Load & engineer real warehouse datasets
            • Warehouse A : logistics_dataset.csv  (Pharma + Electronics, Zones A & B)
            • Warehouse B : logistics_dataset.csv  (Groceries + Apparel, Zones C & D)
                           + Food Report Oct-2022.xlsx  (241 items × 31-day daily usage)
            • Warehouse C : logistics_dataset.csv  (Automotive, all zones)
                           + Consumables Report Oct-2022.xlsx  (142 SKUs, spike patterns)

  STEP 2 — Train one local model per warehouse  (no data leaves the node)
            • Warehouse A → SSD Multi-Scale Regressor
            • Warehouse B → LSTM  (NumPy single-layer, seq-len=12)
            • Warehouse C → Random Forest Regressor  (150 trees)

  STEP 3 — Federated Learning aggregation
            • Each client uploads model params  (NOT raw data)
            • FL Server runs FedAvg, selects best model by R²
            • Global optimization signal broadcast back to all nodes
            • Latency reductions applied locally at each client

  STEP 4 — Generate all 9 output figures
            fig0  Executive dashboard  (KPIs, accuracy cards, radar)
            fig1  Gantt before / after
            fig2  Per-metric grouped bars
            fig3  Time-series latency (rolling avg)
            fig4  Model performance  (R², RMSE, MAE)
            fig5  FL improvement summary  (bars + heatmap)
            fig6  Latency distribution shift  (violin + box)
            fig7  Feature importance  (SSD scales | LSTM gates | RF Gini)
            fig8  Standalone dual radar

Usage
-----
  # From the project directory (same folder as all .py files):
  python main_real.py

Required files (place in same directory or adjust paths below)
--------------------------------------------------------------
  data_loader.py         — real dataset loader & WMS feature engineering
  warehouse_models.py    — SSDRegressor, NumpyLSTM, WMSRandomForest
  federated_learning.py  — FederatedLearningServer, FederatedClient, run_federated_learning
  visualizations.py      — generate_all_plots and all individual plot functions

Raw data paths (edit DATA_PATHS dict if your files are elsewhere)
-----------------------------------------------------------------
  logistics_dataset.csv              — 3,204 rows, 23 logistics KPI columns
  Food Report - Oct. 2022.xlsx       — 241 food items, Oct daily usage
  Consumables Report - Oct. 2022.xlsx — 142 consumable items, Oct daily usage
================================================================================
"""

import sys
import os
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# ── Make all sibling modules importable ───────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from data_loader        import load_all_warehouses
from warehouse_models   import SSDRegressor, NumpyLSTM, WMSRandomForest
from federated_learning import run_federated_learning
from visualizations     import generate_all_plots

# ── Output directory ──────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(THIS_DIR, 'wms_fl_output')

# ── Raw data paths (edit here if needed) ─────────────────────────────────
DATA_PATHS = {
    'logistics':    os.path.join(THIS_DIR, 'logistics_dataset.csv'),
    'food':         os.path.join(THIS_DIR, 'Food Report - Oct. 2022.xlsx'),
    'consumables':  os.path.join(THIS_DIR, 'Consumables Report - Oct. 2022.xlsx'),
}


# ════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  WMS Performance Optimization via Federated Learning")
    print("  REAL DATASET RUN")
    print("  Sources : logistics_dataset.csv | Food Report | Consumables Report")
    print("  Models  : WH-A = SSD Regressor  |  WH-B = LSTM  |  WH-C = Random Forest")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ────────────────────────────────────────────────────────────────────────
    #  STEP 1 — Load & engineer real datasets
    # ────────────────────────────────────────────────────────────────────────
    print("\n[1/4] Loading and engineering real warehouse datasets...")
    df_a, df_b, df_c = load_all_warehouses(
        logistics_path   = DATA_PATHS['logistics'],
        food_path        = DATA_PATHS['food'],
        consumables_path = DATA_PATHS['consumables'],
    )

    print(f"\n  Dataset summary after feature engineering:")
    print(f"  {'Warehouse':<40} {'Rows':>6}  {'Avg Latency':>12}  {'Peak Latency':>12}")
    print(f"  {'-'*74}")
    for label, df in [
        ('WH-A  Pharma + Electronics  (Zones A & B)', df_a),
        ('WH-B  Groceries + Apparel + Food TS',       df_b),
        ('WH-C  Automotive + Consumables',             df_c),
    ]:
        print(f"  {label:<40} {len(df):>6}  "
              f"{df['total_latency_ms'].mean():>11.1f}ms  "
              f"{df['total_latency_ms'].max():>11.0f}ms")

    # Save processed CSVs for reference
    df_a.to_csv(os.path.join(OUTPUT_DIR, 'warehouse_a_engineered.csv'), index=False)
    df_b.to_csv(os.path.join(OUTPUT_DIR, 'warehouse_b_engineered.csv'), index=False)
    df_c.to_csv(os.path.join(OUTPUT_DIR, 'warehouse_c_engineered.csv'), index=False)
    print(f"\n  Engineered CSVs saved to: {OUTPUT_DIR}/")

    # ────────────────────────────────────────────────────────────────────────
    #  STEP 2 — Train local models
    #  Each model trains on its warehouse data only (federated isolation)
    # ────────────────────────────────────────────────────────────────────────
    print("\n[2/4] Training local warehouse models...")

    print("\n  ┌─ Warehouse A — SSD Multi-Scale Regressor ─────────────────────")
    print("  │  Data: Pharma + Electronics items, Zones A & B")
    print("  │  Architecture: 3 SVR branches (coarse/medium/fine) + GB meta-learner")
    model_a = SSDRegressor().fit(df_a)

    print("\n  ┌─ Warehouse B — LSTM (NumPy single-layer) ──────────────────────")
    print("  │  Data: Groceries + Apparel + Food Report 31-day time-series")
    print("  │  Architecture: LSTM seq_len=12, hidden=32, truncated BPTT")
    model_b = NumpyLSTM(epochs=45).fit(df_b)

    print("\n  ┌─ Warehouse C — Random Forest Regressor ────────────────────────")
    print("  │  Data: Automotive logistics + Consumables SKU spike patterns")
    print("  │  Architecture: 150 trees, max_depth=12, min_samples_leaf=5")
    model_c = WMSRandomForest().fit(df_c)

    # Collect metrics dict used across all visualizations
    model_metrics = {
        'A': list(model_a.metrics.values())[0],
        'B': list(model_b.metrics.values())[0],
        'C': list(model_c.metrics.values())[0],
    }

    print(f"\n  Model performance on held-out test sets:")
    print(f"  {'Model':<30} {'R²':>8}  {'RMSE (ms)':>10}  {'MAE (ms)':>10}")
    print(f"  {'-'*62}")
    name_map = {
        'A': 'SSD Regressor     (WH-A)',
        'B': 'LSTM NumPy        (WH-B)',
        'C': 'Random Forest     (WH-C)',
    }
    for k in ['A', 'B', 'C']:
        v = model_metrics[k]
        print(f"  {name_map[k]:<30} {v['r2']:>8.4f}  {v['rmse']:>10.2f}  {v['mae']:>10.2f}")

    # ────────────────────────────────────────────────────────────────────────
    #  STEP 3 — Federated Learning
    # ────────────────────────────────────────────────────────────────────────
    print("\n[3/4] Running Federated Learning pipeline...")
    print("  Privacy guarantee: only model parameters are shared, not raw data.\n")

    results = run_federated_learning(df_a, df_b, df_c, model_a, model_b, model_c)
    opt     = results['opt_signal']

    print(f"\n  FL Optimization Signal (from best model: {opt['recommended_model']}):")
    print(f"  {'Signal':<30} {'Value'}")
    print(f"  {'-'*50}")
    signals = [
        ('Confidence (R²)',          f"{opt['confidence']:.4f}"),
        ('Scan Lag Reduction',       f"{opt['scan_lag_reduction_pct']:.1f}%"),
        ('API Delay Reduction',      f"{opt['api_delay_reduction_pct']:.1f}%"),
        ('DB Latency Reduction',     f"{opt['db_latency_reduction_pct']:.1f}%"),
        ('Total Latency Reduction',  f"{opt['total_latency_reduction_pct']:.1f}%"),
        ('Async Queue Size',         f"{opt['async_queue_size']} messages"),
        ('API Timeout',              f"{opt['api_timeout_ms']} ms"),
        ('DB Pool Size',             f"{opt['db_pool_size']} connections"),
        ('Cache TTL',                f"{opt['cache_ttl_seconds']} seconds"),
        ('Batch Window',             f"{opt['batch_window_ms']:.0f} ms"),
    ]
    for label, val in signals:
        print(f"  {label:<30} {val}")

    print(f"\n  Before → After Latency Summary:")
    print(f"  {'Warehouse':<12} {'Model':<24} {'Before':>10}  {'After':>10}  {'Reduction':>10}")
    print(f"  {'-'*70}")
    for k in ['A', 'B', 'C']:
        r   = results[k]
        b   = r['before']['total_latency_ms']
        a   = r['after']['total_latency_ms']
        pct = (b - a) / b * 100
        print(f"  WH-{k:<9} {r['name']:<24} {b:>9.1f}ms  {a:>9.1f}ms  ↓{pct:.1f}%")

    print(f"\n  Per-component reductions:")
    print(f"  {'Component':<18} {'WH-A':>10}  {'WH-B':>10}  {'WH-C':>10}")
    print(f"  {'-'*52}")
    for comp, label in [
        ('scan_lag_ms',    'Scan Lag'),
        ('api_delay_ms',   'API Delay'),
        ('db_latency_ms',  'DB Latency'),
    ]:
        row = [f"  {label:<18}"]
        for k in ['A', 'B', 'C']:
            b   = results[k]['before'][comp]
            a   = results[k]['after'][comp]
            pct = (b - a) / b * 100
            row.append(f"↓{pct:>5.1f}%    ")
        print("".join(row))

    # ────────────────────────────────────────────────────────────────────────
    #  STEP 4 — Generate all visualizations
    # ────────────────────────────────────────────────────────────────────────
    print("\n[4/4] Generating all 9 visualizations...")
    generate_all_plots(results, model_metrics, model_a, model_b, model_c, OUTPUT_DIR)

    # ────────────────────────────────────────────────────────────────────────
    #  Final summary
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ✅  Pipeline complete!")
    print(f"  📁  All outputs saved to: {OUTPUT_DIR}/")
    print()
    print("  📊  Figures:")
    figures = {
        'fig0_dashboard.png':           'Executive dashboard (KPIs + accuracy cards + radars)',
        'fig1_gantt_before_after.png':  'Gantt: latency components before vs after FL',
        'fig2_metric_comparison.png':   'Per-metric grouped bar comparison',
        'fig3_timeseries.png':          'Rolling-avg latency time-series',
        'fig4_model_performance.png':   'R², RMSE, MAE comparison across models',
        'fig5_fl_improvement.png':      'FL improvement bars + reduction heatmap',
        'fig6_distributions.png':       'Latency distribution shift (violin + box) — all warehouses',
        'fig7_feature_importance.png':  'SSD scale | LSTM gate | RF Gini importances — all warehouses',
        'fig8_radar_performance.png':   'Dual radar: model quality + FL optimization',
    }
    for fname, desc in figures.items():
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            kb = os.path.getsize(fpath) / 1024
            print(f"    ✓  {fname:<38}  {desc}  ({kb:.0f} KB)")
        else:
            print(f"    ✗  {fname:<38}  (not found)")

    print()
    print("  📄  Engineered CSVs:")
    for fname in ['warehouse_a_engineered.csv', 'warehouse_b_engineered.csv',
                  'warehouse_c_engineered.csv']:
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            kb = os.path.getsize(fpath) / 1024
            print(f"    ✓  {fname}  ({kb:.0f} KB)")

    print("=" * 70)

    return results, model_metrics, model_a, model_b, model_c


# ════════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results, model_metrics, model_a, model_b, model_c = main()
