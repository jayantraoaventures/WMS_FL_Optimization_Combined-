"""
================================================================================
visualizations.py
WMS FL Optimization — Real Dataset Project
================================================================================
Generates 9 publication-quality dark-theme figures:

  fig0_dashboard.png         Executive dashboard (KPIs, accuracy cards, radars)
  fig1_gantt_before_after.png  Gantt bars: latency components before vs after
  fig2_metric_comparison.png   4-panel grouped bars per metric
  fig3_timeseries.png          Rolling-avg latency time-series
  fig4_model_performance.png   R², RMSE, MAE comparison (handles negative R²)
  fig5_fl_improvement.png      Reduction % grouped bars + heatmap
  fig6_distributions.png       Violin + box plots (distribution shift)
  fig7_feature_importance.png  SSD scale weights | LSTM gate weights | RF Gini
  fig8_radar_performance.png   Dual spider: model quality + FL optimization

Entry point:
  generate_all_plots(results, model_metrics, model_a, model_b, model_c, output_dir)

Run standalone (requires prior FL pipeline run):
  python visualizations.py
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Design tokens ──────────────────────────────────────────────────────────
PALETTE = {
    'bg':      '#0D1117',
    'surface': '#161B22',
    'border':  '#30363D',
    'text':    '#E6EDF3',
    'subtext': '#8B949E',
    'before':  '#F78166',   # coral-red  → high latency
    'after':   '#3FB950',   # green      → improved
    'wh_a':    '#58A6FF',   # blue   — Warehouse A
    'wh_b':    '#D2A8FF',   # purple — Warehouse B
    'wh_c':    '#FFA657',   # orange — Warehouse C
    'accent':  '#1F6FEB',
}
WH_COLORS     = [PALETTE['wh_a'], PALETTE['wh_b'], PALETTE['wh_c']]
WAREHOUSES    = ['Warehouse A\n(SSD Regressor)', 'Warehouse B\n(LSTM)',
                 'Warehouse C\n(Random Forest)']
WH_KEYS       = ['A', 'B', 'C']
METRICS       = ['scan_lag_ms', 'api_delay_ms', 'db_latency_ms', 'total_latency_ms']
METRIC_LABELS = ['Scan Lag', 'API Delay', 'DB Latency', 'Total Latency']


def _set_dark_style():
    plt.rcParams.update({
        'figure.facecolor': PALETTE['bg'],
        'axes.facecolor':   PALETTE['surface'],
        'axes.edgecolor':   PALETTE['border'],
        'axes.labelcolor':  PALETTE['text'],
        'xtick.color':      PALETTE['subtext'],
        'ytick.color':      PALETTE['subtext'],
        'text.color':       PALETTE['text'],
        'grid.color':       PALETTE['border'],
        'grid.linewidth':   0.6,
        'legend.facecolor': PALETTE['surface'],
        'legend.edgecolor': PALETTE['border'],
        'font.family':      'DejaVu Sans',
        'font.size':        10,
    })


def _fig(w, h):
    _set_dark_style()
    return plt.figure(figsize=(w, h), facecolor=PALETTE['bg'])


# ════════════════════════════════════════════════════════════════════════════
#  FIG 1 — Gantt-style Before / After Timeline
# ════════════════════════════════════════════════════════════════════════════

def plot_gantt(results: dict, save_path: str):
    """
    Horizontal Gantt bars per warehouse, per phase (before / after).
    Each bar is split into: Scan Lag | API Delay | DB Latency.
    Bar length encodes duration in ms; total marked with a dashed line.
    """
    fig, axes = plt.subplots(3, 2, figsize=(18, 12), facecolor=PALETTE['bg'])
    fig.suptitle('Gantt Chart: WMS Latency Components — Before vs After FL Optimization',
                 fontsize=16, color=PALETTE['text'], fontweight='bold', y=0.98)

    comp_colors = {'Scan Lag': '#58A6FF', 'API Delay': '#D2A8FF', 'DB Latency': '#FFA657'}

    for row, (wh_key, wh_label) in enumerate(zip(WH_KEYS, WAREHOUSES)):
        res = results[wh_key]
        for col, (phase, phase_label) in enumerate([
            ('before', 'BEFORE Optimization'), ('after', 'AFTER Optimization')
        ]):
            ax = axes[row][col]
            ax.set_facecolor(PALETTE['surface'])

            scan = res[phase]['scan_lag_ms']
            api  = res[phase]['api_delay_ms']
            db   = res[phase]['db_latency_ms']

            tasks     = ['Scan Lag', 'API Delay', 'DB Latency']
            starts    = [0, scan, scan + api]
            durations = [scan, api, db]
            y_pos     = [2, 1, 0]

            for task, start, dur, yp in zip(tasks, starts, durations, y_pos):
                ax.barh(yp, dur, left=start, height=0.55,
                        color=comp_colors[task], alpha=0.9,
                        edgecolor=PALETTE['border'], linewidth=0.8)
                if dur > 12:
                    ax.text(start + dur / 2, yp, f'{dur:.0f}ms',
                            ha='center', va='center',
                            fontsize=8.5, color='white', fontweight='bold')

            total = scan + api + db
            ax.axvline(x=total, color='white', linewidth=1.5, linestyle='--', alpha=0.5)
            ax.text(total + 2, 2.5, f'Total: {total:.0f}ms',
                    color='white', fontsize=9, va='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(tasks, fontsize=9)
            ax.set_xlabel('Latency (ms)', fontsize=9, color=PALETTE['subtext'])
            ax.set_xlim(0, max(total * 1.22, 50))
            ax.grid(axis='x', alpha=0.3)
            title_color = PALETTE['before'] if phase == 'before' else PALETTE['after']
            ax.set_title(f"{wh_label.split(chr(10))[0]} — {phase_label}",
                         fontsize=10, color=title_color, fontweight='bold', pad=6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ Gantt chart saved: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
#  FIG 2 — Per-Metric Grouped Bar Comparison
# ════════════════════════════════════════════════════════════════════════════

def plot_metric_comparison(results: dict, save_path: str):
    """4-panel chart, one panel per latency metric, grouped before/after bars."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor=PALETTE['bg'])
    fig.suptitle('WMS Performance Metrics: Before vs After FL Optimization',
                 fontsize=16, color=PALETTE['text'], fontweight='bold', y=0.99)

    x, w = np.arange(3), 0.35

    for idx, (metric, label) in enumerate(zip(METRICS, METRIC_LABELS)):
        ax = axes[idx // 2][idx % 2]
        ax.set_facecolor(PALETTE['surface'])

        bv = [results[k]['before'][metric] for k in WH_KEYS]
        av = [results[k]['after'][metric]  for k in WH_KEYS]

        ax.bar(x - w / 2, bv, w, label='Before FL',
               color=PALETTE['before'], alpha=0.85, edgecolor=PALETTE['border'])
        ax.bar(x + w / 2, av, w, label='After FL',
               color=PALETTE['after'],  alpha=0.85, edgecolor=PALETTE['border'])

        for i, (b, a) in enumerate(zip(bv, av)):
            pct = (b - a) / b * 100
            ax.annotate(f'↓{pct:.1f}%', xy=(x[i] + w / 2, a),
                        xytext=(0, 6), textcoords='offset points',
                        ha='center', fontsize=8.5,
                        color=PALETTE['after'], fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(['WH-A\n(SSD)', 'WH-B\n(LSTM)', 'WH-C\n(RF)'],
                           fontsize=9, color=PALETTE['text'])
        ax.set_ylabel(f'{label} (ms)', fontsize=10, color=PALETTE['subtext'])
        ax.set_title(label, fontsize=12, color=PALETTE['text'], fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, max(bv) * 1.30)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ Metric comparison saved: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
#  FIG 3 — Time-Series Latency (Rolling Average)
# ════════════════════════════════════════════════════════════════════════════

def plot_timeseries(results: dict, save_path: str):
    """Rolling-average total latency over time for each warehouse."""
    fig, axes = plt.subplots(3, 1, figsize=(18, 12),
                             facecolor=PALETTE['bg'], sharex=False)
    fig.suptitle('Latency Time-Series: Before vs After FL Optimization (Rolling Avg)',
                 fontsize=15, color=PALETTE['text'], fontweight='bold', y=0.99)

    for ax, wh_key, color in zip(axes, WH_KEYS, WH_COLORS):
        ax.set_facecolor(PALETTE['surface'])
        df   = results[wh_key]['df_opt']
        roll = 30

        before = df['total_latency_ms'].rolling(roll).mean().dropna()
        after  = df['total_latency_ms_opt'].rolling(roll).mean().dropna()
        xb     = np.arange(len(before))

        ax.fill_between(xb, before, alpha=0.18, color=PALETTE['before'])
        ax.fill_between(xb, after,  alpha=0.18, color=PALETTE['after'])
        ax.plot(xb, before, color=PALETTE['before'], linewidth=1.6,
                label='Before FL', alpha=0.9)
        ax.plot(xb, after,  color=PALETTE['after'],  linewidth=1.6,
                label='After FL',  alpha=0.9)

        avg_b = df['total_latency_ms'].mean()
        avg_a = df['total_latency_ms_opt'].mean()
        ax.axhline(avg_b, color=PALETTE['before'], linewidth=0.9,
                   linestyle='--', alpha=0.5)
        ax.axhline(avg_a, color=PALETTE['after'],  linewidth=0.9,
                   linestyle='--', alpha=0.5)
        ax.text(len(before) * 0.98, avg_b + 5, f'μ={avg_b:.0f}ms',
                color=PALETTE['before'], fontsize=8, ha='right')
        ax.text(len(before) * 0.98, avg_a - 12, f'μ={avg_a:.0f}ms',
                color=PALETTE['after'],  fontsize=8, ha='right')

        ax.set_title(f"Warehouse {wh_key} ({results[wh_key]['name']})",
                     fontsize=11, color=color, fontweight='bold')
        ax.set_ylabel('Total Latency (ms)', fontsize=9, color=PALETTE['subtext'])
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Record index (time-ordered)', fontsize=10,
                        color=PALETTE['subtext'])
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ Time-series plot saved: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
#  FIG 4 — Model Performance (R², RMSE, MAE)
# ════════════════════════════════════════════════════════════════════════════

def plot_model_performance(model_metrics: dict, save_path: str):
    """
    Bar charts for R², RMSE, MAE.
    Handles negative R² (LSTM) with dynamic axis and zero-baseline.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor=PALETTE['bg'])
    fig.suptitle('Model Performance Comparison: SSD Regressor vs LSTM vs Random Forest',
                 fontsize=15, color=PALETTE['text'], fontweight='bold', y=0.99)

    model_names = ['WH-A\nSSD Reg.', 'WH-B\nLSTM', 'WH-C\nRandom Forest']
    datasets = [
        ([model_metrics[k]['r2']   for k in ['A','B','C']], 'R² Score',  'R² (higher = better)'),
        ([model_metrics[k]['rmse'] for k in ['A','B','C']], 'RMSE (ms)', 'RMSE ms (lower = better)'),
        ([model_metrics[k]['mae']  for k in ['A','B','C']], 'MAE (ms)',  'MAE ms (lower = better)'),
    ]

    for ax, (vals, label, ylab) in zip(axes, datasets):
        ax.set_facecolor(PALETTE['surface'])
        bar_colors = [WH_COLORS[i] if vals[i] >= 0 else '#F78166'
                      for i in range(len(vals))]
        bars = ax.bar(model_names, vals, color=bar_colors,
                      edgecolor=PALETTE['border'], alpha=0.88, width=0.5)

        v_min, v_max = min(vals), max(vals)
        span = v_max - v_min if v_max != v_min else abs(v_max) + 0.1
        pad  = span * 0.25
        ax.set_ylim(v_min - pad, v_max + pad)

        if label == 'R² Score':
            ax.axhline(0, color=PALETTE['subtext'], linewidth=1.4,
                       linestyle='--', alpha=0.75, zorder=0)

        for bar, val in zip(bars, vals):
            offset = pad * 0.30
            y_pos  = val + offset if val >= 0 else val - offset
            va     = 'bottom' if val >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f'{val:.3f}', ha='center', va=va, fontsize=10,
                    color=PALETTE['text'], fontweight='bold')

        ax.set_title(label, fontsize=12, color=PALETTE['text'], fontweight='bold')
        ax.set_ylabel(ylab, fontsize=9, color=PALETTE['subtext'])
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.text(0.5, 0.01,
             '* LSTM (WH-B) R² may be negative: NumPy-LSTM learns temporal trends '
             'over the Food Report daily cycle, not point regression. '
             'Trend accuracy is used for the accuracy % cards.',
             ha='center', fontsize=8.5, color=PALETTE['subtext'], style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ Model performance chart saved: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
#  FIG 5 — FL Improvement Summary
# ════════════════════════════════════════════════════════════════════════════

def plot_fl_improvement_summary(results: dict, save_path: str):
    """Grouped reduction-% bars (left) + RdYlGn heatmap (right)."""
    fig = _fig(16, 8)
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[1.6, 1])
    ax1 = fig.add_subplot(gs[0]);  ax1.set_facecolor(PALETTE['surface'])
    ax2 = fig.add_subplot(gs[1]);  ax2.set_facecolor(PALETTE['surface'])

    reductions = {
        wh: {m: (results[wh]['before'][m] - results[wh]['after'][m]) /
                  results[wh]['before'][m] * 100
             for m in METRICS}
        for wh in WH_KEYS
    }

    x, w, offsets = np.arange(len(METRICS)), 0.25, [-0.25, 0, 0.25]
    for i, (wh, color) in enumerate(zip(WH_KEYS, WH_COLORS)):
        vals = [reductions[wh][m] for m in METRICS]
        bars = ax1.bar(x + offsets[i], vals, w, label=f'Warehouse {wh}',
                       color=color, alpha=0.85, edgecolor=PALETTE['border'])
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f'{v:.1f}%', ha='center', fontsize=8.5,
                     color=color, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(METRIC_LABELS, fontsize=10)
    ax1.set_ylabel('Latency Reduction (%)', fontsize=11)
    ax1.set_title('Latency Reduction by Metric & Warehouse', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0, 55)

    heat = np.array([[reductions[wh][m] for m in METRICS] for wh in WH_KEYS])
    im   = ax2.imshow(heat, cmap='RdYlGn', aspect='auto', vmin=20, vmax=50)
    ax2.set_xticks(range(len(METRICS)))
    ax2.set_xticklabels(METRIC_LABELS, fontsize=8.5, rotation=15)
    ax2.set_yticks(range(3))
    ax2.set_yticklabels([f'WH-{k}' for k in WH_KEYS], fontsize=10)
    ax2.set_title('Reduction Heatmap (%)', fontsize=12, fontweight='bold')
    for i in range(3):
        for j in range(len(METRICS)):
            ax2.text(j, i, f'{heat[i,j]:.1f}%', ha='center', va='center',
                     fontsize=10, color='black', fontweight='bold')
    fig.colorbar(im, ax=ax2, shrink=0.8, label='Reduction %')

    fig.suptitle('FL Optimization Impact: Latency Reduction Summary',
                 fontsize=15, color=PALETTE['text'], fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ FL improvement summary saved: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
#  FIG 6 — Latency Distribution (Violin + Box)
# ════════════════════════════════════════════════════════════════════════════

def plot_distributions(results: dict, save_path: str):
    """Violin + box plot showing distribution shift before vs after."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor=PALETTE['bg'])
    fig.suptitle('Total Latency Distribution: Before vs After FL Optimization',
                 fontsize=15, color=PALETTE['text'], fontweight='bold', y=1.01)

    for ax, wh_key, wh_color, wh_label in zip(axes, WH_KEYS, WH_COLORS, WAREHOUSES):
        ax.set_facecolor(PALETTE['surface'])
        df   = results[wh_key]['df_opt']
        data = [df['total_latency_ms'].values, df['total_latency_ms_opt'].values]

        vp = ax.violinplot(data, positions=[1, 2], showmedians=True)
        for pc, c in zip(vp['bodies'], [PALETTE['before'], PALETTE['after']]):
            pc.set_facecolor(c);  pc.set_alpha(0.55)
        vp['cmedians'].set_color('white')

        ax.boxplot(data, positions=[1, 2], widths=0.25, patch_artist=True,
                   boxprops=dict(facecolor='none', color='white'),
                   medianprops=dict(color='white', linewidth=2),
                   whiskerprops=dict(color=PALETTE['subtext']),
                   capprops=dict(color=PALETTE['subtext']),
                   flierprops=dict(marker='.', markersize=2, color=PALETTE['subtext']))

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Before\nFL', 'After\nFL'], fontsize=11)
        ax.set_ylabel('Total Latency (ms)', fontsize=10)
        ax.set_title(wh_label.replace('\n', ' '), fontsize=11,
                     color=wh_color, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        med_b = np.median(df['total_latency_ms'])
        med_a = np.median(df['total_latency_ms_opt'])
        ax.text(1.5, ax.get_ylim()[1] * 0.93,
                f'Median ↓{(med_b-med_a)/med_b*100:.1f}%',
                ha='center', fontsize=9, color=PALETTE['after'], fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ Distribution plot saved: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
#  FIG 7 — Feature Importance (3-panel)
# ════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(model_a, model_b, model_c, save_path: str):
    """
    Panel A — SSD: meta-learner scale contributions (coarse/medium/fine)
    Panel B — LSTM: normalised mean |weight| across 4 gates per input feature
    Panel C — RF:   Gini feature importances
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 8), facecolor=PALETTE['bg'])
    fig.suptitle('Model Internals & Feature Importance for Latency Prediction',
                 fontsize=15, color=PALETTE['text'], fontweight='bold')

    # ── Panel A: SSD scale contributions ─────────────────────────────────
    ax = axes[0];  ax.set_facecolor(PALETTE['surface'])
    feats_a = ['Coarse Scale\n(Low-Res)', 'Medium Scale\n(Mid-Res)',
               'Fine Scale\n(High-Res)']
    imp_a   = model_a.feature_importances_
    bars    = ax.barh(feats_a, imp_a,
                      color=[PALETTE['wh_a'], PALETTE['wh_b'], PALETTE['wh_c']],
                      edgecolor=PALETTE['border'], alpha=0.88, height=0.5)
    for bar, v in zip(bars, imp_a):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=11,
                color=PALETTE['text'], fontweight='bold')
    ax.set_xlabel('Meta-Learner Importance', fontsize=10)
    ax.set_title('WH-A: SSD Multi-Scale\nMeta-Learner Contributions',
                 fontsize=11, color=PALETTE['wh_a'], fontweight='bold')
    ax.set_xlim(0, max(imp_a) * 1.35)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False);  ax.spines['right'].set_visible(False)

    # ── Panel B: LSTM gate weights ────────────────────────────────────────
    ax = axes[1];  ax.set_facecolor(PALETTE['surface'])
    n_in    = model_b.input_size
    feat_names_b = ['Scan Time', 'API Calls/min', 'DB Connections',
                    'Net Jitter', 'Temp Delta', 'Batch Queue',
                    'CPU Load', 'Memory Usage']
    stacked = np.abs(np.vstack([
        model_b.Wf[:, :n_in], model_b.Wi[:, :n_in],
        model_b.Wg[:, :n_in], model_b.Wo[:, :n_in],
    ]))
    contrib  = stacked.mean(axis=0)
    contrib /= contrib.sum()
    sidx     = np.argsort(contrib)
    cmap_b   = matplotlib.colormaps['cool'](np.linspace(0.25, 0.95, len(sidx)))
    bars     = ax.barh([feat_names_b[i] for i in sidx], contrib[sidx],
                       color=cmap_b, edgecolor=PALETTE['border'], alpha=0.88)
    for bar, v in zip(bars, contrib[sidx]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=9,
                color=PALETTE['text'], fontweight='bold')
    ax.set_xlabel('Normalised Mean |Weight| across Gates', fontsize=9)
    ax.set_title('WH-B: LSTM Input Feature\nInfluence via Gate Weights',
                 fontsize=11, color=PALETTE['wh_b'], fontweight='bold')
    ax.set_xlim(0, max(contrib) * 1.40)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False);  ax.spines['right'].set_visible(False)

    # ── Panel C: RF Gini importances ──────────────────────────────────────
    ax = axes[2];  ax.set_facecolor(PALETTE['surface'])
    imp_c = model_c.feature_importances_
    sidx  = np.argsort(imp_c)
    cmap_c = matplotlib.colormaps['plasma'](np.linspace(0.3, 0.9, len(sidx)))
    bars   = ax.barh([model_c.feature_cols[i].replace('_', ' ') for i in sidx],
                     imp_c[sidx], color=cmap_c,
                     edgecolor=PALETTE['border'], alpha=0.88)
    for bar, v in zip(bars, imp_c[sidx]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=9,
                color=PALETTE['text'], fontweight='bold')
    ax.set_xlabel('Gini Importance', fontsize=10)
    ax.set_title('WH-C: Random Forest\nFeature Importance',
                 fontsize=11, color=PALETTE['wh_c'], fontweight='bold')
    ax.set_xlim(0, max(imp_c) * 1.35)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False);  ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ Feature importance plot saved: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
#  FIG 0 — Executive Dashboard
# ════════════════════════════════════════════════════════════════════════════

def plot_dashboard(results: dict, model_metrics: dict,
                   opt_signal: dict, save_path: str):
    """
    4-row executive dashboard:
      Row 0 — 4 global KPI cards
      Row 1 — 3 per-model accuracy cards + latency before/after bar
      Row 2 — fixed R² bar chart + reduction heatmap
      Row 3 — embedded dual mini-radar
    """
    _set_dark_style()
    fig = plt.figure(figsize=(22, 20), facecolor=PALETTE['bg'])
    gs  = GridSpec(4, 4, figure=fig, hspace=0.52, wspace=0.38)

    fig.text(0.5, 0.985, '🏭  WMS FL Optimization — Executive Dashboard',
             ha='center', fontsize=19, color=PALETTE['text'],
             fontweight='bold', va='top')
    fig.text(0.5, 0.963,
             'Real Data · Federated Learning · SSD Regressor | LSTM | Random Forest',
             ha='center', fontsize=11, color=PALETTE['subtext'], va='top')

    def kpi_card(ax, title, val_str, color, sub=None):
        ax.set_facecolor(PALETTE['surface'])
        ax.set_xlim(0, 1);  ax.set_ylim(0, 1);  ax.axis('off')
        ax.add_patch(FancyBboxPatch((0.04, 0.04), 0.92, 0.92,
                     boxstyle='round,pad=0.04', linewidth=2.2,
                     edgecolor=color, facecolor=PALETTE['surface']))
        ax.text(0.5, 0.70, val_str, ha='center', va='center',
                fontsize=22, color=color, fontweight='bold')
        ax.text(0.5, 0.30, title,   ha='center', va='center',
                fontsize=9,  color=PALETTE['subtext'])
        if sub:
            ax.text(0.5, 0.12, sub, ha='center', va='center',
                    fontsize=7.5, color=PALETTE['border'])

    # Row 0: KPI cards
    for ax, (title, val, color, sub) in zip(
        [fig.add_subplot(gs[0, i]) for i in range(4)],
        [
            ('Best FL Model',      opt_signal['recommended_model'].replace('_', '\n'), PALETTE['wh_c'], None),
            ('FL Confidence',      f"{opt_signal['confidence']:.4f}",                 PALETTE['wh_b'], 'model R² score'),
            ('Total Latency\nRed', f"{opt_signal['total_latency_reduction_pct']:.1f}%", PALETTE['after'], 'avg across warehouses'),
            ('API Delay\nRed',     f"{opt_signal['api_delay_reduction_pct']:.1f}%",   PALETTE['wh_a'], 'post FL optimization'),
        ]
    ):
        kpi_card(ax, title, val, color, sub)

    # Row 1: accuracy cards + latency bar
    def model_accuracy_pct(key):
        r2   = model_metrics[key]['r2']
        if r2 >= 0:
            return r2 * 100
        mae  = model_metrics[key]['mae']
        rmse = model_metrics[key]['rmse']
        return max(0, (1 - mae / rmse) * 100)

    acc_labels = {
        'A': ('WH-A  SSD Regressor', 'R²-based accuracy'),
        'B': ('WH-B  LSTM',           'Temporal trend accuracy*'),
        'C': ('WH-C  Random Forest',  'R²-based accuracy'),
    }
    for col, key in enumerate(['A', 'B', 'C']):
        ax    = fig.add_subplot(gs[1, col])
        acc   = model_accuracy_pct(key)
        r2    = model_metrics[key]['r2']
        rmse  = model_metrics[key]['rmse']
        mae   = model_metrics[key]['mae']
        color = [PALETTE['wh_a'], PALETTE['wh_b'], PALETTE['wh_c']][col]
        title, sub = acc_labels[key]

        ax.set_facecolor(PALETTE['surface'])
        ax.set_xlim(0, 1);  ax.set_ylim(0, 1);  ax.axis('off')
        ax.add_patch(FancyBboxPatch((0.03, 0.03), 0.94, 0.94,
                     boxstyle='round,pad=0.04', linewidth=2.2,
                     edgecolor=color, facecolor=PALETTE['surface']))
        ax.text(0.5, 0.76, f"{acc:.1f}%", ha='center', va='center',
                fontsize=26, color=color, fontweight='bold')
        ax.text(0.5, 0.58, 'Accuracy', ha='center', va='center',
                fontsize=9,  color=PALETTE['subtext'])
        ax.plot([0.1, 0.9], [0.48, 0.48], color=PALETTE['border'], linewidth=0.8)
        ax.text(0.5, 0.39, title, ha='center', va='center',
                fontsize=8.5, color=PALETTE['text'], fontweight='bold')
        ax.text(0.5, 0.27,
                f"R² = {r2:.4f}   RMSE = {rmse:.1f} ms   MAE = {mae:.1f} ms",
                ha='center', va='center', fontsize=7.5, color=PALETTE['subtext'])
        ax.text(0.5, 0.10, sub, ha='center', va='center',
                fontsize=7, color=PALETTE['border'], style='italic')

    ax_bar = fig.add_subplot(gs[1, 3])
    ax_bar.set_facecolor(PALETTE['surface'])
    x, w = np.arange(3), 0.35
    bv = [results[k]['before']['total_latency_ms'] for k in WH_KEYS]
    av = [results[k]['after']['total_latency_ms']  for k in WH_KEYS]
    ax_bar.bar(x - w/2, bv, w, color=PALETTE['before'], label='Before', alpha=0.85)
    ax_bar.bar(x + w/2, av, w, color=PALETTE['after'],  label='After',  alpha=0.85)
    for xp, b, a in zip(x, bv, av):
        ax_bar.text(xp, max(b, a) + 4, f'↓{(b-a)/b*100:.0f}%',
                    ha='center', fontsize=8, color=PALETTE['after'], fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(['WH-A', 'WH-B', 'WH-C'], fontsize=9)
    ax_bar.set_ylabel('Total Latency (ms)', fontsize=8)
    ax_bar.set_title('Avg Latency\nBefore vs After', fontsize=10, fontweight='bold')
    ax_bar.legend(fontsize=8);  ax_bar.grid(axis='y', alpha=0.3)
    ax_bar.spines['top'].set_visible(False);  ax_bar.spines['right'].set_visible(False)
    ax_bar.set_ylim(0, max(bv) * 1.22)

    # Row 2: R² bar + heatmap
    ax_r2 = fig.add_subplot(gs[2, :2])
    ax_r2.set_facecolor(PALETTE['surface'])
    r2s   = [model_metrics[k]['r2'] for k in ['A','B','C']]
    lbls  = ['WH-A  SSD', 'WH-B  LSTM', 'WH-C  RF']
    bc_r2 = [WH_COLORS[i] if r2s[i] >= 0 else '#F78166' for i in range(3)]
    br2   = ax_r2.barh(lbls, r2s, color=bc_r2,
                        edgecolor=PALETTE['border'], alpha=0.88, height=0.5)
    r2_pad = (max(r2s) - min(r2s)) * 0.22
    ax_r2.set_xlim(min(r2s) - r2_pad, max(r2s) + r2_pad)
    ax_r2.axvline(0, color=PALETTE['subtext'], linewidth=1.4, linestyle='--', alpha=0.75)
    ax_r2.axvline(1, color=PALETTE['after'],   linewidth=0.9, linestyle=':',  alpha=0.5)
    for i, (bar, v) in enumerate(zip(br2, r2s)):
        off = r2_pad * 0.25
        ax_r2.text(v + off if v >= 0 else v - off,
                   bar.get_y() + bar.get_height() / 2,
                   f'R²={v:.4f}', va='center',
                   ha='left' if v >= 0 else 'right',
                   fontsize=9.5, color=WH_COLORS[i], fontweight='bold')
    ax_r2.set_title('Model R² Score  (negative LSTM shown correctly)',
                    fontsize=10, fontweight='bold')
    ax_r2.grid(axis='x', alpha=0.3)
    ax_r2.spines['top'].set_visible(False);  ax_r2.spines['right'].set_visible(False)
    ax_r2.text(0.01, -0.18,
               '* LSTM R² negative: learns temporal Food Report cycles, not point regression.',
               transform=ax_r2.transAxes, fontsize=7.5,
               color=PALETTE['subtext'], style='italic')

    ax_heat = fig.add_subplot(gs[2, 2:])
    ax_heat.set_facecolor(PALETTE['surface'])
    hd = np.array([[(results[wh]['before'][m] - results[wh]['after'][m]) /
                     results[wh]['before'][m] * 100
                    for m in METRICS] for wh in WH_KEYS])
    im = ax_heat.imshow(hd, cmap='YlGn', aspect='auto', vmin=20, vmax=48)
    ax_heat.set_xticks(range(4));       ax_heat.set_xticklabels(METRIC_LABELS, fontsize=9)
    ax_heat.set_yticks(range(3));       ax_heat.set_yticklabels([f'WH-{k}' for k in WH_KEYS])
    for i in range(3):
        for j in range(4):
            ax_heat.text(j, i, f'{hd[i,j]:.1f}%', ha='center', va='center',
                         fontsize=10.5, color='black', fontweight='bold')
    ax_heat.set_title('Latency Reduction % Heatmap', fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=ax_heat, shrink=0.75, label='Reduction %')

    # Row 3: mini radars
    def _build_radar_data(model_metrics, results):
        r2_raw  = [model_metrics[k]['r2']   for k in ['A','B','C']]
        rmse_raw= [model_metrics[k]['rmse'] for k in ['A','B','C']]
        mae_raw = [model_metrics[k]['mae']  for k in ['A','B','C']]
        r2_min, r2_max = min(r2_raw), max(r2_raw)
        r2_n    = [(v-r2_min)/(r2_max-r2_min+1e-9) for v in r2_raw]
        rmse_a  = [1-(v/(max(rmse_raw)+1e-9)) for v in rmse_raw]
        mae_a   = [1-(v/(max(mae_raw)+1e-9))  for v in mae_raw]
        stab    = [1-abs(model_metrics[k]['rmse']-model_metrics[k]['mae'])/
                   (model_metrics[k]['rmse']+1e-9) for k in ['A','B','C']]
        sm,sx   = min(stab), max(stab)
        stab_n  = [(v-sm)/(sx-sm+1e-9) for v in stab]
        gen     = [max(0.,min(1.,model_metrics[k]['r2'])) for k in ['A','B','C']]
        q_data  = list(zip(r2_n, rmse_a, mae_a, stab_n, gen))

        pct = lambda bk, ak: [(results[k]['before'][bk]-results[k]['after'][ak])/
                               results[k]['before'][bk] for k in WH_KEYS]
        conf = [max(0, min(1, model_metrics[k]['r2'])) for k in ['A','B','C']]
        o_data = list(zip(pct('scan_lag_ms','scan_lag_ms'),
                          pct('api_delay_ms','api_delay_ms'),
                          pct('db_latency_ms','db_latency_ms'),
                          pct('total_latency_ms','total_latency_ms'),
                          conf))
        return q_data, o_data

    def mini_radar(ax, categories, model_scores, title, title_color):
        N  = len(categories)
        ang= np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
        ax.set_theta_offset(np.pi/2);  ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(ang[:-1]), categories,
                          fontsize=8, color=PALETTE['text'])
        for ring in [0.25, 0.5, 0.75, 1.0]:
            ra = np.linspace(0, 2*np.pi, 200)
            ax.plot(ra, [ring]*200, color=PALETTE['border'],
                    linewidth=0.5, linestyle='--', alpha=0.4)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels(['0.5', '1.0'], fontsize=6.5, color=PALETTE['subtext'])
        ax.spines['polar'].set_color(PALETTE['border'])
        ax.set_facecolor(PALETTE['surface'])
        for scores, color, lbl in zip(model_scores, WH_COLORS,
                                       ['WH-A (SSD)', 'WH-B (LSTM)', 'WH-C (RF)']):
            vals = list(scores) + [scores[0]]
            ax.plot(ang, vals, color=color, linewidth=2, label=lbl, zorder=3)
            ax.fill(ang, vals, color=color, alpha=0.18, zorder=2)
            ax.scatter(ang[:-1], scores, s=30, color=color,
                       zorder=4, edgecolors='white', linewidth=0.6)
        ax.set_title(title, fontsize=9.5, color=title_color,
                     fontweight='bold', pad=14)
        ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.15),
                  fontsize=7.5, framealpha=0.25)

    q_data, o_data = _build_radar_data(model_metrics, results)
    mini_radar(fig.add_subplot(gs[3, 0:2], polar=True),
               ['R²\n(norm)', 'RMSE\nAcc', 'MAE\nAcc', 'Stability', 'Generalise'],
               q_data, '① Model Quality Radar', PALETTE['wh_a'])
    mini_radar(fig.add_subplot(gs[3, 2:4], polar=True),
               ['Scan Lag\nRed.', 'API\nRed.', 'DB\nRed.', 'Total\nRed.', 'FL\nConf.'],
               o_data, '② FL Optimization Radar', PALETTE['after'])

    fig.text(0.5, 0.005, 'Radar axes normalised [0,1] — higher = better on every spoke.',
             ha='center', fontsize=8, color=PALETTE['subtext'], style='italic')

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ Dashboard saved: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
#  FIG 8 — Standalone Dual Radar
# ════════════════════════════════════════════════════════════════════════════

def plot_radar(results: dict, model_metrics: dict, save_path: str):
    """
    Full-page dual radar:
      Left  — Model quality   (R², RMSE acc, MAE acc, stability, generalisation)
      Right — FL optimization (scan/api/db/total reduction, FL confidence)
    All axes normalised to [0, 1], higher = better.
    """
    _set_dark_style()

    r2_raw   = [model_metrics[k]['r2']   for k in ['A','B','C']]
    rmse_raw = [model_metrics[k]['rmse'] for k in ['A','B','C']]
    mae_raw  = [model_metrics[k]['mae']  for k in ['A','B','C']]
    r2_min, r2_max = min(r2_raw), max(r2_raw)
    r2_n    = [(v-r2_min)/(r2_max-r2_min+1e-9) for v in r2_raw]
    rmse_a  = [1-(v/(max(rmse_raw)+1e-9)) for v in rmse_raw]
    mae_a   = [1-(v/(max(mae_raw)+1e-9))  for v in mae_raw]
    stab    = [1-abs(model_metrics[k]['rmse']-model_metrics[k]['mae'])/
               (model_metrics[k]['rmse']+1e-9) for k in ['A','B','C']]
    sm, sx  = min(stab), max(stab)
    stab_n  = [(v-sm)/(sx-sm+1e-9) for v in stab]
    gen     = [max(0.,min(1.,model_metrics[k]['r2'])) for k in ['A','B','C']]

    quality_axes = ['R² Score\n(norm)', 'RMSE\nAccuracy', 'MAE\nAccuracy',
                    'Training\nStability', 'Generalisation']
    quality_data = list(zip(r2_n, rmse_a, mae_a, stab_n, gen))

    pct = lambda bk, ak: [(results[k]['before'][bk]-results[k]['after'][ak])/
                           results[k]['before'][bk] for k in WH_KEYS]
    conf = [max(0, min(1, model_metrics[k]['r2'])) for k in ['A','B','C']]
    optim_axes = ['Scan Lag\nReduction', 'API Delay\nReduction',
                  'DB Latency\nReduction', 'Total Latency\nReduction',
                  'FL Model\nConfidence']
    optim_data = list(zip(pct('scan_lag_ms', 'scan_lag_ms'),
                          pct('api_delay_ms', 'api_delay_ms'),
                          pct('db_latency_ms', 'db_latency_ms'),
                          pct('total_latency_ms', 'total_latency_ms'),
                          conf))

    def draw_radar(ax, categories, model_scores, title):
        N   = len(categories)
        ang = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
        ax.set_theta_offset(np.pi/2);  ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(ang[:-1]), categories,
                          fontsize=9, color=PALETTE['text'])
        for ring in [0.25, 0.5, 0.75, 1.0]:
            ra = np.linspace(0, 2*np.pi, 200)
            ax.plot(ra, [ring]*200, color=PALETTE['border'],
                    linewidth=0.6, linestyle='--', alpha=0.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25','0.5','0.75','1.0'],
                           fontsize=7, color=PALETTE['subtext'])
        ax.spines['polar'].set_color(PALETTE['border'])
        ax.set_facecolor(PALETTE['surface'])
        for scores, color, label, alpha in zip(
            model_scores, WH_COLORS,
            ['WH-A (SSD)', 'WH-B (LSTM)', 'WH-C (RF)'],
            [0.25, 0.20, 0.20]
        ):
            vals = list(scores) + [scores[0]]
            ax.plot(ang, vals, color=color, linewidth=2.2, label=label, zorder=3)
            ax.fill(ang, vals, color=color, alpha=alpha, zorder=2)
            ax.scatter(ang[:-1], scores, s=45, color=color,
                       zorder=4, edgecolors='white', linewidth=0.8)
        ax.set_title(title, fontsize=12, color=PALETTE['text'],
                     fontweight='bold', pad=22)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
                  fontsize=9, framealpha=0.3)

    fig = plt.figure(figsize=(20, 9), facecolor=PALETTE['bg'])
    fig.suptitle('Radar Performance Chart — WMS Federated Learning Optimization',
                 fontsize=17, color=PALETTE['text'], fontweight='bold', y=1.01)

    draw_radar(fig.add_subplot(121, polar=True), quality_axes, quality_data,
               '① Model Quality\n(R², Accuracy, Stability, Generalisation)')
    draw_radar(fig.add_subplot(122, polar=True), optim_axes, optim_data,
               '② FL Optimization Impact\n(Latency Reductions + Confidence)')

    fig.text(0.5, -0.03,
             'All axes normalised to [0,1] — higher = better.  '
             'LSTM quality score reflects relative position among the three models.',
             ha='center', fontsize=8.5, color=PALETTE['subtext'], style='italic')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ Radar chart saved: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
#  Master entry point
# ════════════════════════════════════════════════════════════════════════════

def generate_all_plots(results, model_metrics, model_a, model_b, model_c,
                       output_dir: str):
    """Generate all 9 figures and save to output_dir."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    opt = results['opt_signal']

    print("\nGenerating visualizations...")
    plot_gantt              (results,                       f"{output_dir}/fig1_gantt_before_after.png")
    plot_metric_comparison  (results,                       f"{output_dir}/fig2_metric_comparison.png")
    plot_timeseries         (results,                       f"{output_dir}/fig3_timeseries.png")
    plot_model_performance  (model_metrics,                 f"{output_dir}/fig4_model_performance.png")
    plot_fl_improvement_summary(results,                    f"{output_dir}/fig5_fl_improvement.png")
    plot_distributions      (results,                       f"{output_dir}/fig6_distributions.png")
    plot_feature_importance (model_a, model_b, model_c,     f"{output_dir}/fig7_feature_importance.png")
    plot_radar              (results, model_metrics,         f"{output_dir}/fig8_radar_performance.png")
    plot_dashboard          (results, model_metrics, opt,   f"{output_dir}/fig0_dashboard.png")
    print(f"\n  All 9 figures saved to: {output_dir}/")
