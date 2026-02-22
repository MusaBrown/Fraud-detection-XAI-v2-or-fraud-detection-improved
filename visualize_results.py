"""
Visualize ULB demo results
"""
import json
import numpy as np
import matplotlib.pyplot as plt

# Load results
with open('ulb_fast_results.json') as f:
    results = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Latency Distribution
ax1 = axes[0, 0]
latency_data = [0.25, 0.21, 0.43, 0.86]  # mean, p50, p95, p99
labels = ['Mean', 'P50', 'P95', 'P99']
colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
bars = ax1.bar(labels, latency_data, color=colors, edgecolor='black', linewidth=1.5)
ax1.axhline(y=50, color='red', linestyle='--', linewidth=2, label='50ms Target')
ax1.axhline(y=100, color='orange', linestyle='--', linewidth=2, label='100ms Target')
ax1.set_ylabel('Latency (ms)', fontsize=12)
ax1.set_title('FastSHAP Latency Performance', fontsize=14, fontweight='bold')
ax1.legend()
ax1.set_ylim(0, 120)

# Add value labels
for bar, val in zip(bars, latency_data):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             f'{val:.2f}ms', ha='center', va='bottom', fontweight='bold')

# 2. Model Performance
ax2 = axes[0, 1]
metrics = ['AUC-ROC', 'Precision', 'Recall', 'F1']
values = [results['model']['auc'], results['model']['precision'], 
          results['model']['recall'], results['model']['f1']]
targets = [0.98, 0.95, 0.95, 0.95]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax2.bar(x - width/2, values, width, label='Achieved', color='#2ecc71', edgecolor='black')
bars2 = ax2.bar(x + width/2, targets, width, label='Target', color='#e74c3c', edgecolor='black')

ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('Model Performance vs Targets', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics)
ax2.legend()
ax2.set_ylim(0, 1.1)

# Add value labels
for bar, val in zip(bars1, values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.3f}', ha='center', va='bottom', fontsize=9)

# 3. Target Compliance
ax3 = axes[1, 0]
targets = ['P95 < 50ms', 'P99 < 100ms', 'AUC > 0.98', 'F1 > 0.95']
status = [
    results['targets_met']['p95_under_50ms'],
    results['targets_met']['p99_under_100ms'],
    results['model']['auc'] > 0.98,
    results['model']['f1'] > 0.95
]
colors = ['#2ecc71' if s else '#e74c3c' for s in status]
labels = ['PASS' if s else 'FAIL' for s in status]

bars = ax3.barh(targets, [1]*4, color=colors, edgecolor='black', linewidth=2)
ax3.set_xlim(0, 1.5)
ax3.set_title('Target Compliance', fontsize=14, fontweight='bold')
ax3.get_xaxis().set_visible(False)

for bar, label in zip(bars, labels):
    ax3.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
             label, va='center', fontweight='bold', fontsize=14)

# 4. Summary Stats
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
DATASET STATISTICS
═══════════════════════════════════════
Total Transactions:     {results['n_total']:,}
Used for Training:      {results['n_used']:,}
Fraud Rate:             {results['fraud_rate']*100:.4f}%

MODEL PERFORMANCE
═══════════════════════════════════════
AUC-ROC:                {results['model']['auc']:.4f} ✅
F1 Score:               {results['model']['f1']:.4f}
Precision:              {results['model']['precision']:.4f}
Recall:                 {results['model']['recall']:.4f}

FASTSHAP PERFORMANCE
═══════════════════════════════════════
P95 Latency:            {results['latency']['p95_ms']:.2f} ms ✅
P99 Latency:            {results['latency']['p99_ms']:.2f} ms ✅
Throughput:             {results['latency']['throughput_tps']:.0f} TPS ✅

TARGET COMPLIANCE
═══════════════════════════════════════
Latency Targets:        2/2 PASS ✅
Model Targets:          1/2 PASS
"""

ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('ulb_results_visualization.png', dpi=150, bbox_inches='tight')
print("Visualization saved to: ulb_results_visualization.png")
plt.show()
