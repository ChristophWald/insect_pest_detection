import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

final_mask_results = pd.read_csv("/user/christoph.wald/u15287/insect_pest_detection/3_2_image_processing_evaluation/data/metrics_final_mask.csv")
final_test_results = pd.read_csv("/user/christoph.wald/u15287/insect_pest_detection/3_2_image_processing_evaluation/data/metrics_final_test.csv")

datasets = [final_mask_results, final_test_results]
titles = ['Final Mask Results', 'Final Test Results']

# Bar width and x-axis positions
bar_width = 0.35
prefixes = final_mask_results['prefix']
x = np.arange(len(prefixes))

# Get a common y-axis limit for TP/FP counts
max_count = max([df[['TP','FP']].to_numpy().max() for df in datasets]) * 1.1

fig, axes = plt.subplots(1, 2, figsize=(14,6), sharey=True)

for i, df in enumerate(datasets):
    ax1 = axes[i]
    
    # Bars for TP and FP
    ax1.bar(x - bar_width/2, df['TP'], width=bar_width, label='TP', color='green')
    ax1.bar(x + bar_width/2, df['FP'], width=bar_width, label='FP', color='red')
    
    # x-axis ticks
    ax1.set_xticks(x)
    ax1.set_xticklabels(prefixes)
    
    # y-axis (counts)
    ax1.set_ylim(0, max_count)
    ax1.set_ylabel('Counts (TP / FP)')
    ax1.set_title(titles[i])
    
    # Dual y-axis for precision
    ax2 = ax1.twinx()
    ax2.plot(x, df['precision'], marker='o', color='blue', label='Precision', linewidth=2)
    ax2.set_ylim(0, 1.0)  # fixed precision scale
    ax2.set_ylabel('Precision', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Combine legends
    bars_labels = ax1.get_legend_handles_labels()
    line_labels = ax2.get_legend_handles_labels()
    ax1.legend(bars_labels[0] + line_labels[0], bars_labels[1] + line_labels[1], loc='upper right')

plt.tight_layout()
plt.savefig("compare_filter_set_absolute.jpg")

diff = pd.DataFrame({
    'prefix': final_mask_results['prefix'],
    'TP_diff': final_test_results['TP'] - final_mask_results['TP'],
    'FP_diff': final_test_results['FP'] - final_mask_results['FP'],
    'precision_diff': final_test_results['precision'] - final_mask_results['precision']
})

x = np.arange(len(diff['prefix']))
bar_width = 0.35

plt.figure(figsize=(10,6))
plt.bar(x - bar_width/2, diff['TP_diff'], width=bar_width, color='green', alpha=0.7, label='ΔTP')
plt.bar(x + bar_width/2, diff['FP_diff'], width=bar_width, color='red', alpha=0.7, label='ΔFP')
plt.axhline(0, color='black', linestyle=':', linewidth=1)
plt.xticks(x, diff['prefix'], rotation=45, ha='right')
plt.ylabel('Difference in Counts')
plt.title('Differences in TP / FP')
plt.legend()
plt.tight_layout()
plt.savefig("diff_TP_FP.jpg")

plt.figure(figsize=(10,6))
plt.bar(x, diff['precision_diff'], width=0.5, color='blue', alpha=0.7, label='ΔPrecision')
plt.axhline(0, color='black', linestyle=':', linewidth=1)
plt.xticks(x, diff['prefix'], rotation=45, ha='right')
plt.ylabel('Difference in Precision')
plt.title('Differences in Precision')
plt.legend()
plt.tight_layout()
plt.savefig("diff_precision.jpg")

