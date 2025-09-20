import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/user/christoph.wald/u15287/insect_pest_detection/3_2_image_processing_evaluation/data_labeled_training_set/filtering_tests_collected_results.csv")
species = ['BRAIIM', 'LIRIBO', 'TRIAVA']
new_labels = ['set1 ...', '+ contour size', '+ box size', '+ box ratio', '+ overlaps', '+ value', 'set2']  
df['Unnamed: 0'] = new_labels
# Colors correspond to curve type
curve_colors = {'p': 'blue', 'TP': 'green', 'FP': 'red'}

for sp in species:
    # Filter rows where any of the three values is not NaN
    mask = df[f'{sp}_p'].notna() | df[f'{sp}_TP'].notna() | df[f'{sp}_FP'].notna()
    df_filtered = df[mask]
    x = df_filtered['Unnamed: 0']

    fig, ax1 = plt.subplots(figsize=(4,5))

    # Left y-axis: Precision
    ax1.plot(x, df_filtered[f'{sp}_p'], marker='o', linestyle='--', markersize = 3, linewidth = 1, color=curve_colors['p'], label='Precision')
    ax1.set_ylabel('Precision', color=curve_colors['p'])
    ax1.tick_params(axis='y', labelcolor=curve_colors['p'])
    ax1.set_xticklabels(x, rotation=45, ha='right')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax1.set_ylim(0.7, 1.0)


    # Right y-axis: Absolute counts (TP and FP)
    ax2 = ax1.twinx()
    ax2.plot(x, df_filtered[f'{sp}_TP'], marker='o', linestyle='--', color=curve_colors['TP'], label='TP')
    ax2.plot(x, df_filtered[f'{sp}_FP'], marker='o', linestyle='--', color=curve_colors['FP'], label='FP')
    ax2.set_ylabel('Counts (TP / FP)')
    ax2.tick_params(axis='y')

    ax2.set_ylim(0, 4000)
    ax1.set_xlabel("filter")

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.title(f'{sp} - filter threshold test')
    plt.tight_layout()
    plt.savefig(f"filter_tests_{sp}.jpg")
    plt.close()
